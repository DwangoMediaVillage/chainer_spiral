from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from builtins import *  # NOQA
from future import standard_library
standard_library.install_aliases()  # NOQA

import copy
from logging import getLogger

import chainer
from chainer import functions as F
import numpy as np

from chainerrl import agent
from chainerrl.misc import async_
from chainerrl.misc.batch_states import batch_states
from chainerrl.misc import copy_param
from chainerrl.recurrent import Recurrent
from chainerrl.recurrent import RecurrentChainMixin
from chainerrl.recurrent import state_kept
from chainerrl.experiments.hooks import StepHook

import cv2

logger = getLogger(__name__)

class SPIRALModel(chainer.Link, RecurrentChainMixin):
    """ SPIRAL Model. """
    def pi_and_v(self, obs):
        """ evaluate the policy and the V-function """
        return NotImplementedError()
    
    def __call__(self, obs):
        return self.pi_and_v(obs)

class SpiralStepHook(StepHook):
    """ Ask the agent to compute reward at the current drawn picture """
    def __init__(self, timestep_limit):
        self.timestep_limit = timestep_limit
    
    def __call__(self, env, agent, step):
        if step % self.timestep_limit == 0:
            agent.compute_reward(env.render(mode='rgb_array'))

class SPIRAL(agent.AttributeSavingMixin, agent.Agent):
    """ SPIRAL: Synthesizing Programs for Images using Reinforced Adversarial Learning.

    See https://arxiv.org/abs/1804.01118

    """
    process_idx = None
    saved_attributes = ['generator', 'discriminator', 'gen_optimizer', 'dis_optimizer']

    def __init__(self, generator, discriminator,
                 gen_optimizer, dis_optimizer,
                 target_data_sampler,
                 in_channel,
                 timestep_limit,
                 rollout_n,
                 act_deterministically=False,
                 gamma=0.9,
                 beta=1e-2,
                 average_entropy_decay=0.999,
                 average_value_decay=0.999,
                 process_idx=0):

        # globally shared model
        self.shared_generator = generator
        self.shared_discriminator = discriminator

        # process specific model
        self.generator = copy.deepcopy(self.shared_generator)
        async_.assert_params_not_shared(self.shared_generator, self.generator)

        self.discriminator = copy.deepcopy(self.shared_discriminator)
        async_.assert_params_not_shared(self.shared_discriminator, self.discriminator)

        self.gen_optimizer = gen_optimizer
        self.dis_optimizer = dis_optimizer
        self.target_data_sampler = target_data_sampler
        self.in_channel = in_channel # image chanel of inputs to the model

        self.timestep_limit = timestep_limit  # time step length of each episode
        self.rollout_n = rollout_n

        self.act_deterministically = act_deterministically
        self.gamma = 0.9
        self.beta = beta

        self.average_value_decay = average_value_decay
        self.average_entropy_decay = average_entropy_decay

        self.t = 0  # time step counter
        self.n = 0  # episode counter

        # buffers to store hist during episodes
        self.past_action_log_prob = {}
        self.past_action_entropy = {}
        self.past_values = {}
        self.past_reward = {}
        self.past_dis_prob = {}
        self.target_data = None  # target picture being sampled by target_data_sampler

        # buffers for get_statistics
        self.stat_l2_loss = 0
        self.stat_average_value = 0
        self.stat_average_entropy = 0
        self.stat_discriminator_acc = 0

    def sync_parameters(self):
        copy_param.copy_param(target_link=self.generator,
                                source_link=self.shared_generator)
        copy_param.copy_param(target_link=self.discriminator,
                                source_link=self.shared_discriminator)

    @property
    def shared_attributes(self):
        return ('shared_generator', 'shared_discriminator', 'gen_optimizer', 'dis_optimizer')

    def __process_obs(self, obs):
        c = obs['image']
        x = obs['position']
        q = obs['prob']

        # image
        c = self.__process_image(c)

        # position
        x = np.asarray(x, dtype=np.float32)
        x = np.reshape(x, (1, 1))

        # prob
        q = np.asarray(q, dtype=np.float32)
        q = np.reshape(q, (1, 1))

        return c, x, q
    
    def __process_image(self, image):
        if self.in_channel == 1:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            image = np.expand_dims(image, -1)

        # normalize from [0, 255] to [0, 1]
        image = np.asarray(image, dtype=np.float32) / 255.0
        image = np.rollaxis(image, -1)
        image = np.expand_dims(image, 0)
        return  image
        
    def __pack_action(self, x, q):
        return {'position': x,
                'pressure': 1.0,
                'color': (0.0, 0.0, 0.0),
                'prob': q }

    def __store_buffer(self, buffer, t, x, concat):
        """ store x to the buffer. Replace or concat """

        if concat:
            buffer[t] = F.concat((buffer[t], x), axis=0)
        else:
            # replace with x
            buffer[t] = x

    def act_and_train(self, obs, r):
        """ Infer action from the observation at each step """

        # parse observation
        state = self.__process_obs(obs)

        # infer by the current policy
        pout, vout = self.generator.pi_and_v(state)

        # Sample actions as scalar values
        x, q = [ p.sample().data[0] for p in pout ]
        
        # t = self.t if (self.t + 1) // self.timestep_limit == 0 else self.t % self.timestep_limit
        if self.t // self.timestep_limit == 0:
            t = self.t
        else:
            t = self.t % self.timestep_limit
        
        if self.process_idx == 0:
            logger.debug("act_and_train at step %s, local_step %s", self.t, t)

        concat = self.t // self.timestep_limit > 0

        entropy = sum([ F.sum(p.entropy) for p in pout ])
        entropy = F.reshape(entropy, (1, 1))
        self.__store_buffer(self.past_action_entropy, t, entropy, concat)

        log_prob = sum([ p.log_prob(a) for p, a in zip(pout, (x, q)) ])
        log_prob = F.reshape(log_prob, (1, 1))
        self.__store_buffer(self.past_action_log_prob, t, log_prob, concat)

        vout = F.reshape(vout, (1, 1))
        self.__store_buffer(self.past_values, t, vout, concat)

        # update stats (average value and entropy )
        self.stat_average_value += (
            (1 - self.average_value_decay) * 
            (float(vout.data[0, 0]) - self.stat_average_value))
        self.stat_average_entropy += (
            (1 - self.average_entropy_decay) *
            (float(entropy.data[0, 0]) - self.stat_average_entropy))
        
        # create action dictionary to the env
        action = self.__pack_action(x, q)

        # update counters
        self.t += 1

        return action

    def compute_reward(self, image):
        """ compute the reward by the discriminator """
        c = self.__process_image(image)
        r = self.discriminator(c)

        # store reward to the buffer
        if self.process_idx == 0:
            logger.debug('compute reward = %s at local_step  %s', r, self.timestep_limit - 1)

        concat = self.t // self.timestep_limit > 1
        self.__store_buffer(self.past_dis_prob, self.timestep_limit - 1, r, concat)

        # sample an image from the dataset, and stores to the buf
        y = self.target_data_sampler()

        if concat:
            self.target_data = F.concat((self.target_data, y), axis=0)
        else:
            self.target_data = y

        # compute L2 loss between target data and drawn picture by the agent
        self.stat_l2_loss += F.mean_squared_error(c, y).data / float(self.rollout_n)

    def stop_episode_and_train(self, obs, r, done=None):
        state = self.__process_obs(obs)
        c, _, _ = state
        self.__update(c)

    def __update(self, c):
        """ update generator and discriminator at the end of drawing """
        R = self.past_dis_prob[self.timestep_limit - 1].data  # prob by the discriminator

        pi_loss = 0
        v_loss = 0

        if self.process_idx == 0:
            logger.debug('Accumulate grads t = %s -> 0', self.t)
        
        for t in reversed(range(self.timestep_limit)):
            R *= self.gamma  # discout factor
            v = self.past_values[t]
            advantage = R - v
            
            # Accumulate gradients of policy
            log_prob = self.past_action_log_prob[t]
            entropy = self.past_action_entropy[t]

            pi_loss -= log_prob * np.asarray(advantage.data)
            pi_loss -= self.beta * entropy

            v_loss += (v - R) ** 2 / 2

        if self.process_idx == 0:
            logger.debug('pi_loss:%s v_loss:%s', pi_loss.data, v_loss.data)

        total_loss = pi_loss + F.reshape(v_loss, pi_loss.data.shape)
        total_loss = F.mean(total_loss, axis=0)  # take mean along batch axis

        # compute gradients of the generator
        self.generator.zerograds()
        total_loss.backward()

        # compute gradients of the discriminator

        # TODO: add spectrum normalization
        dis_prob = self.past_dis_prob[self.timestep_limit - 1]

        loss_dis = F.sum(F.softplus(dis_prob)) + F.sum(F.softplus(-self.discriminator(self.target_data)))
        self.discriminator.zerograds()
        loss_dis.backward()

        # copy the gradients of the local generator to the globally shared model
        self.shared_generator.zerograds()
        copy_param.copy_grad(
            target_link=self.shared_generator, source_link=self.generator)
        
        # copy the gradients of the local discriminator to the globall shared model
        self.shared_discriminator.zerograds()
        copy_param.copy_grad(
            target_link=self.shared_discriminator, source_link=self.discriminator)

        # update the globally shared model
        if self.process_idx == 0:
            logger.debug('Update')

        # Perform asynchronous update
        self.gen_optimizer.update()
        self.dis_optimizer.update()

        self.sync_parameters()
        self.generator.unchain_backward()

        # reset the buffers
        self.past_action_log_prob = {}
        self.past_action_entropy = {}
        self.past_values = {}
        self.past_reward = {}
        self.past_dis_prob = {}
        self.target_data = None

        # reset time step cout
        self.t = 0

    def get_statistics(self):

        # returns statistics after updating. reset stat_l2_loss
        stat_l2_loss = copy.deepcopy(self.stat_l2_loss)
        self.stat_l2_loss = 0.0

        return [
            ('average_value', self.stat_average_value),
            ('average_entropy', self.stat_average_entropy),
            ('l2_loss', stat_l2_loss)
        ]

    def stop_episode(self):
        """ spiral model is a recurrent model """
        if self.process_idx == 0:
            logger.debug('stop_episode: reset state')
        self.generator.reset_state()

    def act(self, obs):
        with chainer.no_backprop_mode():
            state = self.__process_obs(obs)
            pout, _ = self.generator.pi_and_v(state)
            if self.act_deterministically:
                x, q = [ np.argmax(p.log_p.data, axis=1)[0] for p in pout ]
            else:
                x, q = [ p.sample().data[0] for p in pout ]
            
            return self.__pack_action(x, q)

    def load(self, dirname):
        logger.debug('Load parameters from %s', dirname)
        super().load(dirname)
        copy_param.copy_param(target_link=self.shared_generator,
                                source_link=self.generator)
        copy_param.copy_param(target_link=self.shared_discriminator,
                                source_link=self.discriminator)

    