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

import cv2

logger = getLogger(__name__)

class SPIRALModel(chainer.Link, RecurrentChainMixin):
    """ SPIRAL Model. """
    def pi_and_v(self, obs):
        """ evaluate the policy and the V-function """
        return NotImplementedError()
    
    def __call__(self, obs):
        return self.pi_and_v(obs)

class SPIRAL(agent.AttributeSavingMixin, agent.Agent):
    """ SPIRAL: Synthesizing Programs for Images using Reinforced Adversarial Learning.

    See https://arxiv.org/abs/1804.01118

    """
    saved_attributes = ['generator', 'discriminator', 'gen_optimizer', 'dis_optimizer']

    def __init__(self, generator, discriminator,
                 gen_optimizer, dis_optimizer,
                 target_data_sampler,
                 in_channel,
                 act_deterministically=False,
                 gamma=0.9,
                 beta=1e-2,
                 average_entropy_decay=0.999,
                 average_value_decay=0.999):
        self.generator = generator
        self.discriminator = discriminator
        self.gen_optimizer = gen_optimizer
        self.dis_optimizer = dis_optimizer
        self.target_data_sampler = target_data_sampler
        self.in_channel = in_channel # image chanel of inputs to the model

        self.act_deterministically = act_deterministically
        self.gamma = 0.9
        self.beta = beta

        self.average_value_decay = average_value_decay
        self.average_entropy_decay = average_entropy_decay

        self.t = 0  # time step counter

        # buffers to store hist during episodes
        self.past_action_log_prob = {}
        self.past_action_entropy = {}
        self.past_values = {}

        # buffers for get_statistics
        self.stat_l2_loss = None
        self.stat_average_value = 0
        self.stat_average_entropy = 0

    def __process_obs(self, obs):
        c = obs['image']
        x = obs['position']
        q = obs['prob']

        # image
        if self.in_channel == 1:
            c = cv2.cvtColor(c, cv2.COLOR_RGB2GRAY)
            c = np.expand_dims(c, -1)

        # normalize from [0, 255] to [0, 1]
        c = np.asarray(c, dtype=np.float32) / 255.0
        c = np.rollaxis(c, -1)
        c = np.expand_dims(c, 0)

        # position
        x = np.asarray(x, dtype=np.float32)
        x = np.reshape(x, (1, 1))

        # prob
        q = np.asarray(q, dtype=np.float32)
        q = np.reshape(q, (1, 1))

        return c, x, q

    def __pack_action(self, x, q):
        return {'position': x,
                'pressure': 1.0,
                'color': (0.0, 0.0, 0.0),
                'prob': q }

    def act_and_train(self, obs, r):
        """ Infer action from the observation at each step """

        # parse observation
        state = self.__process_obs(obs)

        # infer by the current policy
        pout, vout = self.generator.pi_and_v(state)

        # Sample actions as scalar values
        x, q = [ p.sample().data[0] for p in pout ]
        
        # store to the buffers
        self.past_action_entropy[self.t] = sum([ F.sum(p.entropy) for p in pout ])
        self.past_action_log_prob[self.t] = sum([ p.log_prob(a) for p, a in zip(pout, (x, q)) ])
        self.past_values[self.t] = vout

        # update stats (average value and entropy )
        
        self.stat_average_value += (
            (1 - self.average_value_decay) * 
            (float(self.past_values[self.t].data[0]) - self.stat_average_value))
        self.stat_average_entropy += (
            (1 - self.average_entropy_decay) *
            (float(self.past_action_entropy[self.t].data) - self.stat_average_entropy))
        action = self.__pack_action(x, q)
        self.t += 1
        return action

    def stop_episode_and_train(self, obs, r, done=None):
        state = self.__process_obs(obs)
        c, _, _ = state
        self.__update(c, self.discriminator(c))
        self.generator.reset_state()

    def __update(self, c, dis_out):
        """ update generator and discriminator at the end of drawing """
        R = np.asscalar(dis_out.data)

        pi_loss = 0
        v_loss = 0

        logger.debug('Accumulate grads t = %s -> 0', self.t)
        for t in reversed(range(self.t)):
            R *= self.gamma  # discout factor
            v = self.past_values[t]
            advantage = R - v
            
            # Accumulate gradients of policy
            log_prob = self.past_action_log_prob[t]
            entropy = self.past_action_entropy[t]

            pi_loss -= log_prob * float(advantage.data)
            pi_loss -= self.beta * entropy

            v_loss += (v - R) ** 2 / 2

        logger.debug('pi_loss:%s v_loss:%s', pi_loss.data, v_loss.data)

        total_loss = pi_loss + F.reshape(v_loss, pi_loss.data.shape)

        # compute gradients
        self.generator.zerograds()
        total_loss.backward()
        
        # TODO: copy local grad to the global model
        self.gen_optimizer.update()

        self.past_action_log_prob = {}
        self.past_action_entropy = {}
        self.past_values = {}

        # update discriminator

        # sample y from the target images
        y = self.target_data_sampler()

        # TODO: add spectrum normalization
        loss_dis = F.sum(F.softplus(dis_out)) + F.sum(F.softplus(-self.discriminator(y)))

        # evaluate L2 loss between drawn picture and y
        self.stat_l2_loss = float(F.mean_squared_error(c, y).data)

        self.discriminator.zerograds()
        loss_dis.backward()
        self.dis_optimizer.update()

        # reset time step cout
        self.t = 0

    def get_statistics(self):
        return [
            ('average_value', self.stat_average_value),
            ('average_entropy', self.stat_average_entropy),
            ('l2_loss', self.stat_l2_loss)
        ]

    def stop_episode(self):
        """ spiral model is a recurrent model """
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

    