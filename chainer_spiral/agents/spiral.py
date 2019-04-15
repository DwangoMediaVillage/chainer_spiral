from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import copy
import os
from builtins import *  # NOQA
from logging import getLogger

import chainer
import numpy as np
from chainer import functions as F
from chainerrl import agent
from chainerrl.experiments.hooks import StepHook
from chainerrl.misc import async_, copy_param
from future import standard_library

from chainer_spiral.agents.agent_utils import (ObservationSaver,
                                               compute_auxiliary_reward,
                                               pack_action, preprocess_image,
                                               preprocess_obs)

standard_library.install_aliases()  # NOQA

logger = getLogger(__name__)


class SpiralStepHook(StepHook):
    """ Ask the agent to compute reward at the current drawn picture """

    def __init__(self, max_episode_steps, save_global_step_interval, outdir):
        self.max_episode_steps = max_episode_steps
        self.save_global_step_interval = save_global_step_interval
        self.outdir = outdir

    def __call__(self, env, agent, step):
        # agent.compute_reward is called for each agent
        if agent.t % self.max_episode_steps == 0:
            agent.compute_reward(env.render(mode='rgb_array'))
            env.reset()

        # agent.snap is called once
        if step % self.save_global_step_interval == 0:
            agent.snap(step, self.outdir)


def np_softplus(x):
    return np.maximum(0, x) + np.log(1 + np.exp(-np.abs(-x)))


class SPIRAL(agent.AttributeSavingMixin, agent.Agent):
    """ SPIRAL: Synthesizing Programs for Images using Reinforced Adversarial Learning.

    See https://arxiv.org/abs/1804.01118

    Args:
        generator (SPIRALModel): Generator
        discriminator (chainer.Chain): Discriminator
        gen_optimizer (chainer.Optimizer): optimizer to train generator
        dis_optimizer (chainer.Optimizer): optimizer to train discriminator
        dataset (chainer.dataset.DatasetMixin): dataset to feed a batch data to this agent
        conditional (bool): IF true, the models are assumed to generate / discriminate images with conditional input
        reward_mode (string): method to compute a reward at the end of drawing. 'l2', 'dcgan', or 'wgangp'.
        imsize (int): size of drawn picture image
        max_episode_steps (int): time step length of each drawing process
        rollout_n (int): number of times to rollout the generation process before updating
        gamma (float): discount factor [0, 1]
        beta (float): weight coefficient for the entropy regularization term
        gp_lambda (float): scaling factor of the gradient penalty for WGAN-GP
        lambda_R (float): Scaling parameter of rewards by discriminator
        staying_penalty (float): auxiliary reward to penalize staying at the same position
        empty_drawing_penalty (float): auxiliary reward to penalize drawing nothing
        n_save_final_obs_interval (int): interval to take snapshot of observation
        outdir (str): path to save final observation snapshots
        act_deterministically (bool): If set true, the agent chooses the most probable actions in act()
        average_entropy_decay (float): decay to compute moving average of entropy
        average_value_decay (float): decay to compute moving average of value
        process_idx (int): Index of the process
        pi_loss_coef (float): scaling factor of the loss for policy network
        v_loss_coef (float): scaling factor of the loss for value network
    """

    process_idx = None
    saved_attributes = [
        'generator', 'discriminator', 'gen_optimizer', 'dis_optimizer'
    ]

    def __init__(self,
                 generator,
                 discriminator,
                 gen_optimizer,
                 dis_optimizer,
                 dataset,
                 conditional,
                 reward_mode,
                 imsize,
                 max_episode_steps,
                 rollout_n,
                 gamma,
                 beta,
                 gp_lambda,
                 lambda_R,
                 staying_penalty,
                 empty_drawing_penalty,
                 n_save_final_obs_interval,
                 outdir,
                 act_deterministically=False,
                 average_entropy_decay=0.999,
                 average_value_decay=0.999,
                 process_idx=0,
                 pi_loss_coef=1.0,
                 v_loss_coef=1.0):

        # globally shared model
        self.shared_generator = generator
        self.shared_discriminator = discriminator

        # process specific model
        self.generator = copy.deepcopy(self.shared_generator)
        async_.assert_params_not_shared(self.shared_generator, self.generator)

        self.discriminator = copy.deepcopy(self.shared_discriminator)
        async_.assert_params_not_shared(self.shared_discriminator,
                                        self.discriminator)

        self.gen_optimizer = gen_optimizer
        self.dis_optimizer = dis_optimizer
        self.dataset = dataset
        self.conditional = conditional

        assert reward_mode in ('l2', 'dcgan', 'wgangp')
        self.reward_mode = reward_mode

        self.imsize = imsize
        self.max_episode_steps = max_episode_steps
        self.rollout_n = rollout_n
        self.gamma = gamma
        self.beta = beta
        self.gp_lambda = gp_lambda
        self.lambda_R = lambda_R
        self.staying_penalty = staying_penalty
        self.empty_drawing_penalty = empty_drawing_penalty
        self.n_save_final_obs_interval = n_save_final_obs_interval
        self.outdir = outdir
        self.act_deterministically = act_deterministically
        self.average_entropy_decay = average_entropy_decay
        self.average_value_decay = average_value_decay
        self.pi_loss_coef = pi_loss_coef
        self.v_loss_coef = v_loss_coef

        self.observation_saver = ObservationSaver(self.outdir, self.rollout_n,
                                                  self.imsize)

        # initialize stat
        self.stat_average_value = 0.0
        self.stat_average_entropy = 0.0
        self.update_n = 0  # number of updates

        self.__reset_flags()
        self.__reset_buffers()
        self.__reset_stats()

    def sync_parameters(self):
        copy_param.copy_param(target_link=self.generator,
                              source_link=self.shared_generator)
        copy_param.copy_param(target_link=self.discriminator,
                              source_link=self.shared_discriminator)

    @property
    def shared_attributes(self):
        return ('shared_generator', 'shared_discriminator', 'gen_optimizer',
                'dis_optimizer')

    def __reset_flags(self):
        self.t = 0  # time step counter

    def __reset_buffers(self):
        """ reset internal buffers """
        # buffers to store hist during episodes
        self.past_reward = np.zeros((self.rollout_n, self.max_episode_steps))

        # sampled action to determine drawing or not from the previous point
        self.past_actions = {}
        self.past_action_entropy = {}
        self.past_action_log_prob = {}
        self.past_values = {}
        self.real_data = {}
        self.past_R = {}
        self.fake_data = {}
        self.y_fake = {}
        self.last_x = None
        if self.conditional:
            self.past_conditional_input = {}

    def __reset_stats(self):
        """ reset interval buffers for statistics """
        # buffers for get_statistics
        self.stat_l2_loss = 0
        self.stat_pi_loss = None
        self.stat_v_loss = None
        self.stat_R = None
        self.stat_reward_min = None
        self.stat_reward_max = None
        self.stat_reward_mean = None
        self.stat_reward_std = None
        self.stat_loss_dis = None

        if self.reward_mode == 'wgangp':
            self.stat_loss_gp = None
            self.stat_dis_g = None
        elif self.reward_mode == 'dcgan':
            self.stat_dis_acc = None

    def __get_local_time(self):
        n = self.t // self.max_episode_steps
        if n == 0:
            t = self.t
        else:
            t = self.t % self.max_episode_steps
        return n, t

    def act_and_train(self, obs, r):
        """ Infer action from the observation at each step, and compute auxiliary reward """
        # get local time step at each episode: step t of n-th rollout
        n, t = self.__get_local_time()

        # set the conditional input data
        if self.conditional and t == 0:
            self.past_conditional_input[n] = self.dataset.get_example()

        # preprocess observation
        state = preprocess_obs(obs, self.imsize)

        # get probabilities, sampled actions, and value from the generator
        if self.conditional:
            pout, vout = self.generator.pi_and_v(
                state, self.past_conditional_input[n])
        else:
            pout, vout = self.generator.pi_and_v(state)

        prob, act = pout

        # put inferences to the buffer
        self.past_action_entropy[n, t] = sum([p.entropy for p in prob])
        self.past_action_log_prob[n, t] = sum(
            [p.log_prob(a) for p, a in zip(prob, act)])
        self.past_values[n, t] = vout

        # update stats (moving average of value and entropy)
        self.stat_average_value += (
            (1 - self.average_value_decay) *
            (float(vout.data[0, 0]) - self.stat_average_value))
        self.stat_average_entropy += (
            (1 - self.average_entropy_decay) *
            (float(self.past_action_entropy[n, t].data) -
             self.stat_average_entropy))

        # update counter
        self.t += 1

        # create action dictionary to the env
        act = pack_action(act)
        self.past_actions[n, t] = act

        if self.process_idx == 0:
            logger.debug(
                'act_and_train at step %s, local step %s, local episode %s',
                self.t, t, n)
            logger.debug('taking action %s', act)

        return act

    def compute_reward(self, image):
        """ compute the reward by the discriminator at the end of drawing """
        # store fake data and a paired target data sampled from the dataset
        n = (self.t - 1) // self.max_episode_steps  # number of local episode
        self.fake_data[n] = preprocess_image(image)

        if self.conditional:
            self.real_data[n] = self.past_conditional_input[n]
        else:
            self.real_data[n] = self.dataset.get_example()

        # compute L2 loss between target data and drawn picture by the agent
        l2_loss = F.mean_squared_error(
            self.fake_data[n], self.real_data[n]).data / float(self.rollout_n)
        if n == 0:
            self.stat_l2_loss = l2_loss
        else:
            self.stat_l2_loss += l2_loss

        # compute reward after finishing drawing
        if self.reward_mode == 'l2':
            R = -l2_loss
        else:
            conditional_input = self.past_conditional_input[
                n] if self.conditional else None
            if self.reward_mode == 'dcgan':
                y_fake = self.discriminator(self.fake_data[n],
                                            conditional_input)
                R = np_softplus(y_fake.data).data[0, 0]
                self.y_fake[n] = y_fake
            elif self.reward_mode == 'wgangp':
                y_fake = self.discriminator(self.fake_data[n],
                                            conditional_input)
                R = y_fake.data[0, 0]
                self.y_fake[n] = y_fake
            else:
                raise NotImplementedError()

        # store reward to the buffer
        if self.process_idx == 0:
            logger.debug('compute final reward = %s at local_episode %s', R, n)

        self.past_R[n] = R

        # compute auxiliary reward at the end of drawing process
        self.past_reward = compute_auxiliary_reward(self.past_reward,
                                                    self.past_actions, n,
                                                    self.max_episode_steps,
                                                    self.staying_penalty,
                                                    self.empty_drawing_penalty)

        # reset LSTM states
        self.generator.reset_state()

    def stop_episode_and_train(self, obs, r, done=None):
        if self.process_idx == 0:
            # get local time step at each episode
            n, t = self.__get_local_time()
            logger.debug('update at local episode %s, local step %s', n, t)
        self.__update()

        # saving the final observation images as png
        if self.process_idx == 0 and self.update_n % self.n_save_final_obs_interval == 0:
            self.observation_saver.save(self.fake_data, self.real_data,
                                        self.update_n)

        self.__reset_buffers()
        self.__reset_flags()

    def __update(self):
        """ update generator and discriminator at the end of drawing """
        if self.process_idx == 0:
            logger.debug('Accumulate grads')

        pi_loss = 0
        v_loss = 0

        for n in reversed(range(self.rollout_n)):
            R = self.lambda_R * self.past_R[n]  # prob by the discriminator

            for t in reversed(range(self.max_episode_steps)):
                R *= self.gamma  # discount factor
                R += self.past_reward[n, t]
                v = self.past_values[n, t]
                advantage = R - v

                log_prob = self.past_action_log_prob[n, t]
                entropy = self.past_action_entropy[n, t]

                pi_loss -= log_prob * float(advantage.data)
                pi_loss -= self.beta * entropy

                v_loss += (v - R)**2 / 2

        if self.pi_loss_coef != 1.0:
            pi_loss *= self.pi_loss_coef
        if self.v_loss_coef != 1.0:
            v_loss *= self.v_loss_coef

        # normalize by each step
        pi_loss /= self.max_episode_steps * self.rollout_n
        v_loss /= self.max_episode_steps * self.rollout_n

        total_loss = pi_loss + F.reshape(v_loss, pi_loss.data.shape)

        if self.process_idx == 0:
            logger.debug('pi_loss:%s v_loss:%s', pi_loss.data, v_loss.data)

        # compute gradients of the generator
        self.generator.zerograds()
        total_loss.backward()

        # copy the gradients of the local generator to the globally shared model
        self.shared_generator.zerograds()
        copy_param.copy_grad(target_link=self.shared_generator,
                             source_link=self.generator)

        # update the gobally shared model
        if self.process_idx == 0:
            norm = sum(
                np.sum(np.square(param.grad))
                for param in self.gen_optimizer.target.params())
            logger.debug('grad_norm of generator: %s', norm)
        self.gen_optimizer.update()

        # update the local discriminator
        if self.reward_mode in ('dcgan', 'wgangp'):
            x_fake = F.concat(self.fake_data.values(), axis=0)
            x_real = F.concat(self.real_data.values(), axis=0)
            y_fake = F.concat(self.y_fake.values())

            if self.conditional:
                y_real = self.discriminator(x_real, x_real)
            else:
                y_real = self.discriminator(x_real)

            self.__compute_discriminator_grad(x_real, x_fake, y_real, y_fake)

            # copy the gradients of the local discriminator to the globall shared model
            self.shared_discriminator.zerograds()
            copy_param.copy_grad(target_link=self.shared_discriminator,
                                 source_link=self.discriminator)

            # Perform asynchronous update
            self.dis_optimizer.update()

        self.sync_parameters()
        self.generator.unchain_backward()

        # update statistics
        self.stat_pi_loss = float(pi_loss.data)
        self.stat_v_loss = float(v_loss.data)
        self.stat_R = np.array(list(self.past_R.values())).mean()
        self.stat_reward_min = self.past_reward.min()
        self.stat_reward_max = self.past_reward.max()
        self.stat_reward_mean = self.past_reward.mean()
        self.stat_reward_std = self.past_reward.std()

        # update counter
        self.update_n += 1

    def __compute_discriminator_grad(self, x_real, x_fake, y_real, y_fake):
        """ Compute the loss of discriminator """
        if self.reward_mode == 'wgangp':
            # WGAN-GP with 1 step wasserstein distance sampling
            loss_dis = -F.sum(y_real) / self.rollout_n + F.sum(
                y_fake) / self.rollout_n

            # add gradient panalty to the loss
            eps = np.random.uniform(0, 1,
                                    size=self.rollout_n).astype(np.float32)
            eps = np.reshape(eps, (self.rollout_n, 1, 1, 1))
            x_mid = eps * x_real + (1.0 - eps) * x_fake

            x_mid_v = chainer.Variable(x_mid.data)
            if self.conditional:
                y_mid = self.discriminator(x_mid_v, x_mid_v)
            else:
                y_mid = self.discriminator(x_mid_v)

            dydx = self.discriminator.differentiable_backward(
                np.ones_like(y_mid.data))
            dydx = F.sqrt(F.sum(dydx**2, axis=(1, 2, 3)))
            loss_gp = self.gp_lambda * F.mean_squared_error(
                dydx, np.ones_like(dydx.data))

            # update statistics
            self.stat_loss_dis = float(loss_dis.data)
            self.stat_loss_gp = float(loss_gp.data)
            self.stat_dis_g = float(F.mean(dydx).data)

            # compute grads of the local model
            self.discriminator.zerograds()
            loss_dis.backward()
            loss_gp.backward()

        elif self.reward_mode == 'dcgan':
            # DCGAN
            loss_dis = F.sum(F.softplus(-y_real)) / self.rollout_n + F.sum(
                F.softplus(y_fake)) / self.rollout_n

            # update statistics
            tp = (y_real.data > 0.5).sum()
            fp = (y_fake.data > 0.5).sum()
            fn = (y_real.data <= 0.5).sum()
            tn = (y_fake.data <= 0.5).sum()

            self.stat_loss_dis = float(loss_dis.data)
            self.stat_dis_acc = (tp + tn) / (tp + fp + fn + tn)

            # compute grads of the local model
            self.discriminator.zerograds()
            loss_dis.backward()
        else:
            raise NotImplementedError()

    def get_statistics(self):
        # returns statistics after updating and reset stat_l2_loss

        ret = [('average_value', self.stat_average_value),
               ('average_entropy', self.stat_average_entropy),
               ('l2_loss', self.stat_l2_loss), ('pi_loss', self.stat_pi_loss),
               ('v_loss', self.stat_v_loss), ('R', self.stat_R),
               ('reward_min', self.stat_reward_min),
               ('reward_mean', self.stat_reward_mean),
               ('reward_max', self.stat_reward_max),
               ('reward_std', self.stat_reward_std),
               ('update_n', self.update_n)]

        if self.reward_mode == 'wgangp':
            ret += [('discriminator_grad_panalty', self.stat_loss_gp),
                    ('discriminator_gradient_size', self.stat_dis_g),
                    ('discriminator_loss', self.stat_loss_dis)]
        elif self.reward_mode == 'dcgan':
            ret += [('discriminator_accuracy', self.stat_dis_acc),
                    ('discriminator_loss', self.stat_loss_dis)]

        return ret

    def stop_episode(self):
        """ spiral model is a recurrent model """
        if self.process_idx == 0:
            logger.debug('stop_episode: reset state')
        self.generator.reset_state()

    def act(self, obs, conditional_input=None):
        with chainer.no_backprop_mode():
            state = preprocess_obs(obs, self.imsize)

            if self.conditional:
                if conditional_input is None:
                    conditional_input = self.dataset.get_example(train=False)
                pout, _ = self.generator.pi_and_v(state, conditional_input)
            else:
                pout, _ = self.generator.pi_and_v(state)

            prob, act = pout

            if self.act_deterministically:
                act = [np.argmax(p.log_p.data, axis=1)[0] for p in prob]

            return pack_action(act)

    def load(self, dirname):
        logger.debug('Load parameters from %s', dirname)
        super().load(dirname)
        copy_param.copy_param(target_link=self.shared_generator,
                              source_link=self.generator)
        copy_param.copy_param(target_link=self.shared_discriminator,
                              source_link=self.discriminator)

    def snap(self, step, outdir):
        dirname = os.path.join(outdir, '{}'.format(step))
        self.save(dirname)
        logger.info('Taking snapshot at global step %s to %s', step, dirname)
