from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from builtins import *  # NOQA
from future import standard_library
standard_library.install_aliases()  # NOQA

import os
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

import matplotlib; matplotlib.use('Cairo')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

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
    def __init__(self, timestep_limit, save_global_step_interval, outdir):
        self.timestep_limit = timestep_limit
        self.save_global_step_interval = save_global_step_interval
        self.outdir = outdir

    def __call__(self, env, agent, step):
        # agent.compute_reward is called for each agent
        if agent.t % self.timestep_limit == 0:
            agent.compute_reward(env.render(mode='rgb_array'))
            env.reset()

        # agent.snap is called once
        if step % self.save_global_step_interval == 0:
            agent.snap(step, self.outdir)

          
def np_softplus(x):
    return np.maximum(0, x) + np.log(1 + np.exp(-np.abs(-x)))


class ImageDrawer(object):
    def __init__(self, n, imsize=64):
        self.fig = plt.figure(figsize=(7, 7))
        gs = gridspec.GridSpec(n, 2)
        self.ims_real = []
        self.ims_fake = []
        self.n = n
        for i in range(self.n):
            # target data (fake data)
            ax = plt.subplot(gs[i, 0])
            im = ax.imshow(np.zeros((imsize, imsize)), vmin=0, vmax=1, cmap='gray')
            ax.set_xticks([])
            ax.set_yticks([])
            self.ims_fake.append(im)
            if i == 0:
                ax.set_title('Fake data')

            # generated data (real data)
            ax = plt.subplot(gs[i, 1])
            im = ax.imshow(np.zeros((imsize, imsize)), vmin=0, vmax=1, cmap='gray')
            ax.set_xticks([])
            ax.set_yticks([])
            self.ims_real.append(im)
            if i == 0:
                ax.set_title('Real data')

    def draw_and_save(self, fake_data, real_data, figname, update):
        for i in range(self.n):
            self.ims_fake[i].set_data(fake_data[i][0, 0])
            self.ims_real[i].set_data(real_data[i][0, 0].data)
        self.fig.suptitle(f"Update = {update}")
        plt.savefig(figname)

class SPIRAL(agent.AttributeSavingMixin, agent.Agent):
    """ SPIRAL: Synthesizing Programs for Images using Reinforced Adversarial Learning.

    See https://arxiv.org/abs/1804.01118

    Args:
        generator (SPIRALModel): Generator
        discriminator (chainer.Chain): Discriminator
        gen_optimizer (chainer.Optimizer): optimizer to train generator
        dis_optimizer (chainer.Optimizer): optimizer to train discriminator
        target_data_sampler (func): function to feed a batch
        in_chanel (int): channel of images
        timestep_limit (int): time step length of each drawing process
        rollout_n (int): number of times to rollout the generation process before updating
        obs_pos_dim (int): dimensional size of positional action vector
        conditional (bool): IF true, the models are assumed to generate / discriminate images with conditional input
        act_deterministically (bool): If set true, the agent chooses the most probable actions in act()
        gamma (float): discount factor [0, 1]
        beta (float): weight coefficient for the entropy regularization term
        process_idx (int): Index of the process
        gp_lambda (float): scaling factor of the gradient penalty for WGAN-GP
        continuous_drawing_lambda (float): scaling factor of additional reward to encourage continuous drawing
        empty_drawing_penalty (float): size of negative reward for drawing nothing
        use_wgangp (bool): If true, the discriminator is trained as WGAN-GP
        lambda_R (float): Scaling parameter of rewards by discriminator
        reward_mode (string): method to compute a reward at the end of drawing
        save_final_obs_update_interval (int): number of weight update interval to save the final observation images as a png image.
        outdir (string): the directory to save the final observation images.
        save_final_obs (bool): If true, agent saves the final observation images
        staying_penalty (float): positive value to represent size of the negative reward to penalize staying at the same point
    """
    
    process_idx = None
    saved_attributes = ['generator', 'discriminator', 'gen_optimizer', 'dis_optimizer']

    def __init__(self, generator, discriminator,
                 gen_optimizer, dis_optimizer,
                 target_data_sampler,
                 in_channel,
                 timestep_limit,
                 rollout_n,
                 obs_pos_dim,
                 conditional,
                 act_deterministically=False,
                 gamma=0.9,
                 beta=1e-2,
                 average_entropy_decay=0.999,
                 average_value_decay=0.999,
                 process_idx=0,
                 gp_lambda=10.0,
                 continuous_drawing_lambda=1.0,
                 empty_drawing_penalty=1.0,
                 pi_loss_coef=1.0,
                 v_loss_coef=1.0,
                 lambda_R=1.0,
                 reward_mode='l2',
                 save_final_obs_update_interval=10000,
                 outdir=None,
                 save_final_obs=False,
                 staying_penalty=0.0):

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
        self.obs_pos_dim = obs_pos_dim  # dim size of the position input
        self.conditional = conditional

        self.act_deterministically = act_deterministically
        self.gamma = gamma
        self.beta = beta

        self.gp_lamda = gp_lambda

        self.pi_loss_coef = pi_loss_coef
        self.v_loss_coef = v_loss_coef

        self.lambda_R = lambda_R

        self.continuous_drawing_lambda = continuous_drawing_lambda
        
        assert empty_drawing_penalty >= 0
        self.empty_drawing_penalty = empty_drawing_penalty

        assert staying_penalty >= 0
        self.staying_penalty = staying_penalty

        self.average_value_decay = average_value_decay
        self.average_entropy_decay = average_entropy_decay
        self.stat_average_value = 0.0
        self.stat_average_entropy = 0.0

        assert reward_mode in ('l2', 'dcgan', 'wgangp')
        self.reward_mode = reward_mode

        self.__reset_flags()
        self.__reset_buffers()
        self.__reset_stats()

        # initialize drawer to save the final observation
        self.save_final_obs = save_final_obs
        if self.save_final_obs:
            self.image_drawer = ImageDrawer(self.rollout_n)
            self.save_final_obs_update_interval = save_final_obs_update_interval
            self.outdir_final_obs = os.path.join(outdir, 'final_obs')  # directory to save the final observation
            if not os.path.exists(self.outdir_final_obs):
                os.mkdir(self.outdir_final_obs)


    def sync_parameters(self):
        copy_param.copy_param(target_link=self.generator,
                                source_link=self.shared_generator)
        copy_param.copy_param(target_link=self.discriminator,
                                source_link=self.shared_discriminator)

    @property
    def shared_attributes(self):
        return ('shared_generator', 'shared_discriminator', 'gen_optimizer', 'dis_optimizer')

    def __reset_flags(self):
        self.t = 0  # time step counter
        self.continuous_drawing_step = 0

    def __reset_buffers(self):
        """ reset internal buffers """
        # buffers to store hist during episodes
        self.past_reward = np.zeros((self.rollout_n, self.timestep_limit))

        # sampled action to determine drawing or not from the previous point
        self.past_brush_prob = {}
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

        self.update_n = 0   # number of updates

    def process_obs(self, obs):
        c = obs['image']
        x = obs['position']
        q = obs['prob']

        # image
        c = self.__process_image(c)

        # position
        x /= float(self.obs_pos_dim)
        x = np.asarray(x, dtype=np.float32) 
        x = np.reshape(x, (1, 1))

        # prob
        q = np.asarray(q, dtype=np.float32)
        q = np.reshape(q, (1, 1))

        return c, x, q
    
    def __process_image(self, image):
        if self.in_channel == 1:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)  # unit8, 2d matrix
            image = image.astype(np.float32) / 255.0
            image = np.reshape(image, [1, 1] + list(image.shape))
        else:
            raise NotImplementedError("not implemented in case of channel != 1")
        return  image
    
    def pack_action(self, x, q):
        return {'position': x,
                'pressure': 1.0,
                'color': (0.0, 0.0, 0.0),
                'prob': q }

    def __get_local_time(self):
        n = self.t // self.timestep_limit
        if n  == 0:
            t = self.t
        else:
            t = self.t % self.timestep_limit
        return n, t


    def act_and_train(self, obs, r):
        """ Infer action from the observation at each step """

        # get local time step at each episode
        n, t = self.__get_local_time()

        if self.conditional and t == 0:
            # set the conditional input data
            self.past_conditional_input[n] = self.target_data_sampler()

        # parse observation
        state = self.process_obs(obs)

        # infer by the current policy
        # a1 is position, a2 is the prob to draw line or not
        if self.conditional:
            pout, vout = self.generator.pi_and_v(state, self.past_conditional_input[n])
        else:
            # unconditional generation
            pout, vout = self.generator.pi_and_v(state)

        p1, p2, a1, a2 = pout

        if self.process_idx == 0:
            logger.debug("act_and_train at step %s, local_step %s, local_episode %s", self.t, t, n)

        # calc additional reward during drawing process
        if t > 0:
            if self.past_brush_prob[n, t - 1] and int(a2.data):
                self.continuous_drawing_step += 1
            else:
                self.continuous_drawing_step = 0

        self.past_brush_prob[n, t] = int(a2.data)

        if self.continuous_drawing_step > 0:
            continuous_reward = self.continuous_drawing_lambda * 1.0 / self.continuous_drawing_step
            self.past_reward[n, t] = continuous_reward
        
        entropy = sum([ p.entropy for p in (p1, p2) ])
        self.past_action_entropy[n, t] = entropy

        log_prob = sum([ p.log_prob(a) for p, a in zip((p1, p2), (a1, a2))])
        self.past_action_log_prob[n, t] = log_prob
        
        self.past_values[n, t] = vout

        # update stats (average value and entropy )
        self.stat_average_value += (
            (1 - self.average_value_decay) * 
            (float(vout.data[0, 0]) - self.stat_average_value))
        self.stat_average_entropy += (
            (1 - self.average_entropy_decay) *
            (float(entropy.data) - self.stat_average_entropy))
        
        # create action dictionary to the env
        action = self.pack_action(int(a1.data), int(a2.data))

        if self.process_idx == 0:
            logger.debug("Taking action %s", action)

        # check the last step's pos vs. current pos
        # a1 is position
        if t > 0:
            if int(a1.data) == self.last_x:
                self.past_reward[n, t] -= self.staying_penalty
        self.last_x = int(a1.data)
        
        # update counters
        self.t += 1

        return action


    def compute_reward(self, image):
        """ compute the reward by the discriminator at the end of drawing """
        # store fake data and a paired target data sampled by target data sampler
        n = (self.t - 1) // self.timestep_limit  # number of local episode
        self.fake_data[n] = self.__process_image(image)
        
        if self.conditional:
            self.real_data[n] = self.past_conditional_input[n]
        else:
            self.real_data[n] = self.target_data_sampler()

        # compute L2 loss between target data and drawn picture by the agent
        l2_loss = F.mean_squared_error(self.fake_data[n], self.real_data[n]).data / float(self.rollout_n)
        if n == 0:
            self.stat_l2_loss = l2_loss
        else:
            self.stat_l2_loss += l2_loss
        
        # compute reward after finishing drawing
        if self.reward_mode == 'l2':
            R = -l2_loss
        else:
            conditional_input = self.past_conditional_input[n] if self.conditional else None
            if self.reward_mode == 'dcgan':
                y_fake = self.discriminator(self.fake_data[n], conditional_input)
                R = np_softplus(y_fake.data).data[0, 0]
                self.y_fake[n] = y_fake
            elif self.reward_mode == 'wgangp':
                y_fake = self.discriminator(self.fake_data[n], conditional_input)
                R = y_fake.data[0, 0]
                self.y_fake[n] = y_fake
            else:
                raise NotImplementedError()

        # store reward to the buffer
        if self.process_idx == 0:
            logger.debug('compute final reward = %s at local_episode %s', R, n)
        self.past_R[n] = R

        # add negative reward if the agent did not draw anything
        past_brush_prob = sum([ self.past_brush_prob[n, t] for t in range(self.timestep_limit) ])
        if not past_brush_prob:
            self.past_reward[n, self.timestep_limit - 1] -= self.empty_drawing_penalty

        self.generator.reset_state()

    def stop_episode_and_train(self, obs, r, done=None):
        if self.process_idx == 0:
            # get local time step at each episode
            n, t = self.__get_local_time()
            logger.debug('==== update at local episode %s, local step %s', n, t)
        self.__update()

        # saving the final observation images as png
        if self.process_idx ==0 and self.update_n % self.save_final_obs_update_interval == 0:
            figname = "obs_update_{}.png".format(self.update_n)
            figname = os.path.join(self.outdir_final_obs, figname)
            logger.debug('Saving final observation as %s', figname)
            self.image_drawer.draw_and_save(self.fake_data, self.real_data, figname, self.update_n)


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

            for t in reversed(range(self.timestep_limit)):
                R *= self.gamma  # discount factor
                R += self.past_reward[n, t]
                v = self.past_values[n, t]
                advantage = R - v

                log_prob = self.past_action_log_prob[n, t]
                entropy = self.past_action_entropy[n, t]

                pi_loss -= log_prob * float(advantage.data)
                pi_loss -= self.beta * entropy

                v_loss += (v - R) ** 2 / 2

        if self.pi_loss_coef != 1.0:
            pi_loss *= self.pi_loss_coef
        if self.v_loss_coef != 1.0:
            v_loss *= self.v_loss_coef
        
        # normalize by each step
        pi_loss /= self.timestep_limit * self.rollout_n
        v_loss /= self.timestep_limit * self.rollout_n
        
        total_loss = pi_loss + F.reshape(v_loss, pi_loss.data.shape)

        if self.process_idx == 0:
            logger.debug('pi_loss:%s v_loss:%s', pi_loss.data, v_loss.data)

        # compute gradients of the generator
        self.generator.zerograds()
        total_loss.backward()

        # copy the gradients of the local generator to the globally shared model
        self.shared_generator.zerograds()
        copy_param.copy_grad(
            target_link=self.shared_generator, source_link=self.generator)

        # update the gobally shared model
        if self.process_idx == 0:
            norm = sum(np.sum(np.square(param.grad)) for param in self.gen_optimizer.target.params())
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
            copy_param.copy_grad(
                target_link=self.shared_discriminator, source_link=self.discriminator)

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
            loss_dis = -F.sum(y_real) / self.rollout_n + F.sum(y_fake) / self.rollout_n

            # add gradient panalty to the loss
            eps = np.random.uniform(0, 1, size=self.rollout_n).astype(np.float32)
            eps = np.reshape(eps, (self.rollout_n, 1, 1, 1))
            x_mid = eps * x_real + (1.0 - eps) * x_fake

            x_mid_v = chainer.Variable(x_mid.data)
            if self.conditional:
                y_mid = self.discriminator(x_mid_v, x_mid_v)
            else:
                y_mid = self.discriminator(x_mid_v)

            dydx = self.discriminator.differentiable_backward(np.ones_like(y_mid.data))
            dydx = F.sqrt(F.sum(dydx ** 2, axis=(1, 2, 3)))
            loss_gp = self.gp_lamda * F.mean_squared_error(dydx, np.ones_like(dydx.data))

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
            loss_dis = F.sum(F.softplus(-y_real)) / self.rollout_n + F.sum(F.softplus(y_fake)) / self.rollout_n

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

        ret = [
            ('average_value', self.stat_average_value),
            ('average_entropy', self.stat_average_entropy),
            ('l2_loss', self.stat_l2_loss),
            ('pi_loss', self.stat_pi_loss),
            ('v_loss', self.stat_v_loss),
            ('R', self.stat_R),
            ('reward_min', self.stat_reward_min),
            ('reward_mean', self.stat_reward_mean),
            ('reward_max', self.stat_reward_max),
            ('reward_std', self.stat_reward_std),
            ('update_n', self.update_n)
        ]

        if self.reward_mode == 'wgangp':
            ret += [
                ('discriminator_grad_panalty', self.stat_loss_gp),
                ('discriminator_gradient_size', self.stat_dis_g),
                ('discriminator_loss', self.stat_loss_dis)
            ]
        elif self.reward_mode == 'dcgan':
            ret += [
                ('discriminator_accuracy', self.stat_dis_acc),
                ('discriminator_loss', self.stat_loss_dis)
            ]
        
        return ret


    def stop_episode(self):
        """ spiral model is a recurrent model """
        if self.process_idx == 0:
            logger.debug('stop_episode: reset state')
        self.generator.reset_state()


    def act(self, obs):
        with chainer.no_backprop_mode():
            state = self.process_obs(obs)

            if self.conditional:
                conditional_input = self.target_data_sampler()
                pout, _ = self.generator.pi_and_v(state, conditional_input)
            else:
                pout, _ = self.generator.pi_and_v(state)

            p1, p2, a1, a2 = pout

            if self.act_deterministically:
                a1, a2 = [ np.argmax(p.log_p.data, axis=1)[0] for p in (p1, p2) ]

            return self.pack_action(int(a1.data), int(a2.data))


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


