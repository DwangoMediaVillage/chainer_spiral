"""An example of training SPIRAL with A3C.
"""
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from builtins import *  # NOQA
from future import standard_library
standard_library.install_aliases()  # NOQA
import argparse
import os

# This prevents numpy from using multiple threads
os.environ['OMP_NUM_THREADS'] = '1'  # NOQA

import chainer
from chainer import functions as F
from chainer import links as L
import gym
gym.undo_logger_setup()  # NOQA
import gym.wrappers
import numpy as np

# from chainerrl.agents import a3c
from chainerrl import experiments
from chainerrl import links
from chainerrl import misc
from chainerrl.optimizers.nonbias_weight_decay import NonbiasWeightDecay
from chainerrl.optimizers import rmsprop_async
from chainerrl import policies
from chainerrl.recurrent import RecurrentChainMixin
from chainerrl import v_function

from env import MyPaintEnv
from agents import spiral
from spiral_evaluator import show_drawn_pictures

import chainer.distributions as D

import cv2


def gumbel_softmax_sampling(pi, t=1.0):
    """ performs Gumbel Softmax Sampling for pi """
    p_shape = pi.p.shape

    # sample from uniform dist.
    low = np.array(0, dtype=np.float32)
    high = np.array(1, dtype=np.float32)
    u = D.Uniform(low=low, high=high).sample(sample_shape=p_shape)
    g = -F.log(-F.log(u))

    z = F.softmax((pi.log_p + g) / t)
    return z

class SpiralPi(chainer.Chain):
    """ CNN-LSTM with Autoregressive decodoer. """
    def __init__(self, obs_spaces, action_spaces, in_channel=3):
        self.in_channel = in_channel
        self.obs_pos_dim = obs_spaces.spaces['position'].n
        self.obs_q_dim = obs_spaces.spaces['prob'].n
        self.act_pos_dim = action_spaces.spaces['position'].n
        self.act_q_dim = action_spaces.spaces['prob'].n

        super().__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(self.in_channel, 32, ksize=5)
            self.linear_a1 = L.Linear(1, out_size=16)
            self.linear_a2 = L.Linear(1, out_size=16)
            self.conv2 = L.Convolution2D(32, 32, stride=2, ksize=4)
            self.conv3 = L.Convolution2D(32, 32, stride=2, ksize=4)
            self.conv4 = L.Convolution2D(32, 32, stride=2, ksize=4)

            self.linear_2 = L.Linear(800, 256)
            self.lstm = L.LSTM(256, 256)

            self.z1_conv1 = L.Deconvolution2D(16, 8, stride=2, ksize=4, pad=1)
            self.z1_conv2 = L.Deconvolution2D(8, 4, stride=2, ksize=4, pad=1)
            self.z1_conv3 = L.Deconvolution2D(4, 1, stride=2, ksize=4, pad=1)
            self.z1_linear1 = L.Linear(self.act_pos_dim, out_size=16)
            self.z1_linear2 = L.Linear(256+16, out_size=256)
            self.z2_linear = L.Linear(256, self.act_q_dim)


    def __call__(self, obs):
        """ forward single step. return actions aas categorical distributions. """
        # image and action at the last step
        # a has position (a1) and prob (a2)
        c, a1, a2 = obs
        
        # import ipdb; ipdb.set_trace()
        h_a1 = F.relu(self.linear_a1(a1))
        h_a2 = F.relu(self.linear_a2(a2))
        h_a = F.concat((h_a1, h_a2), axis=1)
        h = F.relu(self.conv1(c))

        # repeat h_a and adds to feature from the conv
        imshape = h.shape[2:]
        h = h + F.reshape(F.tile(h_a, imshape), (1, 32) + imshape)
        
        h = F.relu(self.conv2(h))
        h = F.relu(self.conv3(h))
        h = F.relu(self.conv4(h))

        # TODO: insert ResBlock 3x3
        h = F.expand_dims(F.flatten(h), 0)
        h = F.relu(self.linear_2(h))

        h = self.lstm(h)

        # autoregressive part
        z1 = h

        # location
        h_z1 = F.reshape(z1, (1, 16, 4, 4))
        h_z1 = F.relu(self.z1_conv1(h_z1))
        h_z1 = F.relu(self.z1_conv2(h_z1))
        h_z1 = F.relu(self.z1_conv3(h_z1))
        h_z1 = F.reshape(h_z1, (1, self.act_pos_dim))
        a1 = D.Categorical(logit=h_z1)

        # simple sampling is not differentiable
        h_z1 = F.relu(self.z1_linear1(gumbel_softmax_sampling(a1)))

        z2 = F.relu(self.z1_linear2(F.concat((h_z1, z1), axis=1)))
        h_z2 = F.relu(self.z2_linear(z2))
        a2 = D.Categorical(logit=h_z2)

        return a1, a2


class SpiralV(chainer.Chain):
    """ CNN-LSTM V function """
    def __init__(self, obs_spaces, in_channel=3):
        self.in_channel = in_channel
        self.obs_pos_dim = obs_spaces.spaces['position'].n
        self.obs_q_dim = obs_spaces.spaces['prob'].n  # 1

        super().__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(self.in_channel, 32, ksize=5)
            self.linear_a1 = L.Linear(1, out_size=16)
            self.linear_a2 = L.Linear(1, out_size=16)
            self.conv2 = L.Convolution2D(32, 32, stride=2, ksize=4)
            self.conv3 = L.Convolution2D(32, 32, stride=2, ksize=4)
            self.conv4 = L.Convolution2D(32, 32, stride=2, ksize=4)

            self.linear_2 = L.Linear(800, 128)
            self.lstm = L.LSTM(128, 128)

            self.linear_out = L.Linear(128, 1)

    def __call__(self, obs):
        """ forward single step. return actions aas categorical distributions. """
        # image and action at the last step
        # a has position (a1) and prob (a2)
        c, a1, a2 = obs
        
        h_a1 = F.relu(self.linear_a1(a1))
        h_a2 = F.relu(self.linear_a2(a2))
        h_a = F.concat((h_a1, h_a2), axis=1)
        h = F.relu(self.conv1(c))

        # repeat h_a and adds to feature from the conv
        imshape = h.shape[2:]
        h = h + F.reshape(F.tile(h_a, imshape), (1, 32) + imshape)
        
        h = F.relu(self.conv2(h))
        h = F.relu(self.conv3(h))
        h = F.relu(self.conv4(h))

        # TODO: insert ResBlock 3x3
        h = F.expand_dims(F.flatten(h), 0)
        h = F.relu(self.linear_2(h))

        h = self.lstm(h)

        v = F.relu(self.linear_out(h))
        return v

class SpiralD(chainer.Chain):
    """ Discriminator """
    def __init__(self, in_channel=3):
        self.in_channel = in_channel
        super().__init__()
        with self.init_scope():
            self.c1 = L.Convolution2D(self.in_channel, 32, stride=1, ksize=3, pad=1)
            self.c2 = L.Convolution2D(32, 48, stride=2, ksize=3, pad=1)
            self.c3 = L.Convolution2D(48, 52, stride=1, ksize=3, pad=1)
            self.c4 = L.Convolution2D(52, 64, stride=2, ksize=3, pad=1)
            self.c5 = L.Convolution2D(64, 64, stride=1, ksize=3, pad=1)
            self.c6 = L.Convolution2D(64, 128, stride=2, ksize=3, pad=1)
            self.b2 = L.BatchNormalization(48, use_gamma=False)
            self.b3 = L.BatchNormalization(52, use_gamma=False)
            self.b4 = L.BatchNormalization(64, use_gamma=False)
            self.b5 = L.BatchNormalization(64, use_gamma=False)
            self.b6 = L.BatchNormalization(128, use_gamma=False)
            self.l7 = L.Linear(8 * 8 * 128, 1)
        
    def __call__(self, x):
        h = x
        h = F.leaky_relu(self.c1(h))
        h = F.leaky_relu(self.b2(self.c2(h)))
        h = F.leaky_relu(self.b3(self.c3(h)))
        h = F.leaky_relu(self.b4(self.c4(h)))
        h = F.leaky_relu(self.b5(self.c5(h)))
        h = F.leaky_relu(self.b6(self.c6(h)))
        return self.l7(h)
                

class SPIRALSimpleModel(chainer.ChainList, spiral.SPIRALModel, RecurrentChainMixin):
    """ A simple model """
    def __init__(self, obs_spaces, action_spaces, in_channel=3):
        
        # define policy and value networks
        self.pi = SpiralPi(obs_spaces, action_spaces, in_channel)
        self.v = SpiralV(obs_spaces, in_channel)

        super().__init__(self.pi, self.v)
    
    def pi_and_v(self, state):
        """ Forwarding single step """
        return self.pi(state), self.v(state)

def main():
    import logging

    parser = argparse.ArgumentParser()
    parser.add_argument('processes', type=int)
    parser.add_argument('--logger-level', type=int, default=logging.DEBUG)
    parser.add_argument('--outdir', type=str, default='results',
                            help='Directory path to save output files.'
                                ' If it does not exist, it will be created.')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed [0, 2 ** 32)')
    parser.add_argument('--lr', type=float, default=7e-4)
    parser.add_argument('--rmsprop-epsilon', type=float, default=1e-1)
    parser.add_argument('--weight-decay', type=float, default=0.0)
    parser.add_argument('--steps', type=int, default=20)
    parser.add_argument('--eval-interval', type=int, default=10)
    parser.add_argument('--eval-n-runs', type=int, default=1)
    parser.add_argument('--load', type=str, default='')
    parser.add_argument('--demo', action='store_true', default=False)
    args = parser.parse_args()

    # init a logger
    logging.basicConfig(level=args.logger_level)

    # Set a random seed used in ChainerRL.
    # If you use more than one processes, the results will be no longer
    # deterministic even with the same random seed.
    misc.set_random_seed(args.seed)

    # Set different random seeds for different subprocesses.
    # If seed=0 and processes=4, subprocess seeds are [0, 1, 2, 3].
    # If seed=1 and processes=4, subprocess seeds are [4, 5, 6, 7].
    process_seeds = np.arange(args.processes) + args.seed * args.processes
    assert process_seeds.max() < 2 ** 32

    # create directory to put the results
    args.outdir = experiments.prepare_output_dir(args, args.outdir)

    # define func to create env
    def make_env(process_idx, test):
        env = MyPaintEnv()

        # Use different random seeds for train and test envs
        process_seed = int(process_seeds[process_idx])
        env_seed = 2 ** 32 - 1 - process_seed if test else process_seed
        env.seed(env_seed)
        
        # TODO: implement test mode
        # TODO: implement reward filter?
        
        return env

    sample_env = MyPaintEnv()
    
    # TODO: MyPaintEnv is not wrapped by EnvSpec
    timestep_limit = sample_env.tags['max_episode_steps']
    obs_space = sample_env.observation_space
    action_space = sample_env.action_space

    in_channel = 1

    gen = SPIRALSimpleModel(obs_space, action_space, in_channel)  # generator
    dis = SpiralD(in_channel)  # discriminator

    gen_opt = rmsprop_async.RMSpropAsync(
        lr=args.lr, eps=args.rmsprop_epsilon, alpha=0.99)
    dis_opt = rmsprop_async.RMSpropAsync(
        lr=args.lr, eps=args.rmsprop_epsilon, alpha=0.99)

    gen_opt.setup(gen)
    dis_opt.setup(dis)

    gen_opt.add_hook(chainer.optimizer.GradientClipping(40))
    dis_opt.add_hook(chainer.optimizer.GradientClipping(40))
    if args.weight_decay > 0:
        gen_opt.add_hook(NonbiasWeightDecay(args.weight_decay))
        dis_opt.add_hook(NonbiasWeightDecay(args.weight_decay))

    # target image dataset
    train, test = chainer.datasets.get_mnist(withlabel=False)
    train_iter = chainer.iterators.SerialIterator(train, 1)  # batchsize is 1

    def target_data_sampler():
        """ sample a mnist image """
        y = train_iter.next()[0].data
        y = np.reshape(y, (28, 28))
        y = cv2.resize(y, (sample_env.imsize, sample_env.imsize))
        y = np.reshape(y, (1, 1, sample_env.imsize, sample_env.imsize))
        y = 1.0 - y  # background: black -> white
        return chainer.Variable(y)

    agent = spiral.SPIRAL(gen, dis, gen_opt, dis_opt, target_data_sampler, in_channel)

    if args.load:
        agent.load(args.load)

    if args.demo:
        # IN PROGRESS
        env = make_env(0, True)
        show_drawn_pictures(env, agent, timestep_limit)

    else:
        # train
        # single core for debug
        # TODO: change to train_async
        experiments.train_agent_with_evaluation(
            agent=agent,
            env=make_env(0, False),
            steps=args.steps,
            eval_n_runs=args.eval_n_runs,
            eval_interval=args.eval_interval,
            max_episode_len=timestep_limit,
            outdir=args.outdir
            )

if __name__ == '__main__':
    main()
