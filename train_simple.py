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

import chainer.distributions as D

class SpiralPi(chainer.Chain):
    """ CNN-LSTM with Autoregressive decodoer. """
    def __init__(self, obs_spaces, action_spaces):
        self.obs_inchannel = obs_spaces.spaces['image'].shape[-1]
        self.obs_pos_dim = obs_spaces.spaces['position'].shape[0]
        self.obs_q_dim = 1  # True or false
        self.act_pos_dim = action_spaces.spaces['position'].shape
        self.act_q_dim = 2  # True or false

        super().__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(self.obs_inchannel, 32, ksize=5)
            self.linear_a1 = L.Linear(self.obs_pos_dim, out_size=16)
            self.linear_a2 = L.Linear(self.obs_q_dim, out_size=16)
            self.conv2 = L.Convolution2D(32, 32, stride=2, ksize=4)
            self.conv3 = L.Convolution2D(32, 32, stride=2, ksize=4)
            self.conv4 = L.Convolution2D(32, 32, stride=2, ksize=4)

            self.linear_2 = L.Linear(800, 256)
            self.lstm = L.LSTM(256, 256)

            self.z0_conv1 = L.Deconvolution2D(16, 8, stride=2, ksize=4, pad=1)
            self.z0_conv2 = L.Deconvolution2D(8, 4, stride=2, ksize=4, pad=1)
            self.z0_conv3 = L.Deconvolution2D(4, 1, stride=2, ksize=4, pad=1)
            self.z0_linear1 = L.Linear(1, out_size=16)
            self.z0_linear2 = L.Linear(256-16, out_size=256)



    def __call__(self, obs):
        """ forward single step. returns action """
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

        # autoregressive part
        z0 = h

        # location
        h_z0 = F.reshape(z0, (1, 16, 4, 4))
        h_z0 = F.relu(self.z0_conv1(h_z0))
        h_z0 = F.relu(self.z0_conv2(h_z0))
        h_z0 = F.relu(self.z0_conv3(h_z0))
        h_z0 = F.reshape(h_z0, (1, 32 * 32))

        a1 = D.Categorical(logit=h_z0)

        h_a1 = F.reshape(a1.sample(), (1, 1))  # ERROR! this is int
        h_z0 = F.relu(self.z0_linear1(h_a1))
        z1 = F.relu(self.z0_linear2(F.concat((h_z0, z0), axis=1)))

        import ipdb; ipdb.set_trace()
        # TODO: return actions
        


class SpiralV(chainer.Chain):
    """ CNN-LSTM V function """
    def __init__(self, obs_spaces, action_spaces):
        self.obs_inchannel = obs_spaces.spaces['image'].shape[-1]
        self.obs_pos_dim = obs_spaces.spaces['position'].shape[0]
        self.obs_q_dim = 2

        super().__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(self.obs_inchannel, 32, ksize=5)
            self.linear_a1 = L.Linear(self.obs_pos_dim, out_size=16)
            self.linear_a2 = L.Linear(self.obs_q_dim, out_size=16)
            self.conv2 = L.Convolution2D(32, 32, stride=2)
            self.conv3 = L.Convolution2D(32, 32, stride=2)
            self.conv4 = L.Convolution2D(32, 32, stride=2)
            self.linear_v1 = L.Linear(256, 32)
            self.linear_v2 = L.Linear(32, 1)        
    
    def __call__(self, obs):
        """ forward single step. returns value """
        # image and action at the last step
        # a has position (a1) and prob (a2)
        c, a1, a2 = obs

        h_a1 = F.relu(self.linear_a1(a1))
        h_a2 = F.relu(self.linear_a2(a2))
        h_a = F.concat((h_a1, h_a2), axis=1)
        h = F.relu(self.conv1(c)) + h_a
        h = F.relu(self.conv2(h))
        h = F.relu(self.conv3(h))
        h = F.relu(self.conv4(h))
        h = F.flatten(h)
        h = F.relu(self.linear_v1(h))
        h = F.relu(self.linear_v2(h))
        return h


class SPIRALSimpleModel(chainer.ChainList, spiral.SPIRALModel, RecurrentChainMixin):
    """ A simple model """
    def __init__(self, obs_spaces, action_spaces):
        
        # define policy and value networks
        self.pi = SpiralPi(obs_spaces, action_spaces)
        self.v = SpiralV(obs_spaces, action_spaces)

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

    model = SPIRALSimpleModel(obs_space, action_space)

    opt = rmsprop_async.RMSpropAsync(
        lr=args.lr, eps=args.rmsprop_epsilon, alpha=0.99)
    opt.setup(model)
    opt.add_hook(chainer.optimizer.GradientClipping(40))
    if args.weight_decay > 0:
        opt.add_hook(NonbiasWeightDecay(args.weight_decay))

    agent = spiral.SPIRAL(model, opt)

    # TODO: load a pre-trained weight

    # TODO: implement demo mode

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
