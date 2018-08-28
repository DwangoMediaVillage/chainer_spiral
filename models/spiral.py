from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from builtins import *  # NOQA
from future import standard_library
standard_library.install_aliases()  # NOQA
import os

# This prevents numpy from using multiple threads
os.environ['OMP_NUM_THREADS'] = '1'  # NOQA

import numpy as np
import chainer
from chainerrl.recurrent import RecurrentChainMixin

from chainer import functions as F
from chainer import links as L
from chainer import distributions as D
from agents import spiral


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


class SpiralValue(chainer.Chain):
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

class SpiralDiscriminator(chainer.Chain):
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
        self.v = SpiralValue(obs_spaces, in_channel)

        super().__init__(self.pi, self.v)
    
    def pi_and_v(self, state):
        """ Forwarding single step """
        return self.pi(state), self.v(state)