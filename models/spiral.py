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

from chainerrl.distribution import SoftmaxDistribution


def bw_linear(x_in, x, l):
    return F.matmul(x, l.W)


def bw_convolution(x_in, x, l):
    return F.deconvolution_2d(x, l.W, None, l.stride, l.pad, (x_in.data.shape[2], x_in.data.shape[3]))


def bw_leaky_relu(x_in, x, a):
    return (x_in.data > 0) * x + a * (x_in.data < 0) * x


class AutoregressiveDecoder(chainer.Chain):
    def __init__(self, z_size):
        """ autoregressive decoder """
        self.z_size = z_size
        self.f = F.sigmoid  # activation func
        super().__init__()
        with self.init_scope():
            # location
            self.l1_c1 = L.Convolution2D(8, 16, ksize=3, pad=1)
            self.l1_c2 = L.Convolution2D(16, 16, ksize=3, pad=1)
            self.l1_c3 = L.Deconvolution2D(16, 16, ksize=4, stride=2, pad=1)
            self.l1_c4 = L.Convolution2D(16, 1, ksize=4, stride=1, pad=1)
            self.l1_l1 = L.Linear(1, 16)
            self.l1_l2 = L.Linear(144, self.z_size)

            # prob
            self.l2_l1 = L.Linear(self.z_size, 2)

    def __call__(self, z):
        # decode location
        z1 = z
        h = F.reshape(z1, (1, 8, 4, 4))
        h = self.f(self.l1_c1(h))
        h = self.f(self.l1_c2(h))
        h = self.f(self.l1_c3(h))
        h = self.l1_c4(h)
        h = F.expand_dims(F.flatten(h), 0)
        p1 = D.Categorical(logit=h)
        a1 = p1.sample(1).data  # sampling

        # decode prob
        h_a1 = self.f(self.l1_l1(a1.astype(np.float32)))
        h_a1 = F.concat((z1, h_a1), axis=1)
        z2 = self.f(self.l1_l2(h_a1))

        h = self.f(self.l2_l1(z2))
        p2 = D.Categorical(logit=h)
        a2 = p2.sample(1).data  # sampling
        return p1, p2, a1[0, 0], a2[0, 0]


class SpiralMnistDiscriminator(chainer.Chain):
    def __init__(self, imsize):
        self.imsize = imsize
        super().__init__()
        with self.init_scope():
            self.c1 = L.Convolution2D(1, 16, stride=1, ksize=3, pad=1)
            self.c2 = L.Convolution2D(16, 16, stride=2, ksize=3, pad=1)
            self.c3 = L.Convolution2D(16, 32, stride=1, ksize=2, pad=1)
            self.c4 = L.Convolution2D(32, 48, stride=2, ksize=2, pad=1)
            self.l5 = L.Linear(3 * 3 * 48, 1)
        
    def __call__(self, x):
        self.x = x
        self.h1 = F.leaky_relu(self.c1(self.x))
        self.h2 = F.leaky_relu(self.c2(self.h1))
        self.h3 = F.leaky_relu(self.c3(self.h2))
        self.h4 = F.leaky_relu(self.c4(self.h3))
        return self.l5(self.h4)
                
    def differentiable_backward(self, x):
        g = bw_linear(self.h4, x, self.l5)
        g = F.reshape(g, (x.shape[0], 48, 3, 3))
        g = bw_leaky_relu(self.h4, g, 0.2)
        g = bw_convolution(self.h3, g, self.c4)
        g = bw_leaky_relu(self.h3, g, 0.2)
        g = bw_convolution(self.h2, g, self.c3)
        g = bw_leaky_relu(self.h2, g, 0.2)
        g = bw_convolution(self.h1, g, self.c2)
        g = bw_leaky_relu(self.h1, g, 0.2)
        g = bw_convolution(self.x, g, self.c1)
        return g


class MnistPolicyNet(chainer.Chain):
    def __init__(self, imsize):
        self.imsize = imsize
        self.f = F.relu  # activation func for encoding part
        super().__init__()
        with self.init_scope():
            # image encoding part
            self.e1_c1 = L.Convolution2D(1, 16, ksize=3, pad=1)
            
            # action observation encoding part
            self.e1_l1_a1 = L.Linear(1, 8)
            self.e1_l1_a2 = L.Linear(1, 8)

            # convolution after concat
            self.e2_c1 = L.Convolution2D(16, 16, stride=2, ksize=4, pad=1)
            self.e2_c2 = L.Convolution2D(16, 16, stride=1, ksize=4, pad=1)
            self.e2_l1 = L.Linear(144, 128)
            
            # lstm
            self.lstm = L.LSTM(128, 128)

            # decoding part
            self.decoder = AutoregressiveDecoder(128)

    def __call__(self, obs):
        o_c, o_a1, o_a2 = obs
        # image encoding part
        h_a1 = self.f(self.e1_l1_a1(o_a1))
        h_a2 = self.f(self.e1_l1_a1(o_a2))
        h_a = F.concat((h_a1, h_a2), axis=1)
        h_c = self.f(self.e1_c1(o_c))

        # reshape
        imshape = h_c.shape[2:]
        h = h_c + F.reshape(F.tile(h_a, imshape), (1, 16) + imshape)
        h = self.f(self.e2_c1(h))
        h = self.f(self.e2_c2(h))
        h = F.expand_dims(F.flatten(h), 0)
        h = self.f(self.e2_l1(h))
        
        # lstm
        h = self.lstm(h)

        # output by the decoder
        return self.decoder(h)


class MnistValueNet(chainer.Chain):
    def __init__(self, imsize):
        self.imsize = imsize
        self.f = F.relu  # activation func for encoding part
        super().__init__()
        with self.init_scope():
            # image encoding part
            self.e1_c1 = L.Convolution2D(1, 16, ksize=3, pad=1)
            
            # action observation encoding part
            self.e1_l1_a1 = L.Linear(1, 8)
            self.e1_l1_a2 = L.Linear(1, 8)

            # convolution after concat
            self.e2_c1 = L.Convolution2D(16, 16, stride=2, ksize=4, pad=1)
            self.e2_c2 = L.Convolution2D(16, 16, stride=1, ksize=4, pad=1)
            
            self.lstm = L.LSTM(144, 144)
            self.e2_l1 = L.Linear(144, 1)

    def __call__(self, obs):
        o_c, o_a1, o_a2 = obs
        # image encoding part
        h_a1 = self.f(self.e1_l1_a1(o_a1))
        h_a2 = self.f(self.e1_l1_a1(o_a2))
        h_a = F.concat((h_a1, h_a2), axis=1)
        h_c = self.f(self.e1_c1(o_c))

        # reshape
        imshape = h_c.shape[2:]
        h = h_c + F.reshape(F.tile(h_a, imshape), (1, 16) + imshape)
        h = self.f(self.e2_c1(h))
        h = self.f(self.e2_c2(h))
        h = F.expand_dims(F.flatten(h), 0)
        h = self.lstm(h)
        h = self.e2_l1(h)
        return h


class SpiralToyDiscriminator(chainer.Chain):
    def __init__(self, imsize):
        self.imsize = imsize
        super().__init__()
        with self.init_scope():
            self.l1 = L.Convolution2D(1, 3, stride=1, ksize=2)
            self.l2 = L.Linear(12, 5)
            self.l3 = L.Linear(5, 1)

    def __call__(self, x):
        self.x = x
        self.h1 = F.leaky_relu(self.l1(self.x))
        self.h2 = F.leaky_relu(self.l2(self.h1))
        return self.l3(self.h2)

    def differentiable_backward(self, x):
        g = bw_linear(self.h2, x, self.l3)
        g = bw_leaky_relu(self.h2, g, 0.2)
        g = bw_linear(self.h1, g, self.l2)
        g = F.reshape(g, (x.shape[0], 3, 2, 2))
        g = bw_leaky_relu(self.h1, g, 0.2)
        g = bw_convolution(self.x, g, self.l1)
        return g


class ToyPolicyNet(chainer.Chain):
    def __init__(self, imsize):
        self.f = F.sigmoid
        self.g = F.sigmoid
        super().__init__()
        with self.init_scope():
            self.e1_c1 = L.Linear(imsize * imsize, 30)
            self.e1_a1 = L.Linear(1, 15)
            self.e1_a2 = L.Linear(1, 15)
            self.e2_l1 = L.Linear(30, 30)
            self.lstm = L.LSTM(30, 30)

            self.d_a1_l1 = L.Linear(30, 15)
            self.d_a1_l2 = L.Linear(15, imsize * imsize)

            self.d_a2_l1 = L.Linear(30, 10)
            self.d_a2_l2 = L.Linear(10, 2)


    def __call__(self, obs):
        o_c, o_a1, o_a2 = obs

        h_a1 = self.f(self.e1_a1(o_a1))
        h_a2 = self.f(self.e1_a2(o_a2))
        h_a = F.concat((h_a1, h_a2), axis=1)

        h_c = self.f(self.e1_c1(o_c))
        h = h_c + h_a
        h = self.f(self.e2_l1(h))
        h = self.lstm(h)

        # decoder part
        z1 = h
        z1 = self.f(self.d_a1_l1(z1))
        z1 = self.d_a1_l2(z1)
        p1 = SoftmaxDistribution(z1)

        z2 = h
        z2 = self.f(self.d_a2_l1(z2))
        z2 = self.d_a2_l2(z2)
        p2 = SoftmaxDistribution(z2)

        return p1, p2, p1.sample(), p2.sample()

    
class ToyValueNet(chainer.Chain):
    def __init__(self, imsize):
        self.f = F.sigmoid
        super().__init__()
        with self.init_scope():
            self.e1_c1 = L.Linear(imsize * imsize, 10)
            self.e1_a1 = L.Linear(1, 5)
            self.e1_a2 = L.Linear(1, 5)

            self.e2_l1 = L.Linear(10, 10)
            self.lstm = L.Linear(10, 10)
            self.e2_l2 = L.Linear(10, 1)

    
    def __call__(self, obs):
        o_c, o_a1, o_a2 = obs
        h_a1 = self.f(self.e1_a1(o_a1))
        h_a2 = self.f(self.e1_a2(o_a2))
        h_a = F.concat((h_a1, h_a2), axis=1)
        h_c = self.f(self.e1_c1(o_c))
        h = h_c + h_a
        h = self.f(self.e2_l1(h))
        h = self.lstm(h)
        h = self.e2_l2(h)
        return h

class SpiralMnistModel(chainer.ChainList, spiral.SPIRALModel, RecurrentChainMixin):
    """ Model for mnist drawing """
    def __init__(self, imsize):
        # define policy and value networks
        self.pi = MnistPolicyNet(imsize)
        self.v = MnistValueNet(imsize)
        super().__init__(self.pi, self.v)
    
    def pi_and_v(self, state):
        """ forwarding single step """
        return self.pi(state), self.v(state)


class SpiralToyModel(chainer.ChainList, spiral.SPIRALModel, RecurrentChainMixin):
    """ A simple model """
    def __init__(self, imsize):
        # define policy and value networks
        self.pi = ToyPolicyNet(imsize)
        self.v = ToyValueNet(imsize)
        super().__init__(self.pi, self.v)
    
    def pi_and_v(self, state):
        """ forwarding single step """
        return self.pi(state), self.v(state)

    
