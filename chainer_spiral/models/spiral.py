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
from chainerrl.distribution import SoftmaxDistribution

def bw_linear(x_in, x, l):
    return F.matmul(x, l.W)


def bw_convolution(x_in, x, l):
    return F.deconvolution_2d(x, l.W, None, l.stride, l.pad, (x_in.data.shape[2], x_in.data.shape[3]))


def bw_leaky_relu(x_in, x, a):
    return (x_in.data > 0) * x + a * (x_in.data < 0) * x


class SPIRALModel(chainer.Link, RecurrentChainMixin):
    """ SPIRAL Model. """
    def pi_and_v(self, obs):
        """ evaluate the policy and the V-function """
        return NotImplementedError()
    
    def __call__(self, obs):
        return self.pi_and_v(obs)


class AutoregressiveDecoder(chainer.Chain):
    def __init__(self, z_size):
        """ autoregressive decoder """
        self.z_size = z_size
        self.f = F.sigmoid  # activation func
        super().__init__()
        with self.init_scope():
            # location
            self.l1_c1 = L.Deconvolution2D(64, 32, stride=2, ksize=4, pad=1)
            self.l1_c2 = L.Deconvolution2D(32, 16, stride=2, ksize=4, pad=1)
            self.l1_c3 = L.Deconvolution2D(16, 8, stride=2, ksize=4, pad=1)
            self.l1_c4 = L.Deconvolution2D(8, 8, stride=2, ksize=4, pad=1)
            self.l1_c5 = L.Convolution2D(8, 1, stride=1, ksize=3, pad=1)
            self.l1_l1 = L.Linear(1, 16)
            self.l1_l2 = L.Linear(272, self.z_size)

            # prob
            self.l2_l1 = L.Linear(self.z_size, 2)

    def __call__(self, z):
        # decode location
        z1 = z
        h = F.reshape(z1, (1, 64, 2, 2))
        h = self.f(self.l1_c1(h))
        h = self.f(self.l1_c2(h))
        h = self.f(self.l1_c3(h))
        h = self.f(self.l1_c4(h))
        h = self.l1_c5(h)
        h = F.expand_dims(F.flatten(h), 0)
        p1 = SoftmaxDistribution(h)
        a1 = p1.sample()

        # decode prob
        h_a1 = self.f(self.l1_l1( np.expand_dims(a1.data, 0).astype(np.float32)))
        h_a1 = F.concat((z1, h_a1), axis=1)

        z2 = self.f(self.l1_l2(h_a1))
        h_a2 = self.l2_l1(z2)
        p2 = SoftmaxDistribution(h_a2)
        a2 = p2.sample()

        probs = [p1, p2]
        acts = [a1, a2]

        return probs, acts


class SpiralMnistDiscriminator(chainer.Chain):
    def __init__(self, imsize, conditional):
        self.imsize = imsize
        self.conditional = conditional
        super().__init__()
        with self.init_scope():
            in_channel = 2 if self.conditional else 1
            self.c1 = L.Convolution2D(in_channel, 16, stride=1, ksize=3, pad=1)
            self.c2 = L.Convolution2D(16, 32, stride=2, ksize=3, pad=1)
            self.c3 = L.Convolution2D(32, 48, stride=2, ksize=2, pad=1)
            self.c4 = L.Convolution2D(48, 48, stride=2, ksize=2, pad=1)
            self.c5 = L.Convolution2D(48, 64, stride=2, ksize=2, pad=1)
            self.c6 = L.Convolution2D(64, 64, stride=2, ksize=2, pad=1)
            self.l7 = L.Linear(3 * 3 * 64, 1)
        
    def __call__(self, x, conditional_input=None):
        if self.conditional:
            self.x = F.concat((x, conditional_input), axis=1)
        else:
            self.x = x

        self.h1 = F.leaky_relu(self.c1(self.x))
        self.h2 = F.leaky_relu(self.c2(self.h1))
        self.h3 = F.leaky_relu(self.c3(self.h2))
        self.h4 = F.leaky_relu(self.c4(self.h3))
        self.h5 = F.leaky_relu(self.c5(self.h4))
        self.h6 = F.leaky_relu(self.c6(self.h5))
        return self.l7(self.h6)
                
    def differentiable_backward(self, x):
        g = bw_linear(self.h6, x, self.l7)
        g = F.reshape(g, (x.shape[0], 64, 3, 3))
        g = bw_leaky_relu(self.h6, g, 0.2)
        g = bw_convolution(self.h5, g, self.c6)
        g = bw_leaky_relu(self.h5, g, 0.2)
        g = bw_convolution(self.h4, g, self.c5)
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
    def __init__(self, imsize, conditional):
        self.imsize = imsize
        self.f = F.relu  # activation func for encoding part
        self.conditional = conditional
        super().__init__()
        with self.init_scope():
            # image encoding part
            in_channel = 2 if self.conditional else 1
            self.e1_c1 = L.Convolution2D(in_channel, 32, ksize=5)
            
            # action observation encoding part
            self.e1_l1_a1 = L.Linear(1, 16)
            self.e1_l1_a2 = L.Linear(1, 16)

            # convolution after concat
            self.e2_c1 = L.Convolution2D(32, 32, stride=2, ksize=4)
            self.e2_c2 = L.Convolution2D(32, 32, stride=2, ksize=4)
            self.e2_c3 = L.Convolution2D(32, 32, stride=2, ksize=4)
            self.e2_l1 = L.Linear(800, 256)
            
            # lstm
            self.lstm = L.LSTM(256, 256)

            # decoding part
            self.decoder = AutoregressiveDecoder(256)

    def __call__(self, obs, conditional_input=None):
        o_c, o_a1, o_a2 = obs
        
        if self.conditional:
            # concat image obs and conditional image input
            o_c = F.concat((o_c, conditional_input), axis=1)

        # image encoding part
        h_a1 = self.f(self.e1_l1_a1(o_a1))
        h_a2 = self.f(self.e1_l1_a1(o_a2))
        h_a = F.concat((h_a1, h_a2), axis=1)
        h_c = self.f(self.e1_c1(o_c))

        # reshape
        imshape = h_c.shape[2:]
        h = h_c + F.reshape(F.tile(h_a, imshape), (1, 32) + imshape)
        h = self.f(self.e2_c1(h))
        h = self.f(self.e2_c2(h))
        h = self.f(self.e2_c3(h))
        h = F.expand_dims(F.flatten(h), 0)
        h = self.f(self.e2_l1(h))
        
        # lstm
        h = self.lstm(h)

        # output by the decoder
        return self.decoder(h)


class MnistValueNet(chainer.Chain):
    def __init__(self, imsize, conditional):
        self.imsize = imsize
        self.f = F.relu  # activation func for encoding part
        self.conditional = conditional
        super().__init__()
        with self.init_scope():
            # image encoding part
            in_channel = 2 if self.conditional else 1
            self.e1_c1 = L.Convolution2D(in_channel, 32, ksize=5)
            
            # action observation encoding part
            self.e1_l1_a1 = L.Linear(1, 16)
            self.e1_l1_a2 = L.Linear(1, 16)

            # convolution after concat
            self.e2_c1 = L.Convolution2D(32, 32, stride=2, ksize=4)
            self.e2_c2 = L.Convolution2D(32, 32, stride=2, ksize=4)
            self.e2_c3 = L.Convolution2D(32, 32, stride=2, ksize=4)
            self.e2_l1 = L.Linear(800, 128)
            
            # lstm
            self.lstm = L.LSTM(128, 128)
            self.d1_l1 = L.Linear(128, 1)

    def __call__(self, obs, conditional_input=None):
        o_c, o_a1, o_a2 = obs
        # image encoding part
        
        if self.conditional:
            # concat image obs and conditional image input
            o_c = F.concat((o_c, conditional_input), axis=1)

        h_a1 = self.f(self.e1_l1_a1(o_a1))
        h_a2 = self.f(self.e1_l1_a1(o_a2))
        h_a = F.concat((h_a1, h_a2), axis=1)
        h_c = self.f(self.e1_c1(o_c))

        # reshape
        imshape = h_c.shape[2:]
        h = h_c + F.reshape(F.tile(h_a, imshape), (1, 32) + imshape)
        h = self.f(self.e2_c1(h))
        h = self.f(self.e2_c2(h))
        h = self.f(self.e2_c3(h))
        h = F.expand_dims(F.flatten(h), 0)
        h = self.f(self.e2_l1(h))
        h = self.lstm(h)
        h = self.d1_l1(h)
        return h


class SpiralToyDiscriminator(chainer.Chain):
    def __init__(self, imsize, conditional):
        self.imsize = imsize
        self.conditional = conditional
        super().__init__()
        with self.init_scope():
            in_channel = 2 if self.conditional else 1
            self.l1 = L.Convolution2D(in_channel, 3, stride=1, ksize=2)
            self.l2 = L.Linear(12, 5)
            self.l3 = L.Linear(5, 1)

    def __call__(self, x, conditional_input=None):
        if self.conditional:
            self.x = F.concat((x, conditional_input), axis=1)
        else:
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
    def __init__(self, imsize, conditional):
        self.f = F.sigmoid
        self.g = F.sigmoid
        self.conditional = conditional
        super().__init__()
        with self.init_scope():
            in_size = imsize * imsize * 2 if self.conditional else imsize * imsize
            self.e1_c1 = L.Linear(in_size, 30)
            self.e1_a1 = L.Linear(1, 15)
            self.e1_a2 = L.Linear(1, 15)
            self.e2_l1 = L.Linear(30, 30)
            self.lstm = L.LSTM(30, 30)

            self.d_a1_l1 = L.Linear(30, 15)
            self.d_a1_l2 = L.Linear(15, imsize * imsize)

            self.d_a2_l1 = L.Linear(30, 10)
            self.d_a2_l2 = L.Linear(10, 2)


    def __call__(self, obs, conditional_input):
        o_c, o_a1, o_a2 = obs

        if self.conditional:
            # concat image obs and conditional image input
            o_c = F.concat((o_c, conditional_input), axis=1)

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

        probs = [p1, p2]
        acts = [p1.sample(), p2.sample()]

        return probs, acts

    
class ToyValueNet(chainer.Chain):
    def __init__(self, imsize, conditional):
        self.f = F.sigmoid
        self.conditional = conditional
        super().__init__()
        with self.init_scope():
            in_size = imsize * imsize * 2 if self.conditional else imsize * imsize
            self.e1_c1 = L.Linear(in_size, 10)
            self.e1_a1 = L.Linear(1, 5)
            self.e1_a2 = L.Linear(1, 5)

            self.e2_l1 = L.Linear(10, 10)
            self.lstm = L.Linear(10, 10)
            self.e2_l2 = L.Linear(10, 1)

    
    def __call__(self, obs, conditional_input):
        o_c, o_a1, o_a2 = obs

        if self.conditional:
            # concat image obs and conditional image input
            o_c = F.concat((o_c, conditional_input), axis=1)

        h_a1 = self.f(self.e1_a1(o_a1))
        h_a2 = self.f(self.e1_a2(o_a2))
        h_a = F.concat((h_a1, h_a2), axis=1)
        h_c = self.f(self.e1_c1(o_c))
        h = h_c + h_a
        h = self.f(self.e2_l1(h))
        h = self.lstm(h)
        h = self.e2_l2(h)
        return h


class SpiralMnistModel(chainer.ChainList, SPIRALModel, RecurrentChainMixin):
    """ Model for mnist drawing """
    def __init__(self, imsize, conditional):
        # define policy and value networks
        self.pi = MnistPolicyNet(imsize, conditional)
        self.v = MnistValueNet(imsize, conditional)
        super().__init__(self.pi, self.v)
    
    def pi_and_v(self, state, conditional_input=None):
        """ forwarding single step """
        return self.pi(state, conditional_input), self.v(state, conditional_input)


class SpiralToyModel(chainer.ChainList, SPIRALModel, RecurrentChainMixin):
    """ A simple model """
    def __init__(self, imsize, conditional):
        # define policy and value networks
        self.pi = ToyPolicyNet(imsize, conditional)
        self.v = ToyValueNet(imsize, conditional)
        super().__init__(self.pi, self.v)
    
    def pi_and_v(self, state, conditional_input=None):
        """ forwarding single step """
        return self.pi(state, conditional_input), self.v(state, conditional_input)

    
