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
    saved_attributes = []  # TODO: add model and optimizer

    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer

    def __process_obs(self, obs):
        c = obs['image']
        x = obs['position']
        q = obs['prob']

        # image 
        c = np.asarray(c, dtype=np.float32)
        c = np.rollaxis(c, -1)
        c = np.expand_dims(c, 0)

        # position
        x = np.asarray(x, dtype=np.float32)
        x = np.expand_dims(x, 0)

        # prob
        q = np.asarray(q, dtype=np.float32)
        q = np.reshape(q, [1, 1])

        return c, x, q

    def act_and_train(self, obs, r):
        """ Infer action from the observation at each step """

        # parse observation
        state = self.__process_obs(obs)

        # TODO: update if self.t - self.t_start == self.t_max
        # TODO: store past_rewards, and past_states

        # infer by the current policy
        pout, vout = self.model.pi_and_v(state)






        import ipdb; ipdb.set_trace()


        # random agent for debug
        x = np.random.rand(2)
        q = np.random.rand(1) > 0

        action = {'position': x,
                'pressure': 1.0,
                'color': (0.0, 0.0, 0.0),
                'prob': q}
        return action
    
    def stop_episode_and_train(self, obs, r, done=None):
        """ Update generator and discriminator at the end of drawing """
        pass

    def get_statistics(self):
        # TODO: implement here
        return []

    def stop_episode(self):
        pass

    def act(self, obs):
        # random agent for debug
        x = np.random.rand(2)
        q = np.random.rand(1) > 0

        action = {'position': x,
                'pressure': 1.0,
                'color': (0.0, 0.0, 0.0),
                'prob': q}
        return action