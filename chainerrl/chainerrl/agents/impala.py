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

class IMPALAModel(chainer.Link):
    """IMPALA model."""
    pass

class IMPALA(agent.AttributeSavingMixin, agent.AsyncAgent):
    """IMPALA: Importance Weighted Actor-Learner Architectures

    See https://arxiv.org/abs/1802.01561

    Args:
        model (A3CModel): Model to train # TODO: A3CModel -> IMPALAModel
        optimizer (chainer.Optimizer): optimizer used to train the model
    """
    process_idx = None
    
    def __init__(self, model, optimizer):
        pass
    
    def act(self, obs):
        pass
    
    def act_and_train(self, obs, reward):
        pass
    
    def get_statistics(self):
        pass

    @property
    def shared_attributes(self):
        pass

    def stop_episode(self):
        pass

    def stop_episode_and_train(self, state, reward, done=False):
        pass

    