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

def main():
    import logging

    parser = argparse.ArgumentParser()
    parser.add_argument('--logger-level', type=int, default=logging.DEBUG)

    args = parser.parse_args()

    logging.basicConfig(level=args.logger_level)

    logger = logging.getLogger(__name__)

    logger.debug('training SPIRAL agent with simple shapes')

if __name__ == '__main__':
    main()
