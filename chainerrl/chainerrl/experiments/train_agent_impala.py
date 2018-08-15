from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
from __future__ import absolute_import
from builtins import *  # NOQA
from future import standard_library
standard_library.install_aliases()  # NOQA

import logging
import multiprocessing as mp
import os

from chainerrl.experiments.evaluator import AsyncEvaluator
from chainerrl.misc import async_
from chainerrl.misc import random_seed

def train_loop():
    pass

def train_agent_impala(outdir, processes, make_env,
                       steps=10,
                       agent=None,
                       make_agent=None,
                       logger=None):
    """Run agent asynchronously using multiprocessing, and train the policy on a learner.

    Args:
        outdir (str): Path to the directory to output things.
        processes (int): Number of processes.
        make_env (callable): (process_idx, test) -> Environment.
        steps (int): Number of global time steps for training.
        agent (Agent): Agent to train.
        make_agent (callable): (process_idx) -> Agent
        logger (logging.Logger): Logger used in this function.
    """

    logger = logger or logging.getLogger(__name__)

    # Prevent numpy from using multiple threads
    os.environ['OMP_NUM_THREADS'] = '1'

    if agent is None:
        assert make_agent is not None
        agent = make_agent(0)

    # TODO: add evaluator

    def run_func(process_idx):
        logger.debug("Hi, I'm worker {}".format(process_idx))
    
    async_.run_async(processes, run_func)