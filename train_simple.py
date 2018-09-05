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

import numpy as np
import chainer
import cv2

import gym
gym.undo_logger_setup()  # NOQA
import gym.wrappers

from chainerrl import experiments
from chainerrl import misc
from chainerrl.optimizers.nonbias_weight_decay import NonbiasWeightDecay
from chainerrl.optimizers import rmsprop_async

from environments import MyPaintEnv
from agents import spiral
from models.spiral import SpiralDiscriminator, SPIRALSimpleModel
from utils.arg_utils import load_args, print_args
from utils.stat_utils import get_model_param_sum
from utils.data_utils import get_mnist


def main():
    import logging

    parser = argparse.ArgumentParser()
    parser.add_argument('processes', type=int)
    parser.add_argument('--logger_level', type=int, default=logging.DEBUG)
    parser.add_argument('--outdir', type=str, default='results',
                            help='Directory path to save output files.'
                                ' If it does not exist, it will be created.')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed [0, 2 ** 32)')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--eval_interval', type=int, default=10)
    parser.add_argument('--eval_n_runs', type=int, default=2)
    parser.add_argument('--load', type=str, default='')
    parser.add_argument('--demo', action='store_true', default=False)
    parser.add_argument('--rollout_n', type=int, default=2)
    parser.add_argument('--profile', action='store_true')
    parser.add_argument('--gamma', type=float, default=0.9)
    parser.add_argument('--beta', type=float, default=1e-2)
    parser.add_argument('--gp_lambda', type=float, default=10.0)
    parser.add_argument('--continuous_drawing_lambda', type=float, default=0.1)
    parser.add_argument('--empty_drawing_penalty', type=float, default=1.0)
    parser.add_argument('--max_episode_steps', type=int, default=5)
    parser.add_argument('--save_global_step_interval', type=int, default=10)
    parser.add_argument('--target_label', type=int, default=1)
    parser.add_argument('--lambda_R', type=float, default=1.0)
    parser.add_argument('--gumbel_tmp', type=float, default=0.1)
    parser.add_argument('--reward_mode', default='l2')
    parser.add_argument('--save_final_obs_update_interval', type=int, default=10)
    args = parser.parse_args()

    # init a logger
    logging.basicConfig(level=args.logger_level)

    # load arguments from the load directory
    if args.demo and args.load:
        arg_log = os.path.abspath(
            os.path.join(args.load, os.pardir, 'args.txt')
        )
        args = load_args(args, arg_log, exceptions=('load', 'demo'))
        logging.info('Load arguments: ')
        print_args(args)

    # Set a random seed used in ChainerRL.
    # If you use more than one processes, the results will be no longer
    # deterministic even with the same random seed.
    misc.set_random_seed(args.seed)

    # create directory to put the results
    if not (args.load and args.demo):
        args.outdir = experiments.prepare_output_dir(args, args.outdir)

    # define func to create env
    def make_env(process_idx, test):
        env = MyPaintEnv(max_episode_steps=args.max_episode_steps)  
        return env

    sample_env = MyPaintEnv(max_episode_steps=args.max_episode_steps)
    
    # TODO: MyPaintEnv is not wrapped by EnvSpec
    timestep_limit = sample_env.tags['max_episode_steps']
    obs_space = sample_env.observation_space
    action_space = sample_env.action_space

    in_channel = 1

    gen = SPIRALSimpleModel(obs_space, action_space, in_channel, args.gumbel_tmp)  # generator
    dis = SpiralDiscriminator(in_channel)  # discriminator

    gen_opt = chainer.optimizers.Adam(alpha=args.lr, beta1=0.5)
    dis_opt = chainer.optimizers.Adam(alpha=args.lr, beta1=0.5)

    gen_opt.setup(gen)
    dis_opt.setup(dis)

    gen_opt.add_hook(chainer.optimizer.GradientClipping(40))
    dis_opt.add_hook(chainer.optimizer.GradientClipping(40))
    if args.weight_decay > 0:
        gen_opt.add_hook(NonbiasWeightDecay(args.weight_decay))
        dis_opt.add_hook(NonbiasWeightDecay(args.weight_decay))

    # target image dataset
    train, target_data_sampler = get_mnist(sample_env.imsize, single_class=True, target_label=args.target_label)
    
    save_final_obs = not args.demo
    
    agent = spiral.SPIRAL(
        generator=gen,
        discriminator=dis,
        gen_optimizer=gen_opt,
        dis_optimizer=dis_opt,
        in_channel=in_channel,
        target_data_sampler=target_data_sampler,
        timestep_limit=timestep_limit,
        rollout_n=args.rollout_n,
        gamma=args.gamma,
        beta=args.beta,
        gp_lambda=args.gp_lambda,
        continuous_drawing_lambda=args.continuous_drawing_lambda,
        empty_drawing_penalty=args.empty_drawing_penalty,
        lambda_R=args.lambda_R,
        reward_mode=args.reward_mode,
        save_final_obs_update_interval=args.save_final_obs_update_interval,
        outdir=args.outdir,
        save_final_obs=save_final_obs
    )

    step_hook = spiral.SpiralStepHook(timestep_limit, args.save_global_step_interval, args.outdir)

    if args.load:
        print(f"sum of params before loading: {get_model_param_sum(agent.generator.pi)}")
        agent.load(args.load)
        print(f"sum of params after loading: {get_model_param_sum(agent.generator.pi)}")

    if args.demo:
        from spiral_evaluator import show_drawn_pictures, run_demo

        env = make_env(0, True)
        if args.load:
            savename = os.path.join(args.load, 'result.mp4')
        else:
            savename = os.path.join(args.outdir, 'result.mp4')
        
        run_demo(env, agent, timestep_limit, suptitle=args.load, savename=savename)

    else:
        if args.processes == 1:
            agent.process_idx = 0
            experiments.train_agent_with_evaluation(
                agent=agent,
                outdir=args.outdir,
                env=make_env(0, False),
                steps=args.steps,
                eval_n_runs=args.eval_n_runs,
                eval_interval=args.eval_interval,
                max_episode_len=timestep_limit * args.rollout_n,
                step_hooks=[step_hook],
                save_best_so_far_agent=False
            )
        else:
            experiments.train_agent_async(
                agent=agent,
                outdir=args.outdir,
                processes=args.processes,
                make_env=make_env,
                profile=args.profile,
                steps=args.steps,
                eval_n_runs=args.eval_n_runs,
                eval_interval=args.eval_interval,
                max_episode_len=timestep_limit * args.rollout_n,
                global_step_hooks=[step_hook],
                save_best_so_far_agent=False
            )


if __name__ == '__main__':
    main()
