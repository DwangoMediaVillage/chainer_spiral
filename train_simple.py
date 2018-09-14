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

from environments import MyPaintEnv, ToyEnv
from agents import spiral
from utils.arg_utils import load_args, print_args
from utils.stat_utils import get_model_param_sum
from dataset.mnist_dataset import MnistDataset
from dataset.toy_dataset import ToyDataset
from dataset.emnist_dataset import EmnistDataset
from dataset.jikei_dataset import JikeiDataset
from models.spiral import SpiralMnistModel, SpiralToyModel, SpiralMnistDiscriminator, SpiralToyDiscriminator

def main():
    import logging

    parser = argparse.ArgumentParser()
    parser.add_argument('processes', type=int)
    parser.add_argument('--brush_info_file')
    parser.add_argument('--logger_level', type=int, default=logging.INFO)
    parser.add_argument('--outdir', type=str, default='results',
                            help='Directory path to save output files.'
                                ' If it does not exist, it will be created.')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed [0, 2 ** 32)')
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--steps', type=int, default=1000000)
    parser.add_argument('--eval_interval', type=int, default=100)
    parser.add_argument('--eval_n_runs', type=int, default=1)
    parser.add_argument('--load', type=str, default='')
    parser.add_argument('--demo')
    parser.add_argument('--rollout_n', type=int, default=5)
    parser.add_argument('--profile', action='store_true')
    parser.add_argument('--gamma', type=float, default=1.0)
    parser.add_argument('--beta', type=float, default=1e-2)
    parser.add_argument('--gp_lambda', type=float, default=10.0)
    parser.add_argument('--continuous_drawing_lambda', type=float, default=0.0)
    parser.add_argument('--empty_drawing_penalty', type=float, default=10.0)
    parser.add_argument('--max_episode_steps', type=int, default=3)
    parser.add_argument('--save_global_step_interval', type=int, default=1000)
    parser.add_argument('--lambda_R', type=float, default=1.0)
    parser.add_argument('--reward_mode', default='l2')
    parser.add_argument('--save_final_obs_update_interval', type=int, default=100)
    parser.add_argument('--mnist_target_label', type=int)
    parser.add_argument('--problem', default='toy')
    parser.add_argument('--mnist_binarization', action='store_true')
    parser.add_argument('--demo_savename')
    parser.add_argument('--staying_penalty', type=float, default=0.0)
    parser.add_argument('--conditional', action='store_true')
    parser.add_argument('--jikei_npz')
    parser.add_argument('--emnist_gz')
    args = parser.parse_args()
    print_args(args)

    # init a logger
    logging.basicConfig(level=args.logger_level)

    # check the steps argument is correct
    if args.steps % (args.rollout_n * args.max_episode_steps) != 0:
        logging.warn('steps % (rollout_n * max_episode_steps steps) != 0, so that the training will end with error!')

    # load arguments from the load directory
    if args.demo and args.load:
        arg_log = os.path.abspath(
            os.path.join(args.load, os.pardir, 'args.txt')
        )
        args = load_args(args, arg_log, exceptions=('load', 'demo', 'demo_savename'))
        logging.info('Load arguments: ')
        print_args(args)

    # Set a random seed used in ChainerRL.
    # If you use more than one processes, the results will be no longer
    # deterministic even with the same random seed.
    misc.set_random_seed(args.seed)

    # create directory to put the results
    if not (args.load and args.demo):
        args.outdir = experiments.prepare_output_dir(args, args.outdir)

    # define func to create env, target data sampler, and models
    # TODO: MyPaintEnv is not wrapped by EnvSpec
    if args.problem == 'toy':
        imsize = 3
        in_channel = 1
        obs_pos_dim = imsize * imsize

        def make_env(process_idx, test):
            env = ToyEnv(imsize)
            return env

        if args.conditional:
            train_patterns = [
                (1, 4, 7), (0, 1, 2), (3, 4, 5), (2, 5, 8)
            ]
            test_patterns = [
                (6, 7, 8)
            ]
        else:
            train_patterns = [(1, 4, 7)]
            test_patterns = train_patterns

        dataset = ToyDataset(imsize, train_patterns, test_patterns)
        gen = SpiralToyModel(imsize, args.conditional)
        dis = SpiralToyDiscriminator(imsize, args.conditional)

        def preprocess_image(x):
            """ function to preprocess image from observation """
            x = cv2.cvtColor(x, cv2.COLOR_RGB2GRAY)  # unit8, 2d matrix
            x = x.astype(np.float32) / 255.0
            x = np.reshape(x, [1, 1] + list(x.shape))
            return x

        def preprocess_obs(obs):
            """ function to preprocess observation from ToyEnv """
            c = obs['image']
            x = obs['position']
            q = obs['prob']

            # image
            c = preprocess_image(c)

            # position
            x /= float(obs_pos_dim)
            x = np.asarray(x, dtype=np.float32) 
            x = np.reshape(x, (1, 1))

            # prob
            q = np.asarray(q, dtype=np.float32)
            q = np.reshape(q, (1, 1))
            
            # return state as a tuple
            return c, x, q

    elif args.problem == 'mnist':
        imsize = 64
        in_channel = 1
        pos_resolution = 32
        obs_pos_dim = imsize * imsize

        def make_env(process_idx, test):
            env = MyPaintEnv(max_episode_steps=args.max_episode_steps,
                            imsize=imsize, pos_resolution=pos_resolution, brush_info_file=args.brush_info_file)
            return env

        if args.mnist_target_label is not None:
            dataset = MnistDataset(imsize, True, args.mnist_target_label, args.mnist_binarization)
        else:
            dataset = MnistDataset(imsize, False, binarization=args.mnist_binarization)

        gen = SpiralMnistModel(imsize, args.conditional)
        dis = SpiralMnistDiscriminator(imsize, args.conditional)

        def preprocess_image(x):
            """ function to preprocess image from observation """
            x = cv2.cvtColor(x, cv2.COLOR_RGB2GRAY)  # unit8, 2d matrix
            x = x.astype(np.float32) / 255.0
            x = np.reshape(x, [1, 1] + list(x.shape))
            return x

        def preprocess_obs(obs):
            """ function to preprocess observation from MyPaintEnv with MNIST (black and white images) """
            c = obs['image']
            x = obs['position']
            q = obs['prob']

            # image
            c = preprocess_image(c)

            # position
            x /= float(obs_pos_dim)
            x = np.asarray(x, dtype=np.float32) 
            x = np.reshape(x, (1, 1))

            # prob
            q = np.asarray(q, dtype=np.float32)
            q = np.reshape(q, (1, 1))
            
            # return state as a tuple
            return c, x, q
    
    elif args.problem == 'jikei':
        imsize = 64
        in_channel = 1
        pos_resolution = 32
        obs_pos_dim = imsize * imsize

        def make_env(process_idx, test):
            env = MyPaintEnv(max_episode_steps=args.max_episode_steps,
                            imsize=imsize, pos_resolution=pos_resolution, brush_info_file=args.brush_info_file)
            return env

        assert not args.conditional, "conditional generation for jikei is not implemented!"

        dataset = JikeiDataset(args.jikei_npz)

        gen = SpiralMnistModel(imsize, args.conditional)
        dis = SpiralMnistDiscriminator(imsize, args.conditional)

        def preprocess_image(x):
            """ function to preprocess image from observation """
            x = cv2.cvtColor(x, cv2.COLOR_RGB2GRAY)  # unit8, 2d matrix
            x = x.astype(np.float32) / 255.0
            x = np.reshape(x, [1, 1] + list(x.shape))
            return x

        def preprocess_obs(obs):
            """ function to preprocess observation from MyPaintEnv with MNIST (black and white images) """
            c = obs['image']
            x = obs['position']
            q = obs['prob']

            # image
            c = preprocess_image(c)

            # position
            x /= float(obs_pos_dim)
            x = np.asarray(x, dtype=np.float32) 
            x = np.reshape(x, (1, 1))

            # prob
            q = np.asarray(q, dtype=np.float32)
            q = np.reshape(q, (1, 1))
            
            # return state as a tuple
            return c, x, q

    elif args.problem == 'emnist':
        imsize = 64
        in_channel = 1
        pos_resolution = 32
        obs_pos_dim = imsize * imsize

        def make_env(process_idx, test):
            env = MyPaintEnv(max_episode_steps=args.max_episode_steps,
                            imsize=imsize, pos_resolution=pos_resolution, brush_info_file=args.brush_info_file)
            return env

        assert not args.conditional, "conditional generation for emnist is not implemented!"

        dataset = Emnist(args.emnist_gz)

        gen = SpiralMnistModel(imsize, args.conditional)
        dis = SpiralMnistDiscriminator(imsize, args.conditional)

        def preprocess_image(x):
            """ function to preprocess image from observation """
            x = cv2.cvtColor(x, cv2.COLOR_RGB2GRAY)  # unit8, 2d matrix
            x = x.astype(np.float32) / 255.0
            x = np.reshape(x, [1, 1] + list(x.shape))
            return x

        def preprocess_obs(obs):
            """ function to preprocess observation from MyPaintEnv with MNIST (black and white images) """
            c = obs['image']
            x = obs['position']
            q = obs['prob']

            # image
            c = preprocess_image(c)

            # position
            x /= float(obs_pos_dim)
            x = np.asarray(x, dtype=np.float32) 
            x = np.reshape(x, (1, 1))

            # prob
            q = np.asarray(q, dtype=np.float32)
            q = np.reshape(q, (1, 1))
            
            # return state as a tuple
            return c, x, q


    else:
        raise NotImplementedError()

    # initialize optimizers
    gen_opt = chainer.optimizers.Adam(alpha=args.lr, beta1=0.5)
    dis_opt = chainer.optimizers.Adam(alpha=args.lr, beta1=0.5)

    gen_opt.setup(gen)
    dis_opt.setup(dis)

    gen_opt.add_hook(chainer.optimizer.GradientClipping(40))
    dis_opt.add_hook(chainer.optimizer.GradientClipping(40))
    if args.weight_decay > 0:
        gen_opt.add_hook(NonbiasWeightDecay(args.weight_decay))
        dis_opt.add_hook(NonbiasWeightDecay(args.weight_decay))

    # initialize agent
    save_final_obs = not args.demo

    agent = spiral.SPIRAL(
        generator=gen,
        discriminator=dis,
        gen_optimizer=gen_opt,
        dis_optimizer=dis_opt,
        in_channel=in_channel,
        dataset=dataset,
        preprocess_image_func=preprocess_image,
        preprocess_obs_func=preprocess_obs,
        timestep_limit=args.max_episode_steps,
        rollout_n=args.rollout_n,
        obs_pos_dim=obs_pos_dim,
        conditional=args.conditional,
        gamma=args.gamma,
        beta=args.beta,
        gp_lambda=args.gp_lambda,
        continuous_drawing_lambda=args.continuous_drawing_lambda,
        empty_drawing_penalty=args.empty_drawing_penalty,
        lambda_R=args.lambda_R,
        reward_mode=args.reward_mode,
        save_final_obs_update_interval=args.save_final_obs_update_interval,
        outdir=args.outdir,
        save_final_obs=save_final_obs,
        staying_penalty=args.staying_penalty
    )

    # load from a snapshot
    if args.load:
        agent.load(args.load)

    if args.demo:
        # demo mode
        from spiral_evaluator import demo_static, demo_movie, demo_many
        env = make_env(0, True)

        if args.demo_savename:
            savename = args.demo_savename
        else:
            savedir = args.load if args.load else args.outdir
            if args.demo == 'static':
                savename = os.path.join(savedir, 'static_result.png')
            elif args.demo == 'movie':
                savename = os.path.join(savedir, 'movie_result.mp4')
            elif args.demo == 'many':
                savename = os.path.join(savedir, 'many_result.png')
            else:
                raise NotImplementedError('Invalid demo mode')

        suptitle = args.load if args.load else args.outdir

        # run demo
        if args.demo == 'static':
            demo_static(env, agent, args, savename, suptitle, dataset, plot_act=args.problem != 'toy')
        elif args.demo == 'movie':
            demo_movie(env, agent, args, savename, suptitle, dataset, plot_act=args.problem != 'toy')
        elif args.demo == 'many':
            demo_many(env, agent, args, savename, suptitle, dataset)
        else:
            raise NotImplementedError('Invalid demo mode')
    
    else:
        # training mode
        step_hook = spiral.SpiralStepHook(args.max_episode_steps, args.save_global_step_interval, args.outdir)

        if args.processes == 1:
            agent.process_idx = 0
            env = make_env(0, False)
            experiments.train_agent_with_evaluation(
                agent=agent,
                outdir=args.outdir,
                env=env,
                steps=args.steps,
                eval_n_runs=args.eval_n_runs,
                eval_interval=args.eval_interval,
                max_episode_len=args.max_episode_steps * args.rollout_n,
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
                max_episode_len=args.max_episode_steps * args.rollout_n,
                global_step_hooks=[step_hook],
                save_best_so_far_agent=False
            )


if __name__ == '__main__':
    main()
