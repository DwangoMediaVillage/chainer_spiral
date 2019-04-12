"""An example of training SPIRAL with A3C.
"""
import argparse
import yaml
import os
import logging

# This prevents numpy from using multiple threads
os.environ['OMP_NUM_THREADS'] = '1'  # NOQA

import numpy as np
import chainer

from chainerrl import experiments
from chainerrl import misc
from chainerrl.optimizers.nonbias_weight_decay import NonbiasWeightDecay
from chainerrl.optimizers import rmsprop_async

from chainer_spiral.environments import MyPaintEnv, ToyEnv
from chainer_spiral.agents import SPIRAL, SpiralStepHook
from chainer_spiral.utils.arg_utils import load_args, print_args
from chainer_spiral.dataset import MnistDataset, ToyDataset, EMnistDataset, JikeiDataset, QuickdrawDataset
from chainer_spiral.models import SpiralMnistModel, SpiralToyModel, SpiralMnistDiscriminator, SpiralToyDiscriminator

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='YAML config file')
    parser.add_argument('outdir', type=str, help='directory to put training log')
    parser.add_argument('--profile', action='store_true')
    parser.add_argument('--load', type=str, default='')
    parser.add_argument('--logger_level', type=int, default=logging.INFO)
    args = parser.parse_args()
    print_args(args)
    
    # init a logger
    logging.basicConfig(level=args.logger_level)

    # load yaml config file
    with open(args.config) as f:
        config = yaml.load(f)

    # set random seed
    misc.set_random_seed(config['seed'])

    # create directory to put the results
    args.outdir = experiments.prepare_output_dir(args, args.outdir)
    
    # save config file to outdir
    with open(os.path.join(args.outdir, 'config.yaml'), 'w') as f:
        yaml.dump(config, f, indent=4, default_flow_style=False)

    # define func to create env, target data sampler, and models
    if config['problem'] == 'toy':
        assert config['imsize'] == 3, 'invalid imsize'
        assert config['in_channel'] == 1, 'invalid in_channel'
        
        def make_env(process_idx, test):
            env = ToyEnv(config['imsize'])
            return env
        
        gen = SpiralToyModel(imsize, config['conditional'])
        dis = SpiralToyDiscriminator(imsize, config['conditional'])
        
        if config['conditional']:
            train_patterns = [
                (1, 4, 7), (0, 1, 2), (3, 4, 5), (2, 5, 8)
            ]
            test_patterns = [
                (6, 7, 8)
            ]
        else:
            train_patterns = [(1, 4, 7)]
            test_patterns = train_patterns

        dataset = ToyDataset(config['imsize'], train_patterns, test_patterns)
    
    else:
        # my paint env
        def make_env(process_idx, test):
            env = MyPaintEnv(max_episode_steps=config['max_episode_steps'],
                            imsize=config['imsize'],
                            pos_resolution=config['pos_resolution'],
                            brush_info_file=config['brush_info_file'])
            return env
        
        # generator
        gen = SpiralMnistModel(config['imsize'], config['conditional'])
        dis = SpiralMnistDiscriminator(config['imsize'], config['conditional'])

        if config['problem'] == 'mnist':
            single_label = config['mnist_target_label'] is not None
            dataset = MnistDataset(config['imsize'], single_label, config['mnist_target_label'], config['mnist_binarization'])
        elif config['problem'] == 'emnist':
            dataset = EMnistDataset(config['emnist_gz_images'], config['emnist_gz_labels'], config['emnist_single_label'])
        elif config['problem'] == 'jikei':
            dataset = JikeiDataset(config['jikei_npz'])
        elif config['problem'] == 'quickdraw':
            dataset = QuickdrawDataset(config['quickdraw_npz'])
        else:
            raise NotImplementedError()

    # initialize optimizers
    gen_opt = chainer.optimizers.Adam(alpha=config['lr'], beta1=0.5)
    dis_opt = chainer.optimizers.Adam(alpha=config['lr'], beta1=0.5)

    gen_opt.setup(gen)
    dis_opt.setup(dis)

    gen_opt.add_hook(chainer.optimizer.GradientClipping(40))
    dis_opt.add_hook(chainer.optimizer.GradientClipping(40))
    if config['weight_decay'] > 0:
        gen_opt.add_hook(NonbiasWeightDecay(config['weight_decay']))
        dis_opt.add_hook(NonbiasWeightDecay(config['weight_decay']))

    # init an spiral agent
    agent = SPIRAL(
        generator=gen,
        discriminator=dis,
        gen_optimizer=gen_opt,
        dis_optimizer=dis_opt,
        dataset=dataset,
        conditional=config['conditional'],
        reward_mode=config['reward_mode'],
        imsize=config['imsize'],
        max_episode_steps=config['max_episode_steps'],
        rollout_n=config['rollout_n'],
        gamma=config['gamma'],
        beta=config['beta'],
        gp_lambda=config['gp_lambda'],
        lambda_R=config['lambda_R'],
        staying_penalty=config['staying_penalty'],
        empty_drawing_penalty=config['empty_drawing_penalty'],
        n_save_final_obs_interval=config['n_save_final_obs_interval'],
        outdir=args.outdir
    )

    # load from a snapshot
    if args.load:
        agent.load(args.load)

    # training mode
    max_episode_len = config['max_episode_steps'] * config['rollout_n']
    steps = config['processes'] * config['n_update'] * max_episode_len
    
    save_interval = config['processes'] * config['n_save_interval'] * max_episode_len
    eval_interval = config['processes'] * config['n_eval_interval'] * max_episode_len
    
    step_hook = SpiralStepHook(config['max_episode_steps'], save_interval, args.outdir)

    if config['processes'] == 1:
        # single process for easy to debug
        agent.process_idx = 0
        env = make_env(0, False)

        experiments.train_agent_with_evaluation(
            agent=agent,
            outdir=args.outdir,
            env=env,
            steps=steps,
            eval_n_runs=config['eval_n_runs'],
            eval_interval=eval_interval,
            max_episode_len=max_episode_len,
            step_hooks=[step_hook],
            save_best_so_far_agent=False
        )
    else:
        experiments.train_agent_async(
            agent=agent,
            outdir=args.outdir,
            processes=config['processes'],
            make_env=make_env,
            profile=args.profile,
            steps=steps,
            eval_n_runs=config['eval_n_runs'],
            eval_interval=eval_interval,
            max_episode_len=max_episode_len,
            global_step_hooks=[step_hook],
            save_best_so_far_agent=False
        )

if __name__ == '__main__':
    main()

