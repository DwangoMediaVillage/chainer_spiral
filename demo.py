"""Drawing pictures using a trained model.
"""
import argparse
import logging
import os

import chainer
import yaml

from chainer_spiral.agents import spiral
from chainer_spiral.dataset import (EMnistDataset, JikeiDataset, MnistDataset, QuickdrawDataset,
                                    ToyDataset)
from chainer_spiral.environments import MyPaintEnv, ToyEnv
from chainer_spiral.models import (SpiralMnistDiscriminator, SpiralMnistModel,
                                   SpiralToyDiscriminator, SpiralToyModel)
from chainer_spiral.utils.arg_utils import print_args
from chainer_spiral.utils.evaluators import (demo_many, demo_movie, demo_output_json, demo_static)

# This prevents numpy from using multiple threads
os.environ['OMP_NUM_THREADS'] = '1'  # NOQA


def demo():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', type=str, choices=['static', 'many', 'movie', 'json'])
    parser.add_argument('load', type=str, help='target directory to load trained params')
    parser.add_argument('savename', type=str)
    args = parser.parse_args()
    print_args(args)

    # init a logger
    logging.basicConfig(level=logging.INFO)

    # check load dirtectory exists
    assert os.path.exists(args.load), f"{args.load} does not exist!"

    # load config from load directory
    with open(os.path.join(args.load, os.pardir, 'config.yaml')) as f:
        config = yaml.load(f)

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
            train_patterns = [(1, 4, 7), (0, 1, 2), (3, 4, 5), (2, 5, 8)]
            test_patterns = [(6, 7, 8)]
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
            dataset = MnistDataset(config['imsize'], single_label, config['mnist_target_label'],
                                   config['mnist_binarization'])
        elif config['problem'] == 'emnist':
            dataset = EMnistDataset(config['emnist_gz_images'], config['emnist_gz_labels'],
                                    config['emnist_single_label'])
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
    agent = spiral.SPIRAL(generator=gen,
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
                          outdir=os.path.join(args.load, os.pardir))

    # load from a snapshot
    agent.load(args.load)

    # run demo
    env = make_env(0, True)

    suptitle = args.load

    if args.mode == 'static':
        demo_static(env,
                    agent,
                    config,
                    args.savename,
                    suptitle,
                    dataset,
                    plot_act=config['problem'] != 'toy')
    elif args.mode == 'movie':
        demo_movie(env,
                   agent,
                   config,
                   args.savename,
                   suptitle,
                   dataset,
                   plot_act=config['problem'] != 'toy')
    elif args.mode == 'many':
        demo_many(env, agent, config, args.savename, suptitle, dataset)
    elif args.mode == 'json':
        demo_output_json(env, agent, config, args.savename, dataset)
    else:
        raise NotImplementedError('Invalid demo mode')


if __name__ == '__main__':
    demo()
