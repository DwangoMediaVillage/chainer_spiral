import os

from chainerrl.optimizers import rmsprop_async
from nose.tools import eq_

from chainer_spiral.agents import spiral
from chainer_spiral.dataset.toy_dataset import ToyDataset
from chainer_spiral.environments import ToyEnv
from chainer_spiral.models.spiral import SpiralToyDiscriminator, SpiralToyModel


def init_agent():
    # initialize an agent
    imsize = 3
    ToyEnv(imsize)
    G = SpiralToyModel(imsize, False)
    D = SpiralToyDiscriminator(imsize, False)
    G_opt = rmsprop_async.RMSpropAsync()
    D_opt = rmsprop_async.RMSpropAsync()
    G_opt.setup(G)
    D_opt.setup(D)
    p = [(1, 4, 7)]
    dataset = ToyDataset(imsize, p, p)

    agent = spiral.SPIRAL(generator=G,
                          discriminator=D,
                          gen_optimizer=G_opt,
                          dis_optimizer=D_opt,
                          dataset=dataset,
                          conditional=True,
                          reward_mode='wgangp',
                          imsize=imsize,
                          max_episode_steps=3,
                          rollout_n=1,
                          gamma=0.99,
                          beta=0.001,
                          gp_lambda=10.0,
                          lambda_R=1.0,
                          staying_penalty=10.0,
                          empty_drawing_penalty=1.0,
                          n_save_final_obs_interval=10000,
                          outdir='/tmp/chainer_spiral_test')
    return agent


def test_save_and_load():
    # check the parameters are same between before save and after loading

    # create tmp dir
    save_dir = '/tmp/chainer_spiral_test'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    agent = init_agent()

    # insert some value to the generator
    w = agent.generator.pi.e1_c1.W.data
    w = w * 10
    agent.generator.pi.e1_c1.W.data = w

    # save agent to save_dir
    agent.snap(0, save_dir)

    # re-init agent
    del agent
    agent = init_agent()

    # load parameters from the snap
    agent.load(os.path.join(save_dir, '0'))

    eq_(agent.generator.pi.e1_c1.W.data.sum(), w.sum())
