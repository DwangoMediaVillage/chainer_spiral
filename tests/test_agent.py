from nose.tools import eq_
import os
from agents import spiral
from models.spiral import SpiralDiscriminator, SPIRALSimpleModel
from environments import MyPaintEnv
from chainerrl.optimizers import rmsprop_async

def init_agent():
    # initialize an agent
    env = MyPaintEnv()
    in_channel = 1
    G = SPIRALSimpleModel(env.observation_space, env.action_space, in_channel)
    D = SpiralDiscriminator(in_channel)
    G_opt = rmsprop_async.RMSpropAsync()
    D_opt = rmsprop_async.RMSpropAsync()
    G_opt.setup(G)
    D_opt.setup(D)
    def data_sampler():
        pass
    rollout_n = 1
    timestep_limit = env.tags['max_episode_steps']
    agent = spiral.SPIRAL(G, D, G_opt, D_opt, data_sampler, in_channel, timestep_limit, rollout_n)
    return agent

def test_save_and_load():
    # check the parameters are same between before save and after loading
    
    # create tmp dir
    save_dir = '/tmp/chainer_spiral_test'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    agent = init_agent()

    # insert some value to the generator
    w = agent.generator.pi.conv2.W.data
    w = w * 10
    agent.generator.pi.conv2.W.data = w

    # save agent to save_dir
    agent.snap(0, save_dir)

    # re-init agent
    del agent
    agent = init_agent()

    # load parameters from the snap
    agent.load(os.path.join(save_dir, '0'))

    eq_(agent.generator.pi.conv2.W.data.sum(), w.sum())
