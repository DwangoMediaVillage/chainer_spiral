from enviroments.mypaint_env import MyPaintEnv
from nose.tools import eq_

def test_init_env():
    # initialize environment
    env = MyPaintEnv()

def test_reset_env():
    # canvas can be cleaned
    env = MyPaintEnv()
    
    obs = env.reset()
    tmp = obs['image'].sum()
    
    for _ in range(10):
        obs, _, _, _  = env.step(env.action_space.sample())

    obs = env.reset()
    eq_(tmp, obs['image'].sum())

    for _ in range(10):
        obs, _, _, _  = env.step(env.action_space.sample())

    obs = env.reset()
    eq_(tmp, obs['image'].sum())

if __name__ == '__main__':
    test_reset_env()