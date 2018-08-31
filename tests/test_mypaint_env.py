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

def test_env_after_reset():
    # nothing will be drawn with prob=0 even after reset
    env = MyPaintEnv()
    obs = env.reset()
    tmp = obs['image'].sum()

    act = {'position': 707, 'pressure': 0.9, 'color': (0, 0, 0), 'prob': 1}
    obs, _, _, _ = env.step(act)
    act = {'position': 600, 'pressure': 0.9, 'color': (0, 0, 0), 'prob': 1}
    obs, _, _, _ = env.step(act)

    env.reset()
    act = {'position': 607, 'pressure': 0.9, 'color': (0, 0, 0), 'prob': 0}
    obs, _, _, _ = env.step(act)
    eq_(tmp, obs['image'].sum())

if __name__ == '__main__':
    test_reset_env()