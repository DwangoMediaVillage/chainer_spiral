from chainer_spiral.environments.mypaint_env import MyPaintEnv
from nose.tools import eq_

def test_init_env():
    # initialize environment
    env = MyPaintEnv(brush_info_file='settings/my_simple_brush.myb')

def test_reset_env():
    # canvas can be cleaned
    env = MyPaintEnv(brush_info_file='settings/my_simple_brush.myb')
    
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
    env = MyPaintEnv(brush_info_file='settings/my_simple_brush.myb')
    obs = env.reset()
    tmp = obs['image'].sum()

    act = {'position': 707, 'pressure': 0.9, 'color': (0, 0, 0), 'prob': 1}
    obs, _, _, _ = env.step(act)
    act = {'position': 600, 'pressure': 0.9, 'color': (0, 0, 0), 'prob': 1}
    obs, _, _, _ = env.step(act)

    obs = env.reset()
    eq_(tmp, obs['image'].sum())
    act = {'position': 607, 'pressure': 0.9, 'color': (0, 0, 0), 'prob': 0}
    act = {'position': 300, 'pressure': 0.9, 'color': (0, 0, 0), 'prob': 0}
    obs, _, _, _ = env.step(act)
    eq_(tmp, obs['image'].sum())


def test_env_point_conversion():
    pos_resolution = 32
    imsize = 64
    env = MyPaintEnv(pos_resolution=pos_resolution, imsize=imsize, brush_info_file='settings/my_simple_brush.myb')

    # idx 0 is left top
    x, y = env.convert_x(0)
    eq_(x, env.tile_offset)
    eq_(y, env.tile_offset)

    # idx pos_resolution ** 2 is right bottom
    x, y = env.convert_x(pos_resolution * pos_resolution - 1)
    eq_(x, env.tile_offset + imsize - 2)
    eq_(y, env.tile_offset + imsize - 2)
    

if __name__ == '__main__':
    test_reset_env()
