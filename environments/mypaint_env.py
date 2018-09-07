import gym
import os

from lib import mypaintlib
from lib import tiledsurface
from lib import brush
from lib import pixbufsurface
import numpy as np

import logging
import time
import math

class MyPaintEnv(gym.Env):
    action_space = None
    observation_space = None
    reward_range = None
    viewer = None

    metadata = {
        'render.modes': ['human', 'rgb_array'],
    }

    def __init__(self, logger=None, imsize=64, bg_color=None, max_episode_steps=10, pos_resolution=32, brush_info_file=None):
        """ initialize environment """
        super().__init__()

        # TODO: use EnvSpec?

        # length of a sequence
        self.tags = {'max_episode_steps': max_episode_steps}

        self.logger = logger or logging.getLogger(__name__)

        # starting positing of the pen for each episode
        self.tile_offset = mypaintlib.TILE_SIZE
        self.start_x = 0
        self.start_color = (0, 0, 0)  # initial color of the brush
        self.imsize = imsize
        self.pos_resolution = pos_resolution

        # action space
        self.action_space = gym.spaces.Dict({
            'position': gym.spaces.Discrete(self.pos_resolution ** 2),
            'pressure': gym.spaces.Box(low=0, high=1.0, shape=()),
            'color': gym.spaces.Box(low=0, high=1.0, shape=(3,)),
            'prob': gym.spaces.Discrete(2)
        })

        # observation space
        self.observation_space = gym.spaces.Dict({
            'image': gym.spaces.Box(low=0, high=255, shape=(self.imsize, self.imsize, 3), dtype=np.uint8),
            'position': gym.spaces.Discrete(self.pos_resolution ** 2),
            'pressure': gym.spaces.Box(low=0, high=1.0, shape=()),
            'color': gym.spaces.Box(low=0, high=1.0, shape=(3,)),
            'prob': gym.spaces.Discrete(1)
        })

        # color of the background
        if bg_color is None:
            self.bg_color = (1, 1, 1)
        else:
            self.bg_color = bg_color

        # open brush 
        if brush_info_file is None:
            brush_info_file = os.getenv('BRUSHINFO')
            if brush_info_file is None:
                raise ValueError('You need to specify brush file')

        self.logger.debug('Open brush info from %s', brush_info_file)

        with open(brush_info_file, 'r') as f:
            self.brush_info = brush.BrushInfo(f.read())
        self.brush = brush.Brush(self.brush_info)

        # initialize canvas (surface)
        self.surface = tiledsurface.Surface()

        # reset canvas and set current position of the pen
        self.reset()

    def step(self, action):
        """ draw by action.
        Assuming that action is a dict whose keys are (x, p, c, q)
        x: tuple of float. (x, y) of the next point
        p: float. pressure
        c: tuple of float. (r, g, b). color value of the brush [0, 1] 
        q: integer = (0, 1). a binary flag specifying the type of action: draw or skip to the next point w/o drawing
        """

        x = action['position']
        p = action['pressure']
        r, g, b = action['color']
        q = action['prob']

        color = (float(r), float(g), float(b))

        self.brush_info.set_color_rgb(color)

        # maybe draw
        if q:
            self.__draw(x, float(p))
        else:
            self.brush.reset()
            self.__draw(x, 0.0)
        
        # Fake reward and done flag
        reward = 0.0
        done = False

        # create observation: current drawn picture image and the input action
        ob = {'image': self._get_rgb_array(),
                'position': x,
                'pressure': p,
                'color': color,
                'prob': q
            }
        return ob, reward, done, {}

    def convert_x(self, x):
        """ convert position id -> a point (p1, p2) """
        assert x < self.pos_resolution ** 2
        p1 = (x % self.pos_resolution) / self.pos_resolution * self.imsize + self.tile_offset
        p2 = (x // self.pos_resolution) / self.pos_resolution * self.imsize + self.tile_offset
        return int(p1), int(p2)

    def __draw(self, x, pressure, xtilt=0, ytilt=0, dtime=0.1, viewzoom=1.0, viewrotation=0.0):
        p1, p2 = self.convert_x(x)
        self.surface.begin_atomic()
        self.brush.stroke_to(
            self.surface.backend,
            p1,
            p2,
            pressure,
            xtilt,
            ytilt,
            dtime,
            viewzoom,
            viewrotation
        )
        self.surface.end_atomic()

        # update the current point
        self.x = x

    def reset(self):
        """ clear all the content on the canvas, move the current position to the default """
        self.logger.debug('reset the drawn picture')
        
        # clear content on the canvas
        self.surface.clear()

        # fill the canvas with the background color
        with self.surface.cairo_request(0, 0, self.imsize + self.tile_offset * 2, self.imsize + self.tile_offset * 2) as cr:
            r, g, b = self.bg_color
            cr.set_source_rgb(r, g, b)
            cr.rectangle(self.tile_offset, self.tile_offset, self.imsize + self.tile_offset * 2, self.imsize + self.tile_offset * 2)
            cr.fill()

        # set the pen's initial position
        self.brush.reset()
        self.__draw(self.start_x, 0)
        self.x = self.start_x
        

        # create observation: current drawn picture image and the input action
        ob = {'image': self._get_rgb_array(),
                'position': self.start_x,
                'pressure': 0.0,
                'color': self.start_color,
                'prob': 0
            }
        return ob

    def render(self, mode='human'):
        """ render the current drawn picture image for human """
        if mode == 'human':
            if self.viewer is None:
                from gym.envs.classic_control import rendering
                self.viewer = rendering.SimpleImageViewer()
            self.viewer.imshow(self._get_rgb_array())

        elif mode == 'rgb_array':
            return self._get_rgb_array()
        else:
            raise NotImplementedError


    def _get_rgb_array(self, cut=True):
        """ render the current canvas as a rgb array
        """
        buf = pixbufsurface.render_as_pixbuf(self.surface)
        w = buf.get_width()
        h = buf.get_height()

        # convert uint8 matrix whose shape is [w, h, 4]
        img = np.frombuffer(buf.get_pixels(), np.uint8).reshape(h, w, -1)
        img = img[:, :, :3]  # discard the alpha channel

        # cut out the canvas
        if cut:
            img = img[self.tile_offset:self.tile_offset+self.imsize,
                    self.tile_offset:self.tile_offset+self.imsize, :]

        return img

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
    
    def seed(self, seed=None):
        # TODO: implement here
        pass
        
    
if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(__name__)
    env = MyPaintEnv(logger=logger)

    # drawing something
    env.step({'position': 32 * 3 + 16, 'pressure': 1.0, 'color': (0, 0, 0), 'prob': 0})
    env.step({'position': 32 * 30 + 16, 'pressure': 1.0, 'color': (0, 0, 0), 'prob': 1})

    env.reset()

    env.step({'position': 32 * 3 + 16, 'pressure': 1.0, 'color': (0, 0, 0), 'prob': 0})
    env.step({'position': 32 * 30 + 16, 'pressure': 1.0, 'color': (0, 0, 0), 'prob': 1})

    import matplotlib
    matplotlib.use('MacOSX')
    import matplotlib.pyplot as plt

    obs = env.render('rgb_array')
    plt.imshow(obs)
    plt.show()

    env.close()
