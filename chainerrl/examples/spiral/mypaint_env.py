import gym
import os

from lib import mypaintlib
from lib import tiledsurface
from lib import brush
import numpy as np

import logging

class MyPaintEnv(gym.Env):
    action_space = None
    observation_space = None
    reward_range = None
    viewer = None

    def __init__(self, logger=None, start_x=0, start_y=0):
        """ initialize environment """
        super().__init__()

        self.logger = logger or logging.getLogger(__name__)

        # starting positing of the pen for each episode
        self.start_x = start_x
        self.start_y = start_y

        # open brush 
        brush_info_file = os.getenv('BRUSHINFO')
        if brush_info_file is None:
            raise ValueError('You need to specify brush file by BRUSHINFO')

        self.logger.debug('Open brush info from %s', brush_info_file)

        with open(brush_info_file, 'r') as f:
            self.brush_info = brush.BrushInfo(f.read())
        self.brush = brush.Brush(self.brush_info)

        # initialize canvas (surface)
        self.surface = tiledsurface.Surface()

        # reset canvas and set current position of the pen
        self._reset()

    def _step(self, action):
        """ draw by action.
        Assuming that action is a dict whose keys are (x, y, p, r, g, b, q)
        x: float. x-pos of the next point
        y: float. y-pos of the next point
        p: float. pressure
        r, g, b: float. color value of the brush [0, 1] 
        q: boolean. a binary flag specifying the type of action: draw or skip to the next point w/o drawing
        """
        # update color of the brush
        self.brush_info.set_color_rgb((action['r'], action['g'], action['b']))

        # maybe draw
        if action['q']:
            self.__draw(action['x'], action['y'], action['p'])
        else:
            self.__draw(action['x'], action['y'], 0.0)

    def __draw(self, x, y, p, xtilt=0, ytilt=0, dtime=0.01, viewzoom=1.0, viewrotation=0.0):
        self.surface.begin_atomic()
        self.brush.stroke_to(
            self.surface.backend,
            x,
            y,
            p,
            xtilt,
            ytilt,
            dtime,
            viewzoom,
            viewrotation
        )
        self.surface.end_atomic()

        # update the current point
        self.x = x
        self.y = y

    def _reset(self):
        """ clear all the content on the canvas, move the current position to the default """
        # TODO: clear canvas
        
        # set the pen's position
        self.__draw(self.start_x, self.start_y, 0)
        self.x = self.start_x
        self.y = self.start_y
    
    def _render(self):
        raise NotImplementedError

    def _get_rgb_array(self):
        """ render the current canvas as a rgb array
        """
        buf = self.surface.render_as_pixbuf()
        w = buf.get_width()
        h = buf.get_height()
        # convert uint8 matrix whose shape is [w, h, 4]
        img = np.frombuffer(buf.get_pixels(), np.uint8).reshape(w, h, -1)
        # img = img[:, :, :3]  # dont use alpha channel

        return img
    
    def _close(self):
        raise NotImplementedError
    
    def _seed(self, seed=None):
        raise NotImplementedError
    
if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(__name__)
    env = MyPaintEnv(logger=logger)

    # drawing something
    env._step({'x': 100, 'y': 100, 'p': 0.5, 'r': 0.0, 'g': 0.0, 'b': 0.0, 'q': True})

    # show drawn picture
    img = env._get_rgb_array()

    import matplotlib
    matplotlib.use('MacOSX')  # TODO: check backend of a LINUX server w/o monitor
    import matplotlib.pyplot as plt
    plt.imshow(img)
    plt.show()
