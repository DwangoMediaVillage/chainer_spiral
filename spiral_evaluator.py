from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from builtins import *  # NOQA
from future import standard_library
standard_library.install_aliases()  # NOQA

import numpy as np
import logging


def set_axis_prop(ax, title):
    """ set properties of axis """
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])

def show_drawn_pictures(env, agent, timestep_limit, logger=None):
    """ Make agent to draw and show the drawn picture using matplotlib """
    logger = logger or logging.getLogger(__name__)
    
    obs = env.reset()
    for t in range(timestep_limit):
        a = agent.act(obs)
        obs, r, done, info = env.step(a)

    agent.stop_episode()

    # show image
    # TODO (enhancement): Support Windows
    
    import matplotlib
    matplotlib.use('MacOSX')
    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ims = ax.imshow(obs['image'])
    set_axis_prop(ax, 'Drawn by an agent')
    plt.show()
