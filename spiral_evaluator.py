from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from builtins import *  # NOQA
from future import standard_library
standard_library.install_aliases()  # NOQA

import numpy as np

# TODO (enhancement): Support Windows
import matplotlib

import platform

if 'Darwin' in platform.platform():
    matplotlib.use('MacOSX')
else:
    matplotlib.use('Cairo')

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.animation as anim

def set_axis_prop(ax, title=None):
    """ set properties of axis """
    if title:
        ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal')

def show_drawn_pictures(env, agent, timestep_limit):
    """ Make agent to draw and show the drawn picture using matplotlib """
    
    obs = env.reset()
    for t in range(timestep_limit):
        a = agent.act(obs)
        obs, r, done, info = env.step(a)

    agent.stop_episode()

    # show image
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ims = ax.imshow(obs['image'])
    set_axis_prop(ax, 'Drawn by an agent')
    plt.show()

def run_single_episode(env, agent, timestep_limit):
    obs_hist = {}
    act_hist = {}

    obs = env.reset()
    obs_hist[0] = obs
    
    for t in range(timestep_limit):
        a = agent.act(obs)
        obs, r, done, info = env.step(a)
        obs_hist[t + 1] = obs
        act_hist[t] = a
    
    return obs_hist, act_hist


def set_plot_scale(ax, m=0.1):
    min_x, max_x = ax.get_xlim()
    min_y, max_y = ax.get_ylim()

    x_size = abs(max_x - min_x)
    y_size = abs(max_y - min_y)

    if y_size > x_size:
        diff = y_size - x_size
        min_x -= diff / 2.0
        max_x += diff / 2.0
    else:
        diff = x_size - y_size
        min_y -= diff / 2.0
        max_y += diff / 2.0

    margin = max(x_size, y_size) * m

    min_x -= margin
    min_y -= margin
    max_x += margin
    max_y += margin

    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)
    

def get_colors(n, cmap='cool'):
    cmap = matplotlib.cm.get_cmap(cmap)
    idx = np.linspace(0.0, 1.0, n)
    return cmap(idx)


def plot_action(ax, act, T, init_obs, convert_pos_func, lw=2.0):
    """ plot act to axis for T steps """
    last_prob = init_obs['prob']
    last_x, last_y = convert_pos_func(init_obs['position'])
    colors = get_colors(T)

    for t, color in zip(range(T), colors):
        pos = act[t]['position']
        prob = act[t]['prob']  # draw or not
        
        # convert discrite position index -> (x, y)
        x, y = convert_pos_func(pos)

        # may draw a line
        if prob:
            ax.plot([last_x, x], [-last_y, -y], color=color, lw=lw)

        # update
        last_prob = prob
        last_x, last_y = x, y

    set_plot_scale(ax)


def run_demo(env, agent, timestep_limit, N=4, suptitle=None, savename=None):
    """ demo mode. Agent draws N pictures and visualize drawn pictures as movie and static plots """

    agent.act_deterministically = False

    # get rollout results
    obs_hist, act_hist = [], []
    for n in range(N):
        __obs_hist, __act_hist = run_single_episode(env, agent, timestep_limit)
        obs_hist.append(__obs_hist)
        act_hist.append(__act_hist)
    
    fig = plt.figure(figsize=(7, 7))
    gs = gridspec.GridSpec(N, 3)
    
    ims = []
    for n in range(N):
        # movie
        ax_movie = plt.subplot(gs[n, 0])
        im = ax_movie.imshow(obs_hist[n][0]['image'])  # image at t=0
        set_axis_prop(ax_movie)
        ims.append(im)

        if n == 0: ax_movie.set_title('Movie')

        # plot result
        ax_res = plt.subplot(gs[n, 1])
        res = ax_res.imshow(obs_hist[n][timestep_limit]['image'])
        set_axis_prop(ax_res)

        if n == 0: ax_res.set_title('Final observation')

        # plot actions
        ax_act = plt.subplot(gs[n, 2])
        plot_action(ax_act, act_hist[n], timestep_limit, obs_hist[n][0], env.convert_x)
        set_axis_prop(ax_act)

        if n == 0: ax_act.set_title('Lines colored by ordering')

    # set title
    if suptitle:
        fig.suptitle(suptitle)

    # render as a movie
    def frame_func(t):
        for n, im in enumerate(ims):
            im.set_data(obs_hist[n][t]['image'])
        return ims
    
    ani = anim.FuncAnimation(fig, frame_func, frames=range(timestep_limit), interval=100)

    if savename:
        ani.save(savename)
    else:
        plt.show()


    

    
