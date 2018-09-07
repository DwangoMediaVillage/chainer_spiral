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
matplotlib.use('Cairo')

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.animation as anim

import logging

logger = logging.getLogger(__name__)

def set_axis_prop(ax, title=None):
    """ set properties of axis """
    if title:
        ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal')

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
            ax.plot([last_x, x], [last_y, y], color=color, lw=lw)

        # update
        last_prob = prob
        last_x, last_y = x, y

    set_plot_scale(ax)


def run_single_episode(env, agent, T):
    obs_hist = {}
    act_hist = {}
    obs = env.reset()
    agent.generator.reset_state()
    obs_hist[0] = obs
    
    for t in range(T):
        a = agent.act(obs)
        logger.info('taking action %s', a)
        obs, r, done, info = env.step(a)
        obs_hist[t + 1] = obs
        act_hist[t] = a
    
    agent.generator.reset_state()
    env.reset()
    return obs_hist, act_hist

def run_episode(env, agent, N, T):
    """ rollout N times """
    o, a = [], []
    for n in range(N):
        __o, __a = run_single_episode(env, agent, T)
        o.append(__o)
        a.append(__a)
    return o, a

def run_demo(demo_mode, env, agent, max_episode_steps, savename, suptitle):
    """ Demo mode. Agent draws pictures """    

    T = max_episode_steps

    if demo_mode == 'static':
        N = 5
        obs, act = run_episode(env, agent, N, T)
        fig = plt.figure(figsize=(7, 7))
        gs = gridspec.GridSpec(N, 2)
        for n in range(N):
            # final obs    
            ax_obs = plt.subplot(gs[n, 0])
            ax_obs.imshow(obs[n][T]['image'])
            set_axis_prop(ax_obs)
            if n == 0: ax_obs.set_title('Final observation')
            
            # plot act
            ax_act = plt.subplot(gs[n, 1])
            plot_action(ax_act, act[n], T, obs[n][0], env.convert_x)
            set_axis_prop(ax_act)
            if n == 0: ax_act.set_title('Line colored by order')
        fig.suptitle(suptitle)
        plt.savefig(savename)

    elif demo_mode == 'movie':
        N = 5
        obs, act = run_episode(env, agent, N, T)
        fig = plt.figure(figsize=(7, 7))
        gs = gridspec.GridSpec(N, 3)
        ims = []
        for n in range(N):
            # obs
            ax_movie = plt.subplot(gs[n, 0])
            im = ax_movie.imshow(obs[n][0]['image'])  # image at t=0
            ims.append(im)
            if n == 0: ax_movie.set_title('Movie')
            
            # final obs
            ax_obs = plt.subplot(gs[n, 1])
            ax_obs.imshow(obs[n][T]['image'])
            set_axis_prop(ax_obs)
            if n == 0: ax_obs.set_title('Final observation')
            
            # plot act
            ax_act = plt.subplot(gs[n, 2])
            plot_action(ax_act, act[n], T, obs[n][0], env.convert_x)
            set_axis_prop(ax_act)
            if n == 0: ax_act.set_title('Line colored by order')          
        fig.suptitle(suptitle)
        def frame_func(t):
            for n, im in enumerate(ims):
                im.set_data(obs[n][t]['image'])
            return ims
        ani = anim.FuncAnimation(fig, frame_func, frames=range(0, T + 1), interval=100)
        ani.save(savename)

    elif demo_mode == 'many':
        N = 10
        obs, act = run_episode(env, agent, N * N, T)
        fig = plt.figure(figsize=(7, 7))
        gs = gridspec.GridSpec(N, N)
        n = 0
        for i in range(N):
            for j in range(N):
                ax = plt.subplot(gs[i, j])
                ax.imshow(obs[n][T]['image'])
                set_axis_prop(ax)
                n += 1
        fig.suptitle(suptitle)
        plt.savefig(savename)

