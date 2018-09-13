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

        y *= -1

        # may draw a line
        if prob:
            ax.plot([last_x, x], [last_y, y], color=color, lw=lw)

        # update
        last_prob = prob
        last_x, last_y = x, y

    set_plot_scale(ax)

    
def plot_image(ax, x, vmin=0, vmax=1, cmap='gray', origin='lower'):
    # origin of Cairo backend is different from gtk backend
    return ax.imshow(x, vmin=vmin, vmax=vmax, cmap=cmap, origin=origin)

def run_single_episode(env, agent, max_episode_steps, conditional_input=None):
    obs, act = {} , {}
    o = env.reset()
    obs[0] = o
    agent.generator.reset_state()

    for t in range(max_episode_steps):
        a = agent.act(o, conditional_input=conditional_input)
        logger.debug('taking action %s', a)
        o, _, _, _ = env.step(a)
        obs[t + 1] = o
        act[t] = a

    return obs, act


def run_episode(env, agent, N, max_episode_steps, conditional=False, dataset=None):
    """ rollout N times with agent and env until max_episode steps for each episode. """
    obs, act = [], []
    if conditional: cond_inputs = []
    for n in range(N):
        if conditional:
            # get untrained image data as a conditional input
            conditional_input = dataset.get_example(train=False)
            o, a = run_single_episode(env, agent, max_episode_steps, conditional_input=conditional_input)
        else:
            o, a = run_single_episode(env, agent, max_episode_steps)
        obs.append(o)
        act.append(a)
        if conditional: cond_inputs.append(conditional_input)
    
    if conditional:
        return obs, act, cond_inputs
    else:
        return obs, act


def demo_static(env, agent, args, savename, suptitle, dataset, n_row=5, plot_act=True):
    """ render drawn picture and lines colored by ordering """
    fig = plt.figure(figsize=(7, 7))
    if args.conditional:
        # conditional generation
        gs = gridspec.GridSpec(n_row, 3)
        obs, act, cond = run_episode(env, agent, n_row, args.max_episode_steps, 
                                conditional=True, dataset=dataset)
        for n in range(n_row):
            # conditional input
            ax_cond = plt.subplot(gs[n, 0])
            # conditional input is assumed to be chainer.Variable whose shape is [1, 1, H, W], and value range is [0, 1].
            im = plot_image(ax_cond, cond[n].data[0, 0])
            set_axis_prop(ax_cond)
            if n == 0: ax_cond.set_title('Input')

            # final obs
            ax_obs = plt.subplot(gs[n, 1])
            ax_obs.imshow(obs[n][args.max_episode_steps]['image'], origin=None)
            set_axis_prop(ax_obs)
            if n == 0: ax_obs.set_title('Final observation')
            
            # plot act
            if plot_act:
                ax_act = plt.subplot(gs[n, 2])
                plot_action(ax_act, act[n], args.max_episode_steps, obs[n][0], env.convert_x)
                set_axis_prop(ax_act)
                if n == 0: ax_act.set_title('Line colored by order')
        
    else:
        # unconditional generation
        gs = gridspec.GridSpec(n_row, 2)
        obs, act = run_episode(env, agent, n_row, args.max_episode_steps)
        for n in range(n_row):
            # final obs
            ax_obs = plt.subplot(gs[n, 0])
            ax_obs.imshow(obs[n][args.max_episode_steps]['image'], origin='lower')
            set_axis_prop(ax_obs)
            if n == 0: ax_obs.set_title('Final observation')
            
            # plot act
            if plot_act:
                ax_act = plt.subplot(gs[n, 1])
                plot_action(ax_act, act[n], args.max_episode_steps, obs[n][0], env.convert_x)
                set_axis_prop(ax_act)
                if n == 0: ax_act.set_title('Line colored by order')

    # render as movie
    fig.suptitle(suptitle)
    plt.savefig(savename)

def demo_movie(env, agent, args, savename, suptitle, dataset, n_row=5, plot_act=True):
    """ render movie of drawn picture, final observation, and lines colored by ordering """
    fig = plt.figure(figsize=(7, 7))
    ims = []
    if args.conditional:
        # conditional generation
        gs = gridspec.GridSpec(n_row, 4)
        obs, act, cond = run_episode(env, agent, n_row, args.max_episode_steps, 
                                conditional=True, dataset=dataset)
        for n in range(n_row):
            # conditional input
            ax_cond = plt.subplot(gs[n, 0])
            # conditional input is assumed to be chainer.Variable whose shape is [1, 1, H, W], and value range is [0, 1].
            im = plot_image(ax_cond, cond[n].data[0, 0], origin=None)
            set_axis_prop(ax_cond)
            if n == 0: ax_cond.set_title('Input')

            # obs
            ax_movie = plt.subplot(gs[n, 1])
            im = ax_movie.imshow(obs[n][0]['image'], origin='lower')  # image at t=0
            ims.append(im)
            if n == 0: ax_movie.set_title('Movie')
            
            # final obs
            ax_obs = plt.subplot(gs[n, 2])
            ax_obs.imshow(obs[n][args.max_episode_steps]['image'], origin='lower')
            set_axis_prop(ax_obs)
            if n == 0: ax_obs.set_title('Final observation')
            
            # plot act
            if plot_act:
                ax_act = plt.subplot(gs[n, 3])
                plot_action(ax_act, act[n], args.max_episode_steps, obs[n][0], env.convert_x)
                set_axis_prop(ax_act)
                if n == 0: ax_act.set_title('Line colored by order')
        
    else:
        # unconditional generation
        gs = gridspec.GridSpec(n_row, 3)
        obs, act = run_episode(env, agent, n_row, args.max_episode_steps)
        for n in range(n_row):
            # obs
            ax_movie = plt.subplot(gs[n, 0])
            im = ax_movie.imshow(obs[n][0]['image'], origin='lower')  # image at t=0
            ims.append(im)
            if n == 0: ax_movie.set_title('Movie')
            
            # final obs
            ax_obs = plt.subplot(gs[n, 1])
            ax_obs.imshow(obs[n][args.max_episode_steps]['image'], origin='lower')
            set_axis_prop(ax_obs)
            if n == 0: ax_obs.set_title('Final observation')
            
            # plot act
            if plot_act:
                ax_act = plt.subplot(gs[n, 2])
                plot_action(ax_act, act[n], args.max_episode_steps, obs[n][0], env.convert_x)
                set_axis_prop(ax_act)
                if n == 0: ax_act.set_title('Line colored by order')

    # render as movie
    fig.suptitle(suptitle)
    def frame_func(t):
        for n, im in enumerate(ims):
            im.set_data(obs[n][t]['image'])
        return ims
    ani = anim.FuncAnimation(fig, frame_func, frames=range(0, args.max_episode_steps + 1), interval=100)
    ani.save(savename)


def demo_many(env, agent, args, savename, suptitle, dataset, n_row=10, n_col=5):
    """ render many final observations """
    fig = plt.figure(figsize=(7, 7))
    if args.conditional:
        # conditional generation
        gs = gridspec.GridSpec(n_row, n_col * 2)
        obs, act, cond = run_episode(env, agent, n_row * n_col, args.max_episode_steps, 
                                conditional=True, dataset=dataset)
        n = 0
        for i in range(n_row):
            for j in range(0, n_col, 2):
                # conditional input
                ax = plt.subplot(gs[i, j])
                # conditional input is assumed to be chainer.Variable whose shape is [1, 1, H, W], and value range is [0, 1].
                plot_image(ax, cond[n].data[0, 0])
                set_axis_prop(ax)
                if i == 0: ax.set_title('Input')

                # generated image
                ax = plt.subplot(gs[i, j + 1])
                ax.imshow(obs[n][args.max_episode_steps]['image'], origin='lower')
                set_axis_prop(ax)
                if i == 0: ax.set_title('Reconst')

                n += 1
    else:
        # conditional generation
        gs = gridspec.GridSpec(n_row, n_col * 2)
        obs, act = run_episode(env, agent, n_row * n_col * 2, args.max_episode_steps)
        n = 0
        for i in range(n_row):
            for j in range(n_col * 2):
                ax = plt.subplot(gs[i, j])
                ax.imshow(obs[n][args.max_episode_steps]['image'], origin='lower')
                set_axis_prop(ax)
                n += 1

    # save figure
    fig.suptitle(suptitle)
    plt.savefig(savename)
