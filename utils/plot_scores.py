import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import logging
logging.basicConfig(level=logging.DEBUG)

logger = logging.getLogger(__name__)

import pandas

import math

import matplotlib.gridspec as gridspec

EXCEPT_TAGS = ('steps',
                'episodes',
                'elapsed',
                'mean',
                'median',
                'stdev',
                'max',
                'min')

def set_axis_prop(ax, title):
    ax.set_title(title)
    ax.set_xlabel('step')
    ax.grid()

def plot_score(args):
    logger.debug('plot scores.txt in %s', args.target_dir)

    target = os.path.join(args.target_dir, 'scores.txt')

    assert os.path.exists(target)

    table = pandas.read_table(target)

    data = {}

    for col in table.columns:
        if not col in EXCEPT_TAGS:
            data[col] = np.array(table[col])

    steps = np.array(table['steps'])
    elapsed = np.array(table['elapsed'])

    # number of plots
    N = len(data.keys())
    n_cols = 3
    n_rows = int(math.ceil(N / n_cols))
    
    # init figure
    fig = plt.figure(figsize=(n_cols * 2 * 2, n_rows * 2))
    gs = gridspec.GridSpec(n_rows, n_cols)

    n = 0
    for key, value in data.items():
        ax = plt.subplot(gs[n])
        
        ax.plot(steps, value)
        set_axis_prop(ax, key)

        n += 1

    # plt.suptitle(target)
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    plt.show()



    import ipdb; ipdb.set_trace()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('target_dir')
    parser.add_argument('--savename')
    args = parser.parse_args()

    plot_score(args)