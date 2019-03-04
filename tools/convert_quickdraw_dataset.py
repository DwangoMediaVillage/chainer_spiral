"""Preprocess a npz file of QuickDraw! Dataset (https://github.com/googlecreativelab/quickdraw-dataset)
"""
import argparse
import matplotlib.pyplot as plt
import matplotlib.gridspec as girdspec
import numpy as np
import rdp
from chainer_spiral.environments import MyPaintEnv
import os
from multiprocessing import Pool

def to_abs(pos):
    r_pos = []
    x, y = 0, 0
    for p in pos:
        _x, _y = p
        x += _x
        y += _y
        r_pos.append([x, y])
    r_pos = np.array(r_pos)
    r_pos = (r_pos - r_pos.min()) / (r_pos.max() - r_pos.min())
    return r_pos

def convert_pos(env, x, y):
    x = int(x * env.pos_resolution)
    y = int(y * env.pos_resolution)
    return y * env.pos_resolution + x

def render(data, margin=0.1):
    env = MyPaintEnv(brush_info_file='settings/my_simple_brush.myb')

    # stroke-3 -> abs (normalized) [0, 1]
    data = data.astype(np.float32)
    data[:, :2] = to_abs(data[:, :2])

    # add the first action
    tmp = data[0]
    tmp[-1] = 1
    data = np.concatenate((
        np.expand_dims(tmp, 0), data
        ), axis=0)

    # add margin
    data[:, :2] = data[:, :2] * (1.0 - margin * 2.0) + margin

    # render using env
    env.reset()

    for d in data:
        action = {'position': convert_pos(env, d[0], d[1]),
                'color': (0, 0, 0),
                'pressure': 1.0,
                'prob': 1-d[2]}
        ob, reward, done, info = env.step(action)
    return ob['image']
 
def convert_drawing_seq(data, processes):
   
    with Pool(processes=processes) as pool:
        converted_data = pool.map(render, data)

    return np.array(converted_data)

def convert_quickdraw_dataset(npz_file, savename, processes):
    # load npz file
    data = np.load(npz_file, encoding='latin1')
    train = convert_drawing_seq(data['train'], processes)
    test = convert_drawing_seq(data['test'], processes)
    valid = convert_drawing_seq(data['valid'], processes)
    np.savez_compressed(savename, train=train, test=test, valid=valid, origin=os.path.basename(npz_file))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('npz')
    parser.add_argument('savename')
    parser.add_argument('--processes', type=int, default=2)
    args = parser.parse_args()

    convert_quickdraw_dataset(args.npz, args.savename, args.processes)

