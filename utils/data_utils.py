import chainer
import cv2
import numpy as np
import math
from environments import ToyEnv

def single_class_filter(xs, label):
    assert isinstance(label, int)
    indices = [ i for i, x in enumerate(xs) if x[1] == label ]
    return [ xs[i][0] for i in indices ]

def get_mnist(imsize=64, single_class=False, target_label=None, bin=False):
    """ maybe download mnist dataset, and returns single class data if specified """
    train, _ = chainer.datasets.get_mnist(withlabel=True)
    
    if single_class:
        assert not target_label is None
        train = single_class_filter(train, target_label)
    else:
        train = [ x[0] for x in train ]

    train_iter = chainer.iterators.SerialIterator(train, 1)

    def target_data_sampler():
        y = train_iter.next()[0].data
        y = np.reshape(y, (28, 28))
        y = cv2.resize(y, (imsize, imsize))
        if bin:
            thresh = 0.5
            _, y = cv2.threshold(y, thresh, 1.0, cv2.THRESH_BINARY)
        y = np.reshape(y, (1, 1, imsize, imsize))
        y = 1.0 - y  # background: black -> white
        return chainer.Variable(y)

    return train, target_data_sampler

def get_toydata(imsize):
    """ create a simple image and returns a func to feed data """
    env = ToyEnv(imsize)
    indices = [1, 4, 7]
    
    for index in indices:
        a = {'position': index, 'pressure': 1.0, 'color': (0, 0, 0), 'prob': 1}
        env.step(a)

    train = env.render('rgb_array')
    train = cv2.cvtColor(train, cv2.COLOR_RGB2GRAY)  # (N, N) uint8
    train = train.astype(np.float32) / 255.
    train = train.reshape(1, 1, imsize, imsize).astype(np.float32)
    train_v = chainer.Variable(train)

    # define a func to return train
    def data_sampler():
        return train_v
        
    return train, data_sampler
