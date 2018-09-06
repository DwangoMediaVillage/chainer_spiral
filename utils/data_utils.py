import chainer
import cv2
import numpy as np
import math

def single_class_filter(xs, label):
    assert isinstance(label, int)
    indices = [ i for i, x in enumerate(xs) if x[1] == label ]
    return [ xs[i][0] for i in indices ]

def get_mnist(imsize=64, single_class=False, target_label=None):
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
        thresh = 0.5
        _, y = cv2.threshold(y, thresh, 1.0, cv2.THRESH_BINARY)
        y = np.reshape(y, (1, 1, imsize, imsize))
        y = 1.0 - y  # background: black -> white
        return chainer.Variable(y)

    return train, target_data_sampler

def get_toydata(imsize=3):
    """ create a simple image and returns a func to feed data """
    assert imsize >= 3

    train = np.ones((imsize, imsize))

    # vertical line
    idx = math.floor(imsize / 2)
    train[:, idx] = 0.0
    train_v = train.reshape(1, 1, imsize, imsize).astype(np.float32)
    train_v = chainer.Variable(train_v)

    # define a func to return train
    def data_sampler():
        return train_v
        
    return train, data_sampler
