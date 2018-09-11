import chainer
import cv2
import numpy as np
import math
from environments import ToyEnv
from chainer import functions as F

def single_class_filter(xs, label):
    assert isinstance(label, int)
    indices = [ i for i, x in enumerate(xs) if x[1] == label ]
    return [ xs[i][0] for i in indices ]

def get_mnist(imsize=64, single_class=False, target_label=None, bin=False):
    """ maybe download mnist dataset, and returns single class data if specified """
    train, _ = chainer.datasets.get_mnist(withlabel=True, ndim=2)
    
    if single_class:
        assert not target_label is None
        train = single_class_filter(train, target_label)
    else:
        train = [ x[0] for x in train ]

    train_iter = chainer.iterators.SerialIterator(train, 1)

    def target_data_sampler():
        """ returns an image batch """
        y = train_iter.next()[0] * 255.0
        y = y.astype(np.uint8)
        y = cv2.resize(y, (imsize, imsize))
        if bin:
            thresh = 255 / 2.0
            _, y = cv2.threshold(y, thresh, 1.0, cv2.THRESH_BINARY)
        y = np.reshape(y, (1, 1, imsize, imsize)) / 255.0
        y = y.astype(np.float32)
        y = 1.0 - y  # background: black -> white
        return chainer.Variable(y)

    return train, target_data_sampler

def get_toydata(imsize, multi_pattern=False, channel=1):
    """ create a simple image and returns a func to feed data """

    def create_target_data(indices):
        env = ToyEnv(imsize)
        for index in indices:
            a = {'position': index, 'pressure': 1.0, 'color': (0, 0, 0), 'prob': 1}
            env.step(a)
        x = env.render('rgb_array')
        if channel == 1:
            x = cv2.cvtColor(x, cv2.COLOR_RGB2GRAY)  # (N, N) uint8
        x = x.astype(np.float32) / 255.
        x = x.reshape(1, channel, imsize, imsize).astype(np.float32)
        return chainer.Variable(x)
    
    if multi_pattern:
        # create some patterns
        x1 = create_target_data([1, 4, 7])
        x2 = create_target_data([0, 1, 2])
        x3 = create_target_data([3, 4, 5])
        x4 = create_target_data([2, 5, 8])

        xs = (x1, x2, x3, x4)
        def data_sampler():
            """ returns one of xs randomly """
            return xs[np.random.randint(len(xs))]
        return F.concat(xs, axis=0).data, data_sampler
    else:
        # carete a pattern
        x = create_target_data([1, 4, 7])
        def data_sampler():
            return x
        return x.data, data_sampler
