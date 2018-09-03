import chainer
import cv2
import numpy as np

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