import chainer
import cv2
import numpy as np


class MnistDataset(chainer.dataset.DatasetMixin):
    """ MNIST dataset. get_exmaple() returns a batch which has a converted mnist image 
        as chainer.Variable whose value range is [0, 1]

    Args:
        imsize (int): Size of converted image to make a batch
        single_class (bool): If True, it gives images of a specific class
        target_label (int): Number of class (0, 1, ... ,9). Should be used with single_class
        binarization (bool): If True, it gives binarized images
    """

    def __init__(self,
                 imsize,
                 single_class=False,
                 target_label=None,
                 binarization=False):
        self.imsize = imsize
        self.single_class = single_class
        self.target_label = target_label
        self.binarization = binarization

        # may be download mnist dataset
        self.train, self.test, self.train_iter, self.test_iter = self.__get_mnist(
        )

    def __len__(self):
        return len(self.train)

    def get_example(self, train=True):
        """ return a batch """
        if train:
            x = self.train_iter.next()
        else:
            x = self.test_iter.next()
        return self.__preprocess_image(x)

    def __filter_single_class(self, xs, label):
        assert label >= 0 and label <= 9
        indices = [i for i, x in enumerate(xs) if x[1] == label]
        return [xs[i] for i in indices]

    def __get_mnist(self):
        train, test = chainer.datasets.get_mnist(withlabel=True, ndim=2)

        if self.single_class:
            assert self.target_label is not None, "target_label should be specified"
            train = self.__filter_single_class(train, self.target_label)
            test = self.__filter_single_class(test, self.target_label)

        train_iter = chainer.iterators.SerialIterator(train, 1)
        test_iter = chainer.iterators.SerialIterator(test, 1)

        return train, test, train_iter, test_iter

    def __preprocess_image(self, x):
        """ convert a mnist image to a batch """
        x = x[0][0]
        x = np.reshape(x, (28, 28))
        x = cv2.resize(x, (self.imsize, self.imsize))
        if self.binarization:
            thresh = 255 / 2.0
            _, x = cv2.threshold(x, thresh, 1.0, cv2.THRESH_BINARY)
        x = np.reshape(x, (1, 1, self.imsize, self.imsize))
        x = 1.0 - x  # background black -> white
        return chainer.Variable(x)
