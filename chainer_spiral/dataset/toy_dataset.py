import chainer
import cv2
import numpy as np

from chainer_spiral.environments import ToyEnv


class ToyDataset(chainer.dataset.DatasetMixin):
    """ Toy data dataset. get_exmaple() returns a batch which has a converted mnist image 
        as chainer.Variable whose value range is [0, 1]

    Args:
        imsize (int): Size of converted image to make a batch
        target_patterns (tuple): tuple of actions to create target images
    """

    def __init__(self, imsize, train_patterns, test_patterns):
        self.imsize = imsize
        self.train_patterns = train_patterns
        self.test_patterns = test_patterns
        self.train, self.test = self.__get_data()

    def __get_data(self):
        train = [self.__create_target_data(pattern) for pattern in self.train_patterns]
        test = [self.__create_target_data(pattern) for pattern in self.test_patterns]
        return train, test

    def __create_target_data(self, pattern):
        env = ToyEnv(self.imsize)
        for index in pattern:
            a = {'position': index, 'pressure': 1.0, 'color': (0, 0, 0), 'prob': 1}
            env.step(a)
        return env.render('rgb_array')

    def __len__(self):
        return len(self.train)

    def get_example(self, train=True):
        if train:
            i = np.random.randint(len(self.train))
            return self.__preprocess_image(self.train[i])
        else:
            i = np.random.randint(len(self.test))
            return self.__preprocess_image(self.test[i])

    def __preprocess_image(self, x):
        x = cv2.cvtColor(x, cv2.COLOR_RGB2GRAY)
        x = x.astype(np.float32) / 255.
        x = x.reshape(1, 1, self.imsize, self.imsize)
        return chainer.Variable(x)
