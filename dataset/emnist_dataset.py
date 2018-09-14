import chainer
import numpy as np
import gzip
import cv2

class EmnistDataset(chainer.dataset.DatasetMixin):
    """ Koten Jikei dataset. get_exmaple() returns a batch which has a converted mnist image 
        as chainer.Variable whose value range is [0, 1]

    Args:
        imsize (int): Size of converted image to make a batch
        single_class (bool): If True, it gives images of a specific class
        target_label (int): Number of class (0, 1, ... ,9). Should be used with single_class
        binarization (bool): If True, it gives binarized images
    """

    def __init__(self, gz_filename):
        self.train = self.__load_emnist(gz_filename)
        self.N = self.train.shape[0]

    def __load_emnist(self, gz_filename):
        with gzip.open(gz_filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
        return data.reshape(-1, 28 * 28)

    def __len__(self):
        return len(self.train)

    def get_example(self, train=True):
        """ return a batch """
        ind = np.random.randint(self.N)
        return self.__preprocess_image(self.train[ind])

    def __preprocess_image(self, x):
        """ convert an image to a batch """
        x = x.reshape(28, 28).T
        x = cv2.resize(x, (64, 64))
        x = x.astype(np.float32) / 255.0
        x = np.expand_dims(x, 0)
        x = np.expand_dims(x, 0)
        x = 1.0 - x  # background black -> white
        return chainer.Variable(x)



