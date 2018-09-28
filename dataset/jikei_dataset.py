import chainer
import numpy as np

class JikeiDataset(chainer.dataset.DatasetMixin):
    """ Koten Jikei dataset. get_exmaple() returns a batch which has a converted mnist image 
        as chainer.Variable whose value range is [0, 1]
    """

    def __init__(self, npz_filename):
        self.train, self.label = self.__load_jikei(npz_filename)
        self.N = self.train.shape[0]

    def __len__(self):
        return self.N

    def get_example(self, train=True):
        """ return a batch """
        ind = np.random.randint(self.N)
        return self.__preprocess_image(self.train[ind])

    def __load_jikei(self, npz_filename):
        data = np.load(npz_filename)
        return data['img'], data['tag']

    def __preprocess_image(self, x):
        """ convert an image to a batch """
        x = x.astype(np.float32)
        x = np.expand_dims(x, 0)
        x = np.expand_dims(x, 0)
        return chainer.Variable(x)



