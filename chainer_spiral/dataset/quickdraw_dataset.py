import os
import numpy as np
import chainer

class QuickdrawDataset(chainer.dataset.DatasetMixin):
    """ QuickDraw! dataset from https://github.com/googlecreativelab/quickdraw-dataset
        get_exmaple() returns a batch which has a converted image as chainer.Varible whose value range is [0, 1]
        
    Args:
        data_file (string): path of npz file converted by `convert_quickdraw_dataset.py`        
    """


    def __init__(self, data_file):
        assert os.path.exists(data_file), f"{data_file} does not exist!"
        data = np.load(data_file)
        self.train = data['train']
        self.test = data['test']

    def __len__(self):
        return len(self.train)
    
    def get_example(self, train=True):
        """ return single image batch """
        if train:
            i = np.random.randint(len(self.train))
            data = self.train[i]
        else:
            i = np.random.randint(len(self.test))
            data = self.test[i]
        return self._preprocess_data(data)
    
    def _preprocess_data(self, data):
        data = data[:, :, 0].astype(np.float32) / 255.0
        data = np.reshape(data, [1, 1] + list(data.shape))
        return chainer.Variable(data)

