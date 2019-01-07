import chainer
import numpy as np
import gzip
import cv2

class EMnistDataset(chainer.dataset.DatasetMixin):
    """ EMNIST dataset. get_exmaple() returns a batch which has a converted emnist image 
        as chainer.Variable whose value range is [0, 1]

    Args:
        imsize (int): Size of converted image to make a batch
        single_class (bool): If True, it gives images of a specific class
        target_label (int): Number of class (0, 1, ... ,9). Should be used with single_class
        binarization (bool): If True, it gives binarized images
    """

    def __init__(self, gz_images, gz_labels, single_label=False):
        self.images, self.labels = self.__load_emnist(gz_images, gz_labels)
        if single_label:
            self.images, self.labels = self.__limit_by_single_label(self.images, self.labels)
        else:
            self.images, self.labels = self.__limit_by_labels(self.images, self.labels)
        self.N = self.images.shape[0]

    def __limit_by_single_label(self, images, labels, target_label=11):
        res_images, res_labels = [], []
        for image, label in zip(images, labels):
            if label == target_label:
                res_images.append(image)
                res_labels.append(label)
        return np.array(res_images), np.array(res_lagels)

    def __limit_by_labels(self, images, labels, target_label=11):
        res_images, res_labels = [], []
        for image, label in zip(images, labels):
            if label < target_label:
                # emnist's labels: [1, 2, ...]
                res_images.append(image)
                res_labels.append(label)
        return np.array(res_images), np.array(res_labels)

    def __load_emnist(self, gz_images, gz_labels):
        # load image
        with gzip.open(gz_images, 'rb') as f:
            images = np.frombuffer(f.read(), np.uint8, offset=16)
        with gzip.open(gz_labels, 'rb') as f:
            labels = np.frombuffer(f.read(), np.uint8, offset=8)
        return images.reshape(-1, 28 * 28), labels

    def __len__(self):
        return self.N

    def get_example(self, train=True):
        """ return a batch """
        ind = np.random.randint(self.N)
        return self.__preprocess_image(self.images[ind])

    def __preprocess_image(self, x):
        """ convert an image to a batch """
        x = x.reshape(28, 28).T
        x = cv2.resize(x, (64, 64))
        x = x.astype(np.float32) / 255.0
        x = np.expand_dims(x, 0)
        x = np.expand_dims(x, 0)
        x = 1.0 - x  # background black -> white
        return chainer.Variable(x)



