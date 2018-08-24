import chainer

train, test = chainer.datasets.get_mnist(withlabel=False, rgb_format=True)
batchsize = 2
train_iter = chainer.iterators.SerialIterator(train, batchsize)

import ipdb; ipdb.set_trace()