import numpy as np
import chainer
import chainer.distributions as D
from chainer import functions as F

def gumbel_max_sampling(pi):
    """ performs Gumbel-Max trick for pi """
    p_shape = pi.p.shape

    # sample from uniform dist.
    low = np.array(0, dtype=np.float32)
    high = np.array(1, dtype=np.float32)
    u = D.Uniform(low=low, high=high).sample(sample_shape=p_shape)
    g = -F.log(-F.log(u))
    
    z = g + pi.log_p
    return F.argmax(z)

def gumbel_softmax_sampling(pi, t=1.0):
    """ performs Gumbel Softmax Sampling for pi """
    p_shape = pi.p.shape

    # sample from uniform dist.
    low = np.array(0, dtype=np.float32)
    high = np.array(1, dtype=np.float32)
    u = D.Uniform(low=low, high=high).sample(sample_shape=p_shape)
    g = -F.log(-F.log(u))

    z = F.softmax((pi.log_p + g) / t)
    return z

x = chainer.Variable(np.random.rand(1, 10).astype(np.float32))
pi = D.Categorical(logit=x)
z = gumbel_max_sampling(pi)
z = gumbel_softmax_sampling(pi)
