from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import utilities as util
import time

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

def batch_normalization(size):
    x = tf.random_uniform([size, size, 8], -1, 1)
    return tf.layers.batch_normalization(x)


def run(size, backend):
    if backend == 'gpu' and util.can_use_gpu():
        doWarmup = True
    else:
        assert backend == 'cpu'
        doWarmup = False

    backend = util.get_backend(backend)
    with tf.device(backend):
        sess = tf.Session()

        if doWarmup:
            sess.run(batch_normalization(size))

        start = time.time()
        sess.run(batch_normalization(size))
        end = time.time()
        runtime = end - start
        print(runtime)


if __name__ == "__main__":
    run(10, 'gpu')







