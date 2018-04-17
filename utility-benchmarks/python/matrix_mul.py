from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import utilities as util
import time

config = tf.ConfigProto()
config.gpu_options.allow_growth = True


def matrix_mul(size):
    a = tf.random_uniform([size, size], -1, 1)
    b = tf.random_uniform([size, size], -1, 1)
    tf.matmul(a, b)


def run(size, backend):
    if backend == 'gpu' and util.can_use_gpu():
        doWarmup = True
    else:
        assert backend == 'cpu', 'Please specify either cpu or gpu as backend options'
        doWarmup = False

    backend = util.get_backend(backend)
    with tf.device(backend):
        sess = tf.Session()

        if doWarmup:
            sess.run(matrix_mul(size))

        start = time.time()
        sess.run(matrix_mul(size))
        end = time.time()
        runtime = end - start
        print(runtime)


if __name__ == "__main__":
    run(10, 'cpu')
