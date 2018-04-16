from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import utilities as util

config = tf.ConfigProto()
config.gpu_options.allow_growth = True


def matrix_mul(size, warming_up):
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
            sess.run(matrix_mul(size, True))

            #  Add performance benchmarking here!
            sess.run(matrix_mul(size, False))
        else:
            sess.run(matrix_mul(size, False))


if __name__ == "__main__":
    run(10, 'gpu')
