from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import utilities as util
import numpy as np
import time

config = tf.ConfigProto()
config.gpu_options.allow_growth = True


def get_pooling_op(x, pool_size, strides, option):
    if option == 'max':
        return tf.layers.max_pooling1d(x, pool_size, strides, 'same')
    else:
        assert option == 'avg', 'Pooling option not found: ' + option
        return tf.layers.average_pooling1d(x, pool_size, strides, 'same')


def run(size, option, params, backend):
    if backend == 'gpu' and util.can_use_gpu():
        doWarmup = True
    else:
        assert backend == 'cpu'
        doWarmup = False

    output_depth = params['depth']
    x_shape = np.asarray([size, size, output_depth], dtype=np.int32)
    field_size = params['field_size']
    stride = params['stride']
    x = tf.random_uniform(x_shape, -1, 1)
    pool = get_pooling_op(x, field_size, stride, option)

    backend = util.get_backend(backend)
    with tf.device(backend):
        sess = tf.Session()

        if doWarmup:
            sess.run(pool)

        start = time.time()
        sess.run(pool)
        end = time.time()
        runtime = end - start
        print(runtime)


if __name__ == "__main__":
    run(100, 'max', {'depth': 10, 'field_size': 2, 'stride': 2}, 'cpu')