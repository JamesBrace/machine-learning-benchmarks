from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import utilities as util
import numpy as np

config = tf.ConfigProto()
config.gpu_options.allow_growth = True


def get_pooling_op(x, pool_size, strides, option):
    if option == 'max':
        return tf.layers.max_pooling2d(x, pool_size, strides, 'same')
    else:
        assert option == 'avg', 'Pooling option not found: ' + option
        return tf.layers.average_pooling2d(x, pool_size, strides, 'same')


def run(size, option, params, backend):
    if backend == 'gpu' and util.can_use_gpu():
        doWarmup = True
    else:
        assert backend == 'cpu'
        doWarmup = False

    output_depth = params.depth
    x_shape = np.asarray([size, size, output_depth], dtype=np.int32)
    field_size = params.field_size
    stride = params.stride
    x = tf.random_uniform(x_shape, -1, 1)
    pool = get_pooling_op(x, field_size, stride, option)

    backend = util.get_backend(backend)
    with tf.device(backend):
        sess = tf.Session()

        if doWarmup:
            sess.run(pool)

            #  Add performance benchmarking here!
            sess.run(pool)
        else:
            sess.run(pool)


if __name__ == "__main__":
