from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import utilities as util

config = tf.ConfigProto()
config.gpu_options.allow_growth = True


def get_unary_op(x, option):
    unary_ops = {
        'log': tf.log(x),
        'exp': tf.exp(x),
        'neg': tf.negative(x),
        'ceil': tf.ceil(x),
        'floor': tf.floor(x),
        'log1p': tf.log1p(x),
        'sqrt': tf.sqrt(x),
        'square': tf.square(x),
        'abs': tf.abs(x),
        'relu': tf.nn.relu(x),
        'elu': tf.nn.elu(x),
        'selu': tf.nn.selu(x),
        'leakyRelu': tf.nn.leaky_relu(x),
        'prelu': tf.keras.layers.PReLU(x),
        'sigmoid': tf.sigmoid(x),
        'sin': tf.sin(x),
        'cos': tf.cos(x),
        'tan': tf.tan(x),
        'asin': tf.asin(x),
        'acos': tf.acos(x),
        'atan': tf.atan(x),
        'sinh': tf.sinh(x),
        'cosh': tf.cosh(x),
        'tanh': tf.tanh(x),
    }

    assert option in unary_ops, 'Unary option not found: ' + option
    return unary_ops[option]


def run(size, option, backend):
    if backend == 'gpu' and util.can_use_gpu():
        doWarmup = True
    else:
        assert backend == 'cpu'
        doWarmup = False

    x = tf.random_uniform([size, size], -1, 1)
    reduction = get_unary_op(x, option)

    backend = util.get_backend(backend)
    with tf.device(backend):
        sess = tf.Session()

        if doWarmup:
            sess.run(reduction)

            #  Add performance benchmarking here!
            sess.run(reduction)
        else:
            sess.run(reduction)


if __name__ == "__main__":
