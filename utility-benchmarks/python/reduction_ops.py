from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import utilities as util

config = tf.ConfigProto()
config.gpu_options.allow_growth = True


def get_reduction_op(x, option):
    reductions = {
        'argMax': tf.arg_max(x),
        'argMin': tf.arg_min(x),
        'sum': tf.reduce_sum(x),
        'mean': tf.reduce_mean(x),
        'logSumExp': tf.reduce_logsumexp(x)
    }

    assert option in reductions, 'Reduction option not found: ' + option
    return reductions[option]


def run(size, option, backend):
    if backend == 'gpu' and util.can_use_gpu():
        doWarmup = True
    else:
        assert backend == 'cpu'
        doWarmup = False


    x = tf.random_uniform([size, size], -1, 1)
    reduction = get_reduction_op(x, option)

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
