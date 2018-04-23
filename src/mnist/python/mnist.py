from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

import tensorflow as tf
import dataset

config = tf.ConfigProto()
config.gpu_options.allow_growth = True


def cnn_model_fn(features, labels, mode, params):
    with tf.device(params['backend']):

        # Input Layer
        if params['backend'] == "/cpu:0":
            input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])
        else:
            input_layer = tf.reshape(features["x"], [-1, 1, 28, 28])

        '''
        Convolutional Layer #1
        ===========================

        Takes the pre-processed images as the input layer, applies a filter of 5x5x32
        to each image and adjust the padding so that the output remains the same size.
        Afterwards, performs ReLU activation.
        '''
        conv1 = tf.layers.conv2d(
            inputs=input_layer,
            filters=32,
            kernel_size=[5, 5],
            padding="same",
            activation=tf.nn.relu)

        # Pooling Layer #1
        pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

        '''
        Convolutional Layer #2
        ===========================

        Takes the pooled layer as the input layer, applies a filter of 5x5x64
        to each image and adjust the padding so that the output remains the same size.
        Afterwards, performs ReLU activation.
        '''
        conv2 = tf.layers.conv2d(
            inputs=pool1,
            filters=64,
            kernel_size=[5, 5],
            padding="same",
            activation=tf.nn.relu)

        # Pooling Layer #2
        pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

        '''
        Dense Layer + Dropout + Output
        ===========================

        The first step here is to flatten the output of the pool array into 1D arrays. 
        Afterwards we feed the flattened array to a fully connected layer with 1024 units.
        Lastly we perform dropout in order to prevent over-fitting. We use a dropout rate of 0.4
        '''
        pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
        dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
        dropout = tf.layers.dropout(
            inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

        # Logits Layer
        logits = tf.layers.dense(inputs=dropout, units=10)

        predictions = {
            # Generate predictions (for PREDICT and EVAL mode)
            "classes": tf.argmax(input=logits, axis=1),
            # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
            # `logging_hook`.
            "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
        }

        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

        # Calculate Loss (for both TRAIN and EVAL modes)
        loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

        # Configure the Training Op (for TRAIN mode)
        if mode == tf.estimator.ModeKeys.TRAIN:
            optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
            train_op = optimizer.minimize(
                loss=loss,
                global_step=tf.train.get_global_step())
            return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

        # Add evaluation metrics (for EVAL mode)
        eval_metric_ops = {
            "accuracy": tf.metrics.accuracy(
                labels=labels, predictions=predictions["classes"])}

        return tf.estimator.EstimatorSpec(
            mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def main(unused):

    if FLAGS.run_gpu:
        backend = "/device:GPU:0"
    else:
        backend = "/cpu:0"

    mnist_classifier = tf.estimator.Estimator(
        model_fn=cnn_model_fn,
        model_dir=FLAGS.model_dir,
        params={
            'backend': backend,
        })

    if FLAGS.mode == 'train' or FLAGS.mode == 'both':
        ds = dataset.train(FLAGS.data_dir)

        # Train the model
        train_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": ds['images']},
            y=ds['labels'],
            batch_size=FLAGS.batch_size,
            num_epochs=FLAGS.train_epochs,
            shuffle=True)

        mnist_classifier.train(
            input_fn=train_input_fn,
            steps=200
        )

    if FLAGS.mode == 'predict' or FLAGS.mode == 'both':
        ds = dataset.test(FLAGS.data_dir)

        # Predict test set
        pred_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": ds['images']},
            shuffle=False
        )

        mnist_classifier.predict(input_fn=pred_input_fn)


class MNISTArgParser(argparse.ArgumentParser):
    def __init__(self):
        super(MNISTArgParser, self).__init__()

        self.add_argument(
            '--run_gpu',
            action='store_true',
            help='If set, run in GPU mode.')
        self.add_argument(
            '--mode',
            type=str,
            choices=['train', 'predict', 'both'],
            default='train',
            help='Mode to run in. Either train or predict')
        self.add_argument(
            '--batch_size',
            type=int,
            default=64,
            help='Number of images to process in a batch')
        self.add_argument(
            '--data_dir',
            type=str,
            default='/tmp/mnist_data',
            help='Path to directory containing the MNIST dataset')
        self.add_argument(
            '--model_dir',
            type=str,
            default='/tmp/mnist_model',
            help='The directory where the model will be stored.')
        self.add_argument(
            '--train_epochs',
            type=int,
            default=40,
            help='Number of epochs to create_NN.')
        self.add_argument(
            '--export_dir',
            type=str,
            help='The directory where the exported SavedModel will be stored.')


if __name__ == "__main__":
    parser = MNISTArgParser()
    tf.logging.set_verbosity(tf.logging.INFO)
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)