from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import dataset
import math

# from tensorflow.python.client import device_lib
# local_device_protos = device_lib.list_local_devices()
# print(local_device_protos)



"""""""""""
CONSTANTS
"""""""""""
TRAIN_SIZE = 10000
TEST_SIZE = 2000
LEARNING_RATE = .001
BATCH_SIZE = 64
IMAGE_SIZE = 28
IMAGE_DEPTH = 1
EPOCHS = 1


class MNIST:
    def __init__(self, backend, batch_size=BATCH_SIZE, train_epochs=EPOCHS, train_size=TRAIN_SIZE, test_size=TEST_SIZE):
        self.train_data = dataset.train("/tmp/mnist_data")
        self.test_data = dataset.test("/tmp/mnist_data")

        if backend == 'gpu':
            backend = "/device:GPU:0"
        else:
            assert backend == 'cpu', 'Invalid backend specified: %s' % backend
            backend = "/cpu:0"

        self.classifier = tf.estimator.Estimator(
            model_fn=self.cnn_model_fn,
            params={
                'backend': backend,
            })

        self.train_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": self.train_data['images'][:train_size]},
            y=self.train_data['labels'][:train_size],
            batch_size=batch_size,
            num_epochs=train_epochs,
            shuffle=True)

        # Predict test set
        self.pred_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": self.test_data['images'][:test_size]},
            shuffle=False,
            batch_size=batch_size,
        )

    @staticmethod
    def cnn_model_fn(features, labels, mode, params):
        with tf.device(params['backend']):

            print(features["x"])

            input_layer = tf.reshape(features["x"], [-1, IMAGE_SIZE, IMAGE_SIZE, IMAGE_DEPTH])

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
                strides=1,
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

            '''
            Predict Mode
            ===========================
            '''
            if mode == tf.estimator.ModeKeys.PREDICT:
                predictions = {
                    # Generate predictions (for PREDICT and EVAL mode)
                    "classes": tf.argmax(input=logits, axis=1),
                    # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
                    # `logging_hook`.
                    "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
                }
                return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

            '''
            Train Mode
            ===========================
            '''
            # Calculate Loss (for both TRAIN and EVAL modes)
            loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

            # Configure the Training Op (for TRAIN mode)
            if mode == tf.estimator.ModeKeys.TRAIN:
                optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
                train_op = optimizer.minimize(
                    loss=loss,
                    global_step=tf.train.get_global_step())
                return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

            '''
            Eval Mode
            ===========================
            '''
            predictions = {
                # Generate predictions (for PREDICT and EVAL mode)
                "classes": tf.argmax(input=logits, axis=1),
                # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
                # `logging_hook`.
                "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
            }

            # Add evaluation metrics (for EVAL mode)
            eval_metric_ops = {
                "accuracy": tf.metrics.accuracy(
                    labels=labels, predictions=predictions["classes"])}

            return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

    def train(self, epochs=EPOCHS):
        self.classifier.train(
            input_fn=self.train_input_fn,
            steps=int(math.ceil(TRAIN_SIZE/BATCH_SIZE)) * epochs,
        )

    def predict(self):
        self.classifier.predict(input_fn=self.pred_input_fn)


def init(backend, train_size=TRAIN_SIZE, test_size=TEST_SIZE):
    m = MNIST(backend, train_size, test_size)
    return m









