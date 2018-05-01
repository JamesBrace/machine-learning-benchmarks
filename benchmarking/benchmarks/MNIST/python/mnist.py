from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.layers import Input, Convolution2D, MaxPooling2D, Activation, Dropout, Reshape, Dense
from keras import optimizers
from keras.models import Model

import tensorflow as tf
import dataset

"""""""""""
DEFAULTS
"""""""""""
TRAIN_SIZE = 1000
TEST_SIZE = 1000
LEARNING_RATE = .001
BATCH_SIZE = 64
IMAGE_SIZE = 28
IMAGE_DEPTH = 1
EPOCHS = 1


class MNIST:
    def __init__(self, backend):

        self.train_data = dataset.train("/tmp/mnist_data")
        self.test_data = dataset.test("/tmp/mnist_data")
        self.backend = backend

        self.train_images = self.train_data['images'][:TRAIN_SIZE]
        self.train_labels = self.train_data['labels'][:TRAIN_SIZE]

        self.test_images = self.test_data['images'][:TRAIN_SIZE]
        self.test_labels = self.test_data['labels'][:TRAIN_SIZE]

        if backend == 'gpu':
            self.device = "/device:GPU:0"
        else:
            assert backend == 'cpu', 'Invalid backend specified: %s' % backend
            self.device = "/cpu:0"

        self.model = MNIST.create_model()

    @staticmethod
    def create_model(input_shape=(IMAGE_SIZE, IMAGE_SIZE, IMAGE_DEPTH)):
        img_input = Input(shape=input_shape)

        x = Convolution2D(32, (5, 5), strides=(1, 1), padding='same', name='conv1')(img_input)
        x = Activation('relu', name='relu_conv1')(x)
        x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool1')(x)

        x = Convolution2D(64, (5, 5), strides=(1, 1), padding='same', name='conv2')(x)
        x = Activation('relu', name='relu_conv2')(x)
        x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool2')(x)

        x = Reshape([-1, 7 * 7 * 64])(x)
        x = Dense(1024)(x)
        x = Activation('relu', name='relu_conv1')(x)
        x = Dropout(0.4)(x)
        x = Dense(10)(x)

        inputs = img_input

        model = Model(inputs, x, name='squeezenet')

        optimizer = optimizers.adam(lr=0.001)
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def train(self, train_steps=EPOCHS):
        with tf.device(self.device):
            self.model.fit(self.train_images, self.train_labels, batch_size=BATCH_SIZE, epochs=train_steps)

    def predict(self):
        with tf.device(self.device):
            self.model.evaluate(x=self.test_images, y=self.test_labels, batch_size=BATCH_SIZE)


def init(backend):
    return MNIST(backend)









