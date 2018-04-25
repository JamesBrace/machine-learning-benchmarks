from keras.layers import Input, Convolution2D, MaxPooling2D, Activation, concatenate, Dropout, warnings
from keras.layers import GlobalAveragePooling2D
from keras import optimizers
from keras.models import Model
from keras.datasets import cifar10
from keras.preprocessing.text import one_hot
import tensorflow as tf
import numpy as np

"""""""""
CONSTANTS
"""""""""
TRAIN_SIZE = 50000
TEST_SIZE = 10000
BATCH_SIZE = 64
TRAIN_STEPS = 1

sq1x1 = "squeeze1x1"
exp1x1 = "expand1x1"
exp3x3 = "expand3x3"
relu = "relu_"


class SqueezeNet:

    def __init__(self, backend):
        self.train_images = []
        self.train_labels = np.zeros((TRAIN_SIZE, 10))
        self.test_images = []
        self.test_labels = np.zeros((TEST_SIZE, 10))

        self.load_data()
        self.backend = backend

        if backend == 'cpu':
            self.device = '/cpu:0'
        else:
            assert backend == 'gpu', 'Invalid backend: %s' % backend
            self.device = "/device:GPU:0"

        self.model = SqueezeNet.create_model()

    # Modular function for Fire Node
    @staticmethod
    def fire_module(x, fire_id, squeeze=16, expand=64):
        s_id = 'fire' + str(fire_id) + '/'

        channel_axis = 3

        x = Convolution2D(squeeze, (1, 1), padding='valid', name=s_id + sq1x1)(x)
        x = Activation('relu', name=s_id + relu + sq1x1)(x)

        left = Convolution2D(expand, (1, 1), padding='valid', name=s_id + exp1x1)(x)
        left = Activation('relu', name=s_id + relu + exp1x1)(left)

        right = Convolution2D(expand, (3, 3), padding='same', name=s_id + exp3x3)(x)
        right = Activation('relu', name=s_id + relu + exp3x3)(right)

        x = concatenate([left, right], axis=channel_axis, name=s_id + 'concat')
        return x

    # Original SqueezeNet from paper.
    @staticmethod
    def create_model(input_shape=(32, 32, 3)):

        img_input = Input(shape=input_shape)

        x = Convolution2D(64, (2, 2), strides=(1, 1), padding='same', name='conv1')(img_input)
        x = Activation('relu', name='relu_conv1')(x)
        x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool1')(x)

        x = SqueezeNet.fire_module(x, fire_id=2, squeeze=16, expand=64)
        x = SqueezeNet.fire_module(x, fire_id=3, squeeze=16, expand=64)
        x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool3')(x)

        x = SqueezeNet.fire_module(x, fire_id=4, squeeze=32, expand=128)
        x = SqueezeNet.fire_module(x, fire_id=5, squeeze=32, expand=128)
        x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool5')(x)

        x = SqueezeNet.fire_module(x, fire_id=6, squeeze=48, expand=192)
        x = SqueezeNet.fire_module(x, fire_id=7, squeeze=48, expand=192)
        x = SqueezeNet.fire_module(x, fire_id=8, squeeze=64, expand=256)
        x = SqueezeNet.fire_module(x, fire_id=9, squeeze=64, expand=256)
        x = Convolution2D(10, (1, 1), strides=(1, 1), padding='same', name='conv2')(x)

        x = GlobalAveragePooling2D()(x)

        inputs = img_input

        model = Model(inputs, x, name='squeezenet')

        optimizer = optimizers.adam(lr=0.001)
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def load_data(self):
        (self.train_images, train_labels), (self.test_images, test_labels) = cifar10.load_data()

        train_labels = train_labels.flatten()
        test_labels = test_labels.flatten()

        self.train_labels[np.arange(TRAIN_SIZE), train_labels] = 1
        self.test_labels[np.arange(TEST_SIZE), test_labels] = 1

    def train(self, train_steps=TRAIN_STEPS):
        with tf.device(self.device):
            self.model.fit(self.train_images, self.train_labels, batch_size=BATCH_SIZE, epochs=train_steps)

    def predict(self):
        with tf.device(self.device):
            self.model.evaluate(x=self.test_images, y=self.test_labels, batch_size=64)


def init(backend):
    return SqueezeNet(backend)







