from keras.layers import Input, Convolution2D, MaxPooling2D, Activation, concatenate, Dropout, warnings
from keras.layers import GlobalAveragePooling2D
from keras import optimizers
from keras.models import Model
from keras.datasets import cifar10
import tensorflow as tf

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
        self.train_images, self.train_labels, \
        self.test_images, self.test_labels = SqueezeNet.load_data()
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
        x = Convolution2D(10, (1, 1), strides=(1,1), padding='same', name='conv2')

        x = GlobalAveragePooling2D()(x)

        inputs = img_input

        model = Model(inputs, x, name='squeezenet')

        optimizer = optimizers.adam(lr=0.001)
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    @staticmethod
    def load_data():
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()

        train_images = []
        train_labels = []
        for x in range(TRAIN_SIZE):
            train_images.append(tf.to_float(x_train[x]))
            train_labels.append(tf.reshape(tf.one_hot(y_train[x], 10), [10]))

        test_images = []
        test_labels = []
        for x in range(TEST_SIZE):
            test_images.append(tf.to_float(x_test[x]))
            test_labels.append(tf.reshape(tf.one_hot(y_test[x], 10), [10]))

        return train_images, train_labels, test_images, test_labels


    def train(model, set, set_size = BATCH_SIZE, train_steps = TRAIN_STEPS):
        for i in range(train_steps):


    def predict():


def init():
    return SqueezeNet()







