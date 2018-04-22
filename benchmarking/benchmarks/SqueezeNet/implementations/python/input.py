import tensorflow as tf
from keras.datasets import cifar10
import numpy as np


class Pipeline(object):
    def __init__(self, type, sess):
        self.sess = sess
        self.batch_size = 64

        (x_train, y_train), (x_test, y_test) = cifar10.load_data()

        if type == 'train':
            labels = []
            images = []
            for x in range(1000):
                images.append(tf.to_float(x_train[x]))
                labels.append(tf.reshape(tf.one_hot(y_train[x], 10, dtype=tf.int32), [10]))

            dataset = tf.data.Dataset.from_tensor_slices((images,  labels))
            self.is_training = True
        else:

            for x in range(len(x_test)):
                x_test[x] = np.asarray(x_test[x], dtype=np.float32)
                y_test[x] = tf.one_hot(y_test[x])

            dataset = tf.data.Dataset.from_tensor_slices((x_test,  y_test))
            self.is_training = False

        iterator = dataset.make_one_shot_iterator()

        self.data = iterator.get_next()
