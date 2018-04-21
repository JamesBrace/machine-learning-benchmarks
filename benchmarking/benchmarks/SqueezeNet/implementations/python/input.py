import tensorflow as tf
from keras.datasets import cifar10


class Pipeline(object):
    def __init__(self, type, sess):
        self.sess = sess
        self.batch_size = 64

        (x_train, y_train), (x_test, y_test) = cifar10.load_data()

        if type == 'train':
            dataset = tf.data.Dataset.from_tensor_slices((x_train,  y_train))
            self.is_training = True
        else:
            dataset = tf.data.Dataset.from_tensor_slices((x_test,  y_test))
            self.is_training = False

        iterator = dataset.make_one_shot_iterator()

        self.data = iterator.get_next()
