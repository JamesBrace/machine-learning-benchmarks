from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import shutil
import gzip

import numpy as np
from six.moves import urllib
import tensorflow as tf


def read32(bytestream):
    """Read 4 bytes from bytestream as an unsigned 32-bit integer."""
    dt = np.dtype(np.uint32).newbyteorder('>')
    return np.frombuffer(bytestream.read(4), dtype=dt)[0]


def check_image_file_header(filename):
    """Validate that filename corresponds to images for the MNIST dataset."""
    with tf.gfile.Open(filename, 'rb') as f:
        magic = read32(f)
        rows = read32(f)
        cols = read32(f)
        # if magic != 2051:
        #     raise ValueError('Invalid magic number %d in MNIST file %s' % (magic,
        #                                                                    f.name))
        # if rows != 28 or cols != 28:
        #     raise ValueError(
        #         'Invalid MNIST file %s: Expected 28x28 images, found %dx%d' %
        #         (f.name, rows, cols))


def check_labels_file_header(filename):
    """Validate that filename corresponds to labels for the MNIST dataset."""
    with tf.gfile.Open(filename, 'rb') as f:
        magic = read32(f)
        num_items = read32(f)
        if magic != 2049:
            raise ValueError('Invalid magic number %d in MNIST file %s' % (magic,
                                                                           f.name))


def download(directory, filename):
    """Download (and unzip) a file from the MNIST dataset if not already done."""
    filepath = os.path.join(directory, filename)
    if tf.gfile.Exists(filepath):
        return filepath
    if not tf.gfile.Exists(directory):
        tf.gfile.MakeDirs(directory)
    url = 'http://yann.lecun.com/exdb/mnist/' + filename + '.gz'
    zipped_filepath = filepath + '.gz'
    print('Downloading %s to %s' % (url, zipped_filepath))
    urllib.request.urlretrieve(url, zipped_filepath)
    with gzip.open(zipped_filepath, 'rb') as f_in, open(filepath, 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)
    os.remove(zipped_filepath)
    return filepath


def dataset(type, directory, images_file, labels_file):
    images_file = download(directory, images_file)
    labels_file = download(directory, labels_file)

    check_image_file_header(images_file)
    check_labels_file_header(labels_file)

    def decode_image(image_):
        # Normalize from [0, 255] to [0.0, 1.0]
        image_ = tf.decode_raw(image_, tf.uint8)
        image_ = tf.cast(image_, tf.float32)
        image_ = tf.reshape(image_, [784])
        return image_ / 255.0

    def decode_label(label_):
        label_ = tf.decode_raw(label_, tf.uint8)  # tf.string -> [tf.uint8]
        label_ = tf.reshape(label_, [])  # label is a scalar
        label_ = tf.to_int32(label_)
        # label = tf.one_hot(label, 10)
        return label_

    images = tf.data.FixedLengthRecordDataset(
        images_file, 28 * 28, header_bytes=16).map(decode_image)
    labels = tf.data.FixedLengthRecordDataset(
        labels_file, 1, header_bytes=8).map(decode_label)

    # create image iterator
    image_iter = images.make_one_shot_iterator()
    next_image = image_iter.get_next()

    # create label iterator
    label_iter = labels.make_one_shot_iterator()
    next_label = label_iter.get_next()

    image_array = []
    label_array = []

    # Create a default in-process session.
    with tf.Session() as sess:
        if type == 'train':
            x = 10000
        else:
            x = 2000

        for i in range(x):
            image = sess.run(next_image)
            label = sess.run(next_label)
            image_array.append(image)
            label_array.append(label)

    image_array = np.asarray(image_array, np.float32)
    label_array = np.asarray(label_array, np.int32)

    return {'images': image_array, 'labels': label_array}


def train(directory):
    """tf.data.Dataset object for MNIST training data."""
    return dataset('train', directory, 'train-images-idx3-ubyte', 'train-labels-idx1-ubyte')


def test(directory):
    """tf.data.Dataset object for MNIST test data."""
    return dataset('test', directory, 't10k-images-idx3-ubyte', 't10k-labels-idx1-ubyte')
