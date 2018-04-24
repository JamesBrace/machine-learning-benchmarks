import tensorflow as tf
import input
from squeezenet import Squeezenet_CIFAR
import model_deploy
import time
import json
import argparse
import numpy as np
import math

"""""""""
Arguments 
"""""""""
parser = argparse.ArgumentParser(description='Runner script for python implementation of MNIST CNN.')
parser.add_argument(
    '--backend',
    type=str,
    choices=['cpu', 'gpu'],
    required=True,
    help=''' Backend to for model be ran on. Either 'gpu' or 'cpu' '''
)
parser.add_argument(
    '--output',
    type=str,
    required=True,
    help=''' Output file name for the benchmark results '''
)

"""""""""
CONSTANTS
"""""""""
WARMUP_STEPS = 1
EPOCHS = 1
TRAINING_SIZE = 10000
TEST_SIZE = 1000
BATCH_SIZE = 64


def runner(params):
    backend = params.backend
    output_file = params.output

    print("Info: Setting up the graph")
    with tf.Graph().as_default():
        network = Squeezenet_CIFAR()

        deploy_config = _configure_deployment(backend)
        sess = tf.Session(config=_configure_session())

        with tf.device(deploy_config.variables_device()):
            global_step = tf.train.create_global_step()

        with tf.device(deploy_config.optimizer_device()):
            optimizer = tf.train.AdamOptimizer(learning_rate=0.001)

        '''Inputs'''
        with tf.device(deploy_config.inputs_device()), tf.name_scope('inputs'):
            print("Info: Gather datasets")
            pipeline = input.Pipeline('train', sess)

            images = []
            labels = []
            for x in range(TRAINING_SIZE):
                image, label = pipeline.data
                images.append(image)
                labels.append(label)

            image_splits = tf.split(
                value=images,
                num_or_size_splits=deploy_config.num_clones,
                name='split_images'
            )

            label_splits = tf.split(
                value=labels,
                num_or_size_splits=deploy_config.num_clones,
                name='split_labels'
            )

        '''Model Creation'''
        print("Info: Compiling model")
        model_dp = model_deploy.deploy(
            config=deploy_config,
            model_fn=_clone_fn,
            optimizer=optimizer,
            kwargs={
                'images': image_splits,
                'labels': label_splits,
                'index_iter': iter(range(deploy_config.num_clones)),
                'network': network,
                'is_training': pipeline.is_training,
                'batch_size': BATCH_SIZE
            }
        )

        train_op = tf.group(
            model_dp.train_op
        )

        '''Model Initialization'''
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(init_op)

        starting_step = sess.run(global_step)

        if backend == 'gpu':
            print("Info: Warming up GPU")
            sess.run(train_op)

        '''Main Loop'''
        print("Info: Running training")

        start = time.time()
        for train_step in range(starting_step, int(math.ceil(TRAINING_SIZE / BATCH_SIZE)) * EPOCHS):
            sess.run(train_op)
        end = time.time()

        train_time = (end - start) / EPOCHS

        print("Training time average: %s" % train_time)

        # start = time.time()
        # benchmark.predict()
        # end = time.time()


def _clone_fn(images,
              labels,
              index_iter,
              network,
              is_training,
              batch_size):

    clone_index = next(index_iter)

    # Perform batch stuff here
    images = images[clone_index][:batch_size]
    labels = labels[clone_index][:batch_size]

    unscaled_logits = network.build(images, is_training)

    tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=unscaled_logits)
    predictions = tf.argmax(unscaled_logits, 1, name='predictions')
    return {
        'predictions': predictions,
        'images': images,
    }


def _configure_deployment(backend):
    if backend == 'cpu':
        return model_deploy.DeploymentConfig(clone_on_cpu=True)
    else:
        assert backend == 'gpu', "Invalid backend: %s" % backend
        return model_deploy.DeploymentConfig()


def _configure_session():
    gpu_config = tf.GPUOptions()
    config = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_config)
    config.gpu_options.allow_growth = True
    return config


if __name__ == '__main__':
    args = parser.parse_args()
    runner(args)

