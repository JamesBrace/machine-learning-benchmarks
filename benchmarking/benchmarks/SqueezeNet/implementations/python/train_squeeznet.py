import os
import tensorflow as tf
import input
from squeezenet import Squeezenet_CIFAR
import metrics
import model_deploy


def runner(params):
    backend = params['backend']
    mode = params['mode']

    max_train_steps = 1000

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
            pipeline = input.Pipeline(mode, sess)

            images = []
            labels = []
            for x in range(1000):
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
        model_dp = model_deploy.deploy(
            config=deploy_config,
            model_fn=_clone_fn,
            optimizer=optimizer,
            kwargs={
                'images': image_splits,
                'labels': label_splits,
                'index_iter': iter(range(deploy_config.num_clones)),
                'network': network,
                'is_training': pipeline.is_training
            }
        )

        '''Metrics'''
        train_metrics = metrics.Metrics(
            labels=labels,
            clone_predictions=[clone.outputs['predictions']
                               for clone in model_dp.clones],
            device=deploy_config.variables_device(),
            name='training'
        )

        train_op = tf.group(
            model_dp.train_op,
            train_metrics.update_op
        )

        '''Model Initialization'''
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(init_op)

        starting_step = sess.run(global_step)

        '''Main Loop'''
        for train_step in range(starting_step, max_train_steps):
            sess.run(train_op, feed_dict=images)


def _clone_fn(images,
              labels,
              index_iter,
              network,
              is_training):

    clone_index = next(index_iter)
    images = images[clone_index]
    labels = labels[clone_index]

    unscaled_logits = network.build(images, is_training)
    tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=unscaled_logits)
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


runner({'backend': 'gpu', 'mode': 'train'})

