import squeezenet
import time
import json
import argparse

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
WARMUP_EPOCHS = 1
TRAIN_EPOCHS = 1
TRAIN_SIZE = 1000
TEST_SIZE = 1000
BATCH_SIZE = 64


def runner(params):
    backend = params.backend
    output_file = params.output

    print("Info: Setting up the model")
    model = squeezenet.init(backend)

    print("Info: Model created")
    print("Info: Backend being used is %s" % backend)

    if backend == 'gpu':
        print("Info: Warming up GPU")
        model.train(WARMUP_EPOCHS)

    print("Info: Starting training benchmark")
    start = time.time()
    model.train()
    end = time.time()

    print('Info: Finished training benchmark!')

    train_time = (end - start) / TRAIN_EPOCHS

    print("Info: Training time was %s" % str(train_time))

    print("Info: Starting prediction testing")

    start = time.time()
    model.predict()
    end = time.time()

    print("Info: Finished prediction testing")

    test_time = end - start

    print("Info: Testing time was %s" % str(test_time))

    data = {'benchmark': 'SqueezeNet', 'backend': backend, 'implementation': 'Python', 'train': train_time,
            'test': test_time, 'train_size': TRAIN_SIZE, 'training_steps': TRAIN_EPOCHS, 'test_size': TEST_SIZE}
    print(json.dumps(data, separators=(',', ':')))

    file = open(output_file, "a+")
    file.write(json.dumps(data, separators=(',', ':')))
    file.write("\n")


if __name__ == '__main__':
    args = parser.parse_args()
    runner(args)
