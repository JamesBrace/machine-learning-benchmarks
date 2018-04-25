import mnist
import time
import json
import argparse

"""
Arguments 
"""
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
TRAINING_SIZE = 50000
TEST_SIZE = 10000


def runner(params):
    backend = params.backend
    output = params.output

    print("Info: Initializing model")

    benchmark = mnist.init(backend, train_size=TRAINING_SIZE, test_size=TEST_SIZE)

    if backend == 'gpu':
        print("Info: Warming up GPU")
        benchmark.train(EPOCHS)

    print("Info: Starting training benchmark")
    start = time.time()
    benchmark.train(EPOCHS)
    end = time.time()
    print("Info: Finished training benchmark")

    train_time = (end - start) / EPOCHS
    print("Training time average: %s" % train_time)

    print("Info: Starting testing benchmark")
    start = time.time()
    benchmark.predict()
    end = time.time()
    print("Info: Finished testing benchmark")

    test_time = end - start

    print("Test time: %s" % str(test_time))

    data = {'benchmark': 'MNIST', 'backend': backend, 'implementation': 'Python', 'train': train_time,
            'test': test_time, 'train_size': TRAINING_SIZE, 'training_steps': EPOCHS, 'test_size': TEST_SIZE}
    print(json.dumps(data, separators=(',', ':')))

    file = open(output, "a+")
    file.write(json.dumps(data, separators=(',', ':')))
    file.write("\n")


if __name__ == '__main__':
    args = parser.parse_args()
    runner(args)

