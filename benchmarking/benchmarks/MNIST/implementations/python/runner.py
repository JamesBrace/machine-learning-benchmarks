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


"""
CONSTANTS
"""
WARMUP_STEPS = 1
TRAINING_STEPS = 5
TRAINING_SIZE = 50000
TEST_SIZE = 10000


def runner(params):
    print(params)
    backend = params.backend
    output = params.output

    benchmark = mnist.init(backend, train_size=TRAINING_SIZE, test_size=TEST_SIZE)

    if backend == 'gpu':
        benchmark.train(WARMUP_STEPS)

    start = time.time()
    benchmark.train(TRAINING_STEPS)
    end = time.time()

    train_time = (end - start)/TRAINING_STEPS
    print("Training time average: %s" % train_time)

    start = time.time()
    benchmark.predict()
    end = time.time()

    test_time = end - start

    print("Test time: %s" % str(test_time))

    data = {'benchmark': 'MNIST', 'backend': backend, 'implementation': 'Python', 'train': train_time,
            'test': test_time, 'train_size': TRAINING_SIZE, 'training_steps': TRAINING_STEPS, 'test_size': TEST_SIZE}
    print(json.dumps(data, separators=(',', ':')))

    file = open(output, "a+")
    file.write(json.dumps(data, separators=(',', ':')))
    file.write("\n")


if __name__ == '__main__':
    args = parser.parse_args()
    runner(args)

