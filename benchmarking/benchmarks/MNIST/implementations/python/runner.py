import mnist
import time
import json


def runner(params):
    backend = params['backend']
    mode = params['mode']

    benchmark = mnist.init(backend)

    start = time.time()
    benchmark.train()
    end = time.time()

    if mode == 'train':
        print("Computation time: %s" % str(end - start))

        data = {'status': '1', 'options': 'train( %s )' % backend, 'time': "%s" % str((end - start) / 1000),
                'output': '0'}
        print(json.dumps(data, separators=(',',':')))
        return

    start = time.time()
    benchmark.predict()
    end = time.time()

    print("Computation time: %s" % str(end - start))

    data = {'status': '1', 'options': 'predict( %s )' % backend, 'time': "%s" % str((end - start) / 1000),
            'output': '0'}
    print(json.dumps(data, separators=(',', ':')))
