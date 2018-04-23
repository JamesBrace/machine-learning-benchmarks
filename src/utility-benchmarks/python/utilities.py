from tensorflow.python.client import device_lib


def get_backend(backend):
    if backend == 'gpu':
        return "/device:GPU:0"
    else:
        return "/cpu:0"


def can_use_gpu():
    devices = device_lib.list_local_devices()
    gpus_exist = [True for x in devices if x.device_type == 'GPU']
    assert len(gpus_exist) > 0, "No CUDA-compatible GPUs available!"
    return True
