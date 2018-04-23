from squeezenet.python.networks.squeezenet import Squeezenet_CIFAR

catalogue = dict()


def register(cls):
    catalogue.update({cls.name: cls})


register(Squeezenet_CIFAR)
