import numpy as np
from tensorflow.keras.datasets import mnist, cifar10


def load_mnist():
    (train, _), (_, _) = mnist.load_data()
    train = np.expand_dims(train, axis=-1)
    train = train.astype('float32')
    # train = train / 127.5 - 1  # [0,255]->[-1,1]
    train = train / 255  # [0,255]->[0,1]

    return train


def load_cifar10():
    (train, _), (_, _) = cifar10.load_data()
    train = train.astype('float32')
    # train = train / 127.5 - 1  # [0,255]->[-1,1]
    train = train / 255  # [0,255]->[0,1]

    return train
