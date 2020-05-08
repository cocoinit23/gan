import numpy as np

import datasets
from gan.gan import GAN
from gan.dcgan import DCGAN
from generator import Generator


def train():
    # dataset = datasets.load_mnist()
    dataset = datasets.load_cifar10()

    # gan = GAN(dataset=dataset, epochs=30000, batch=256, save_interval=1000)
    gan = DCGAN(dataset=dataset, epochs=30000, batch=256, save_interval=1000)

    gan.train()


def generate():
    gan_name = 'gan'
    generator = Generator(gan_name=gan_name, num=100)
    # generator.save_plot(row=5, col=5)
    # generator.save_fake()
    generator.interpolate(steps=20)


if __name__ == '__main__':
    # np.random.seed(0)

    train()
    # generate()

    print('Done!')
