import datasets
from gan.gan import GAN


def train():
    dataset = datasets.load_mnist()
    path = 'gan'

    gan = GAN(dataset=dataset, path=path, epochs=30000, batch=256, save_interval=1000)
    gan.train()


if __name__ == '__main__':
    train()
