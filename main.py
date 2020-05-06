import datasets
from gan.gan import GAN
from generator import Generator


def train():
    dataset = datasets.load_mnist()
    path = 'gan'

    gan = GAN(dataset=dataset, path=path, epochs=30000, batch=256, save_interval=1000)
    gan.train()


def generate():
    path = 'gan'
    generator = Generator(path=path, num=25)
    # generator.save_plot(row=5, col=5)
    # generator.save_fake()
    generator.interpolate()


if __name__ == '__main__':
    # train()
    generate()
