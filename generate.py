import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from PIL import Image


class Generator:
    def __init__(self, path, latent_dim=100, num=25):
        self.latent_dim = latent_dim
        self.num = num
        self.path = path

        self.model = os.path.join(self.path, 'model/best.h5')
        self.generator = load_model(self.model)

        self.z = self.latent_vector()
        self.fake_img = self.generator.predict(self.z)

    def latent_vector(self):
        latent = np.random.randn(self.latent_dim * self.num)
        latent = latent.reshape(self.num, self.latent_dim)

        return latent

    def save_plot(self, row, col):
        for i in range(row * col):
            plt.subplot(row, col, i + 1)
            plt.imshow(self.fake_img[i, :, :, 0], cmap='gray')
            plt.axis('off')

        # plt.show()
        save_name = os.path.join(self.path, 'fake/plot.jpg')
        plt.savefig(save_name)

    def save_fake(self):
        for i in range(self.num):
            save_name = os.path.join(self.path, 'fake/%04d.jpg' % (i + 1))
            img = np.clip(self.fake_img[i] * 255, 0, 255).squeeze()
            img = Image.fromarray(img.astype(np.uint8))
            img.save(save_name)

    def interpolate(self):
        z = self.latent_vector()
        steps = 20
        ratio = np.linspace(0, 1, steps)

        morph = []
        for i in range(len(z)):
            for r in ratio:
                v = r * z[i] + (1 - r) * z[i - 1]
                morph.append(v)

        morph_img = self.generator.predict(np.asarray(morph))

        gif = []
        for idx in range(len(morph_img)):
            img = np.clip(morph_img[idx] * 255, 0, 255).squeeze()
            gif.append(Image.fromarray(img.astype(np.uint8)))

        save_name = os.path.join(self.path, 'fake/morphing.gif')
        gif[0].save(save_name, save_all=True, append_images=gif[1:], loop=1)


if __name__ == '__main__':
    path = 'gan'
    generator = Generator(path=path, num=25)
    # generator.save_plot(row=5, col=5)
    # generator.save_fake()
    generator.interpolate()
