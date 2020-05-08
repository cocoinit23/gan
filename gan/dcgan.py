from gan.gan import GAN
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, LeakyReLU, Reshape, Conv2D, Conv2DTranspose, Dropout


class DCGAN(GAN):
    def __init__(self, dataset, epochs=30000, batch=256, save_interval=1000):
        super().__init__(dataset, epochs, batch, save_interval)
        self.path = 'results/dcgan'

    def build_discriminator(self):
        model = Sequential(name='DCGAN_Discriminator')
        model.add(Conv2D(64, (3, 3), padding='same', input_shape=self.img_shape))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Conv2D(128, (3, 3), strides=(2, 2), padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Conv2D(128, (3, 3), strides=(2, 2), padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Conv2D(256, (3, 3), strides=(2, 2), padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Flatten())
        model.add(Dropout(0.4))
        model.add(Dense(1, activation='sigmoid'))

        model.summary()

        return model

    def build_generator(self):
        model = Sequential(name='DCGAN_Generator')
        n_nodes = 256 * self.img_shape[0] // 4 * self.img_shape[1] // 4
        model.add(Dense(n_nodes, input_dim=self.latent_dim))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Reshape((self.img_shape[0] // 4, self.img_shape[1] // 4, 256)))
        model.add(Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Conv2D(self.img_shape[2], (3, 3), activation='sigmoid', padding='same'))

        model.summary()

        return model
