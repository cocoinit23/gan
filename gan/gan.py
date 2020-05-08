import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, LeakyReLU, BatchNormalization, Reshape
from tensorflow.keras.optimizers import Adam


class GAN(object):
    def __init__(self, dataset, epochs=30000, batch=256, save_interval=1000):
        self.dataset = dataset
        self.path = 'results/gan'

        self.img_shape = (dataset.shape[1], dataset.shape[2], dataset.shape[3])
        print("Dataset shape:", dataset.shape)
        assert self.img_shape[0] % 4 == 0 or self.img_shape[1] % 4 == 0, "Input shape must be a multiple of 4."
        assert self.img_shape[2] == 1 or self.img_shape[2] == 3, "Input channels must be 1 or 3."

        self.latent_dim = 100

        self.epochs = epochs
        self.batch = batch
        self.save_interval = save_interval

        self.loss = 'binary_crossentropy'
        self.optimizer = Adam(lr=0.0002, beta_1=0.5)
        self.metrics = ['accuracy']

        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss=self.loss, optimizer=self.optimizer, metrics=self.metrics)

        self.generator = self.build_generator()
        self.combined = self.build_combined()
        self.combined.compile(loss=self.loss, optimizer=self.optimizer)

    def latent_vector(self):
        latent = np.random.randn(self.latent_dim * self.batch)
        latent = latent.reshape(self.batch, self.latent_dim)

        return latent

    def real_samples(self):
        idx = np.random.randint(0, self.dataset.shape[0], self.batch)
        x = self.dataset[idx]
        y = np.ones((self.batch, 1))

        return x, y

    def fake_samples(self):
        z = self.latent_vector()
        x = self.generator.predict(z)
        y = np.zeros((self.batch, 1))

        return x, y

    def combined_samples(self):
        x = self.latent_vector()
        y = np.ones((self.batch, 1))

        return x, y

    def build_discriminator(self):
        model = Sequential(name='GAN_Discriminator')
        model.add(Flatten(input_shape=self.img_shape))
        model.add(Dense(512))
        model.add(LeakyReLU())
        model.add(Dense(256))
        model.add(LeakyReLU())
        model.add(Dense(1, activation='sigmoid'))

        model.summary()

        return model

    def build_generator(self):
        model = Sequential(name='GAN_Generator')
        model.add(Dense(256, input_shape=(self.latent_dim,)))
        model.add(LeakyReLU())
        model.add(BatchNormalization())
        model.add(Dense(512))
        model.add(LeakyReLU())
        model.add(BatchNormalization())
        model.add(Dense(1024))
        model.add(LeakyReLU())
        model.add(BatchNormalization())
        model.add(Dense(np.prod(self.img_shape), activation='sigmoid'))
        model.add(Reshape(self.img_shape))

        model.summary()

        return model

    def build_combined(self):
        self.discriminator.trainable = False

        model = Sequential(name='Combined')
        model.add(self.generator)
        model.add(self.discriminator)

        model.summary()

        return model

    def plot(self, fake_img, epoch):
        row, col = 5, 5

        for i in range(row * col):
            plt.subplot(row, col, i + 1)
            img = fake_img[i]
            if self.img_shape[2] == 1:
                plt.imshow(img[:, :, 0], cmap='gray')
            else:
                plt.imshow(img)
            plt.axis('off')

        save_name = os.path.join(self.path, 'progress/%05d.png' % epoch)
        plt.savefig(save_name)
        plt.close()

    def train(self):
        g_loss_min = 10 ** 10
        for epoch in tqdm(range(self.epochs)):
            x_real, y_real = self.real_samples()
            d_loss_real = self.discriminator.train_on_batch(x_real, y_real)

            x_fake, y_fake = self.fake_samples()
            d_loss_fake = self.discriminator.train_on_batch(x_fake, y_fake)

            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            x_combined, y_combined = self.combined_samples()
            g_loss = self.combined.train_on_batch(x_combined, y_combined)

            if epoch % self.save_interval == 0:
                print('epoch=%d, D_loss=%f, D_acc=%f, G_loss=%f' % (epoch, d_loss[0], d_loss[1], g_loss))
                self.plot(x_fake, epoch)

                save_name = os.path.join(self.path, 'model/%05d.h5' % epoch)
                self.generator.save(save_name)

                if epoch != 0 and g_loss < g_loss_min:
                    g_loss_min = g_loss
                    best_name = os.path.join(self.path, 'model/best.h5')
                    self.generator.save(best_name)
                    print('Best G_loss updated : %f' % g_loss_min)
