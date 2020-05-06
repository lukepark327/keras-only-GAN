"""
Ref. Advanced Deep Learning with Keras by Atienza, Rowel.
"""

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, BatchNormalization, Conv2DTranspose, Conv2D, Flatten, Activation, LeakyReLU
from keras.models import Model, load_model
from keras.optimizers import RMSprop

import numpy as np
# import argparse  # TODO: params

from visualization import plot_images


def build_generator(inputs, img_size):
    img_resize = img_size // 4

    x = inputs
    x = Dense(img_resize * img_resize * 128)(x)
    x = Reshape((img_resize, img_resize, 128))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2DTranspose(filters=128, kernel_size=5, strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2DTranspose(filters=64, kernel_size=5, strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2DTranspose(filters=32, kernel_size=5, strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2DTranspose(filters=1, kernel_size=5, strides=1, padding='same')(x)
    x = Activation('sigmoid')(x)

    generator = Model(inputs, x, name='generator')
    return generator


def build_discriminator(inputs):
    x = inputs
    x = LeakyReLU(alpha=0.2)(x)
    x = Conv2D(filters=32, kernel_size=5, strides=2, padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Conv2D(filters=64, kernel_size=5, strides=2, padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Conv2D(filters=128, kernel_size=5, strides=2, padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Conv2D(filters=256, kernel_size=5, strides=1, padding='same')(x)
    x = Flatten()(x)
    x = Dense(1)(x)
    x = Activation('sigmoid')(x)

    discriminator = Model(inputs, x, name='discriminator')
    return discriminator


def train_models(models, x_train, **kwargs):
    (batch_size, latent_size, steps, img_save_interval, name) = 64, 100, 40000, 500, 'gan'  # Default
    if 'batch_size' in kwargs:
        batch_size = kwargs['batch_size']
    if 'latent_size' in kwargs:
        latent_size = kwargs['latent_size']
    if 'steps' in kwargs:
        steps = kwargs['steps']
    if 'img_save_interval' in kwargs:
        img_save_interval = kwargs['img_save_interval']
    if 'name' in kwargs:
        name = kwargs['name']

    (generator, discriminator, adversarial) = models

    img_save_noise_vectors = np.random.uniform(-1.0, 1.0, size=[16, latent_size])  # Temp. set 16

    train_size = x_train.shape[0]  # Number of training dataset
    for i in range(steps):
        # Train discriminator
        rand_idxes = np.random.randint(0, train_size, size=batch_size)
        real_imgs = x_train[rand_idxes]
        noise_vectors = np.random.uniform(-1.0, 1.0, size=[batch_size, latent_size])
        fake_imgs = generator.predict(noise_vectors)

        x = np.concatenate([real_imgs, fake_imgs])
        y = np.concatenate([np.ones([batch_size]), np.zeros([batch_size])]).reshape(-1, 1)
        loss, acc = discriminator.train_on_batch(x, y)
        log = "%05d: [discriminator loss: %f, acc: %f]" % (i, loss, acc)

        # Train adversarial
        x = np.random.uniform(-1.0, 1.0, size=[batch_size, latent_size])  # noise vectors
        y = np.ones([batch_size, 1])
        loss, acc = adversarial.train_on_batch(x, y)
        log = "%s [adversarial loss: %f, acc: %f]" % (log, loss, acc)
        # print(log)
        print(log, end='\r')

        # img_save
        if (i + 1) % img_save_interval == 0:
            show = False
            if (i + 1) == steps:
                show = True

            plot_images(generator, noise_input=img_save_noise_vectors, show=show, step=(i + 1), name=name)

    generator.save(name + ".h5")


if __name__ == "__main__":
    model_name = 'dcgan_mnist'
    img_shape = (img_row, img_col, img_dim) = (28, 28, 1)
    z_shape = (z_size, ) = (100, )
    lr = 2e-4
    decay = 6e-8

    # Loading MNIST dataset
    (x_train, _), (_, _) = mnist.load_data()
    x_train = x_train.reshape(x_train.shape[0], img_row, img_col, img_dim)
    x_train = x_train.astype('float32') / 255.

    # Discriminator
    inputs = Input(shape=img_shape, name='discriminator_input')
    discriminator = build_discriminator(inputs)
    optimizer = RMSprop(lr=lr, decay=decay)
    discriminator.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    discriminator.summary()

    # Generator
    inputs = Input(shape=z_shape, name='generator_input')
    generator = build_generator(inputs, img_shape[0])
    generator.summary()

    # Adversarial
    discriminator.trainable = False  # Fix weights  # Boolean flag at compiling

    inputs = Input(shape=z_shape, name='generator_input')
    adversarial = Model(inputs, discriminator(generator(inputs)), name=model_name)
    optimizer = RMSprop(lr=lr * 0.5, decay=decay * 0.5)
    adversarial.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    adversarial.summary()

    # Training
    models = (generator, discriminator, adversarial)
    train_models(models, x_train, batch_size=64, latent_size=z_size, steps=40000, img_save_interval=500, name=model_name)
