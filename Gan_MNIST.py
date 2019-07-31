from keras.datasets import mnist
from keras.layers import *
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential, Model
from keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt
import math

(x_train, _), (_, _) = mnist.load_data()

x_train = x_train.reshape((*x_train.shape, 1))
x_train = (x_train.astype('float32')-127.5)/127.5

TOTAL_EPOCHS = 50
BATCH_SIZE = 256
NO_OF_BATCHES = math.ceil(x_train.shape[0] / float(BATCH_SIZE))
HALF_BATCH = int(BATCH_SIZE/2)
NOISE_DIM = 100
adam = Adam(lr=2e-4, beta_1=0.5)

generator = Sequential()
generator.add(Dense(7 * 7 * 128, input_shape=(NOISE_DIM,)))
generator.add(Reshape((7, 7, 128)))
generator.add(LeakyReLU(0.2))
generator.add(BatchNormalization())
generator.add(UpSampling2D())
generator.add(Conv2D(64, kernel_size=(5, 5), padding='same'))
generator.add(LeakyReLU(0.2))
generator.add(BatchNormalization())
generator.add(UpSampling2D())
generator.add(Conv2D(1, kernel_size=(5, 5), padding='same', activation='tanh'))
generator.compile(loss='binary_crossentropy', optimizer=adam)

discriminator = Sequential()
discriminator.add(Conv2D(64, kernel_size=(5, 5), strides=(2, 2),
                         padding='same', input_shape=(28, 28, 1)))
discriminator.add(LeakyReLU(0.2))
discriminator.add(Conv2D(128, kernel_size=(5, 5), strides=(2, 2),
                         padding='same'))
discriminator.add(LeakyReLU(0.2))
discriminator.add(Flatten())
discriminator.add(Dense(1, activation='sigmoid'))
discriminator.compile(loss='binary_crossentropy', optimizer=adam)

discriminator.trainable = False
gan_input = Input(shape=(NOISE_DIM,))
generated_img = generator(gan_input)
gan_output = discriminator(generated_img)

model = Model(gan_input, gan_output)
model.compile(loss='binary_crossentropy', optimizer=adam)


def save_img(epoch, sample=100):
    noise = np.random.normal(0, 1, size=(sample, NOISE_DIM))
    generated_imgs = generator.predict(noise)
    generated_imgs = generated_imgs.reshape(sample, 28, 28)
    plt.figure(figsize=(10, 10))
    for i in range(sample):
        plt.subplot(10, 10, i + 1)
        plt.imshow(generated_imgs[i], interpolation='nearest', cmap='gray')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(f'images/gan_output_epoch{epoch+1}.png')
    plt.show()


for epoch in range(TOTAL_EPOCHS):
    epoch_d_loss = 0.
    epoch_g_loss = 0.
    for step in range(NO_OF_BATCHES):
        idx = np.random.randint(0, x_train.shape[0], HALF_BATCH)
        real_img = x_train[idx]
        noise = np.random.normal(0, 1, size=(HALF_BATCH, NOISE_DIM))
        fake_img = generator.predict(noise)

        real_y = np.ones((HALF_BATCH, 1)) * .9
        fake_y = np.zeros((HALF_BATCH, 1))
        d_loss_real = discriminator.train_on_batch(real_img, real_y)
        d_loss_fake = discriminator.train_on_batch(fake_img, fake_y)
        d_loss = .5 * d_loss_fake + .5 * d_loss_real
        epoch_d_loss += d_loss

        noise = np.random.normal(0, 1, size=(BATCH_SIZE, NOISE_DIM))
        ground_truth_y = np.ones((BATCH_SIZE, 1))
        g_loss = model.train_on_batch(noise, ground_truth_y)
        epoch_g_loss += g_loss

    if (epoch + 1) % 5 == 0:
        generator.save(f'model/gan_generator_{epoch+1}.h5')
        save_img(epoch)
