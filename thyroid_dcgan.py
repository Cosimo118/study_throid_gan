import numpy as np

from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D,Conv2DTranspose
from keras.models import Sequential, Model
from keras.optimizers import Adam
import matplotlib.pyplot as plt


def loadData():
    pic = np.load(r'E:\F\Dataset\thyroid_down_left_sharp.npy')
    return pic

class LSGAN():
    def __init__(self):
        self.img_rows = 192
        self.img_cols = 192
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 100

        traindata = loadData()
        self.traindata = traindata
        self.traindata = self.traindata.reshape(-1, self.img_rows, self.img_cols, self.channels).astype(np.float32)

        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='mse',
                                   optimizer=optimizer,
                                   metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()

        # The generator takes noise as input and generated imgs
        frozen_D = Model(
            inputs=self.build_discriminator().inputs,
            outputs=self.build_discriminator().outputs)
        frozen_D.trainable = False

        z = Input(shape=(self.latent_dim,))
        img = self.generator(z)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The valid takes generated images as input and determines validity
        valid = frozen_D(img)

        # The combined model  (stacked generator and discriminator)
        # Trains generator to fool discriminator
        self.combined = Model(z, valid)
        # (!!!) Optimize w.r.t. MSE loss instead of crossentropy
        self.combined.compile(loss='mse', optimizer=optimizer)

    def build_discriminator(self):
        depth = 64
        model = Sequential()
        input_shape = (self.img_rows, self.img_cols, self.channels)
        model.add(Conv2D(depth , 5, strides = 2, input_shape = input_shape, padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))

        model.add(Conv2D(depth*2 , 5, strides=2, padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))

        model.add(Conv2D(depth*4, 5, strides=2, padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))

        model.add(Conv2D(depth*8, 5, strides=1, padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))

        model.add(Flatten())
        model.add(Dense(1))
        model.add(Activation('sigmoid'))
        model.summary()

        img = Input(shape=self.img_shape)
        validity = model(img)

        return Model(img, validity)

    def build_generator(self):

        model = Sequential()
        depth = 64
        dim = 48

        model.add(Dense(dim*dim*depth,input_dim=100))
        model.add(BatchNormalization(momentum=0.9))
        model.add(Activation('relu'))
        model.add(Reshape((dim,dim,depth)))

        model.add(UpSampling2D())
        model.add(Conv2DTranspose(int(depth/2), 5, padding = 'same'))
        model.add(BatchNormalization(momentum=0.9))
        model.add(Activation('relu'))

        model.add(UpSampling2D())
        model.add(Conv2DTranspose(int(depth/4), 5, padding = 'same'))
        model.add(BatchNormalization(momentum=0.9))
        model.add(Activation('relu'))

        model.add(Conv2DTranspose(1, 5, padding='same'))
        model.add(Activation('tanh'))
        model.summary()


        noise = Input(shape=(self.latent_dim,))
        img = model(noise)


        return Model(noise, img)

    def train(self,train_steps = 2000 , batch_size = 30, save_interval = 50):
        for i in range(train_steps):
            images_train = self.traindata[np.random.randint(0,self.traindata.shape[0], size = batch_size),:,:,:]
            noise = np.random.uniform(-1.0, 1.0 , size = [batch_size, 100])
            images_fake = self.generator.predict(noise, batch_size = 5)
            real = np.random.random_sample((batch_size,1))*0.1+0.8
            fake = np.random.random_sample((batch_size,1))*0.1

            d_loss_real = self.discriminator.train_on_batch(images_train, real)
            d_loss_fake = self.discriminator.train_on_batch(images_fake,fake)
            d_loss = 0.5*(np.add(d_loss_real,d_loss_fake))

            noise = np.random.uniform(-1.0, 1.0, size = [batch_size, 100])
            g_loss = self.combined.train_on_batch(noise, real)

            print("%d,[D loss :%f, acc :%.2f%%] [G loss: %f]" %(i,d_loss[0],100*d_loss[1],g_loss))

            if (i+1)%save_interval==0:
                noise_plot = np.random.uniform(-1.0, 1.0, size = [1,100])
                image_plot = self.generator.predict(noise_plot)
                image_plot = np.reshape(image_plot, [self.img_rows, self.img_cols])
                plt.imshow(image_plot, cmap = 'gray')
                plt.savefig('thyroid_%d.png'%i)

if __name__ == '__main__':
    thyroid_lsgan = LSGAN()
    thyroid_lsgan.train(train_steps=1000,batch_size = 10, save_interval=50)