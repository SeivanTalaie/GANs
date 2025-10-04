########################### Libraries ###########################

import numpy as np 
import matplotlib.pyplot as plt 
from keras.datasets import cifar10
from keras.layers import Dense, Reshape, BatchNormalization, Input, Flatten, LeakyReLU
from keras.models import Sequential, Model
from keras.optimizers import Adam


########################### Generator ###########################

image_shape=(32,32,3)

def build_generator():
    
    noise_shape=(100,) #Latent Vector shape
    
    model = Sequential()
    model.add(Dense(256, input_shape=noise_shape))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    
    model.add(Dense(1024))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    
    model.add(Dense(np.prod(image_shape),activation="tanh"))
    model.add(Reshape(image_shape))
    
    noise=Input(shape=noise_shape)
    img=model(noise)
    
    return Model(noise, img)


########################### Discriminator ###########################

def build_discriminator():
    
    model= Sequential()
    model.add(Flatten(input_shape=image_shape))
    
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    
    model.add(Dense(256))
    model.add(LeakyReLU(alpha=0.2))
    
    model.add(Dense(1, activation="sigmoid"))
    
    img=Input(shape=image_shape)
    validity=model(img)
    
    return Model(img, validity)


########################### Train Function ###########################

def train(epochs, batch_size=128, save_interval=50):
    
    (x_train,_),(_,_) = cifar10.load_data()
    
    x_train = (x_train.astype("float32")-127.5) / 127.5 
    
    # x_train = np.expand_dims(x_train, axis=3)
    
    half_batch = int(batch_size / 2)
    
    for epoch in range(epochs):
        
        # first we train the discriminator
        
        indx=np.random.randint(0, x_train.shape[0], half_batch)
        imgs = x_train[indx]  
        
        noise = np.random.normal(0, 1, (half_batch, 100))
        
        gen_imgs = generator.predict(noise)
        
        d_loss_real = discriminator.train_on_batch(imgs, np.ones((half_batch,1)))
        d_loss_fake = discriminator.train_on_batch(gen_imgs, np.zeros((half_batch,1)))
        
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake) 

        # second we train the generator
        
        noise = np.random.normal(0, 1, (batch_size, 100)) 
        
        valid_y = np.array([1] * batch_size)

        g_loss = combined.train_on_batch(noise, valid_y)

        print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

        if epoch % save_interval == 0:
            save_imgs(epoch)


#################### Get Prediction every n epoch ####################

def save_imgs(epoch):
    r, c = 2, 2
    noise = np.random.normal(0, 1, (r * c, 100))
    gen_imgs = generator.predict(noise)

    # Rescale images 0 - 1
    gen_imgs = 0.5 * gen_imgs + 0.5

    fig, axs = plt.subplots(r, c)
    cnt = 0
    for i in range(r):
        for j in range(c):
            axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')
            axs[i,j].axis('off')
            cnt += 1
    fig.savefig("F:/Python/GAN/cifar10/cifar10_%d.png" % epoch)
    plt.close()


########################## Define optimizer ##########################

optimizer = Adam(0.0002, 0.5)

#################### Build and Compile GAN Model #####################

discriminator = build_discriminator()
discriminator.compile(loss='binary_crossentropy',
    optimizer=optimizer,
    metrics=['accuracy']) 
    
generator = build_generator()
generator.compile(loss='binary_crossentropy', optimizer=optimizer)


z = Input(shape=(100,))
img = generator(z)

discriminator.trainable = False  

valid = discriminator(img)  #Validity check on the generated image

combined = Model(z, valid)
combined.compile(loss='binary_crossentropy', optimizer=optimizer)


########################### Train the GAN ###########################

train(epochs=60000, batch_size=64, save_interval=500)

generator.save('generator_model.h5')
    
