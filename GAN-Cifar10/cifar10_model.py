#################################################

import numpy as np 
import matplotlib.pyplot as plt 
from keras.layers import Dense, Flatten, Dropout, BatchNormalization, Conv2D, Conv2DTranspose, LeakyReLU, Reshape
from keras.datasets import cifar10
from keras.optimizers import Adam 
from keras.models import Sequential


img_input_shape= (32,32,3)
latent_dim=100

#################################################

def generator(latent_dim=100):
    
    n_nodes= 256*4*4
    
    model=Sequential()
    
    model.add(Dense(n_nodes,input_dim=latent_dim))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Reshape((4,4,256)))
    
    model.add(Conv2DTranspose(256, (4,4), strides=(2,2), padding='same')) # (8,8,256)
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    
    model.add(Conv2DTranspose(128,(4,4), strides=(2,2), padding='same')) # (16,16,128)
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    
    model.add(Conv2DTranspose(128,(4,4), strides=(2,2), padding='same')) # (32,32,128)
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    
    model.add(Conv2D(3,(8,8), padding="same", activation="tanh")) # (32,32,3)
    
    return model


generator_sample = generator(latent_dim)
generator_sample.summary()

#################################################

def discriminator(input_shape=(32,32,3)):
    
    model=Sequential()
    
    model.add(Conv2D(128, kernel_size=(4,4), strides=(2,2), padding="same", input_shape=input_shape))
    model.add(LeakyReLU(alpha=0.2))
    
    model.add(Conv2D(128, kernel_size=(4,4), strides=(2,2), padding="same"))
    model.add(LeakyReLU(alpha=0.2))
    
    model.add(Conv2D(64, kernel_size=(4,4), padding="same"))
    model.add(LeakyReLU(alpha=0.2))
    
    model.add(Flatten())
    model.add(Dropout(0.2))
    model.add(Dense(1,activation="sigmoid"))
    
    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(opt, loss="binary_crossentropy", metrics=["accuracy"])
    return model    

discriminator_sample = discriminator()
discriminator_sample.summary()

#################################################

def gan_model(generator, discriminator):
    discriminator.trainable = False
    
    model=Sequential()
    model.add(generator)
    model.add(discriminator)
    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt)
    return model

#################################################

def load_real_images():
    (x_train,_),(_,_) = cifar10.load_data()
    x_train = (x_train.astype("float32") - 127.5) / 127.5 #[-1 , +1]
    
    return x_train

#################################################

def generate_real_images(x_train, n_images):
    
    n_img = np.random.randint(0, x_train.shape[0], n_images)
    
    real_images = x_train[n_img]
    real_labels = np.ones((n_images,1)) 
    
    return real_images, real_labels

#################################################

def generate_latent_points(latent_dim, n_samples):
    
    noise = np.random.normal(0,1,(n_samples,latent_dim))
    return noise

#################################################

def generate_fake_images(generator, latent_dim, n_samples):
    
    noise = generate_latent_points(latent_dim, n_samples) # shape: [n_samples, latent_dim]
    
    fake_images= generator.predict(noise)
    fake_labels= np.zeros((n_samples,1))
    
    return fake_images, fake_labels

#################################################

def train(g_model, d_model, gan_model, x_train, latent_dim, batch_size, epochs, save_interval):
    
    batch_per_epoch = int(x_train.shape[0]/batch_size) 
    half_batch = int(batch_size/2)
    
    for i in range(epochs):
        for j in range(batch_per_epoch):
            
            ### train discriminator ###
            
            x_real, y_real = generate_real_images(x_train, half_batch)
            
            d_loss_real,_ = d_model.train_on_batch(x_real, y_real)
            
            x_fake, y_fake = generate_fake_images(g_model, latent_dim, half_batch)
            
            d_loss_fake,_ = d_model.train_on_batch(x_fake, y_fake)
            
            ### train generator ###
            
            x_gan = generate_latent_points(latent_dim, batch_size)
            
            y_gan = np.ones((batch_size, 1))
            
            g_loss = gan_model.train_on_batch(x_gan, y_gan)
            
            print(f"Epoch {i}: {j+1}/{batch_per_epoch} __ [d1: {d_loss_real:.2f} _ d2: {d_loss_fake:.2f} _ g_loss: {g_loss:.2f}] ")
            
        if i % save_interval == 0 :
            
            noise = np.random.normal(0, 1, (4, 100))
            gen_imgs = g_model.predict(noise)
            gen_imgs = 0.5 * gen_imgs + 0.5
            
            plt.figure(figsize=(10,10))
            plt.subplot(221)
            plt.imshow(gen_imgs[0])
            
            plt.subplot(222)
            plt.imshow(gen_imgs[1])
            
            plt.subplot(223)
            plt.imshow(gen_imgs[2])
            
            plt.subplot(224)
            plt.imshow(gen_imgs[3])
            plt.savefig(f"F:/Python/GAN/cifar10/epoch_{i}.png")
            plt.close()



#################################################

discriminator_part = discriminator()

generator_part = generator(latent_dim=latent_dim)

gan_network = gan_model(generator_part, discriminator_part)

x_train = load_real_images()

batch_size= 128
epochs= 15
save_interval= 1

train(generator_part, discriminator_part, gan_network,
      x_train, latent_dim, batch_size, epochs, save_interval)

