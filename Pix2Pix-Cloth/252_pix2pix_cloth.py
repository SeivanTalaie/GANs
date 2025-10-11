#################################################

import os
import glob
import cv2
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

#################################################

#Resizing images, if needed
SIZE_X = 256
SIZE_Y = 256

#Capture training image info as a list
tar_images = []
tar_image_list = glob.glob("F:/Python/Datasets/cloth segmentation/img/*.jpg")

for path in tar_image_list:
    img = cv2.imread(path, 1)       
    img = cv2.resize(img, (SIZE_Y, SIZE_X), interpolation = cv2.INTER_NEAREST)
    tar_images.append(img)
       
#Convert list to array for machine learning processing        
tar_images = np.array(tar_images)

#Capture mask/label info as a list
src_images = [] 
src_images_list = glob.glob("F:/Python/Datasets/cloth segmentation/masks/*.png")

for path in src_images_list:
    mask = cv2.imread(path, 1)       
    mask = cv2.resize(mask, (SIZE_Y, SIZE_X), interpolation = cv2.INTER_NEAREST)  #Otherwise ground truth changes due to interpolation
    src_images.append(mask)
        
#Convert list to array for machine learning processing          
src_images = np.array(src_images)

# print(np.unique(src_images))

#################################################

n_samples = 3
for i in range(n_samples):
	plt.subplot(2, n_samples, 1 + i)
	plt.axis('off')
	plt.imshow(src_images[i])
# plot target image
for i in range(n_samples):
	plt.subplot(2, n_samples, 1 + n_samples + i)
	plt.axis('off')
	plt.imshow(tar_images[i])
plt.show()

#################################################

x_train, x_test, y_train, y_test = train_test_split(src_images,
                                                    tar_images,
                                                    test_size=0.2,
                                                    random_state=12 )

rand_num=np.random.randint(0,x_train.shape[0])
plt.figure(figsize=(10,10))
plt.subplot(121)
plt.imshow(x_train[rand_num])
plt.subplot(122)
plt.imshow(y_train[rand_num])
plt.show()

#################################################

from pix2pix_model import define_discriminator, define_generator, define_gan, train
# define input shape based on the loaded dataset
image_shape = src_images.shape[1:]
# define the models
d_model = define_discriminator(image_shape)
g_model = define_generator(image_shape)
# define the composite model
gan_model = define_gan(g_model, d_model, image_shape)

#################################################

#Define data
# load and prepare training images
data_train = [x_train, y_train]
data_test = [x_test, y_test]

def preprocess_data(data):
	# load compressed arrays
	# unpack arrays
	X1, X2 = data[0], data[1]
	# scale from [0,255] to [-1,1]
	X1 = (X1 - 127.5) / 127.5
	X2 = (X2 - 127.5) / 127.5
	return [X1, X2]

dataset = preprocess_data(data_train)
data_test = preprocess_data(data_test)

[x_test1, y_test1] = data_test

#################################################

from datetime import datetime 

start1 = datetime.now() 
train(d_model, g_model, gan_model, dataset, x_test1, y_test1, n_epochs=1000, n_batch=1) 

stop1 = datetime.now()
#Execution time of the model 
execution_time = stop1-start1
print("Execution time is: ", execution_time)

g_model.save('cloth_generator.h5')

#################################################

from keras.models import load_model
from numpy.random import randint
from numpy import vstack

model = load_model('F:/Python/GAN/cloth_pix2pix/model_003800.h5',
                   compile=False)

# plot source, generated and target images
def plot_images(src_img, gen_img, tar_img):
	images = vstack((src_img, gen_img, tar_img))
	# scale from [-1,1] to [0,1]
	images = (images + 1) / 2.0
	titles = ['Input-segm-img', 'Output-Generated', 'target_img']
	# plot images row by row
	for i in range(len(images)):
		# define subplot
		plt.subplot(1, 3, 1 + i)
		# turn off axis
		plt.axis('off')
		# plot raw pixel data
		plt.imshow(images[i,:,:,0], cmap='gray')
		# show title
		plt.title(titles[i])
	plt.show()


[X1, X2] = dataset
# select random example
ix = randint(0, len(X1), 1)
src_image, tar_image = X1[ix], X2[ix]
# generate image from source
gen_image = model.predict(src_image)
# plot all three images
plot_images(src_image, gen_image, tar_image)

#################################################

test_src_img = cv2.imread("F:/test.png", 1)       
test_src_img = cv2.resize(test_src_img, (SIZE_Y, SIZE_X), interpolation = cv2.INTER_NEAREST)
plt.imshow(test_src_img)
test_src_img = (test_src_img - 127.5) / 127.5
test_src_img = np.expand_dims(test_src_img, axis=0)

# generate image from source 
gen_test_image = model.predict(test_src_img)
cv2.imwrite("ss.png", gen_test_image[0])
gen_test_image = (gen_test_image[0] +1) /2
#pyplot.imshow(test_src_img[0, :,:,0], cmap='gray')
plt.imshow(gen_test_image)

#################################################

test_data = [x_test, y_test]
test_data = preprocess_data(test_data)

[x_test1, y_test1] = test_data

indx = randint(0, len(x_test1),1)
src_image, tar_image = x_test1[indx], y_test1[indx]

# generate image from source

gen_test_image = model.predict(src_image)

gen_test_image = (gen_test_image[0] + 1) / 2
src_image = (src_image[0]+1)/2
tar_image = (tar_image[0]+1)/2

# src_image = cv2.cvtColor(tar_image, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(12,12))
plt.subplot(131)
plt.imshow(src_image )
plt.title("source image")

plt.subplot(132)
plt.imshow(tar_image )
plt.title("target image")

plt.subplot(133)
plt.imshow(gen_test_image)
plt.title("generated image")
plt.show()

