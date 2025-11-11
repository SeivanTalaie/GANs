########################### Libraries ###########################

import os
import glob
import cv2
import numpy as np
from datetime import datetime 
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from pix2pix_model import define_discriminator, define_generator, define_gan, train

######################## Create Datasets ########################

SIZE_X = 256
SIZE_Y = 256

tar_images = []
tar_image_list = glob.glob("F:/Python/Datasets/FoodSeg103/Images/img_dir/train/*.jpg")

for path in tar_image_list[:350]:
    img = cv2.imread(path, 1)     
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (SIZE_Y, SIZE_X), interpolation = cv2.INTER_NEAREST)
    tar_images.append(img)
          
tar_images = np.array(tar_images)

src_images = [] 
src_images_list = glob.glob("F:/Python/Datasets/FoodSeg103/Images/ann_dir/train/*.png")

for path in src_images_list[:350]:
    mask = cv2.imread(path, 1)   
    # mask = cv2.cvtColor(mask, cv.COLOR_BGR2RGB)    
    mask = cv2.resize(mask, (SIZE_Y, SIZE_X), interpolation = cv2.INTER_NEAREST)
    src_images.append(mask)
               
src_images = np.array(src_images)

# print(np.unique(src_images))

########################## Sanity Check #########################

n_samples = 3

for i in range(n_samples):
	plt.subplot(2, n_samples, 1 + i)
	plt.axis('off')
	plt.imshow(src_images[i])

for i in range(n_samples):
	plt.subplot(2, n_samples, 1 + n_samples + i)
	plt.axis('off')
	plt.imshow(tar_images[i])
plt.show()

########################### Split Data ########################## 

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

########################## Define Model #########################

image_shape = src_images.shape[1:]

d_model = define_discriminator(image_shape)
g_model = define_generator(image_shape)

gan_model = define_gan(g_model, d_model, image_shape)

data_train = [x_train, y_train]

data_test = [x_test, y_test]

######################### Preprocessing #########################

def preprocess_data(data):
	X1, X2 = data[0], data[1]

	X1 = (X1 - 127.5) / 127.5
	X2 = (X2 - 127.5) / 127.5
	return [X1, X2]

dataset = preprocess_data(data_train)
data_test = preprocess_data(data_test)

[x_test1, y_test1] = data_test

########################## Train Model ##########################

start1 = datetime.now() 

train(d_model, g_model, gan_model, dataset, x_test1, y_test1, n_epochs=1000, n_batch=1) 

stop1 = datetime.now()

execution_time = stop1-start1
print("Execution time is: ", execution_time)

g_model.save('food_generator.h5')

##################### Test Model Performance ####################

from keras.models import load_model
from numpy.random import randint
from numpy import vstack

model = load_model('F:/Python/GAN/cloth_pix2pix/model_003800.h5',
                   compile=False)

def plot_images(src_img, gen_img, tar_img):
	images = vstack((src_img, gen_img, tar_img))

	images = (images + 1) / 2.0
	titles = ['Input-segm-img', 'Output-Generated', 'target_img']

	for i in range(len(images)):
		plt.subplot(1, 3, 1 + i)
		plt.axis('off')
		plt.imshow(images[i,:,:,0], cmap='gray')
		plt.title(titles[i])
	plt.show()


[X1, X2] = dataset

ix = randint(0, len(X1), 1)
src_image, tar_image = X1[ix], X2[ix]
gen_image = model.predict(src_image)

plot_images(src_image, gen_image, tar_image)



test_data = [x_test, y_test]
test_data = preprocess_data(test_data)

[x_test1, y_test1] = test_data

indx = randint(0, len(x_test1),1)
src_image, tar_image = x_test1[indx], y_test1[indx]
gen_test_image = model.predict(src_image)

gen_test_image = (gen_test_image[0] + 1) / 2
src_image = (src_image[0]+1)/2
tar_image = (tar_image[0]+1)/2

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



