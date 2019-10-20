#!/usr/bin/env python
# coding: utf-8

# In[1]:

# new feature selection for MNIST dataset
# labels (index) as before (no change), see notebook 'data_mnist'

# version data_mnist_comp: max features (150 x 3 = 450)
# the version was extended and used to create data with max features (200 x 3 = 600)


# In[ ]:

import gzip
import numpy as np
import matplotlib.pyplot as plt
import copy
from scipy import ndimage, misc

threshold = 180
num_angles = 230


# In[2]:

# produce a raster (random)
# random seed: inserted only later
np.random.seed(30)
raster = np.zeros((num_angles, 5))
raster[:, 0] = np.random.randint(0, 360, num_angles)
raster[:, 1] = np.random.randint(0, 27, num_angles) # choose a row
raster[:, 2] = np.random.randint(0, 27, num_angles)
raster[:, 3] = np.random.randint(0, 27, num_angles)
raster[:, 4] = np.random.randint(0, 18, num_angles) # initial position (column) for cutting out samples of length 10, between 0 and 18


# In[5]:

# READ AND GET FEATURES TRAINING DATA

f = gzip.open('train-images-idx3-ubyte.gz','r')
num_images = 60000 #number of images to read out
image_size = 28 #image size

f.read(16) #related to position of image
buf = f.read(image_size * image_size * num_images)
data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
data = data.reshape(num_images, image_size, image_size, 1)

res = np.zeros((num_images, num_angles * 3, 10))
res_2 = np.zeros((num_images, num_angles * 3))
res_3 = np.zeros((num_images, num_angles * 3))

for z in range(num_images):

    image_binary = np.zeros((image_size, image_size))
    image_binary_turned = np.zeros((image_size, image_size))

    store = np.empty((num_angles * 3, 10))

    image = np.asarray(data[z]).squeeze() #python array with 28 x 28 pixel values 
    for i, angle in enumerate(raster[:, 0]):
        image_turned = ndimage.rotate(image, angle, reshape=False)
        for a in range(image_size):
            image_binary_turned[a , :] = [0 if i < threshold else 1 for i in image_turned[a,:]]

        event_rows = np.zeros((3, 10)) # 1 times 10 bins long
        for c, start in enumerate(raster[i, 1:4]):
        #start = raster[i, 1]
            for b in range(10):
                if (image_binary_turned[int(start), (b + int(raster[i, 4]))] < image_binary_turned[int(start), (b + 1 + int(raster[i, 4]))]) and (np.size(np.nonzero(event_rows[c, :])) == 0):
                    event_rows[c, b] = 1
        if i == 0:
            store = event_rows
        if i > 0:
            store = np.concatenate((store, event_rows), axis = 0)
            
    res[z, :, :] = store
    events = np.nonzero(store)
    for d in range(np.shape(events)[1]):
        res_2[z, events[0][d]] = events[1][d]
        res_3[z, events[0][d]] = 1

np.save('spikes_all_.txt', res)
np.save('spike_times_all_.txt', res_2)
np.save('spike_weights_all_.txt', res_3)


# In[6]:

# READ AND GET FEATURES TEST DATA

f = gzip.open('t10k-images-idx3-ubyte.gz','r')

image_size = 28 #image size
num_images = 10000 #number of images to read out

f.read(16) #related to position of image
buf = f.read(image_size * image_size * num_images)
data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
data = data.reshape(num_images, image_size, image_size, 1)

res = np.zeros((num_images, num_angles * 3, 10))
res_2 = np.zeros((num_images, num_angles * 3))
res_3 = np.zeros((num_images, num_angles * 3))

for z in range(num_images):

    image_binary = np.zeros((image_size, image_size))
    image_binary_turned = np.zeros((image_size, image_size))

    store = np.empty((num_angles * 3, 10))

    image = np.asarray(data[z]).squeeze() #python array with 28 x 28 pixel values 
    for i, angle in enumerate(raster[:, 0]):
        image_turned = ndimage.rotate(image, angle, reshape=False)
        for a in range(image_size):
            image_binary_turned[a , :] = [0 if i < threshold else 1 for i in image_turned[a,:]]

        event_rows = np.zeros((3, 10)) # 1 times 10 bins long
        for c, start in enumerate(raster[i, 1:4]):
        #start = raster[i, 1]
            for b in range(10):
                if (image_binary_turned[int(start), (b + int(raster[i, 4]))] < image_binary_turned[int(start), (b + 1 + int(raster[i, 4]))]) and (np.size(np.nonzero(event_rows[c, :])) == 0):
                    event_rows[c, b] = 1
        if i == 0:
            store = event_rows
        if i > 0:
            store = np.concatenate((store, event_rows), axis = 0)
            
    res[z, :, :] = store
    events = np.nonzero(store)
    for d in range(np.shape(events)[1]):
        res_2[z, events[0][d]] = events[1][d]
        res_3[z, events[0][d]] = 1

np.save('spikes_all_test_.txt', res)
np.save('spike_times_all_test_.txt', res_2)
np.save('spike_weights_all_test_.txt', res_3)


# In[ ]:



