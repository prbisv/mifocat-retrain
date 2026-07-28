# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 09:48:15 2024

@author: ramad
"""

import os
import numpy as np
import keras
from matplotlib import pyplot as plt
import glob
import random
import os
from custom_datagen import imageLoader
import tensorflow as tf


####################################################
train_img_dir = "D:/Intelligent Multimedia Network/Research/Riset Bu Dini/Dataset/acdc17/acdc17/Data 3D 128 - Split/train/images/"
train_mask_dir = "D:/Intelligent Multimedia Network/Research/Riset Bu Dini/Dataset/acdc17/acdc17/Data 3D 128 - Split/train/masks/"

img_list = os.listdir(train_img_dir)
msk_list = os.listdir(train_mask_dir)

num_images = len(os.listdir(train_img_dir))

img_num = random.randint(0,num_images-1)
test_img = np.load(train_img_dir+img_list[img_num])
test_mask = np.load(train_mask_dir+msk_list[img_num])
test_mask = np.argmax(test_mask, axis=3)

n_slice=random.randint(0, test_mask.shape[2])
plt.figure(figsize=(12, 8))

plt.subplot(221)
plt.imshow(test_img[:,:,n_slice, 0], cmap='gray')
plt.title('Image flair')
# plt.subplot(222)
# plt.imshow(test_img[:,:,n_slice, 1], cmap='gray')
# plt.title('Image t1ce')
# plt.subplot(223)
# plt.imshow(test_img[:,:,n_slice, 2], cmap='gray')
# plt.title('Image t2')
plt.subplot(222)
plt.imshow(test_mask[:,:,n_slice])
plt.title('Mask')
plt.show()