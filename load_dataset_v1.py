# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 07:29:18 2024

@author: ramad
"""

import numpy as np
import nibabel as nib
import glob
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from tifffile import imsave
from scipy.ndimage import zoom

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()


def resize_volume(data, new_shape=(128, 128, 128)):
    """
    Mengubah ukuran volume data dengan interpolasi.

    Parameters:
    - data: numpy.ndarray, array 3D yang akan diubah ukurannya.
    - new_shape: tuple, ukuran baru untuk output array.

    Returns:
    - resized_data: numpy.ndarray, array 3D dengan ukuran baru.
    """
    # Menghitung faktor skala untuk setiap dimensi
    scale_factors = [n / float(o) for n, o in zip(new_shape, data.shape)]
    
    # Menggunakan fungsi zoom dari scipy untuk mengubah ukuran data
    # mode='nearest' untuk menghindari artifacts di luar batas data asli
    resized_data = zoom(data, zoom=scale_factors, mode='nearest')
    
    return resized_data


TRAIN_DATASET_PATH = 'D:/Intelligent Multimedia Network/Research/Riset Bu Dini/Dataset/acdc17/acdc17/trainingACDC17/'

test_image_diastole=nib.load(TRAIN_DATASET_PATH + 'patient002/patient002_frame01.nii.gz').get_fdata()
print(test_image_diastole.max())

test_image_diastole=scaler.fit_transform(test_image_diastole.reshape(-1, test_image_diastole.shape[-1])).reshape(test_image_diastole.shape)

test_mask_diastole=nib.load(TRAIN_DATASET_PATH + 'patient002/patient002_frame01_gt.nii.gz').get_fdata()
test_mask_diastole=test_mask_diastole.astype(np.uint8)

print("Nilai unik mask diastole = ", np.unique(test_mask_diastole))


test_image_sistole=nib.load(TRAIN_DATASET_PATH + 'patient002/patient002_frame12.nii.gz').get_fdata()
print(test_image_sistole.max())

test_image_sistole=scaler.fit_transform(test_image_sistole.reshape(-1, test_image_sistole.shape[-1])).reshape(test_image_sistole.shape)

test_mask_sistole=nib.load(TRAIN_DATASET_PATH + 'patient002/patient002_frame12_gt.nii.gz').get_fdata()
test_mask_sistole=test_mask_sistole.astype(np.uint8)

print("Nilai unik mask sistole = ",np.unique(test_mask_sistole))


import random
# n_slice=random.randint(0, test_mask_diastole.shape[2])

n_slice = 2

plt.figure(figsize=(12, 8))

plt.subplot(231)
plt.imshow(test_image_diastole[:,:,n_slice], cmap='gray')
plt.title('Image Diastole')
plt.subplot(232)
plt.imshow(test_mask_diastole[:,:,n_slice])
plt.title('Mask Diastole')
plt.subplot(233)
plt.imshow(test_image_sistole[:,:,n_slice])
plt.title('Image Sistole')
plt.subplot(234)
plt.imshow(test_mask_sistole[:,:,n_slice])
plt.title('Mask Sistole')
plt.show()

resized_image_diastole = resize_volume(test_image_diastole)
resized_mask_diastole = resize_volume(test_mask_diastole)
resized_image_sistole = resize_volume(test_image_sistole)
resized_mask_sistole = resize_volume(test_mask_sistole)

resized_diastole_4d = np.expand_dims(resized_image_diastole, axis=3)
resized_sistole_4d = np.expand_dims(resized_image_sistole, axis=3)


import random
n_slice=random.randint(0, resized_mask_diastole.shape[2])

# n_slice = 2

plt.figure(figsize=(12, 8))

plt.subplot(231)
plt.imshow(resized_diastole_4d[:,:,n_slice, 0], cmap='gray')
plt.title('Image Diastole Resized')
plt.subplot(232)
plt.imshow(resized_mask_diastole[:,:,n_slice])
plt.title('Mask Diastole Resized')
plt.subplot(233)
plt.imshow(resized_sistole_4d[:,:,n_slice,0], cmap='gray')
plt.title('Image Sistole Resized')
plt.subplot(234)
plt.imshow(resized_mask_sistole[:,:,n_slice])
plt.title('Mask Sistole Resized')
plt.show()


TEST_DATASET_PATH = 'D:/Intelligent Multimedia Network/Research/Riset Bu Dini/Dataset/acdc17/acdc17/testingACDC17_baru/'

list_img_ed = sorted(glob.glob(TEST_DATASET_PATH + '*/patient*_frame01.nii.gz'))
list_mask_ed = sorted(glob.glob(TEST_DATASET_PATH + '*/patient*_frame01_gt.nii.gz'))


for img in range(len(list_img_ed)):
    
    print("Citra dan mask ED nomor = ", img+1)
    
    #load citra
    temp_img_ed = nib.load(list_img_ed[img]).get_fdata()
    temp_img_ed = scaler.fit_transform(temp_img_ed.reshape(-1, temp_img_ed.shape[-1])).reshape(temp_img_ed.shape)
    
    # Memanggil citra mask
    temp_mask_ed = nib.load(list_mask_ed[img]).get_fdata().astype(np.uint8)
    print(np.unique(temp_mask_ed))
    
    resized_img_ed = resize_volume(temp_img_ed)
    resized_mask_ed = resize_volume(temp_mask_ed)
    
    resized_img_ed_1 = np.expand_dims(resized_img_ed, axis=3)
    
    val, counts = np.unique(resized_mask_ed, return_counts=True)
    
    if (1 - (counts[0]/counts.sum())) > 0.01:
        print(f"Berhasil menyimpan citra ED {img} menjadi 3d numpy (.npy)")
        
        print("Nilai maksimum dalam mask: ", np.max(resized_mask_ed))
        
        resized_mask_ed = to_categorical(resized_mask_ed, num_classes=5)
        
        # resized_temp_mask = np.expand_dims(resized_temp_mask, axis=3)
        
        # Save the processed data
        np.save('D:/Intelligent Multimedia Network/Research/Riset Bu Dini/Dataset/acdc17/acdc17/Data 3D 128 Testing/ED/images/image_'+str(img+101)+'_ed.npy', resized_img_ed_1)
        np.save('D:/Intelligent Multimedia Network/Research/Riset Bu Dini/Dataset/acdc17/acdc17/Data 3D 128 Testing/ED/masks/mask_'+str(img+101)+'_ed.npy', resized_mask_ed)
    else:
        print("Gagal")
        
print("/n")
print("Proses Generate Citra ED Sudah Selesai")
print("/n")



# list_img_es = sorted(glob.glob(TRAIN_DATASET_PATH + '*/patient*_frame*.nii.gz'))
# list_mask_es = sorted(glob.glob(TRAIN_DATASET_PATH + '*/patient*_frame*_gt.nii.gz'))

# list_img_es_1 = [img for img in list_img_es if "frame01" not in img]
# list_mask_es_1 = [mask for mask in list_mask_es if "frame01" not in mask]

# for img in range(len(list_img_es_1)):
    
#     print("Citra dan mask ES nomor = ", img+1)
    
#     #load citra
#     temp_img_es = nib.load(list_img_es_1[img]).get_fdata()
#     temp_img_es = scaler.fit_transform(temp_img_es.reshape(-1, temp_img_es.shape[-1])).reshape(temp_img_es.shape)
    
#     # Memanggil citra mask
#     temp_mask_es = nib.load(list_mask_es_1[img]).get_fdata().astype(np.uint8)
#     print(np.unique(temp_mask_es))
    
#     resized_img_es = resize_volume(temp_img_es)
#     resized_mask_es = resize_volume(temp_mask_es)
    
#     resized_img_es_1 = np.expand_dims(resized_img_es, axis=3)
    
#     val, counts = np.unique(resized_mask_es, return_counts=True)
    
#     if (1 - (counts[0]/counts.sum())) > 0.01:
#         print(f"Berhasil menyimpan citra ES {img} menjadi 3d numpy (.npy)")
        
#         print("Nilai maksimum dalam mask: ", np.max(resized_mask_es))
        
#         resized_mask_es = to_categorical(resized_mask_es, num_classes=5)
        
#         # resized_temp_mask = np.expand_dims(resized_temp_mask, axis=3)
        
#         # Save the processed data
#         np.save('D:/Intelligent Multimedia Network/Research/Riset Bu Dini/Dataset/acdc17/acdc17/Data 3D 128/ES/images/image_'+str(img+1)+'_es.npy', resized_img_es_1)
#         np.save('D:/Intelligent Multimedia Network/Research/Riset Bu Dini/Dataset/acdc17/acdc17/Data 3D 128/ES/masks/mask_'+str(img+1)+'_es.npy', resized_mask_es)
#     else:
#         print("Gagal")
        
# print("/n")
# print("Proses Generate Citra ES Sudah Selesai")
# print("/n")












