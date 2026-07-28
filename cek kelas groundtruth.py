# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 14:42:20 2024

@author: ramad
"""

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

mask_path = "D:/Intelligent Multimedia Network/Research/Riset Bu Dini/Dataset/acdc17/acdc17/Data 2D/ED/Data Per Pasien Testing 2D/Pasien 101/groundtruth/Pasien101_2_ed_gt.png"

mask = cv.imread(mask_path, 0)


# Verifikasi nilai unik pada mask
unique_values = np.unique(mask)
print("Unique values in the mask:", unique_values)

# Membuat kelas berdasarkan nilai unik yang diidentifikasi
kelas1 = np.where(mask == unique_values[1], 255, 0).astype(np.uint8) if len(unique_values) > 1 else np.zeros_like(mask)
kelas2 = np.where(mask == unique_values[2], 255, 0).astype(np.uint8) if len(unique_values) > 2 else np.zeros_like(mask)
kelas3 = np.where(mask == unique_values[3], 255, 0).astype(np.uint8) if len(unique_values) > 3 else np.zeros_like(mask)

cv.imwrite("D:/Intelligent Multimedia Network/Research/Riset Bu Dini/Dataset/acdc17/acdc17/Data 2D/ED/Data Per Pasien Testing 2D/Pasien 101/gt_rv_101_2.png", kelas1)
cv.imwrite("D:/Intelligent Multimedia Network/Research/Riset Bu Dini/Dataset/acdc17/acdc17/Data 2D/ED/Data Per Pasien Testing 2D/Pasien 101/gt_myo_101_2.png", kelas2)
cv.imwrite("D:/Intelligent Multimedia Network/Research/Riset Bu Dini/Dataset/acdc17/acdc17/Data 2D/ED/Data Per Pasien Testing 2D/Pasien 101/gt_lv_101_2.png", kelas3)

plt.figure(figsize=(12, 8))
# plt.subplot(231)
# plt.title('Citra Asli')
# plt.imshow(mask, cmap='gray')  # Menampilkan citra dalam grayscale

plt.subplot(231)
plt.title('Groundtruth')
plt.imshow(mask, cmap='gray')

plt.subplot(232)
plt.title('RV')
plt.imshow(kelas1, cmap='gray')

plt.subplot(233)
plt.title('Myo')
plt.imshow(kelas2, cmap='gray')

plt.subplot(234)
plt.title('LV')
plt.imshow(kelas3, cmap='gray')

plt.show()
