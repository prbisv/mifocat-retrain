# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 13:17:23 2024

@author: ramad
"""

import cv2 as cv
import numpy as np
import os

direktori_gt = "D:/Intelligent Multimedia Network/Research/Riset Bu Dini/Dataset/acdc17/acdc17/Data 2D/ED/Data Train Resize 128 ED/groundtruth/"

simpan_gt_myo = "D:/Intelligent Multimedia Network/Research/Riset Bu Dini/Dataset/acdc17/acdc17/Data 2D/ED/Data Train Resize 128 ED/gt_myo/"


def get_mask_modification_time(item):
    item_path = os.path.join(direktori_gt, item)
    return os.path.getmtime(item_path)

def pisah_kelas_myo(sorted_masks):
    
    for filenames in sorted_masks:
        if filenames.endswith('.png'):
            file_path = os.path.join(direktori_gt, filenames)
            print(file_path)
        
            mask = cv.imread(file_path, 0)
            
            # Verifikasi nilai unik pada mask
            unique_values = np.unique(mask)
            print("Unique values in the mask:", unique_values)
            
            
             # Menentukan mask berdasarkan jumlah nilai unik
            if len(unique_values) == 3:
                # Jika ada 3 nilai unik, simpan citra untuk kelas 1
                kelas_target = np.where(mask == unique_values[1], 255, 0).astype(np.uint8)
            elif len(unique_values) == 4:
                # Jika ada 4 nilai unik, simpan citra untuk kelas 2
                kelas_target = np.where(mask == unique_values[2], 255, 0).astype(np.uint8)
            else:
                # Jika tidak memenuhi kondisi di atas, gunakan mask nol
                kelas_target = np.zeros_like(mask)
            
            
            
            # kelas2 = np.where(mask == unique_values[2], 255, 0).astype(np.uint8) if len(unique_values) > 2 else np.zeros_like(mask)
            
            base = os.path.splitext(filenames)[0]
            name_file_mask = os.path.join(simpan_gt_myo, str(base) + ".png")
            
            #simpan citra
            cv.imwrite(name_file_mask, kelas_target)
            
            print(f"Citra {filenames} sudah diproses\n")
            
    print("Proses Resize Citra mask Myocardium Sudah Selesai \n")
    
    

def main():
    
    get_masks = os.listdir(direktori_gt)
    sorted_masks = sorted(get_masks, key=get_mask_modification_time)
    
    pisah_kelas_myo(sorted_masks)
    

if __name__ == "__main__":
    main()

