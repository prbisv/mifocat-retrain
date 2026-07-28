# -*- coding: utf-8 -*-
"""
Created on Thu May 16 20:46:28 2024

@author: ramad
"""

import nrrd
import numpy as np
from PIL import Image
import os
import glob


direktori_path = "D:/Intelligent Multimedia Network/Research/Riset Bu Dini/Dataset/miccai2012/miccai2012/human_dataset/"
direktori_simpan = "D:/Intelligent Multimedia Network/Research/Riset Bu Dini/Dataset/miccai2012/miccai2012/human_dataset/data test 2d/"

def membuat_direktori_2d(id_pasien, path_direktori):
    
    direktori_pasien = os.path.join(path_direktori, f"Pasien {id_pasien}")
    direktori_citra = os.path.join(direktori_pasien, "images")
    direktori_gt = os.path.join(direktori_pasien, "groundtruth myocardium")
    
    #membuat direktori jika belum ada 
    os.makedirs(direktori_pasien, exist_ok=True)
    os.makedirs(direktori_citra, exist_ok=True)
    os.makedirs(direktori_gt, exist_ok=True)
    
    return direktori_pasien, direktori_citra, direktori_gt


def ubah_nrrd_ke_png(list_img_miccai12, list_mask_miccai12, axis=2):
    
    print("Proses Slice 3D nrrd menjadi 2D citra png")
    print("--------------------------------------------------------------------------\n")
    
    for index, (img_path, mask_path) in enumerate(zip(list_img_miccai12, list_mask_miccai12)):
        
        nomor_pasien_img = os.path.basename(img_path).split('_')[0][1:]
        nomor_pasien_mask = os.path.basename(mask_path).split('_')[0][1:]
        
        print(f"Memproses img pasien nomor = {nomor_pasien_img}")
        print(f"Memproses mask pasien nomor = {nomor_pasien_mask}")
        
        direktori_pasien, direktori_citra, direktori_gt = membuat_direktori_2d(nomor_pasien_img, direktori_simpan)
        
        data_img, header_img = nrrd.read(img_path)
        data_mask, header_mask = nrrd.read(mask_path)
    
        num_slices = data_img.shape[axis]
    
        print(f"Total slice = {num_slices}")
        
        for i in range(num_slices):
            
            if axis == 0:
                slice_img = data_img[i, :, :]
                slice_mask = data_mask[i, :, :]
            elif axis == 1:
                slice_img = data_img[:, i, :]
                slice_mask = data_mask[:, i, :]
            else:
                slice_img = data_img[:, :, i]
                slice_mask = data_mask[:, :, i]
            
            min_val_img = np.min(slice_img)
            max_val_img = np.max(slice_img)
        
            if min_val_img != max_val_img:
                slice_img = (slice_img - min_val_img) / (max_val_img - min_val_img) * 255
            else:
                slice_img = np.zeros_like(slice_img)
            
            slice_img = slice_img.astype(np.uint8)
    
            img = Image.fromarray(slice_img)
    
            img.save(os.path.join(direktori_citra, f'p{nomor_pasien_img}_{i+1}.png'))
            
            
            
            min_val_mask = np.min(slice_mask)
            max_val_mask = np.max(slice_mask)
        
            if min_val_mask != max_val_mask:
                slice_mask = (slice_mask - min_val_mask) / (max_val_mask - min_val_mask) * 255
            else:
                slice_mask = np.zeros_like(slice_mask)
            
            slice_mask = slice_mask.astype(np.uint8)
    
            mask = Image.fromarray(slice_mask)
    
            mask.save(os.path.join(direktori_gt, f'p{nomor_pasien_mask}_mask_{i+1}.png'))
            
        print(f"Pasien {nomor_pasien_img} sudah diproses... \n")
    
    print("Proses Selesai...")
    
def main():
    
    list_img_miccai12 = sorted(glob.glob(direktori_path + 'p*_de.nrrd'))
    list_mask_miccai12 = sorted(glob.glob(direktori_path + 'p*_myo.nrrd'))
    
    ubah_nrrd_ke_png(list_img_miccai12, list_mask_miccai12, axis=2)
    
if __name__ == "__main__":
    main()