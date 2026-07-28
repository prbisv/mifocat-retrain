# -*- coding: utf-8 -*-
"""
Created on Fri May 17 17:59:00 2024

@author: ramad
"""

import nrrd
import numpy as np
from PIL import Image
import os
import glob


direktori_path = "D:/Intelligent Multimedia Network/Research/Riset Bu Dini/Dataset/miccai2012/miccai2012/human_dataset/training/"
direktori_simpan = "D:/Intelligent Multimedia Network/Research/Riset Bu Dini/Dataset/miccai2012/miccai2012/human_dataset/data train 2d/"

def membuat_direktori_2d(id_pasien, path_direktori):
    
    direktori_pasien = os.path.join(path_direktori, f"Pasien {id_pasien}")
    direktori_citra = os.path.join(direktori_pasien, "images")
    direktori_gt_myo = os.path.join(direktori_pasien, "groundtruth myocardium")
    direktori_gt_infarct = os.path.join(direktori_pasien, "groundtruth infarct")
    
    #membuat direktori jika belum ada 
    os.makedirs(direktori_pasien, exist_ok=True)
    os.makedirs(direktori_citra, exist_ok=True)
    os.makedirs(direktori_gt_myo, exist_ok=True)
    os.makedirs(direktori_gt_infarct, exist_ok=True)
    
    return direktori_pasien, direktori_citra, direktori_gt_myo, direktori_gt_infarct


def ubah_nrrd_ke_png(list_img_miccai12, list_myo_miccai12, list_infarct_miccai12, axis=2):
    
    print("Proses Slice 3D nrrd menjadi 2D citra png")
    print("--------------------------------------------------------------------------\n")
    
    for index, (img_path, myo_path, infarct_path) in enumerate(zip(list_img_miccai12, list_myo_miccai12, list_infarct_miccai12)):
        
        nomor_pasien_img = os.path.basename(img_path).split('_')[0][1:]
        nomor_pasien_myo = os.path.basename(myo_path).split('_')[0][1:]
        nomor_pasien_infarct = os.path.basename(infarct_path).split('_')[0][1:]
        
        print(f"Memproses img pasien nomor = {nomor_pasien_img}")
        print(f"Memproses mask myo pasien nomor = {nomor_pasien_myo}")
        print(f"Memproses mask infarct pasien nomor = {nomor_pasien_infarct}")
        
        direktori_pasien, direktori_citra, direktori_gt_myo, direktori_gt_infarct = membuat_direktori_2d(nomor_pasien_img, direktori_simpan)
        
        data_img, header_img = nrrd.read(img_path)
        data_myo, header_myo = nrrd.read(myo_path)
        data_infarct, header_infarct = nrrd.read(infarct_path)
    
        num_slices = data_img.shape[axis]
    
        print(f"Total slice = {num_slices}")
        
        for i in range(num_slices):
            
            if axis == 0:
                slice_img = data_img[i, :, :]
                slice_myo = data_myo[i, :, :]
                slice_infarct = data_infarct[i, :, :]
            elif axis == 1:
                slice_img = data_img[:, i, :]
                slice_myo = data_myo[:, i, :]
                slice_infarct = data_infarct[:, i, :]
            else:
                slice_img = data_img[:, :, i]
                slice_myo = data_myo[:, :, i]
                slice_infarct = data_infarct[:, :, i]
            
            min_val_img = np.min(slice_img)
            max_val_img = np.max(slice_img)
        
            if min_val_img != max_val_img:
                slice_img = (slice_img - min_val_img) / (max_val_img - min_val_img) * 255
            else:
                slice_img = np.zeros_like(slice_img)
            
            slice_img = slice_img.astype(np.uint8)
    
            img = Image.fromarray(slice_img)
    
            img.save(os.path.join(direktori_citra, f'p{nomor_pasien_img}_{i+1}.png'))
            
            
            
            min_val_myo = np.min(slice_myo)
            max_val_myo = np.max(slice_myo)
        
            if min_val_myo != max_val_myo:
                slice_myo = (slice_myo - min_val_myo) / (max_val_myo - min_val_myo) * 255
            else:
                slice_myo = np.zeros_like(slice_myo)
            
            slice_myo = slice_myo.astype(np.uint8)
    
            mask_myo = Image.fromarray(slice_myo)
    
            mask_myo.save(os.path.join(direktori_gt_myo, f'p{nomor_pasien_myo}_mask_{i+1}_myo.png'))
            
            
            min_val_infarct = np.min(slice_infarct)
            max_val_infarct = np.max(slice_infarct)
        
            if min_val_infarct != max_val_infarct:
                slice_infarct = (slice_infarct - min_val_infarct) / (max_val_infarct - min_val_infarct) * 255
            else:
                slice_infarct = np.zeros_like(slice_infarct)
            
            slice_infarct = slice_infarct.astype(np.uint8)
    
            mask_infarct = Image.fromarray(slice_infarct)
    
            mask_infarct.save(os.path.join(direktori_gt_infarct, f'p{nomor_pasien_infarct}_mask_{i+1}_infarct.png'))
            
        print(f"Pasien {nomor_pasien_img} sudah diproses... \n")
    
    print("Proses Selesai...")
    
def main():
    
    list_img_miccai12 = sorted(glob.glob(direktori_path + 'p*_de.nrrd'))
    list_myo_miccai12 = sorted(glob.glob(direktori_path + 'p*_myo.nrrd'))
    list_infarct_miccai12 = sorted(glob.glob(direktori_path + 'p*_infarct.nrrd'))
    
    ubah_nrrd_ke_png(list_img_miccai12, list_myo_miccai12, list_infarct_miccai12, axis=2)
    
if __name__ == "__main__":
    main()