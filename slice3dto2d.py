# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 10:12:30 2024

@author: ramad
"""

import cv2 as cv
import numpy as np
import os
import nibabel as nib
import glob
import re

dataset_path = "D:/Intelligent Multimedia Network/Research/Riset Bu Dini/Dataset/acdc17/acdc17/testingACDC17_baru/"

direktori_pasien_ed = "D:/Intelligent Multimedia Network/Research/Riset Bu Dini/Dataset/acdc17/acdc17/Data 2D/ED/Data Per Pasien Testing 2D/"
direktori_pasien_es = "D:/Intelligent Multimedia Network/Research/Riset Bu Dini/Dataset/acdc17/acdc17/Data 2D/ES/Data Per Pasien Testing 2D/"


direktori_resize_ed = "D:/Intelligent Multimedia Network/Research/Riset Bu Dini/Dataset/acdc17/acdc17/Data 2D/ED/"
direktori_resize_es = "D:/Intelligent Multimedia Network/Research/Riset Bu Dini/Dataset/acdc17/acdc17/Data 2D/ES/"


def membuat_direktori_2d_ed(id_pasien, path_direktori):
    
    direktori_pasien = os.path.join(path_direktori, f"Pasien {id_pasien}")
    direktori_citra = os.path.join(direktori_pasien, "images")
    direktori_gt = os.path.join(direktori_pasien, "groundtruth")
    
    #membuat direktori jika belum ada 
    os.makedirs(direktori_pasien, exist_ok=True)
    os.makedirs(direktori_citra, exist_ok=True)
    os.makedirs(direktori_gt, exist_ok=True)
    
    return direktori_pasien, direktori_citra, direktori_gt

def membuat_direktori_2d_ed_testing(id_pasien, path_direktori):
    
    direktori_pasien = os.path.join(path_direktori, f"Pasien {id_pasien}")
    direktori_citra = os.path.join(direktori_pasien, "images")
    direktori_gt = os.path.join(direktori_pasien, "groundtruth")
    direktori_rv = os.path.join(direktori_pasien, "groundtruth right ventricel")
    direktori_myo = os.path.join(direktori_pasien, "groundtruth myocardium")
    direktori_lv = os.path.join(direktori_pasien, "groundtruth left ventricel")
    
    
    #membuat direktori jika belum ada 
    os.makedirs(direktori_pasien, exist_ok=True)
    os.makedirs(direktori_citra, exist_ok=True)
    os.makedirs(direktori_gt, exist_ok=True)
    os.makedirs(direktori_rv, exist_ok=True)
    os.makedirs(direktori_myo, exist_ok=True)
    os.makedirs(direktori_lv, exist_ok=True)
    
    return direktori_pasien, direktori_citra, direktori_gt, direktori_rv, direktori_myo, direktori_lv


def membuat_direktori_2d_es(id_pasien, path_direktori):
    
    direktori_pasien = os.path.join(path_direktori, f"Pasien {id_pasien}")
    direktori_citra = os.path.join(direktori_pasien, "images")
    direktori_gt = os.path.join(direktori_pasien, "groundtruth")
    
    #membuat direktori jika belum ada 
    os.makedirs(direktori_pasien, exist_ok=True)
    os.makedirs(direktori_citra, exist_ok=True)
    os.makedirs(direktori_gt, exist_ok=True)
    
    return direktori_pasien, direktori_citra, direktori_gt

def slice_ed(list_img_ed, list_mask_ed):
    
    print("Proses Slice 3D menjadi 2D citra ED")
    print("--------------------------------------------------------------------------\n")
    
    for index, (img_path, mask_path) in enumerate(zip(list_img_ed, list_mask_ed), start=101):
        
        
        print(f"Citra ED nomor = {index:03}")
        print(f"GT ED nomor = {index:03}")
        print("\n")
        
        
        direktori_pasien, direktori_citra, direktori_gt = membuat_direktori_2d_ed(index, direktori_pasien_ed)
        
        imgs = nib.load(img_path).get_fdata()
        masks = nib.load(mask_path).get_fdata()
        
        num_slices = imgs.shape[2]
        
        print("Shape of imgs:", imgs.shape)
        print("Shape of masks:", masks.shape)
        
        for slice_index in range(num_slices):
            
            print(f"Nomor indeks = {slice_index}")
            
            img_slice = imgs[:, :, slice_index].astype('float32')
            mask_slice = masks[:, :, slice_index].astype('float32')
            
            # Normalisasi data ke rentang 0-255 dan konversi ke uint8
            img_slice = cv.normalize(img_slice, None, 0, 255, cv.NORM_MINMAX)
            mask_slice = cv.normalize(mask_slice, None, 0, 255, cv.NORM_MINMAX)

            img_slice = img_slice.astype('uint8')
            mask_slice = mask_slice.astype('uint8')
            
            
            simpan_slice_citra = os.path.join(direktori_citra, f"Pasien{index}_{slice_index+1}_ed" + ".png")
            simpan_slice_gt = os.path.join(direktori_gt, f"Pasien{index}_{slice_index+1}_ed_gt" + ".png")
            
            cv.imwrite(simpan_slice_citra, img_slice)
            cv.imwrite(simpan_slice_gt, mask_slice)
            
        
        print("\n")
        print(f"Proses Penyimpanan Citra Pasien {index:03}")
        print(f"Proses Penyimpanan Groundtruth Pasien {index:03}")
        print("\n")
    
    print("--------------------------------------------------------------------------")
    print("Proses Slice 3D menjadi 2D citra ED sudah selesai\n")
    
def slice_es(list_img_es, list_mask_es):
    
    print("Proses Slice 3D menjadi 2D citra ES")
    print("--------------------------------------------------------------------------\n")
    
    for index, (img_path, mask_path) in enumerate(zip(list_img_es, list_mask_es), start=1):
        
        
        print(f"Citra ES nomor = {index:03}")
        print(f"GT ES nomor = {index:03}")
        print("\n")
        
        
        direktori_pasien, direktori_citra, direktori_gt = membuat_direktori_2d_es(index, direktori_pasien_es)
        
        imgs = nib.load(img_path).get_fdata()
        masks = nib.load(mask_path).get_fdata()
        
        num_slices_img = imgs.shape[2]
        num_slices_mask = masks.shape[2]
        
        print("Shape of imgs:", imgs.shape)
        print("Shape of masks:", masks.shape)
        
        for slice_index in range(num_slices_img):
            
            print(f"Nomor indeks = {slice_index}")
            
            img_slice = imgs[:, :, slice_index].astype('float32')
            # mask_slice = masks[:, :, slice_index].astype('float32')
            
            # Normalisasi data ke rentang 0-255 dan konversi ke uint8
            img_slice = cv.normalize(img_slice, None, 0, 255, cv.NORM_MINMAX)
            # mask_slice = cv.normalize(mask_slice, None, 0, 255, cv.NORM_MINMAX)

            img_slice = img_slice.astype('uint8')
            # mask_slice = mask_slice.astype('uint8')
            
            
            simpan_slice_citra = os.path.join(direktori_citra, f"Pasien{index}_{slice_index+1}_es" + ".png")
            # simpan_slice_gt = os.path.join(direktori_gt, f"Pasien{index}_{slice_index+1}_es_gt" + ".png")
            
            cv.imwrite(simpan_slice_citra, img_slice)
            # cv.imwrite(simpan_slice_gt, mask_slice)
        
        for slice_index in range(num_slices_mask):
            
            print(f"Nomor indeks = {slice_index}")
            
            # img_slice = imgs[:, :, slice_index].astype('float32')
            mask_slice = masks[:, :, slice_index].astype('float32')
            
            # Normalisasi data ke rentang 0-255 dan konversi ke uint8
            # img_slice = cv.normalize(img_slice, None, 0, 255, cv.NORM_MINMAX)
            mask_slice = cv.normalize(mask_slice, None, 0, 255, cv.NORM_MINMAX)

            # img_slice = img_slice.astype('uint8')
            mask_slice = mask_slice.astype('uint8')
            
            
            # simpan_slice_citra = os.path.join(direktori_citra, f"Pasien{index}_{slice_index+1}_es" + ".png")
            simpan_slice_gt = os.path.join(direktori_gt, f"Pasien{index}_{slice_index+1}_es_gt" + ".png")
            
            # cv.imwrite(simpan_slice_citra, img_slice)
            cv.imwrite(simpan_slice_gt, mask_slice)
            
        
        print("\n")
        print(f"Proses Penyimpanan Citra ES Pasien {index:03}")
        print(f"Proses Penyimpanan Groundtruth ES Pasien {index:03}")
        print("\n")
    
    print("--------------------------------------------------------------------------")
    print("Proses Slice 3D menjadi 2D citra ES sudah selesai\n")

def resize_img_ed(sorted_imgs, sorted_masks):
    
    print("Proses Resize Citra 2D citra ED")
    print("--------------------------------------------------------------------------\n")
    
    # #untuk data train
    direktori_resize = os.path.join(direktori_resize_ed, "Data Train Resize 128 ED")
    direktori_img = os.path.join(direktori_resize, "images")
    direktori_gt = os.path.join(direktori_resize, "groundtruth")
    
    # #membuat direktori jika belum ada 
    os.makedirs(direktori_resize, exist_ok=True)
    os.makedirs(direktori_img, exist_ok=True)
    os.makedirs(direktori_gt, exist_ok=True)
    
    for img_path, mask_path in zip(sorted_imgs, sorted_masks):
        
        img = cv.imread(img_path, 0)
        mask = cv.imread(mask_path, 0)
        
        img_resize = cv.resize(img, (128, 128))
        mask_resize = cv.resize(mask, (128, 128), interpolation = cv.INTER_NEAREST)
        
        #inialisasi nama file berdasarkan nama file sebelumnya
        base_img = os.path.splitext(os.path.basename(img_path))[0]
        base_mask = os.path.splitext(os.path.basename(mask_path))[0]
        
        
        name_file_img = os.path.join(direktori_img, str(base_img) + ".png")
        name_file_mask = os.path.join(direktori_gt, str(base_mask) + ".png")
            
        #simpan citra
        cv.imwrite(name_file_img, img_resize)
        cv.imwrite(name_file_mask, mask_resize)
        
    print("Proses Resize Citra 2D citra ED Selesai")
    print("--------------------------------------------------------------------------\n")
        
    
def resize_img_es(sorted_imgs, sorted_masks):
    
    print("Proses Resize Citra 2D citra ES")
    print("--------------------------------------------------------------------------\n")
    
    direktori_resize = os.path.join(direktori_resize_es, "Data Train Resize 128 ES")
    direktori_img = os.path.join(direktori_resize, "images")
    direktori_gt = os.path.join(direktori_resize, "groundtruth")
    
    #membuat direktori jika belum ada 
    os.makedirs(direktori_resize, exist_ok=True)
    os.makedirs(direktori_img, exist_ok=True)
    os.makedirs(direktori_gt, exist_ok=True)
    
    for img_path, mask_path in zip(sorted_imgs, sorted_masks):
        
        img = cv.imread(img_path, 0)
        mask = cv.imread(mask_path, 0)
        
        img_resize = cv.resize(img, (128, 128))
        mask_resize = cv.resize(mask, (128, 128), interpolation = cv.INTER_NEAREST)
        
        #inialisasi nama file berdasarkan nama file sebelumnya
        base_img = os.path.splitext(os.path.basename(img_path))[0]
        base_mask = os.path.splitext(os.path.basename(mask_path))[0]
        
        
        name_file_img = os.path.join(direktori_img, str(base_img) + ".png")
        name_file_mask = os.path.join(direktori_gt, str(base_mask) + ".png")
            
        #simpan citra
        cv.imwrite(name_file_img, img_resize)
        cv.imwrite(name_file_mask, mask_resize)
        
    print("Proses Resize Citra 2D citra ES Selesai")
    print("--------------------------------------------------------------------------\n")
       
def resize_img_ed_test(sorted_imgs, sorted_masks):
    
    print("Proses Resize Citra 2D citra ED Testing")
    print("--------------------------------------------------------------------------\n")
    
    # #untuk data train
    direktori_resize = os.path.join(direktori_resize_ed, "Data Test Resize 128 ED")
    # direktori_img = os.path.join(direktori_resize, "images")
    # direktori_gt = os.path.join(direktori_resize, "groundtruth")
    
    # #membuat direktori jika belum ada 
    os.makedirs(direktori_resize, exist_ok=True)
    # os.makedirs(direktori_img, exist_ok=True)
    # os.makedirs(direktori_gt, exist_ok=True)
    
    for img_path, mask_path in zip(sorted_imgs, sorted_masks):
        
        #untuk data testing
        match = re.search(r'Pasien(\d+)_', os.path.basename(img_path))
        if match:
            nomor_pasien = match.group(1)
        else:
            print(f"Tidak dapat mengekstrak nomor pasien dari {img_path}")
            continue
        
        direktori_pasien, direktori_img, direktori_gt, direktori_rv, direktori_myo, direktori_lv = membuat_direktori_2d_ed_testing(nomor_pasien, direktori_resize)
        
        img = cv.imread(img_path, 0)
        mask = cv.imread(mask_path, 0)
        
        img_resize = cv.resize(img, (128, 128))
        mask_resize = cv.resize(mask, (128, 128), interpolation = cv.INTER_NEAREST)
        
        unique_values = np.unique(mask_resize)
        print("Unique values in the mask:", unique_values)

        # Membuat kelas berdasarkan nilai piksel yang ditentukan
        kelas1 = np.where(mask_resize == 85, 255, 0).astype(np.uint8) if 85 in unique_values else np.zeros_like(mask_resize)
        kelas2 = np.where(mask_resize == 170, 255, 0).astype(np.uint8) if 170 in unique_values else np.zeros_like(mask_resize)
        kelas3 = np.where(mask_resize == 255, 255, 0).astype(np.uint8) if 255 in unique_values else np.zeros_like(mask_resize)
    
        
        #inialisasi nama file berdasarkan nama file sebelumnya
        base_img = os.path.splitext(os.path.basename(img_path))[0]
        base_mask = os.path.splitext(os.path.basename(mask_path))[0]
        
        
        name_file_img = os.path.join(direktori_img, str(base_img) + ".png")
        name_file_mask = os.path.join(direktori_gt, str(base_mask) + ".png")
        
        name_file_mask_kelas1 = os.path.join(direktori_rv, str(base_mask) + "_rv"+".png")
        name_file_mask_kelas2 = os.path.join(direktori_myo, str(base_mask) + "_myo"+".png")
        name_file_mask_kelas3 = os.path.join(direktori_lv, str(base_mask) + "_lv"+".png")
        
        
        #simpan citra
        cv.imwrite(name_file_img, img_resize)
        cv.imwrite(name_file_mask, mask_resize)
        cv.imwrite(name_file_mask_kelas1, kelas1)
        cv.imwrite(name_file_mask_kelas2, kelas2)
        cv.imwrite(name_file_mask_kelas3, kelas3)
        
        
    print("Proses Resize Citra 2D citra ED Selesai")
    print("--------------------------------------------------------------------------\n")
        

def main():
    
    #mengurutkan citra ed
    list_img_ed = sorted(glob.glob(dataset_path + '*/patient*_frame01.nii.gz'))
    list_mask_ed = sorted(glob.glob(dataset_path + '*/patient*_frame01_gt.nii.gz'))
    
    #mengurutkan citra es
    list_img_es = sorted(glob.glob(dataset_path + '*/patient*_frame*.nii.gz'))
    list_mask_es = sorted(glob.glob(dataset_path + '*/patient*_frame*_gt.nii.gz'))

    list_img_es_1 = [img for img in list_img_es if "frame01" not in img]
    list_mask_es_1 = [mask for mask in list_mask_es if "frame01" not in mask]
    
    #proses slice citra ed
    # slice_ed(list_img_ed, list_mask_ed)
    
    #proses slice citra es
    # slice_es(list_img_es_1, list_mask_es_1)
    
    list_img_ed_2d = sorted(glob.glob(direktori_pasien_ed + '*/images/Pasien*_*_ed.png'))
    list_mask_ed_2d = sorted(glob.glob(direktori_pasien_ed + '*/groundtruth/Pasien*_*_ed_gt.png'))
    
    list_img_es_2d = sorted(glob.glob(direktori_pasien_es + '*/images/Pasien*_*_es.png'))
    list_mask_es_2d = sorted(glob.glob(direktori_pasien_es + '*/groundtruth/Pasien*_*_es_gt.png'))
    
    #resize slice citra ed
    # resize_img_ed(list_img_ed_2d, list_mask_ed_2d)
    
    #resize slice citra es
    # resize_img_es(list_img_es_2d, list_mask_es_2d)
    
    #resize slice citra ed testing
    resize_img_ed_test(list_img_ed_2d, list_mask_ed_2d)

if __name__ == "__main__" :
    main()