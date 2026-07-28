# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 12:46:15 2024

@author: ramad
"""

import cv2 as cv
import numpy as np
import os
import re

def create_folder(patient_id, saved_directory):
    loss_dir = os.path.join(saved_directory, "focalctc")
    patient_dir = os.path.join(loss_dir, f"Pasien {patient_id}")
    
    # membuat direktori jika belum ada 
    os.makedirs(loss_dir, exist_ok=True)
    os.makedirs(patient_dir, exist_ok=True)
    
    return loss_dir, patient_dir

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split('(\d+)', s)]


def combined_images(sorted_imgs, sorted_myo, sorted_rv, sorted_lv, image_dir, myo_dir, rv_dir, lv_dir, saved_directory):
    for index, (img, myo_mask, rv_mask, lv_mask) in enumerate(zip(sorted_imgs, sorted_myo, sorted_rv, sorted_lv)):
        match_img = re.search(r'Pasien(\d+)_', img)
        match_myo = re.search(r'Pasien(\d+)_', myo_mask)
        match_rv = re.search(r'Pasien(\d+)_', rv_mask)
        match_lv = re.search(r'Pasien(\d+)_', lv_mask)
        
        if match_img and match_myo and match_rv and match_lv and \
           match_img.group(1) == match_myo.group(1) == match_rv.group(1) == match_lv.group(1):
            patient_id = match_img.group(1)
        else:
            print(f"Tidak dapat mengekstrak nomor pasien dari images = {match_img}, myo = {match_myo}, rv = {match_rv}, lv = {match_lv}\n")
            continue
        
        loss_dir, patient_dir = create_folder(patient_id, saved_directory)
        
        img_path = os.path.join(image_dir, img)
        myo_mask_path = os.path.join(myo_dir, myo_mask)
        rv_mask_path = os.path.join(rv_dir, rv_mask)
        lv_mask_path = os.path.join(lv_dir, lv_mask)
        
        print(f"images = {img_path}")
        print(f"myo = {myo_mask_path}")
        print(f"rv = {rv_mask_path}")
        print(f"lv = {lv_mask_path}\n")
        
        imgs = cv.imread(img_path, 0)
        myo_mask = cv.imread(myo_mask_path, 0)
        rv_mask = cv.imread(rv_mask_path, 0)
        lv_mask = cv.imread(lv_mask_path, 0)
        
        img_rgb = cv.cvtColor(imgs, cv.COLOR_GRAY2BGR)
        
        myo_mask_1 = myo_mask > 0
        rv_mask_1 = rv_mask > 0
        lv_mask_1 = lv_mask > 0
        
        # img_rgb[lv_mask_1] = [139, 51, 204]  # magenta
        img_rgb[myo_mask_1] = [235, 183, 0]  # cyan
        # img_rgb[rv_mask_1] = [255, 0, 17]  # pink
        
        name_file_images = os.path.join(patient_dir, f"Pasien{patient_id}_{index+1}_img.png")
        
        cv.imwrite(name_file_images, img_rgb)
        
        print(f"Gambar {index+1} Pasien {patient_id} sudah diproses... \n")

def main():
    for i in range(101, 151):  
        image_dir = f"D:/Intelligent Multimedia Network/Research/Riset Bu Dini/Dataset/acdc17/acdc17/Data 2D/ED/Data Test Resize 128 ED/Pasien {i}/images/"
        myo_dir = f"D:/Intelligent Multimedia Network/Research/Riset Bu Dini/Dataset/acdc17/acdc17/Data 2D/ED/Hasil Predict 2D versi 3/Hasil Predict Versi 2/transunet/focalctc/Prediksi Pasien/Pasien {i}/prediksi myocardium/"
        rv_dir = f"D:/Intelligent Multimedia Network/Research/Riset Bu Dini/Dataset/acdc17/acdc17/Data 2D/ED/Hasil Predict 2D versi 3/Hasil Predict Versi 2/transunet/focalctc/Prediksi Pasien/Pasien {i}/prediksi right ventricel/"
        lv_dir = f"D:/Intelligent Multimedia Network/Research/Riset Bu Dini/Dataset/acdc17/acdc17/Data 2D/ED/Hasil Predict 2D versi 3/Hasil Predict Versi 2/transunet/focalctc/Prediksi Pasien/Pasien {i}/prediksi left ventricel/"
        
        saved_directory = "D:/Intelligent Multimedia Network/Research/Riset Bu Dini/Dataset/acdc17/acdc17/Data 2D/ED/Hasil Predict 2D versi 3/Gabung citra dan myo/transunet/"
        
        if not os.path.exists(myo_dir) or not os.path.exists(rv_dir) or not os.path.exists(lv_dir):
            print(f"Jalur direktori tidak ditemukan untuk Pasien {i}.")
            continue
        
        get_imgs = os.listdir(image_dir)
        sorted_imgs = sorted(get_imgs, key=natural_sort_key)
        
        get_myo = os.listdir(myo_dir)
        sorted_myo = sorted(get_myo, key=natural_sort_key)
        
        get_rv = os.listdir(rv_dir)
        sorted_rv = sorted(get_rv, key=natural_sort_key)
        
        get_lv = os.listdir(lv_dir)
        sorted_lv = sorted(get_lv, key=natural_sort_key)
        
        combined_images(sorted_imgs, sorted_myo, sorted_rv, sorted_lv, image_dir, myo_dir, rv_dir, lv_dir, saved_directory)

if __name__ == "__main__":
    main()
