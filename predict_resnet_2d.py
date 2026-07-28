# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 11:38:20 2024

@author: ramad
"""

from keras.models import load_model
import cv2 as cv
import os
import numpy as np
import glob
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
import glob
import re
import segmentation_models as sm


direktori_citra = "D:/Intelligent Multimedia Network/Research/Riset Bu Dini/Dataset/acdc17/acdc17/Data 2D/ED/Data Test Resize 128 ED/"
direktori_prediksi_ed = "D:/Intelligent Multimedia Network/Research/Riset Bu Dini/Dataset/acdc17/acdc17/Data 2D/ED/Hasil Predict ED/resnet/batch8/"

def get_image_modification_time(item):
    item_path = os.path.join(direktori_citra, item)
    return os.path.getmtime(item_path)


def get_mask_modification_time(item):
    item_path = os.path.join(direktori_mask, item)
    return os.path.getmtime(item_path)

def get_predict_modification_time(item):
    item_path = os.path.join(direktori_prediksi, item)
    return os.path.getmtime(item_path)


def membuat_direktori_2d_ed_prediksi(id_pasien, path_direktori):
    
    direktori_pasien = os.path.join(path_direktori, f"Pasien {id_pasien}")
    direktori_prediksi = os.path.join(direktori_pasien, "citra prediksi")
    direktori_rv = os.path.join(direktori_pasien, "prediksi right ventricel")
    direktori_myo = os.path.join(direktori_pasien, "prediksi myocardium")
    direktori_lv = os.path.join(direktori_pasien, "prediksi left ventricel")
    
    
    #membuat direktori jika belum ada 
    os.makedirs(direktori_pasien, exist_ok=True)
    os.makedirs(direktori_prediksi, exist_ok=True)
    os.makedirs(direktori_rv, exist_ok=True)
    os.makedirs(direktori_myo, exist_ok=True)
    os.makedirs(direktori_lv, exist_ok=True)
    
    return direktori_pasien, direktori_prediksi,  direktori_rv, direktori_myo, direktori_lv




def prediksi_citra(imgs, model):
    
    BACKBONE3 = 'resnet34'
    preprocess_input3 = sm.get_preprocessing(BACKBONE3)
    
    for img_path in imgs:
        
        #untuk data predict
        match = re.search(r'Pasien(\d+)_', os.path.basename(img_path))
        if match:
            nomor_pasien = match.group(1)
        else:
            print(f"Tidak dapat mengekstrak nomor pasien dari {img_path}")
            continue
        
        direktori_pasien, direktori_prediksi, direktori_rv, direktori_myo, direktori_lv = membuat_direktori_2d_ed_prediksi(nomor_pasien, direktori_prediksi_ed)
        
        img = cv.imread(img_path, 0)
        img_array = np.array(img)
            
        test_img = np.repeat(img_array[..., np.newaxis], 3, axis=-1)
            
            
        test_img_input=np.expand_dims(test_img, 0)

        test_img_input1 = preprocess_input3(test_img_input)
        test_pred1 = model.predict(test_img_input1)
        predicted_img = np.argmax(test_pred1, axis=3)[0,:,:]
        
        # Normalisasi ke rentang 0 - 255
        predicted_img_norm = (predicted_img / predicted_img.max()) * 255
        predicted_img_uint8 = predicted_img_norm.astype(np.uint8)
        
        
        base_img = os.path.splitext(os.path.basename(img_path))[0]
        
        name_file_predict = os.path.join(direktori_prediksi, str(base_img) + "_predict"+".png")
        name_file_rv = os.path.join(direktori_rv, str(base_img) + "_rv_predict"+".png")
        name_file_myo = os.path.join(direktori_myo, str(base_img) + "_myo_predict"+".png")
        name_file_lv = os.path.join(direktori_lv, str(base_img) + "_lv_predict"+".png")
        
        cv.imwrite(name_file_predict, predicted_img_uint8)
        
        # Menyimpan citra untuk kelas RV
        if 1 in predicted_img:
            predicted_rv = (predicted_img == 1).astype(np.uint8) * 255
            cv.imwrite(name_file_rv, predicted_rv)
        else:
            cv.imwrite(name_file_rv, np.zeros_like(predicted_img_uint8))

        # Menyimpan citra untuk kelas Myo
        if 2 in predicted_img:
            predicted_myo = (predicted_img == 2).astype(np.uint8) * 255
            cv.imwrite(name_file_myo, predicted_myo)
        else:
            cv.imwrite(name_file_myo, np.zeros_like(predicted_img_uint8))

        # Menyimpan citra untuk kelas LV
        if 3 in predicted_img:
            predicted_lv = (predicted_img == 3).astype(np.uint8) * 255
            cv.imwrite(name_file_lv, predicted_lv)
        else:
            cv.imwrite(name_file_lv, np.zeros_like(predicted_img_uint8))
            
            
        
        print(f"Citra {base_img} sudah diproses")
        

    print("Proses Prediksi Selesai....")
        
        
    
def main():
    
    model = load_model('D:/Intelligent Multimedia Network/Research/Riset Bu Dini/Dataset/acdc17/acdc17/Data 2D/ED/Model/model_resnet_cardiac_ed_batch8_categoricalcrossentropy_lr001.hdf5', compile=False)
    list_img_ed_2d = sorted(glob.glob(direktori_citra + '*/images/Pasien*_*_ed.png'))
    
    prediksi_citra(list_img_ed_2d, model)
    
    
if __name__ == "__main__":
    main()