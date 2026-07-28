# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 04:46:12 2023

@author: ramad
"""

from keras.models import load_model
import numpy as np
import random
from matplotlib import pyplot as plt
import os
import cv2 as cv
from hausdorff.hausdorff import hausdorff_distance
from hitung_dice_2d import dice_coefficient
import csv

direktori_citra_ed = "D:/Intelligent Multimedia Network/Research/Riset Bu Dini/Dataset/acdc17/acdc17/Data 3D 128 Testing/ED/images/part 2/"
direktori_gt_ed  = "D:/Intelligent Multimedia Network/Research/Riset Bu Dini/Dataset/acdc17/acdc17/Data 3D 128 Testing/ED/masks/part 2/"


simpan_prediksi_ed = "D:/Intelligent Multimedia Network/Research/Riset Bu Dini/Dataset/acdc17/acdc17/Predict ACDC 2017/ED/unet/"   
simpan_gt_ed = "D:/Intelligent Multimedia Network/Research/Riset Bu Dini/Dataset/acdc17/acdc17/GT ACDC 2017 2D Hasil Preprocessing/ED/"


direktori_prediksi_ed_myo = "D:/Intelligent Multimedia Network/Research/Riset Bu Dini/Dataset/acdc17/acdc17/Predict ACDC 2017/ED/unet/Pasien 121/Predict Myo/"
direktori_gt_ed_myo = "D:/Intelligent Multimedia Network/Research/Riset Bu Dini/Dataset/acdc17/acdc17/GT ACDC 2017 2D Hasil Preprocessing/ED/Pasien 121/Groundtruth Myo/"

simpan_csv = "D:/Intelligent Multimedia Network/Research/Riset Bu Dini/Dataset/acdc17/acdc17/Predict ACDC 2017/ED/unet/Pasien 121/"

def get_image_modification_time(item):
    item_path = os.path.join(direktori_citra_ed, item)
    return os.path.getmtime(item_path)

def get_mask_modification_time(item):
    item_path = os.path.join(direktori_gt_ed, item)
    return os.path.getmtime(item_path)

def get_predict_myo_modification_time(item):
    item_path = os.path.join(direktori_prediksi_ed_myo, item)
    return os.path.getmtime(item_path)

def get_gt_myo_modification_time(item):
    item_path = os.path.join(direktori_gt_ed_myo, item)
    return os.path.getmtime(item_path)

def membuat_direktori_gt(patient_id, base_save_dir):
    """
    Membuat direktori utama berdasarkan iterasi nama pasien dan sub-direktori untuk prediksi keseluruhan dan per kelas.
    Mengembalikan path direktori yang telah dibuat atau yang sudah ada beserta path untuk prediksi keseluruhan dan per kelas.
    """
    direktori_pasien = os.path.join(base_save_dir, f"Pasien {patient_id}")
    direktori_gt_keseluruhan = os.path.join(direktori_pasien, "Groundtruth Keseluruhan")
    direktori_rv = os.path.join(direktori_pasien, "Groundtruth RV")
    direktori_myo = os.path.join(direktori_pasien, "Groundtruth Myo")
    direktori_lv = os.path.join(direktori_pasien, "Groundtruth LV")# Asumsi ada 3 kelas
    
    # Membuat direktori jika belum ada
    os.makedirs(direktori_pasien, exist_ok=True)
    os.makedirs(direktori_gt_keseluruhan, exist_ok=True)
    # os.makedirs(direktori_rv, exist_ok=True)
    os.makedirs(direktori_myo, exist_ok=True)
    # os.makedirs(direktori_lv, exist_ok=True)
    
    return direktori_pasien, direktori_gt_keseluruhan, direktori_myo



def membuat_direktori_prediksi(patient_id, base_save_dir):
    """
    Membuat direktori utama berdasarkan iterasi nama pasien dan sub-direktori untuk prediksi keseluruhan dan per kelas.
    Mengembalikan path direktori yang telah dibuat atau yang sudah ada beserta path untuk prediksi keseluruhan dan per kelas.
    """
    direktori_pasien = os.path.join(base_save_dir, f"Pasien {patient_id}")
    direktori_prediksi_keseluruhan = os.path.join(direktori_pasien, "Predict Keseluruhan")
    direktori_rv = os.path.join(direktori_pasien, "Predict RV")
    direktori_myo = os.path.join(direktori_pasien, "Predict Myo")
    direktori_lv = os.path.join(direktori_pasien, "Predict LV")# Asumsi ada 3 kelas
    
    # Membuat direktori jika belum ada
    os.makedirs(direktori_pasien, exist_ok=True)
    os.makedirs(direktori_prediksi_keseluruhan, exist_ok=True)
    os.makedirs(direktori_rv, exist_ok=True)
    os.makedirs(direktori_myo, exist_ok=True)
    os.makedirs(direktori_lv, exist_ok=True)
    
    
    
    return direktori_pasien, direktori_prediksi_keseluruhan, direktori_rv, direktori_myo, direktori_lv

def slice_gt(mask_ed):
    
    for index, filenames in enumerate(mask_ed):
        if filenames.endswith('.npy'):
            file_path = os.path.join(direktori_gt_ed, filenames)
            print(file_path)
            
            id_pasien = index + 133
            
            direktori_pasien, direktori_gt_keseluruhan, direktori_myo = membuat_direktori_gt(id_pasien, simpan_gt_ed)
            
            img = np.load(file_path)
            
            test_mask_argmax=np.argmax(img, axis=3)
            
            for slice_index in range(test_mask_argmax.shape[2]):
                
                print(f"Citra {slice_index + 1} Masih dalam Proses")
                
                plt.imshow(test_mask_argmax[:, :, slice_index])  # Menampilkan citra dalam grayscale
                plt.axis('off')  # Menyembunyikan sumbu
                plt.savefig(os.path.join(direktori_gt_keseluruhan, f'img_{slice_index + 1}.png'), bbox_inches='tight', pad_inches=0)  # Menyimpan citra
                plt.close()
                # plt.title('Citra Prediksi Myocardium')
                
                plt.imshow(test_mask_argmax[:, :, slice_index] == 2, cmap='gray')  # Menampilkan citra dalam grayscale
                plt.axis('off')  # Menyembunyikan sumbu
                plt.savefig(os.path.join(direktori_myo, f'img_{slice_index + 1}_myo.png'), bbox_inches='tight', pad_inches=0)  # Menyimpan citra
                plt.close()
                
    print("Proses Selesai")
                
                
                
            
            
def prediksi_ed(img_ed, model):
    
    for index, filenames in enumerate(img_ed):
        if filenames.endswith('.npy'):
            file_path = os.path.join(direktori_citra_ed, filenames)
            print(file_path)
            
            id_pasien = index + 101
            
            direktori_pasien, direktori_prediksi_keseluruhan, direktori_rv, direktori_myo, direktori_lv = membuat_direktori_prediksi(id_pasien, simpan_prediksi_ed)
            
            img = np.load(file_path)
            
            img_input = np.expand_dims(img, axis=0)
            img_prediction = model.predict(img_input)
            prediction_argmax = np.argmax(img_prediction, axis=4)[0,:,:,:]
            
            for slice_index in range(prediction_argmax.shape[2]):
                # Mengambil slice saat ini
                current_slice = prediction_argmax[:, :, slice_index]
               
                
                print(f"Citra {slice_index + 1} Masih dalam Proses")
                
               
                # plt.title('Citra Prediksi Keseluruhan')
                plt.imshow(current_slice)  # Menampilkan citra dalam grayscale
                plt.axis('off')  # Menyembunyikan sumbu
                plt.savefig(os.path.join(direktori_prediksi_keseluruhan, f'img_{slice_index + 1}.png'), bbox_inches='tight', pad_inches=0)  # Menyimpan citra
                plt.close()
                # plt.title('Citra Prediksi Myocardium')
                
                plt.imshow(prediction_argmax[:, :, slice_index] == 1, cmap='gray')  # Menampilkan citra dalam grayscale
                plt.axis('off')  # Menyembunyikan sumbu
                plt.savefig(os.path.join(direktori_rv, f'img_{slice_index + 1}_rv.png'), bbox_inches='tight', pad_inches=0)  # Menyimpan citra
                plt.close()
                
                
                plt.imshow(prediction_argmax[:, :, slice_index] == 2, cmap='gray')  # Menampilkan citra dalam grayscale
                plt.axis('off')  # Menyembunyikan sumbu
                plt.savefig(os.path.join(direktori_myo, f'img_{slice_index + 1}_myo.png'), bbox_inches='tight', pad_inches=0)  # Menyimpan citra
                plt.close()
                
                plt.imshow(prediction_argmax[:, :, slice_index] == 3, cmap='gray')  # Menampilkan citra dalam grayscale
                plt.axis('off')  # Menyembunyikan sumbu
                plt.savefig(os.path.join(direktori_lv, f'img_{slice_index + 1}_lv.png'), bbox_inches='tight', pad_inches=0)  # Menyimpan citra
                plt.close()
                
                # plt.show()
                
    print("Proses Selesai")
    
def hitung_evaluasi_metrik(imgs, masks):
    
    with open(simpan_csv + 'hasil_evaluasi_metrik_myocardium_Pasien 121.csv', mode='w', newline='') as file:
        writer = csv.writer(file, delimiter=';')
        writer.writerow(['Citra Ke', 'Dice','Hausdorff Distance'])
            
        if len(imgs) != len(masks):
            print("Jumlah citra predict dan mask tidak sama. Proses evaluasi tidak dilanjutkan.")
            return
        
        dice = 0
        hausdorff = 0
        
        for predict_filename, mask_filename in zip(imgs, masks):
            if predict_filename.endswith('.png') and mask_filename.endswith('.png'):
                
                file_path_predict = os.path.join(direktori_prediksi_ed_myo, predict_filename)
                file_path_mask = os.path.join(direktori_gt_ed_myo, mask_filename)
                
                print(file_path_predict)
                print(file_path_mask)
                
                img_predict_myo = cv.imread(file_path_predict, 0)
                img_gt_myo = cv.imread(file_path_mask, 0)
                
                
                dice_myo_gt = np.where(img_gt_myo == 255, 1, 0)
                dice_myo_predict = np.where(img_predict_myo == 255, 1, 0)
        
                dice_myo = dice_coefficient(dice_myo_gt, dice_myo_predict)
        
                hd_myo_gt = np.where(img_gt_myo == 255, 1, 0)
                hd_myo_predict = np.where(img_predict_myo == 255, 1, 0)

                hd_myo = hausdorff_distance(hd_myo_gt, hd_myo_predict, distance='euclidean')
                
                dice += dice_myo
                hausdorff += hd_myo
        
        
                dice_myo_str = str(dice_myo).replace('.', ',')
                hd_myo_str = str(hd_myo).replace('.', ',')
        

                # Menulis hasil ke file CSV
                writer.writerow([f"citra {predict_filename}", dice_myo_str, hd_myo_str])

                print(f"Citra {predict_filename}:")
                print("Nilai Dice myo = ", dice_myo)
                print("Nilai Hausdorff myo = ", hd_myo)
                print("\n")
                
                print(f"Citra {predict_filename} sudah dihitung... \n")
         
        avg_dice = dice / len(imgs)
        avg_hausdorff = hausdorff / len(imgs)
        
        avg_dice_myo_str = str(avg_dice).replace('.', ',')
        avg_hd_myo_str = str(avg_hausdorff).replace('.', ',')
         
        writer.writerow(["Mean Dice", "Mean Hausdorff Distance"])
        writer.writerow([avg_dice_myo_str, avg_hausdorff])
   
    print("Perhitungan Selesai....")
    
    
            
            

def main():
    
    model_ed = load_model('D:/Intelligent Multimedia Network/Research/Riset Bu Dini/Dataset/acdc17/model/train_unet_3d_ed_1.hdf5',compile=False)
    
    get_img_ed = os.listdir(direktori_citra_ed)
    get_gt_ed = os.listdir(direktori_gt_ed)
    
    sorted_images_ed = sorted(get_img_ed, key=get_image_modification_time)
    sorted_gt_ed = sorted(get_gt_ed, key=get_mask_modification_time)
    
    
    # prediksi_ed(sorted_images_ed, model_ed)
    
    # slice_gt(sorted_gt_ed)
    
    get_predict_ed_myo = os.listdir(direktori_prediksi_ed_myo)
    get_gt_ed_myo = os.listdir(direktori_gt_ed_myo)
    
    sorted_predict_ed_myo = sorted(get_predict_ed_myo, key=get_predict_myo_modification_time)
    sorted_gt_ed_myo = sorted(get_gt_ed_myo, key=get_gt_myo_modification_time)
    
    hitung_evaluasi_metrik(sorted_predict_ed_myo, sorted_gt_ed_myo)
    

if __name__ == "__main__":
    main()
    
    

# model_ed = load_model('D:/Intelligent Multimedia Network/Research/Riset Bu Dini/Dataset/acdc17/model/train_unet_3d_ed_1.hdf5',compile=False)


# img_num = 101

# test_img = np.load("D:/Intelligent Multimedia Network/Research/Riset Bu Dini/Dataset/acdc17/acdc17/Data 3D 128 Testing/ED/images/image_"+str(img_num)+"_ed.npy")
# test_mask = np.load("D:/Intelligent Multimedia Network/Research/Riset Bu Dini/Dataset/acdc17/acdc17/Data 3D 128 Testing/ED/masks/mask_"+str(img_num)+"_ed.npy")
# test_mask_argmax=np.argmax(test_mask, axis=3)


# test_img_input = np.expand_dims(test_img, axis=0)
# test_prediction = model_ed.predict(test_img_input)
# test_prediction_argmax=np.argmax(test_prediction, axis=4)[0,:,:,:]

# for slice_index in range(test_prediction_argmax.shape[2]):
#     # Mengambil slice saat ini
#     current_slice = test_prediction_argmax[:, :, slice_index]
#     plt.figure(figsize=(12, 8))
#     plt.subplot(231)
#     plt.title('Citra Prediksi')
#     # Menyimpan slice saat ini sebagai citra 2D grayscale
#     plt.imshow(current_slice, cmap='gray')  # Menampilkan citra dalam grayscale
#     plt.subplot(232)
#     plt.title('Ground Truth')
#     plt.imshow(test_mask_argmax[:, :, slice_index])
#     plt.subplot(233)
#     plt.title('Ground Truth Myo')
#     plt.imshow(test_mask_argmax[:, :, slice_index] == 2, cmap='gray')
#     # plt.axis('off')  # Menyembunyikan sumbu
#     # plt.savefig(f'image_{img_num}_prediction_2d_slice_{slice_index}.png', bbox_inches='tight', pad_inches=0)  # Menyimpan citra
#     plt.show()







# # n_slice=random.randint(0, test_prediction_argmax.shape[2])
# # print("slice ke = ", n_slice)
# n_slice = 2
# plt.figure(figsize=(12, 8))
# plt.subplot(231)
# plt.title('Citra Asli')
# plt.imshow(test_img[:,:,n_slice,0], cmap='gray')
# plt.subplot(232)
# plt.title('Groundtruth')
# plt.imshow(test_mask_argmax[:,:,n_slice])
# plt.subplot(233)
# plt.title('Citra Prediksi')
# plt.imshow(test_prediction_argmax[:,:, n_slice])
# plt.subplot(234)
# plt.title('Citra Prediksi Right Ventrical')
# plt.imshow(test_prediction_argmax[:,:, n_slice] == 1) 
# plt.subplot(235)
# plt.title('Citra Prediksi Myocardium')
# plt.imshow(test_prediction_argmax[:,:, n_slice] == 2)
# plt.subplot(236)
# plt.title('Citra Prediksi Lef Ventrical')
# plt.imshow(test_prediction_argmax[:,:, n_slice] == 3) 
# plt.show()