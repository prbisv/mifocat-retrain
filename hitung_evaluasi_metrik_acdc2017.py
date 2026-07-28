# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 21:55:30 2024

@author: ramad
"""



from keras.models import load_model
import cv2 as cv
import os
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from hitung_iou_2d import iou_2d
from hausdorff.hausdorff import hausdorff_distance
from hitung_dice_2d import dice_coefficient
from hitung_f1score_2d import f1_score_2d
from hitung_surface_distance_2d import surfd
from hitung_mcc_2d import mcc_2d

import csv
    
for i in range(101, 151):
    
    direktori_predict_ed_myo = f"D:/Intelligent Multimedia Network/Research/Riset Bu Dini/Dataset/acdc17/acdc17/Data 2D/ED/Hasil Predict 2D versi 3/Hasil Predict Versi 2/transunet/msefocalctc/Prediksi Pasien/Pasien {i}/prediksi myocardium/"
    direktori_predict_ed_rv = f"D:/Intelligent Multimedia Network/Research/Riset Bu Dini/Dataset/acdc17/acdc17/Data 2D/ED/Hasil Predict 2D versi 3/Hasil Predict Versi 2/transunet/msefocalctc/Prediksi Pasien/Pasien {i}/prediksi right ventricel/"
    direktori_predict_ed_lv = f"D:/Intelligent Multimedia Network/Research/Riset Bu Dini/Dataset/acdc17/acdc17/Data 2D/ED/Hasil Predict 2D versi 3/Hasil Predict Versi 2/transunet/msefocalctc/Prediksi Pasien/Pasien {i}/prediksi left ventricel/"


    direktori_gt_ed_myo = f"D:/Intelligent Multimedia Network/Research/Riset Bu Dini/Dataset/acdc17/acdc17/Data 2D/ED/Data Test Resize 128 ED/Pasien {i}/groundtruth myocardium/"
    direktori_gt_ed_rv = f"D:/Intelligent Multimedia Network/Research/Riset Bu Dini/Dataset/acdc17/acdc17/Data 2D/ED/Data Test Resize 128 ED/Pasien {i}/groundtruth right ventricel/"
    direktori_gt_ed_lv = f"D:/Intelligent Multimedia Network/Research/Riset Bu Dini/Dataset/acdc17/acdc17/Data 2D/ED/Data Test Resize 128 ED/Pasien {i}/groundtruth left ventricel/"



    simpan_csv = f"D:/Intelligent Multimedia Network/Research/Riset Bu Dini/Dataset/acdc17/acdc17/Data 2D/ED/Hasil Predict 2D versi 3/Hasil Predict Versi 2/transunet/msefocalctc/Prediksi Pasien/Pasien {i}/"

    
    def hitung_evaluasi_metrik_myo(imgs, masks):
    
        iou_list_myo = []
        dice_list_myo = []
        # mcc_list_myo = []
        # f1_score_list_myo = []
        hausdorff_list_myo = []
        surface_distance_list_myo = []
    
        with open(simpan_csv + f'metrik_evaluasi_transunet_myo_ed_Pasien {i}_acdc2017.csv', mode='w', newline='') as file:
            writer = csv.writer(file, delimiter=';')
            writer.writerow(['Citra Ke', 'IoU', 'Dice','Hausdorff Distance', 'Surface Distance'])
            
            if len(imgs) != len(masks):
                print("Jumlah citra predict dan mask tidak sama. Proses evaluasi tidak dilanjutkan.")
                return
        
        
            for predict_filename, mask_filename in zip(imgs, masks):
                if predict_filename.endswith('.png') and mask_filename.endswith('.png'):
                
                    file_path_predict = os.path.join(direktori_predict_ed_myo, predict_filename)
                    file_path_mask = os.path.join(direktori_gt_ed_myo, mask_filename)
                
                    print(file_path_predict)
                    print(file_path_mask)
                
                    img_predict_myo_ed = cv.imread(file_path_predict, 0)
                    img_gt_myo_ed = cv.imread(file_path_mask, 0)
                
                    # Konversi citra ke format biner (0 dan 1)
                    bin_predict = np.where(img_predict_myo_ed == 255, 1, 0)
                    bin_gt = np.where(img_gt_myo_ed == 255, 1, 0)
                
                
                    iou_myo_ed = iou_2d(bin_gt, bin_predict)
                
                    dice_myo_ed = dice_coefficient(bin_gt, bin_predict)
                
                    # f1_score_myo_ed = f1_score_2d(bin_gt, bin_predict)
                    # mcc_myo_ed = mcc_2d(bin_gt, bin_predict)
                
                    hd_myo_ed = hausdorff_distance(bin_gt, bin_predict, distance='euclidean')
                
                    surface_myo_ed = surfd(bin_gt, bin_predict, sampling=1, connectivity=1)
                    mean_surface_myo_ed = np.mean(surface_myo_ed)
                
                    iou_list_myo.append(iou_myo_ed)
                    dice_list_myo.append(dice_myo_ed)
                    # f1_score_list.append(f1_score_myo_ed)
                    # mcc_list.append(mcc_myo_ed)
                    hausdorff_list_myo.append(hd_myo_ed)
                    surface_distance_list_myo.append(mean_surface_myo_ed)
        
                    iou_myo_ed_str = str(iou_myo_ed).replace('.', ',')
                    dice_myo_ed_str = str(dice_myo_ed).replace('.', ',')
                    # f1_myo_ed_str = str(f1_score_myo_ed).replace('.', ',')
                    # mcc_myo_ed_str = str(mcc_myo_ed).replace('.', ',')
                    hd_myo_ed_str = str(hd_myo_ed).replace('.', ',')
                    mean_surface_myo_ed_str = str(mean_surface_myo_ed).replace('.', ',')
                
                
                    # iou += iou_myo_ed
                    # dice += dice_myo_ed
                    # f1_score += f1_score_myo_ed
                    # hausdorff += hd_myo_ed
                    # surface += surface_myo_ed
                
        
        
                    # iou_myo_ed_str = str(iou_myo_ed).replace('.', ',')
                    # dice_myo_ed_str = str(dice_myo_ed).replace('.', ',')
                    # f1_myo_ed_str = str(f1_score_myo_ed).replace('.', ',')
                    # hd_myo_ed_str = str(hd_myo_ed).replace('.', ',')
                    # surface_myo_ed_str = str(surface_myo_ed).replace('.', ',')
        

                    # Menulis hasil ke file CSV
                    writer.writerow([f"citra {predict_filename}", iou_myo_ed_str, dice_myo_ed_str, hd_myo_ed_str, mean_surface_myo_ed_str])

                    print(f"Citra {predict_filename}:")
                    print("Nilai IoU Myo ED = ", iou_myo_ed)
                    print("Nilai Dice Myo ED = ", dice_myo_ed)
                    # print("Nilai F1 Myo ED = ", f1_score_myo_ed)
                    # print("Nilai MCC Myo ED = ", mcc_myo_ed)
                    print("Nilai Hausdorff Myo ED = ", hd_myo_ed)
                    print("Nilai Surface Myo ED = ", mean_surface_myo_ed)
                    print("\n")
                
                    print(f"Citra {predict_filename} sudah dihitung... \n")
        
            avg_iou = np.mean(iou_list_myo)
            avg_dice = np.mean(dice_list_myo)
            # avg_f1 = np.mean(f1_score_list)
            # avg_mcc = np.mean(mcc_list)
            avg_hausdorff = np.mean(hausdorff_list_myo)
            avg_surface = np.mean(surface_distance_list_myo)
        
            avg_iou_myo_ed_str = str(avg_iou).replace('.', ',')
            avg_dice_myo_ed_str = str(avg_dice).replace('.', ',')
            # avg_f1_myo_ed_str = str(avg_f1).replace('.', ',')
            # avg_mcc_myo_ed_str = str(avg_mcc).replace('.', ',')
            avg_hd_myo_ed_str = str(avg_hausdorff).replace('.', ',')
            avg_surface_myo_ed_str = str(avg_surface).replace('.', ',')
         
            writer.writerow(["Mean IoU", "Mean Dice","Mean Hausdorff Distance", "Mean Surface Distance"])
            writer.writerow([avg_iou_myo_ed_str, avg_dice_myo_ed_str, avg_hd_myo_ed_str, avg_surface_myo_ed_str])
        
            return iou_list_myo, dice_list_myo, hausdorff_list_myo, surface_distance_list_myo
        
            print("Perhitungan Nilai Metrik Myocardium Sudah Selesai...\n")

    def hitung_evaluasi_metrik_rv(imgs, masks):
    
        iou_list_rv = []
        dice_list_rv = []
        # mcc_list_rv = []
        # f1_score_list_rv = []
        hausdorff_list_rv = []
        surface_distance_list_rv = []
    
        with open(simpan_csv + f'metrik_evaluasi_transunet_rv_ed_Pasien {i}_acdc2017.csv', mode='w', newline='') as file:
            writer = csv.writer(file, delimiter=';')
            writer.writerow(['Citra Ke', 'IoU', 'Dice','Hausdorff Distance', 'Surface Distance'])
            
            if len(imgs) != len(masks):
                print("Jumlah citra predict dan mask tidak sama. Proses evaluasi tidak dilanjutkan.")
                return
        
        
            for predict_filename, mask_filename in zip(imgs, masks):
                if predict_filename.endswith('.png') and mask_filename.endswith('.png'):
                
                    file_path_predict = os.path.join(direktori_predict_ed_rv, predict_filename)
                    file_path_mask = os.path.join(direktori_gt_ed_rv, mask_filename)
                
                    print(file_path_predict)
                    print(file_path_mask)
                
                    img_predict_rv_ed = cv.imread(file_path_predict, 0)
                    img_gt_rv_ed = cv.imread(file_path_mask, 0)
                
                    # Konversi citra ke format biner (0 dan 1)
                    bin_predict = np.where(img_predict_rv_ed == 255, 1, 0)
                    bin_gt = np.where(img_gt_rv_ed == 255, 1, 0)
                
                
                    iou_rv_ed = iou_2d(bin_gt, bin_predict)
                
                    dice_rv_ed = dice_coefficient(bin_gt, bin_predict)
                
                    # f1_score_rv_ed = f1_score_2d(bin_gt, bin_predict)
                    # mcc_rv_ed = mcc_2d(bin_gt, bin_predict)
                
                    hd_rv_ed = hausdorff_distance(bin_gt, bin_predict, distance='euclidean')
                
                    surface_rv_ed = surfd(bin_gt, bin_predict, sampling=1, connectivity=1)
                    mean_surface_rv_ed = np.mean(surface_rv_ed)
                
                    iou_list_rv.append(iou_rv_ed)
                    dice_list_rv.append(dice_rv_ed)
                    # f1_score_list.append(f1_score_rv_ed)
                    # mcc_list.append(mcc_rv_ed)
                    hausdorff_list_rv.append(hd_rv_ed)
                    surface_distance_list_rv.append(mean_surface_rv_ed)
        
                    iou_rv_ed_str = str(iou_rv_ed).replace('.', ',')
                    dice_rv_ed_str = str(dice_rv_ed).replace('.', ',')
                    # f1_rv_ed_str = str(f1_score_rv_ed).replace('.', ',')
                    # mcc_rv_ed_str = str(mcc_rv_ed).replace('.', ',')
                    hd_rv_ed_str = str(hd_rv_ed).replace('.', ',')
                    mean_surface_rv_ed_str = str(mean_surface_rv_ed).replace('.', ',')
                
                
                    # iou += iou_rv_ed
                    # dice += dice_rv_ed
                    # f1_score += f1_score_rv_ed
                    # hausdorff += hd_rv_ed
                    # surface += surface_rv_ed
                
        
        
                    # iou_rv_ed_str = str(iou_rv_ed).replace('.', ',')
                    # dice_rv_ed_str = str(dice_rv_ed).replace('.', ',')
                    # f1_rv_ed_str = str(f1_score_rv_ed).replace('.', ',')
                    # hd_rv_ed_str = str(hd_rv_ed).replace('.', ',')
                    # surface_myo_ed_str = str(surface_rv_ed).replace('.', ',')
        

                    # Menulis hasil ke file CSV
                    writer.writerow([f"citra {predict_filename}", iou_rv_ed_str, dice_rv_ed_str, hd_rv_ed_str, mean_surface_rv_ed_str])

                    print(f"Citra {predict_filename}:")
                    print("Nilai IoU rv ED = ", iou_rv_ed)
                    print("Nilai Dice rv ED = ", dice_rv_ed)
                    # print("Nilai F1 rv ED = ", f1_score_rv_ed)
                    # print("Nilai MCC rv ED = ", mcc_rv_ed)
                    print("Nilai Hausdorff rv ED = ", hd_rv_ed)
                    print("Nilai Surface rv ED = ", mean_surface_rv_ed)
                    print("\n")
                
                    print(f"Citra {predict_filename} sudah dihitung... \n")
        
            avg_iou = np.mean(iou_list_rv)
            avg_dice = np.mean(dice_list_rv)
            # avg_f1 = np.mean(f1_score_list_rv)
            # avg_mcc = np.mean(mcc_list_rv)
            avg_hausdorff = np.mean(hausdorff_list_rv)
            avg_surface = np.mean(surface_distance_list_rv)
        
            avg_iou_rv_ed_str = str(avg_iou).replace('.', ',')
            avg_dice_rv_ed_str = str(avg_dice).replace('.', ',')
            # avg_f1_rv_ed_str = str(avg_f1).replace('.', ',')
            # avg_mcc_rv_ed_str = str(avg_mcc).replace('.', ',')
            avg_hd_rv_ed_str = str(avg_hausdorff).replace('.', ',')
            avg_surface_rv_ed_str = str(avg_surface).replace('.', ',')
         
            writer.writerow(["Mean IoU", "Mean Dice","Mean Hausdorff Distance", "Mean Surface Distance"])
            writer.writerow([avg_iou_rv_ed_str, avg_dice_rv_ed_str, avg_hd_rv_ed_str, avg_surface_rv_ed_str])
        
            return iou_list_rv, dice_list_rv, hausdorff_list_rv, surface_distance_list_rv
        
            print("Perhitungan Nilai Metrik Right Ventricle Sudah Selesai.../n")
        
    def hitung_evaluasi_metrik_lv(imgs, masks):
    
        iou_list_lv = []
        dice_list_lv = []
        # mcc_list_lv = []
        # f1_score_list_lv = []
        hausdorff_list_lv = []
        surface_distance_list_lv = []
    
        with open(simpan_csv + f'metrik_evaluasi_transunet_lv_ed_Pasien {i}_acdc2017.csv', mode='w', newline='') as file:
            writer = csv.writer(file, delimiter=';')
            writer.writerow(['Citra Ke', 'IoU', 'Dice','Hausdorff Distance', 'Surface Distance'])
            
            if len(imgs) != len(masks):
                print("Jumlah citra predict dan mask tidak sama. Proses evaluasi tidak dilanjutkan.")
                return
        
        
            for predict_filename, mask_filename in zip(imgs, masks):
                if predict_filename.endswith('.png') and mask_filename.endswith('.png'):
                
                    file_path_predict = os.path.join(direktori_predict_ed_lv, predict_filename)
                    file_path_mask = os.path.join(direktori_gt_ed_lv, mask_filename)
                
                    print(file_path_predict)
                    print(file_path_mask)
                
                    img_predict_lv_ed = cv.imread(file_path_predict, 0)
                    img_gt_lv_ed = cv.imread(file_path_mask, 0)
                
                    # Konversi citra ke format biner (0 dan 1)
                    bin_predict = np.where(img_predict_lv_ed == 255, 1, 0)
                    bin_gt = np.where(img_gt_lv_ed == 255, 1, 0)
                
                
                    iou_lv_ed = iou_2d(bin_gt, bin_predict)
                
                    dice_lv_ed = dice_coefficient(bin_gt, bin_predict)
                
                    # f1_score_lv_ed = f1_score_2d(bin_gt, bin_predict)
                    # mcc_lv_ed = mcc_2d(bin_gt, bin_predict)
                
                    hd_lv_ed = hausdorff_distance(bin_gt, bin_predict, distance='euclidean')
                
                    surface_lv_ed = surfd(bin_gt, bin_predict, sampling=1, connectivity=1)
                    mean_surface_lv_ed = np.mean(surface_lv_ed)
                
                    iou_list_lv.append(iou_lv_ed)
                    dice_list_lv.append(dice_lv_ed)
                    # f1_score_list.append(f1_score_lv_ed)
                    # mcc_list.append(mcc_lv_ed)
                    hausdorff_list_lv.append(hd_lv_ed)
                    surface_distance_list_lv.append(mean_surface_lv_ed)
        
                    iou_lv_ed_str = str(iou_lv_ed).replace('.', ',')
                    dice_lv_ed_str = str(dice_lv_ed).replace('.', ',')
                    # f1_lv_ed_str = str(f1_score_lv_ed).replace('.', ',')
                    # mcc_lv_ed_str = str(mcc_lv_ed).replace('.', ',')
                    hd_lv_ed_str = str(hd_lv_ed).replace('.', ',')
                    mean_surface_lv_ed_str = str(mean_surface_lv_ed).replace('.', ',')
                
                
                    # iou += iou_lv_ed
                    # dice += dice_lv_ed
                    # f1_score += f1_score_lv_ed
                    # hausdorff += hd_lv_ed
                    # surface += surface_lv_ed
                
        
        
                    # iou_lv_ed_str = str(iou_lv_ed).replace('.', ',')
                    # dice_lv_ed_str = str(dice_lv_ed).replace('.', ',')
                    # f1_lv_ed_str = str(f1_score_lv_ed).replace('.', ',')
                    # hd_lv_ed_str = str(hd_lv_ed).replace('.', ',')
                    # surface_myo_ed_str = str(surface_lv_ed).replace('.', ',')
        

                    # Menulis hasil ke file CSV
                    writer.writerow([f"citra {predict_filename}", iou_lv_ed_str, dice_lv_ed_str, hd_lv_ed_str, mean_surface_lv_ed_str])

                    print(f"Citra {predict_filename}:")
                    print("Nilai IoU lv ED = ", iou_lv_ed)
                    print("Nilai Dice lv ED = ", dice_lv_ed)
                    # print("Nilai F1 lv ED = ", f1_score_lv_ed)
                    # print("Nilai MCC lv ED = ", mcc_lv_ed)
                    print("Nilai Hausdorff lv ED = ", hd_lv_ed)
                    print("Nilai Surface lv ED = ", mean_surface_lv_ed)
                    print("\n")
                
                    print(f"Citra {predict_filename} sudah dihitung... \n")
        
            avg_iou = np.mean(iou_list_lv)
            avg_dice = np.mean(dice_list_lv)
            # avg_f1 = np.mean(f1_score_list_lv)
            # avg_mcc = np.mean(mcc_list_lv)
            avg_hausdorff = np.mean(hausdorff_list_lv)
            avg_surface = np.mean(surface_distance_list_lv)
        
            avg_iou_lv_ed_str = str(avg_iou).replace('.', ',')
            avg_dice_lv_ed_str = str(avg_dice).replace('.', ',')
            # avg_f1_lv_ed_str = str(avg_f1).replace('.', ',')
            # avg_mcc_lv_ed_str = str(avg_mcc).replace('.', ',')
            avg_hd_lv_ed_str = str(avg_hausdorff).replace('.', ',')
            avg_surface_lv_ed_str = str(avg_surface).replace('.', ',')
         
            writer.writerow(["Mean IoU", "Mean Dice","Mean Hausdorff Distance", "Mean Surface Distance"])
            writer.writerow([avg_iou_lv_ed_str, avg_dice_lv_ed_str, avg_hd_lv_ed_str, avg_surface_lv_ed_str])
        
            return iou_list_lv, dice_list_lv, hausdorff_list_lv, surface_distance_list_lv
        
            print("Perhitungan Nilai Metrik Left Ventricle Sudah Selesai...\n")
    
    def plot_visualize(iou_list_myo, dice_list_myo, hausdorff_list_myo, surface_list_myo, iou_list_rv, dice_list_rv, hausdorff_list_rv, surface_list_rv, iou_list_lv, dice_list_lv, hausdorff_list_lv, surface_list_lv) :
        
        fig, ax = plt.subplots()
        
        boxplot = ax.boxplot([iou_list_myo, iou_list_rv, iou_list_lv], labels=['Myocardium', 'Right Ventricle', 'Left Ventricle'], patch_artist=True)
    
        # Mengatur warna untuk box plot
        colors = ['red', 'yellow', 'green']
        for patch, color in zip(boxplot['boxes'], colors):
            patch.set_facecolor(color)
    
        ax.set_title(f'Mean IoU ACDC 2017')
        ax.set_ylabel('Score')
        plt.savefig(os.path.join(simpan_csv, f'boxplot_mean_iou_pasien {i}_acdc2017.png'))
        plt.show()
        plt.close()
        
        fig, ax = plt.subplots()
        
        boxplot = ax.boxplot([dice_list_myo, dice_list_rv, dice_list_lv], labels=['Myocardium', 'Right Ventricle', 'Left Ventricle'], patch_artist=True)
    
        # Mengatur warna untuk box plot
        colors = ['red', 'yellow', 'green']
        for patch, color in zip(boxplot['boxes'], colors):
            patch.set_facecolor(color)
    
        ax.set_title(f'Mean Dice ACDC 2017')
        ax.set_ylabel('Score')
        plt.savefig(os.path.join(simpan_csv, f'boxplot_mean_dice_pasien {i}_acdc2017.png'))
        plt.show()
        plt.close()
        
        fig, ax = plt.subplots()
        
        boxplot = ax.boxplot([hausdorff_list_myo, hausdorff_list_rv, hausdorff_list_lv], labels=['Myocardium', 'Right Ventricle', 'Left Ventricle'], patch_artist=True)
    
        # Mengatur warna untuk box plot
        colors = ['red', 'yellow', 'green']
        for patch, color in zip(boxplot['boxes'], colors):
            patch.set_facecolor(color)
    
        ax.set_title(f'Mean Hausdorff Distance ACDC 2017')
        ax.set_ylabel('Score')
        plt.savefig(os.path.join(simpan_csv, f'boxplot_mean_hd_pasien {i}_acdc2017.png'))
        plt.show()
        plt.close()
        
        fig, ax = plt.subplots()
        
        boxplot = ax.boxplot([surface_list_myo, surface_list_rv, surface_list_lv], labels=['Myocardium', 'Right Ventricle', 'Left Ventricle'], patch_artist=True)
    
        # Mengatur warna untuk box plot
        colors = ['red', 'yellow', 'green']
        for patch, color in zip(boxplot['boxes'], colors):
            patch.set_facecolor(color)
    
        ax.set_title(f'Mean Surface Distance ACDC 2017')
        ax.set_ylabel('Score')
        plt.savefig(os.path.join(simpan_csv, f'boxplot_mean_sd_pasien {i}_acdc2017.png'))
        plt.show()
        plt.close()
        
    
        
        # fig, ax = plt.subplots()
        
        # boxplot = ax.boxplot([hausdorff_list, surface_distance_list], labels=['Mean Hausdorff Distance', 'Mean Surface Distance'], patch_artist=True)
    
        # # Mengatur warna untuk box plot
        # colors = ['magenta', 'cyan']
        # for patch, color in zip(boxplot['boxes'], colors):
            #     patch.set_facecolor(color)
    
        #     ax.set_title('Evaluation Metrics (Distance Metrics)')
        #     ax.set_ylabel('Score')
        #     plt.savefig(os.path.join(simpan_csv, f'boxplot_distance metrics_pasien {i}_myocardium.png'))
        #     plt.show()
        #     plt.close()
        
        print("Visualisasi Selesai.... \n")
    
    def main():
    
        get_masks_myo = os.listdir(direktori_gt_ed_myo)
        sorted_masks_myo = sorted(get_masks_myo, key=lambda x: x.lower())
    
        get_masks_rv = os.listdir(direktori_gt_ed_rv)
        sorted_masks_rv = sorted(get_masks_rv, key=lambda x: x.lower())
    
        get_masks_lv = os.listdir(direktori_gt_ed_lv)
        sorted_masks_lv = sorted(get_masks_lv, key=lambda x: x.lower())
    
        get_predict_myo = os.listdir(direktori_predict_ed_myo)
        sorted_predicts_myo = sorted(get_predict_myo, key=lambda x: x.lower())
    
        get_predict_rv = os.listdir(direktori_predict_ed_rv)
        sorted_predicts_rv = sorted(get_predict_rv, key=lambda x: x.lower())
    
        get_predict_lv = os.listdir(direktori_predict_ed_lv)
        sorted_predicts_lv = sorted(get_predict_lv, key=lambda x: x.lower())
    
        iou_list_myo, dice_list_myo, hausdorff_list_myo, surface_list_myo = hitung_evaluasi_metrik_myo(sorted_predicts_myo, sorted_masks_myo)
        iou_list_rv, dice_list_rv, hausdorff_list_rv, surface_list_rv = hitung_evaluasi_metrik_rv(sorted_predicts_rv, sorted_masks_rv)
        iou_list_lv, dice_list_lv, hausdorff_list_lv, surface_list_lv = hitung_evaluasi_metrik_lv(sorted_predicts_lv, sorted_masks_lv)
    
        plot_visualize(iou_list_myo, dice_list_myo, hausdorff_list_myo, surface_list_myo, iou_list_rv, dice_list_rv, hausdorff_list_rv, surface_list_rv, iou_list_lv, dice_list_lv, hausdorff_list_lv, surface_list_lv)
    
    
    if __name__ == "__main__":
        main()
    
    
      
    