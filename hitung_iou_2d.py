# -*- coding: utf-8 -*-
"""
Created on Sun Jan 28 15:47:03 2024

@author: ramad
"""

from keras.metrics import MeanIoU
import numpy as np


def iou_2d(groundtruth, predict):
    
    
    # Ubah nilai dalam gambar
    # groundtruth = np.where(groundtruth == 255, 1, 0)
    # predict = np.where(predict == 255, 1, 0)

    # Inisialisasi hitungan untuk gambar saat ini
    tp = 0
    fp = 0
    fn = 0

    # Hitung tp, fp, dan fn untuk setiap piksel
    # for i in range(groundtruth.shape[0]):
    #     for j in range(groundtruth.shape[1]):
    #         if groundtruth[i][j] == 1 and predict[i][j] == 1:
    #             tp += 1
    #         elif groundtruth[i][j] == 0 and predict[i][j] == 1:
    #             fp += 1
    #         elif groundtruth[i][j] == 1 and predict[i][j] == 0:
    #             fn += 1
    print(groundtruth.shape)
    print(predict.shape)
    
    for i in range(groundtruth.shape[0]):
        for j in range(groundtruth.shape[1]):
            try:
                if groundtruth[i][j] == 1 and predict[i][j] == 1:
                    tp += 1
                elif groundtruth[i][j] == 0 and predict[i][j] == 1:
                    fp += 1
                elif groundtruth[i][j] == 1 and predict[i][j] == 0:
                    fn += 1
            except IndexError as e:
                print(f"Error at index ({i}, {j}): {e}")
                break

    # Perhitungan IoU untuk gambar saat ini
    if tp + fp + fn > 0:
        iou = tp / (tp + fp + fn)
    else:
        iou = 0
        
    return iou
    
    
    # intersection = np.logical_and(groundtruth, predict)
    # union = np.logical_or(groundtruth, predict)
    # # Kondisi untuk menghindari pembagian 0
    # if np.sum(union) == 0:
    #    iou = 0
    # else:
    #    iou = np.sum(intersection) / np.sum(union)
    
    # return iou
    # Inisialisasi hitungan untuk gambar saat ini

    