# -*- coding: utf-8 -*-
"""
Created on Fri May 17 21:26:53 2024

@author: ramad
"""

import numpy as np


def f1_score_2d(groundtruth, predict):
    
    # Ubah nilai dalam gambar
    # groundtruth = np.where(groundtruth == 255, 1, 0)
    # predict = np.where(predict == 255, 1, 0)

    # Inisialisasi hitungan untuk gambar saat ini
    tp = 0
    fp = 0
    fn = 0
    
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
        f1_score = tp / (tp + 0.5*(fp + fn))
    else:
        f1_score = 0
        
    return f1_score