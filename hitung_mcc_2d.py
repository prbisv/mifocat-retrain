# -*- coding: utf-8 -*-
"""
Created on Fri May 17 23:21:33 2024

@author: ramad
"""

import numpy as np

def mcc_2d(groundtruth, predict):
    
    # Hitung nilai TP, TN, FP, FN
    tp = np.sum((groundtruth == 1) & (predict == 1)).astype(np.float64)
    tn = np.sum((groundtruth == 0) & (predict == 0)).astype(np.float64)
    fp = np.sum((groundtruth == 0) & (predict == 1)).astype(np.float64)
    fn = np.sum((groundtruth == 1) & (predict == 0)).astype(np.float64)
    
    # Perhitungan MCC
    numerator = (tp * tn) - (fp * fn)
    denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) + 1e-7)  # Tambahkan epsilon kecil untuk mencegah pembagian nol
    
    if denominator == 0:
        mcc = 0
    else:
        mcc = numerator / denominator
    
    return mcc