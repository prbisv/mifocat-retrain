# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 21:28:28 2024

@author: ramad
"""

import numpy as np

def dice_coefficient(groundtruth, predict):
    intersection = np.sum(groundtruth * predict)
    
    sum_groundtruth = np.sum(groundtruth)
    sum_predict = np.sum(predict)

    
    if sum_groundtruth + sum_predict == 0:
        return 0  

    # Hitung koefisien Dice
    dice_coeff = (2 * intersection) / (sum_groundtruth + sum_predict)

    
    if np.isnan(dice_coeff):
        return 0  

    return dice_coeff
    
    