# -*- coding: utf-8 -*-
"""
Created on Fri May 17 21:31:12 2024

@author: ramad
"""

import numpy as np
from scipy.ndimage import morphology

def surfd(input1, input2, sampling=1, connectivity=1):
    
    input_1 = np.atleast_1d(input1.astype(bool))
    input_2 = np.atleast_1d(input2.astype(bool))
    
    conn = morphology.generate_binary_structure(input_1.ndim, connectivity)

    S = np.logical_xor(input_1, morphology.binary_erosion(input_1, conn))
    Sprime = np.logical_xor(input_2, morphology.binary_erosion(input_2, conn))
    
    dta = morphology.distance_transform_edt(~S, sampling)
    dtb = morphology.distance_transform_edt(~Sprime, sampling)
    
    sds = np.concatenate([dta[Sprime != 0].ravel(), dtb[S != 0].ravel()])
    
    if sds.size == 0:
        return 0
    
    return sds
