# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 10:55:16 2024

@author: ramad
"""

import SimpleITK as sitk
import os
from multiprocessing import pool
import pickle5 as pickle
import numpy as np
from skimage.transform import resize


def resize_image(image, old_spacing, new_spacing, order=3):
    new_shape = (int(np.round(old_spacing[0]/new_spacing[0]*float(image.shape[0]))),
                 int(np.round(old_spacing[1]/new_spacing[1]*float(image.shape[1]))),
                 int(np.round(old_spacing[2]/new_spacing[2]*float(image.shape[2]))))
    return resize(image, new_shape, order=order, mode='edge')

str_to_ind = {'DCM':0, 'HCM':1, 'MINF':2, 'NOR':3, 'RV':4}
ind_to_str = {}
for k in str_to_ind.keys():
    ind_to_str[str_to_ind[k]] = k
    

def view_patient_raw_data(patient, width=400, height=400):
    import batchviewer
    a = []
    a.append(patient['ed_data'][None])
    a.append(patient['ed_gt'][None])
    a.append(patient['es_data'][None])
    a.append(patient['es_gt'][None])
    batchviewer.view_batch(np.vstack(a), width, height)


def convert_to_one_hot(seg):
    vals = np.unique(seg)
    res = np.zeros([len(vals)] + list(seg.shape), seg.dtype)
    for c in range(len(vals)):
        res[c][seg == c] = 1
    return res


def preprocess_image(itk_image, is_seg=False, spacing_target=(1, 0.5, 0.5), keep_z_spacing=False):
    spacing = np.array(itk_image.GetSpacing())[[2, 1, 0]]
    image = sitk.GetArrayFromImage(itk_image).astype(float)
    if keep_z_spacing:
        spacing_target = list(spacing_target)
        spacing_target[0] = spacing[0]
    if not is_seg:
        order_img = 3
        if not keep_z_spacing:
            order_img = 1
        image = resize_image(image, spacing, spacing_target, order=order_img).astype(np.float32)
        image -= image.mean()
        image /= image.std()
    else:
        tmp = convert_to_one_hot(image)
        vals = np.unique(image)
        results = []
        for i in range(len(tmp)):
            results.append(resize_image(tmp[i].astype(float), spacing, spacing_target, 1)[None])
        image = vals[np.vstack(results).argmax(0)]
    return image

def load_dataset(ids=range(101), root_dir="D:/Intelligent Multimedia Network/Research/Riset Bu Dini/Dataset/acdc17/acdc17/preprocessed dataset 3d/"):
    with open(os.path.join(root_dir, "patient_info.pkl"), 'r') as f:
        patient_info = cPickle.load(f)

    data = {}
    for i in ids:
        if os.path.isfile(os.path.join(root_dir, "pat_%03.0d.npy"%i)):
            a = np.load(os.path.join(root_dir, "pat_%03.0d.npy"%i), mmap_mode='r')
            data[i] = {}
            data[i]['height'] = patient_info[i]['height']
            data[i]['weight'] = patient_info[i]['weight']
            data[i]['pathology'] = patient_info[i]['pathology']
            data[i]['ed_data'] = a[0, :]
            data[i]['ed_gt'] = a[1, :]
            data[i]['es_data'] = a[2, :]
            data[i]['es_gt'] = a[3, :]
    return data

def process_patient(args):
    id, patient_info, folder, folder_out, keep_z_spc = args
    #print id
    # if id in [286, 288]:
    #     return
    patient_folder = os.path.join(folder, "patient%03.0d"%id)
    if not os.path.isdir(patient_folder):
        return
    images = {}

    fname = os.path.join(patient_folder, "patient%03.0d_frame%02.0d.nii.gz" % (id, patient_info[id]['ed']))
    if os.path.isfile(fname):
        images["ed"] = sitk.ReadImage(fname)
    fname = os.path.join(patient_folder, "patient%03.0d_frame%02.0d_gt.nii.gz" % (id, patient_info[id]['ed']))
    if os.path.isfile(fname):
        images["ed_seg"] = sitk.ReadImage(fname)
    fname = os.path.join(patient_folder, "patient%03.0d_frame%02.0d.nii.gz" % (id, patient_info[id]['es']))
    if os.path.isfile(fname):
        images["es"] = sitk.ReadImage(fname)
    fname = os.path.join(patient_folder, "patient%03.0d_frame%02.0d_gt.nii.gz" % (id, patient_info[id]['es']))
    if os.path.isfile(fname):
        images["es_seg"] = sitk.ReadImage(fname)

    print(id, images["es_seg"].GetSpacing())

    for k in images.keys():
        #print k
        images[k] = preprocess_image(images[k], is_seg=(k == "ed_seg" or k == "es_seg"),
                                     spacing_target=(10, 1.25, 1.25), keep_z_spacing=keep_z_spc)

    img_as_list = []
    for k in ['ed', 'ed_seg', 'es', 'es_seg']:
        if k not in images.keys():
            print(id, "has missing key:", k)
        img_as_list.append(images[k][None])
    try:
        all_img = np.vstack(img_as_list)
    except:
        print(id, "has a problem with spacings")
    np.save(os.path.join(folder_out, "pat_%03.0d" % id), all_img.astype(np.float32))
    
    
def generate_patient_info(folder):
    patient_info={}
    for id in range(151):
        fldr = os.path.join(folder, 'patient%03.0d'%id)
        print("Ditemukan Folder", id)
        if not os.path.isdir(fldr):
            print("could not find dir of patient ", id)
            continue
        nfo = np.loadtxt(os.path.join(fldr, "Info.cfg"), dtype=str, delimiter=': ')
        patient_info[id] = {}
        patient_info[id]['ed'] = int(nfo[0, 1])
        patient_info[id]['es'] = int(nfo[1, 1])
        patient_info[id]['height'] = float(nfo[3, 1])
        patient_info[id]['pathology'] = nfo[2, 1]
        patient_info[id]['weight'] = float(nfo[5, 1])
    return patient_info

def run_preprocessing(folder, folder_out, keep_z_spacing=True):
    patient_info = generate_patient_info(folder)

    if not os.path.isdir(folder_out):
        os.mkdir(folder_out)
    with open(os.path.join(folder_out, "patient_info.pkl"), 'wb') as f:  # Ubah 'w' menjadi 'wb'
        pickle.dump(patient_info, f)


    # beware of z spacing!!! see process_patient for more info!
    ids = range(101)
    p = pool.Pool(8)
    p.map(process_patient, zip(ids, [patient_info]*101, [folder]*101, [folder_out]*101, [keep_z_spacing]*101))
    p.close()
    p.join()
    
def main():
    
    folder="D:/Intelligent Multimedia Network/Research/Riset Bu Dini/Dataset/acdc17/acdc17/trainingACDC17/"
    folder_out = "D:/Intelligent Multimedia Network/Research/Riset Bu Dini/Dataset/acdc17/acdc17/ACDC_forReal_orig_Z/"
    
    run_preprocessing(folder, folder_out, True)
    
if __name__ == "__main__":
    main()
    



    
    
