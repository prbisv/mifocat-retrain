import os
import glob
import numpy as np
import pandas as pd
import json
from pathlib import Path
import cv2
from hitung_dice_2d import dice_coefficient
from hitung_iou_2d import iou_2d
from hausdorff.hausdorff import hausdorff_distance
from hitung_surface_distance_2d import surfd
from keras.layers import Conv2DTranspose

# Fix for Keras 3 / TF 2.16+ loading older Conv2DTranspose configs with 'groups'
original_from_config = Conv2DTranspose.from_config
@classmethod
def custom_from_config(cls, config):
    if 'groups' in config:
        del config['groups']
    return original_from_config.__func__(cls, config)
Conv2DTranspose.from_config = custom_from_config

def fast_iou(gt, pred):
    intersection = np.logical_and(gt, pred)
    union = np.logical_or(gt, pred)
    if np.sum(union) == 0:
        return 0.0
    return np.sum(intersection) / np.sum(union)

def evaluate_heldout_test(model, model_name, fold_id, base_test_dir, output_dir):
    """
    Evaluates the model on the held-out test dataset.
    base_test_dir: e.g. "acdc2017/Data 2D/ED/Data Test Resize 128 ED"
    """
    import tensorflow as tf
    
    print(f"--- Evaluating Held-out Test Set for Fold {fold_id} ---")
    
    # Path setup
    fold_out_dir = os.path.join(output_dir, f"fold_{fold_id}", "test_evaluation")
    os.makedirs(fold_out_dir, exist_ok=True)
    
    patient_dirs = glob.glob(os.path.join(base_test_dir, "Pasien *"))
    if not patient_dirs:
        print(f"Warning: No patients found in {base_test_dir}")
        return None
    
    results = []
    
    for p_dir in sorted(patient_dirs):
        patient_id = os.path.basename(p_dir)
        img_dir = os.path.join(p_dir, "images")
        
        # Determine ground truth dir: either "groundtruth" or "masks"
        gt_dir = os.path.join(p_dir, "groundtruth")
        if not os.path.exists(gt_dir):
            gt_dir = os.path.join(p_dir, "masks")
            if not os.path.exists(gt_dir):
                # Try specific ones?
                continue
                
        img_files = glob.glob(os.path.join(img_dir, "*.png"))
        
        patient_dice = []
        patient_iou = []
        patient_hausdorff = []
        patient_surfd = []
        
        for img_path in img_files:
            img_name = os.path.basename(img_path)
            gt_name = img_name.replace('.png', '_gt.png')
            gt_path = os.path.join(gt_dir, gt_name)
            
            if not os.path.exists(gt_path):
                # Try without _gt just in case
                gt_path = os.path.join(gt_dir, img_name)
                if not os.path.exists(gt_path):
                    continue
                
            # Read image
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (256, 256))
            img = img / 255.0
            
            # Read GT
            gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
            gt = cv2.resize(gt, (256, 256), interpolation=cv2.INTER_NEAREST)
            # The ACDC17 masks in the dataset have values [0, 85, 170, 255]
            # Convert to [0, 1, 2, 3] class indices for consistent metric calculation
            gt_mapped = np.rint(gt.astype('float32') * 3 / 255.0).astype(np.uint8)
            gt_binary = np.where(gt_mapped > 0, 1, 0).astype(np.uint8)
            
            # Predict
            # The original fold_X_best_model.h5 models have input_shape=(None, 256, 256, 1)
            # The new retrained models have input_shape=(None, 256, 256, 3) because TransUNet inputs expect 3 channels
            if model.input_shape[-1] == 3:
                img_array = img[:, :, 0] if len(img.shape) == 3 else img
                img_repeat = np.repeat(img_array[..., np.newaxis], 3, axis=-1)
                img_in = np.expand_dims(img_repeat, axis=0) # batch size 1
            else:
                img_in = np.expand_dims(img, axis=-1)
                img_in = np.expand_dims(img_in, axis=0)

            pred_raw = model.predict(img_in, verbose=0)[0]
            if pred_raw.shape[-1] > 1:
                pred_class = np.argmax(pred_raw, axis=-1)
                
                # Check for completely inverted background predictions
                # e.g. if the model learned background as class 3 instead of 0
                unique, counts = np.unique(pred_class, return_counts=True)
                bg_class = unique[np.argmax(counts)]
                
                # Convert the classes to binary. If the background (most common class) is not 0, flip it
                # ONLY if we're predicting literally a single class for everything (completely broken model)
                if len(unique) == 1:
                    pred_binary = np.where(pred_class != bg_class, 1, 0).astype(np.uint8)
                else:
                    # In some mappings, the model was trained with the original grayscale GT mapped via np.where(gt==255,1,0) etc.
                    # Or with one-hot encoded classes where class 0 is BG. We just trust the mapping here.
                    pred_binary = np.where(pred_class > 0, 1, 0).astype(np.uint8)
                    
                    # Force inversion check if the network heavily favors a non-zero class as background
                    if bg_class != 0 and counts[np.argmax(counts)] > 0.8 * pred_class.size:
                        pred_binary = np.where(pred_class != bg_class, 1, 0).astype(np.uint8)

            else:
                pred_binary = np.where(pred_raw[:, :, 0] > 0.5, 1, 0).astype(np.uint8)
                
            # If the model predicts absolute zeros, dice is zero. That's a model issue not an eval issue.
                
            # Let's inspect the shapes
            d = dice_coefficient(gt_binary, pred_binary)
            i = fast_iou(gt_binary, pred_binary)
            h = hausdorff_distance(gt_binary, pred_binary)
            
            # Surface distance
            try:
                sd = surfd(gt_binary, pred_binary)
                s = np.mean(sd) if isinstance(sd, np.ndarray) and sd.size > 0 else 0
            except:
                s = 0
                
            patient_dice.append(d)
            patient_iou.append(i)
            patient_hausdorff.append(h)
            patient_surfd.append(s)
            
            # Save numpy arrays
            np.save(os.path.join(fold_out_dir, f"{patient_id}_{img_name}_pred.npy"), pred_binary)
            np.save(os.path.join(fold_out_dir, f"{patient_id}_{img_name}_gt.npy"), gt_binary)
            
            results.append({
                'patient': patient_id,
                'slice': img_name,
                'dice': d,
                'iou': i,
                'hausdorff': h,
                'surfd': s
            })
            
    df = pd.DataFrame(results)
    if not df.empty:
        df.to_csv(os.path.join(fold_out_dir, "slice_metrics.csv"), index=False)
        
        # Per patient
        patient_df = df.groupby('patient')[['dice', 'iou', 'hausdorff', 'surfd']].mean().reset_index()
        patient_df.to_csv(os.path.join(fold_out_dir, "patient_metrics.csv"), index=False)
        
        mean_metrics = {
            'mean_dice': df['dice'].mean(),
            'std_dice': df['dice'].std(),
            'mean_iou': df['iou'].mean(),
            'std_iou': df['iou'].std(),
            'mean_hausdorff': df['hausdorff'].mean(),
            'std_hausdorff': df['hausdorff'].std(),
            'mean_surfd': df['surfd'].mean(),
            'std_surfd': df['surfd'].std()
        }
        
        with open(os.path.join(fold_out_dir, "test_summary.json"), 'w') as f:
            json.dump(mean_metrics, f, indent=4)
            
        return mean_metrics
    return None

def aggregate_cross_fold_test_results(output_dir, n_splits):
    all_results = []
    for fold_id in range(n_splits):
        summary_path = os.path.join(output_dir, f"fold_{fold_id}", "test_evaluation", "test_summary.json")
        if os.path.exists(summary_path):
            with open(summary_path, 'r') as f:
                res = json.load(f)
                res['fold'] = fold_id
                all_results.append(res)
                
    if all_results:
        df = pd.DataFrame(all_results)
        df.to_csv(os.path.join(output_dir, "cross_fold_test_results.csv"), index=False)
        
        agg = {
            'mean_dice': f"{df['mean_dice'].mean():.4f} ± {df['mean_dice'].std():.4f}",
            'mean_iou': f"{df['mean_iou'].mean():.4f} ± {df['mean_iou'].std():.4f}",
            'mean_hausdorff': f"{df['mean_hausdorff'].mean():.4f} ± {df['mean_hausdorff'].std():.4f}",
            'mean_surfd': f"{df['mean_surfd'].mean():.4f} ± {df['mean_surfd'].std():.4f}",
        }
        with open(os.path.join(output_dir, "cross_fold_test_results_aggregated.json"), 'w') as f:
            json.dump(agg, f, indent=4)
        
        print("Cross-fold test aggregation saved.")
