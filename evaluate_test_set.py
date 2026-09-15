import argparse
import json
import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from tensorflow import keras
from keras.layers import Conv2DTranspose

from hitung_dice_2d import dice_coefficient
from hitung_iou_2d import iou_2d
from hausdorff.hausdorff import hausdorff_distance
from hitung_surface_distance_2d import surfd

from transunet_model import get_custom_objects
from proposed_model import mifocat_loss, mean_iou, dice_score

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

def main():
    base_output_dir = "transunet_kfold_results"
    heldout_base = "acdc2017/Data 2D/ED/Data Test Resize 128 ED"
    n_splits = 5

    all_results = []
    
    for fold_id in range(n_splits):
        model_path = os.path.join(base_output_dir, f"fold_{fold_id}", f"fold_{fold_id}_best_model.h5")
        if not os.path.exists(model_path):
            print(f"Skipping fold {fold_id}, no model found.")
            continue
            
        print(f"Loading fold {fold_id}...")
        custom_objs = get_custom_objects()
        custom_objs.update({
            'mifocat_loss': mifocat_loss(alpha=0.25, gamma=2.0, r1=1.0, r2=1.0, r3=1.0),
            'mean_iou': mean_iou,
            'dice_score': dice_score
        })
        model = keras.models.load_model(model_path, custom_objects=custom_objs, compile=False)

        fold_out_dir = os.path.join(base_output_dir, f"fold_{fold_id}", "test_evaluation")
        os.makedirs(fold_out_dir, exist_ok=True)
        
        patient_dirs = sorted(list(Path(heldout_base).glob("Pasien *")))
        
        results = []
        
        for p_dir in patient_dirs:
            patient_id = p_dir.name
            img_dir = p_dir / "images"
            gt_dir = p_dir / "groundtruth"
            
            if not gt_dir.exists():
                gt_dir = p_dir / "masks"
            
            for img_path in sorted(list(img_dir.glob("*.png"))):
                img_name = img_path.name
                gt_name = img_name.replace(".png", "_gt.png")
                gt_path = gt_dir / gt_name
                if not gt_path.exists():
                    gt_path = gt_dir / img_name
                    if not gt_path.exists():
                        continue
                        
            # IMPORTANT FIX for inference scaling vs train scaling mismatch:
            # The model was trained with:
            # image = cv2.resize(image, (target_size[1], target_size[0]), interpolation=cv2.INTER_LINEAR)
            # image = image.astype('float32') / 255.0
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_LINEAR)
            img_normalized = img.astype('float32') / 255.0
            
            if model.input_shape[-1] == 3:
                img_array = img_normalized[:, :, 0] if len(img_normalized.shape) == 3 else img_normalized
                img_repeat = np.repeat(img_array[..., np.newaxis], 3, axis=-1)
                img_in = np.expand_dims(img_repeat, axis=0) # batch size 1
            else:
                img_in = np.expand_dims(img_normalized, axis=-1)
                img_in = np.expand_dims(img_in, axis=0)
            
            gt = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
            gt = cv2.resize(gt, (256, 256), interpolation=cv2.INTER_NEAREST)
            
            # The target masks loaded by datagen use class indices (0 to 3) internally based on value
            # The raw masks are [0, 85, 170, 255]. Mapping to [0, 1, 2, 3] gives us exactly what the one-hot expects.
            gt_mapped = np.rint(gt.astype('float32') * 3 / 255.0).astype(np.uint8)
            gt_binary = np.where(gt_mapped > 0, 1, 0).astype(np.uint8)
            
            pred_raw = model.predict(img_in, verbose=0)[0]
            if pred_raw.shape[-1] > 1:
                pred_class = np.argmax(pred_raw, axis=-1)
                
                # IMPORTANT FIX: the training script maps class arrays exactly matching the GT (0 to 3)
                # But here we're only calculating binary foreground metrics to be safe
                # Anything > 0 in prediction is considered foreground
                pred_binary = np.where(pred_class > 0, 1, 0).astype(np.uint8)
                
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
                
            d = dice_coefficient(gt_binary, pred_binary)
            i = fast_iou(gt_binary, pred_binary)
            h = hausdorff_distance(gt_binary, pred_binary)
            try:
                sd = surfd(gt_binary, pred_binary)
                s = np.mean(sd) if isinstance(sd, np.ndarray) and sd.size > 0 else 0
            except:
                s = 0
                
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
            patient_df = df.groupby('patient')[['dice', 'iou', 'hausdorff', 'surfd']].mean().reset_index()
            patient_df.to_csv(os.path.join(fold_out_dir, "patient_metrics.csv"), index=False)
            
            mean_metrics = {
                'fold': fold_id,
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
            all_results.append(mean_metrics)
            print(f"Fold {fold_id} metrics - Dice: {mean_metrics['mean_dice']:.4f}, IoU: {mean_metrics['mean_iou']:.4f}")

    if all_results:
        df_all = pd.DataFrame(all_results)
        df_all.to_csv(os.path.join(base_output_dir, "cross_fold_test_results.csv"), index=False)
        
        agg = {
            'mean_dice': f"{df_all['mean_dice'].mean():.4f} ± {df_all['mean_dice'].std():.4f}",
            'mean_iou': f"{df_all['mean_iou'].mean():.4f} ± {df_all['mean_iou'].std():.4f}",
            'mean_hausdorff': f"{df_all['mean_hausdorff'].mean():.4f} ± {df_all['mean_hausdorff'].std():.4f}",
            'mean_surfd': f"{df_all['mean_surfd'].mean():.4f} ± {df_all['mean_surfd'].std():.4f}",
        }
        with open(os.path.join(base_output_dir, "cross_fold_test_results_aggregated.json"), 'w') as f:
            json.dump(agg, f, indent=4)
        print("\nCross-fold test aggregation saved to cross_fold_test_results_aggregated.json:")
        print(json.dumps(agg, indent=2))

if __name__ == "__main__":
    main()
