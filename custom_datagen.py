# -*- coding: utf-8 -*-
"""
Custom Data Generators for MIFOCAT Cardiac Segmentation

Supports:
1. Simple .npy loading (legacy)
2. Fold-aware loading for k-fold cross-validation

@author: ramad
Modified for k-fold CV support
"""

import os
import json
from pathlib import Path
from typing import List, Tuple, Optional, Generator
import numpy as np

import cv2

from math import ceil

try:
    import tensorflow as tf
    from tensorflow import keras
    from proposed_model import mifocat_components
except ImportError:
    tf = None
    keras = None


def load_img(img_dir: str, img_list: List[str], target_size: Tuple[int, int] = (256, 256),
             is_mask: bool = False, num_classes: int = 4) -> np.ndarray:
    """
    Load images from directory or full paths (supports .npy and .png formats).
    All images are resized to target_size for consistent batch shapes.
    
    Args:
        img_dir: Base directory path (can be None/empty if img_list contains full paths)
        img_list: List of image filenames or full paths
        target_size: Tuple of (height, width) to resize all images to. Default (256, 256)
        is_mask: If True, converts to one-hot encoding for segmentation masks
        num_classes: Number of classes for one-hot encoding (only used if is_mask=True)
        
    Returns:
        Stacked numpy array of shape:
        - Images: (N, height, width, 1)
        - Masks: (N, height, width, num_classes) if is_mask=True
    """
    
    images = []
    for i, image_name in enumerate(img_list):
        ext = image_name.split('.')[-1].lower()
        
        # Check if image_name is already a full path
        if os.path.isabs(image_name):
            file_path = image_name
        else:
            file_path = os.path.join(img_dir, image_name)

        try:
            if ext == 'npy':
                image = np.load(file_path)
                # Resize if necessary
                if image.shape[:2] != target_size:
                    image = cv2.resize(image, (target_size[1], target_size[0]), 
                                     interpolation=cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR)
            elif ext in ['png', 'jpg', 'jpeg']:
                # For PNG/JPG, use cv2 for reading
                image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                if image is None:
                    print(f"✗ Warning: Could not load {file_path}")
                    continue
                # Resize to target size
                image = cv2.resize(image, (target_size[1], target_size[0]), 
                                 interpolation=cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR)
                # Normalize to [0, 1] range for images, keep integer for masks
                if not is_mask:
                    image = image.astype('float32') / 255.0
            else:
                continue
            
            images.append(image)
        except Exception as e:
            print(f"✗ Error loading {file_path}: {e}")
            continue
    
    if not images:
        raise ValueError(f"No images were loaded from {img_list}")
    
    # Stack images
    stacked = np.array(images)
    
    if is_mask:
        # Convert masks to one-hot encoding
        # PNG exports commonly encode four classes as 0, 85, 170, 255,
        # while NumPy masks may already contain class IDs 0, 1, 2, 3.
        if stacked.max() > num_classes - 1:
            stacked = np.rint(
                stacked.astype('float32') * (num_classes - 1) / 255.0
            ).astype('int32')
        # Shape: (N, H, W) -> (N, H, W, num_classes)
        one_hot = np.zeros((stacked.shape[0], stacked.shape[1], stacked.shape[2], num_classes), dtype='float32')
        for c in range(num_classes):
            one_hot[..., c] = (stacked == c).astype('float32')
        return one_hot
    else:
        # Add channel dimension for grayscale images
        # Shape: (N, H, W) -> (N, H, W, 1)
        return np.expand_dims(stacked, axis=-1)


def imageLoader(img_dir: str, img_list: List[str], 
                mask_dir: str, mask_list: List[str], 
                batch_size: int, target_size: Tuple[int, int] = (256, 256),
                num_classes: int = 4) -> Generator:
    """
    Legacy infinite batch generator for Keras training.
    
    Args:
        img_dir: Directory containing image .npy files
        img_list: List of image filenames
        mask_dir: Directory containing mask .npy files
        mask_list: List of mask filenames
        batch_size: Batch size for yielding
        target_size: Tuple of (height, width) to resize images to
        num_classes: Number of segmentation classes for one-hot encoding
        
    Yields:
        Tuple (X, Y) of batch-sized numpy arrays
        X: (batch, height, width, 1)
        Y: (batch, height, width, num_classes)
    """
    L = len(img_list)
    
    # Keras needs infinite generator
    while True:
        batch_start = 0
        batch_end = batch_size

        while batch_start < L:
            limit = min(batch_end, L)
            
            X = load_img(img_dir, img_list[batch_start:limit], target_size=target_size, is_mask=False)
            Y = load_img(mask_dir, mask_list[batch_start:limit], target_size=target_size, is_mask=True, num_classes=num_classes)

            yield (X, Y)
            
            batch_start += batch_size   
            batch_end += batch_size


class FoldAwareDataLoader:
    """
    Fold-aware data loader for k-fold cross-validation.
    
    Filters patient-level data based on fold metadata, ensuring all slices
    of a patient remain in the same split (train/val/test).
    """
    
    def __init__(self, base_dir: str, fold_metadata_path: str):
        """
        Initialize fold-aware loader.
        
        Args:
            base_dir: Base directory containing 2D sliced patient data
                     (assumes structure: base_dir/Pasien <id>/images/)
            fold_metadata_path: Path to kfold_metadata.json from split_data.py
        """
        self.base_dir = Path(base_dir)
        self.fold_metadata_path = Path(fold_metadata_path)
        self.metadata = self._load_metadata()
    
    def _load_metadata(self) -> dict:
        """Load k-fold metadata JSON."""
        if not self.fold_metadata_path.exists():
            raise FileNotFoundError(f"Metadata not found: {self.fold_metadata_path}")
        
        with open(self.fold_metadata_path, 'r') as f:
            return json.load(f)
    
    def get_fold_patients(self, fold_id: int, split: str = 'train') -> List[str]:
        """
        Get list of patient IDs for a specific fold and split.
        
        Args:
            fold_id: Fold index (0 to n_splits-1)
            split: 'train', 'val', or 'test'
            
        Returns:
            List of patient ID strings
        """
        if fold_id >= len(self.metadata['folds']):
            raise ValueError(f"fold_id {fold_id} exceeds n_splits={self.metadata['n_splits']}")
        
        fold_info = self.metadata['folds'][fold_id]
        
        if split not in fold_info:
            raise ValueError(f"split '{split}' not in fold_info. Available: {fold_info.keys()}")
        
        return fold_info[split]
    
    def get_file_list(self, patient_ids: List[str], 
                     image_subdir: str = 'images') -> Tuple[List[str], str]:
        """
        Get list of image filenames for given patients.
        
        Assumes folder structure: base_dir/Pasien <id>/<image_subdir>/*.(npy|png|jpg)
        
        Args:
            patient_ids: List of patient IDs
            image_subdir: Subdirectory name within patient folder (default 'images')
            
        Returns:
            Tuple of (file_list, directory_path) for use with load_img()
        """
        print(f"[FoldAwareDataLoader.get_file_list] Starting file list collection")
        print(f"[FoldAwareDataLoader.get_file_list] Base directory: {self.base_dir}")
        print(f"[FoldAwareDataLoader.get_file_list] Base directory exists: {self.base_dir.exists()}")
        print(f"[FoldAwareDataLoader.get_file_list] Number of patient IDs to process: {len(patient_ids)}")
        print(f"[FoldAwareDataLoader.get_file_list] First few patient IDs: {patient_ids[:3]}")
        
        # List what's actually in base_dir
        if self.base_dir.exists():
            print(f"[FoldAwareDataLoader.get_file_list] Contents of base directory:")
            for item in sorted(self.base_dir.iterdir())[:10]:
                print(f"  - {item.name} {'(dir)' if item.is_dir() else ''}")
        
        file_list = []
        patient_dirs = {}  # Map patient_id -> directory path
        
        for pid in patient_ids:
            patient_dir = self.base_dir / f"Pasien {pid}" / image_subdir
            
            if not patient_dir.exists():
                # Check what does exist for this patient
                patient_base = self.base_dir / f"Pasien {pid}"
                if patient_base.exists():
                    print(f"[FoldAwareDataLoader] Patient dir exists: {patient_base}, but missing subdir '{image_subdir}'")
                    print(f"[FoldAwareDataLoader] Contents: {list(patient_base.iterdir())}")
                else:
                    print(f"[FoldAwareDataLoader] ✗ Patient directory not found: {patient_dir}")
                    print(f"[FoldAwareDataLoader]   Expected pattern: {self.base_dir}/Pasien {pid}/{image_subdir}")
                continue
            
            patient_dirs[pid] = patient_dir
            
            # Get all supported image files in this patient's directory
            # Store FULL PATHS, not just filenames
            for ext in ['*.npy', '*.png', '*.jpg', '*.jpeg']:
                files = sorted([str(f) for f in patient_dir.glob(ext)])  # Full path
                file_list.extend(files)
        
        # Return the base directory (will be used to extract relative paths if needed)
        if patient_dirs:
            print(f"[FoldAwareDataLoader.get_file_list] ✓ Found {len(patient_dirs)} valid patients, {len(file_list)} total files")
            return file_list, str(self.base_dir)
        else:
            raise ValueError(f"✗ CRITICAL: No valid patient IDs found! Checked {len(patient_ids)} patient IDs in {self.base_dir}")

    def get_paired_file_list(self, patient_ids: List[str],
                            image_subdir: str = 'images',
                            mask_subdir: str = 'groundtruth') -> Tuple[List[str], List[str]]:
        """
        Get image/mask file paths matched by filename, not by directory glob
        order. In this dataset each patient's `images/` and `groundtruth/`
        folders commonly hold different slice counts (e.g. `Pasien1_5_ed.png`
        with no `Pasien1_5_ed_gt.png`, or vice versa) — pairing by index
        position (as get_file_list()'s two independent calls implicitly do
        when zipped downstream) silently mismatches images with the wrong
        mask, or desyncs batch sizes entirely once one list runs out first,
        which surfaces as a "Invalid input shapes" Keras crash mid-epoch.
        Matching by stem, and only keeping slices present in both folders,
        is the only way to keep X/Y aligned.

        Returns:
            Tuple of (img_files, mask_files): same length, same order, each
            index is a genuine (image, mask) pair.
        """
        def stem_key(filename: str) -> str:
            name = Path(filename).stem
            if name.lower().endswith('_gt'):
                name = name[:-3]
            return name

        img_files: List[str] = []
        mask_files: List[str] = []
        matched_patients = 0
        dropped_img_only = 0
        dropped_mask_only = 0

        for pid in patient_ids:
            img_dir = self.base_dir / f"Pasien {pid}" / image_subdir
            mask_dir = self.base_dir / f"Pasien {pid}" / mask_subdir
            if not img_dir.exists() or not mask_dir.exists():
                continue

            img_map = {}
            for ext in ('*.npy', '*.png', '*.jpg', '*.jpeg'):
                for f in img_dir.glob(ext):
                    img_map[stem_key(f.name)] = f
            mask_map = {}
            for ext in ('*.npy', '*.png', '*.jpg', '*.jpeg'):
                for f in mask_dir.glob(ext):
                    mask_map[stem_key(f.name)] = f

            common = sorted(set(img_map) & set(mask_map))
            if not common:
                continue

            matched_patients += 1
            dropped_img_only += len(img_map.keys() - mask_map.keys())
            dropped_mask_only += len(mask_map.keys() - img_map.keys())
            for key in common:
                img_files.append(str(img_map[key]))
                mask_files.append(str(mask_map[key]))

        if not img_files:
            raise ValueError(
                f"✗ CRITICAL: No matching image/mask pairs found! Checked {len(patient_ids)} patient IDs in {self.base_dir}"
            )

        print(f"[FoldAwareDataLoader.get_paired_file_list] ✓ {matched_patients} patients, "
              f"{len(img_files)} matched image/mask pairs "
              f"(dropped {dropped_img_only} images with no mask, {dropped_mask_only} masks with no image)")
        return img_files, mask_files

    def get_generators(self, fold_id: int, batch_size: int = 8,
                      image_subdir: str = 'images',
                      mask_subdir: str = 'groundtruth',
                      target_size: Tuple[int, int] = (256, 256)) -> Tuple[Generator, Generator, int, int]:
        """
        Get train and validation generators for a specific fold.
        
        Args:
            fold_id: Fold index
            batch_size: Batch size for generators
            image_subdir: Subdirectory containing images
            mask_subdir: Subdirectory containing masks
            target_size: Tuple of (height, width) to resize all images to
            
        Returns:
            Tuple of (train_generator, val_generator, train_steps, val_steps)
        """
        print(f"\n[FoldAwareDataLoader.get_generators] ===== STARTING FOLD {fold_id} =====")
        print(f"[FoldAwareDataLoader.get_generators] Image subdir: {image_subdir}, Mask subdir: {mask_subdir}")
        print(f"[FoldAwareDataLoader.get_generators] Target size: {target_size}")
        print(f"[FoldAwareDataLoader.get_generators] Base directory: {self.base_dir}")
        
        # Get patient lists
        train_patients = self.get_fold_patients(fold_id, 'train')
        val_patients = self.get_fold_patients(fold_id, 'val')
        
        print(f"[FoldAwareDataLoader] Fold {fold_id}: "
              f"{len(train_patients)} train patients, {len(val_patients)} val patients")
        print(f"[FoldAwareDataLoader] Train patients: {train_patients[:5]}{'...' if len(train_patients) > 5 else ''}")
        print(f"[FoldAwareDataLoader] Val patients: {val_patients[:5]}{'...' if len(val_patients) > 5 else ''}")
        
        # Get matched (image, mask) pairs — same length and order by construction,
        # unlike two independent get_file_list() calls zipped by index.
        train_img_files, train_mask_files = self.get_paired_file_list(train_patients, image_subdir, mask_subdir)
        val_img_files, val_mask_files = self.get_paired_file_list(val_patients, image_subdir, mask_subdir)

        print(f"[FoldAwareDataLoader] Train: {len(train_img_files)} images, {len(train_mask_files)} masks")
        print(f"[FoldAwareDataLoader] Val: {len(val_img_files)} images, {len(val_mask_files)} masks")
        
        # Create generators (pass empty string for directories since files are full paths)
        train_gen = imageLoader(
            "", train_img_files,
            "", train_mask_files,
            batch_size,
            target_size=target_size,
            num_classes=4  # ACDC2017: background, RV, myocardium, LV
        )
        val_gen = imageLoader(
            "", val_img_files,
            "", val_mask_files,
            batch_size,
            target_size=target_size,
            num_classes=4  # ACDC2017: background, RV, myocardium, LV
        )
        
        # Use ceil so we do not silently drop the final partial batch; integer division was collapsing to 1 step
        train_steps = max(1, ceil(len(train_img_files) / float(batch_size)))
        val_steps = max(1, ceil(len(val_img_files) / float(batch_size)))

        return train_gen, val_gen, train_steps, val_steps


class MIFOCATGradientMonitor(keras.callbacks.Callback if keras is not None else object):
    """
    Records the global L2 gradient norm of each MIFOCAT loss component
    (MSE, focal, categorical cross-entropy) on one fixed training batch,
    once per epoch. The batch is fixed at construction time and is
    independent of the generator driving the actual optimizer step, so the
    measurement isolates "how much gradient signal does each component
    produce" from epoch to epoch, on the same inputs.
    """

    def __init__(self, fixed_x: np.ndarray, fixed_y: np.ndarray, output_path,
                 alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.fixed_x = tf.constant(fixed_x)
        self.fixed_y = tf.constant(fixed_y)
        self.alpha = alpha
        self.gamma = gamma
        self.output_path = Path(output_path)
        self.history = []

    def on_epoch_end(self, epoch, logs=None):
        with tf.GradientTape(persistent=True) as tape:
            y_pred = self.model(self.fixed_x, training=True)
            components = mifocat_components(self.fixed_y, y_pred, alpha=self.alpha, gamma=self.gamma)

        record = {'epoch': epoch + 1}
        for name, component_loss in components.items():
            grads = [g for g in tape.gradient(component_loss, self.model.trainable_variables) if g is not None]
            record[name] = float(tf.linalg.global_norm(grads).numpy()) if grads else 0.0
        del tape

        self.history.append(record)
        print(f"[MIFOCATGradientMonitor] Epoch {epoch + 1}: "
              f"mse={record['mse']:.6f} focal={record['focal']:.6f} cat={record['cat']:.6f}")

    def on_train_end(self, logs=None):
        with open(self.output_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        print(f"[MIFOCATGradientMonitor] Saved gradient history: {self.output_path}")