# -*- coding: utf-8 -*-
"""
Data Splitting Utility for MIFOCAT Cardiac Segmentation Research

Supports two modes:
1. Simple train/val split (legacy, backward-compatible)
2. Stratified k-fold cross-validation (recommended for robust evaluation)

@author: ramad
Modified for k-fold CV support
"""

import os
import json
import argparse
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
import splitfolders
from sklearn.model_selection import StratifiedKFold


class CardiacDataSplitter:
    """
    Handles both simple and k-fold splitting strategies for cardiac MRI data.
    
    Patient-level stratification ensures all 2D slices of a patient remain in same fold,
    preventing data leakage across train/val/test splits.
    """
    
    def __init__(self, input_folder: str, output_folder: str, seed: int = 42):
        """
        Initialize splitter with input/output paths.
        
        Args:
            input_folder: Root directory containing patient data (assumes 2D PNG structure)
            output_folder: Directory to save split results
            seed: Random seed for reproducibility
        """
        self.input_folder = Path(input_folder)
        self.output_folder = Path(output_folder)
        self.seed = seed
        os.makedirs(self.output_folder, exist_ok=True)
    
    def simple_split(self, ratio: Tuple[float, float] = (0.90, 0.10)) -> None:
        """
        Legacy mode: Simple train/validation split using splitfolders library.
        
        Args:
            ratio: Tuple of (train_ratio, val_ratio), e.g., (0.90, 0.10)
        """
        print(f"[SIMPLE SPLIT] Input: {self.input_folder}")
        print(f"[SIMPLE SPLIT] Output: {self.output_folder}")
        print(f"[SIMPLE SPLIT] Ratio: {ratio}")
        
        splitfolders.ratio(
            str(self.input_folder),
            output=str(self.output_folder),
            seed=self.seed,
            ratio=ratio,
            group_prefix=None
        )
        print("[SIMPLE SPLIT] ✓ Split complete")
    
    def extract_patient_ids(self) -> List[str]:
        """
        Extract unique patient IDs from input folder structure.
        
        Assumes folder structure: input_folder/Pasien <id>/images/ or similar
        Extracts numeric patient IDs from folder names.
        
        Returns:
            Sorted list of unique patient IDs (strings)
        """
        patient_ids = set()
        for item in self.input_folder.iterdir():
            if item.is_dir():
                # Try to extract patient ID from "Pasien <id>" or similar naming
                parts = item.name.split()
                if len(parts) >= 2 and parts[0].lower() in ['pasien', 'patient', 'pat']:
                    try:
                        patient_id = parts[1]
                        patient_ids.add(patient_id)
                    except (IndexError, ValueError):
                        pass
        
        return sorted(list(patient_ids))
    
    def get_stratification_labels(self, patient_ids: List[str]) -> np.ndarray:
        """
        Generate stratification labels for k-fold split.
        
        Currently uses patient ID hash % 3 to preserve rough class balance.
        In production, integrate with pathology info from Info.cfg if available.
        
        Args:
            patient_ids: List of patient IDs
            
        Returns:
            Array of stratification labels (0, 1, or 2)
        """
        labels = np.array([hash(pid) % 3 for pid in patient_ids])
        return labels
    
    def kfold_split(self, n_splits: int = 5, val_ratio: float = 0.15) -> None:
        """
        Stratified k-fold split at patient level.
        
        Generates k folds, each with:
        - Train set: (100 - val_ratio*100)% of patients
        - Val set: val_ratio*100% of patients
        - Metadata saved to JSON for reproducibility
        
        Args:
            n_splits: Number of folds (default 5 for ~150 patients → ~30 per fold)
            val_ratio: Fraction of training data reserved for validation per fold (default 0.15)
        """
        patient_ids = self.extract_patient_ids()
        
        if not patient_ids:
            print("[K-FOLD] ✗ No patients found in input folder")
            return
        
        print(f"[K-FOLD] Found {len(patient_ids)} patients: {patient_ids}")
        print(f"[K-FOLD] Creating {n_splits}-fold split with val_ratio={val_ratio}")
        
        stratification_labels = self.get_stratification_labels(patient_ids)
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.seed)
        
        fold_metadata = {
            "n_splits": n_splits,
            "seed": self.seed,
            "val_ratio": val_ratio,
            "total_patients": len(patient_ids),
            "folds": []
        }
        
        for fold_idx, (train_test_idx, val_idx) in enumerate(skf.split(patient_ids, stratification_labels)):
            train_test_patients = [patient_ids[i] for i in train_test_idx]
            val_patients = [patient_ids[i] for i in val_idx]
            
            # Further split train_test into train and test (or keep external test)
            n_val_train = max(1, int(len(train_test_patients) * val_ratio))
            train_patients = train_test_patients[:-n_val_train]
            test_patients = train_test_patients[-n_val_train:]
            
            fold_info = {
                "fold_id": fold_idx,
                "train": train_patients,
                "val": val_patients,
                "test": test_patients,
                "train_count": len(train_patients),
                "val_count": len(val_patients),
                "test_count": len(test_patients)
            }
            fold_metadata["folds"].append(fold_info)
            
            print(f"[K-FOLD] Fold {fold_idx}: "
                  f"train={len(train_patients)}, val={len(val_patients)}, test={len(test_patients)} patients")
        
        # Save metadata
        metadata_path = self.output_folder / "kfold_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(fold_metadata, f, indent=2)
        
        print(f"[K-FOLD] ✓ Metadata saved to {metadata_path}")
    
    def create_fold_directories(self, n_splits: int = 5) -> None:
        """
        Create fold-specific directory structure from metadata.
        
        Assumes kfold_metadata.json exists; creates:
        - fold_0/train/, fold_0/val/, fold_0/test/
        - fold_1/train/, fold_1/val/, fold_1/test/
        - ...
        
        Args:
            n_splits: Number of folds (must match metadata)
        """
        metadata_path = self.output_folder / "kfold_metadata.json"
        
        if not metadata_path.exists():
            print(f"[CREATE-FOLD-DIRS] ✗ Metadata not found: {metadata_path}")
            return
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        print(f"[CREATE-FOLD-DIRS] Creating directories for {len(metadata['folds'])} folds")
        
        for fold_info in metadata['folds']:
            fold_id = fold_info['fold_id']
            fold_dir = self.output_folder / f"fold_{fold_id}"
            
            for split in ['train', 'val', 'test']:
                split_dir = fold_dir / split
                os.makedirs(split_dir, exist_ok=True)
            
            print(f"[CREATE-FOLD-DIRS] ✓ Created fold_{fold_id}/ with train/, val/, test/")


def main():
    """Command-line interface for data splitting."""
    parser = argparse.ArgumentParser(
        description="Split cardiac MRI data for training (simple or k-fold)"
    )
    parser.add_argument(
        '--input',
        type=str,
        default='D:/Intelligent Multimedia Network/Research/Riset Pak Dedi/Dataset Olah/Dataset Olah Pertama/',
        help='Input folder with patient data'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='D:/Intelligent Multimedia Network/Research/Riset Pak Dedi/Dataset Olah/Dataset Olah Kedua/Citra Asli/',
        help='Output folder for split results'
    )
    parser.add_argument(
        '--mode',
        type=str,
        choices=['simple', 'kfold'],
        default='simple',
        help='Split mode: simple (90/10) or kfold (k-fold CV)'
    )
    parser.add_argument(
        '--n-splits',
        type=int,
        default=5,
        help='Number of folds for k-fold mode (default 5)'
    )
    parser.add_argument(
        '--val-ratio',
        type=float,
        default=0.15,
        help='Validation ratio within training set for k-fold (default 0.15)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    parser.add_argument(
        '--create-dirs',
        action='store_true',
        help='Create fold directories (only for k-fold mode, requires metadata.json)'
    )
    
    args = parser.parse_args()
    
    splitter = CardiacDataSplitter(args.input, args.output, seed=args.seed)
    
    if args.mode == 'simple':
        splitter.simple_split(ratio=(0.90, 0.10))
    elif args.mode == 'kfold':
        splitter.kfold_split(n_splits=args.n_splits, val_ratio=args.val_ratio)
        if args.create_dirs:
            splitter.create_fold_directories(n_splits=args.n_splits)


if __name__ == '__main__':
    main()