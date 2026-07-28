# -*- coding: utf-8 -*-
"""
K-Fold Cross-Validation Training Wrapper for MIFOCAT

This script orchestrates k-fold cross-validation training:
1. Loads fold metadata from split_data.py
2. Iterates through each fold
3. Trains model and evaluates on validation set
4. Aggregates metrics across folds for robust performance estimate

Usage:
    python train_kfold_wrapper.py --fold-metadata /path/to/kfold_metadata.json \
        --model unet --n-folds 5 --epochs 50

@author: ramad
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
from datetime import datetime
import sys

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
except ImportError:
    print("Warning: TensorFlow not installed. Metrics aggregation will still work.")
    tf = None

from custom_datagen import FoldAwareDataLoader
from proposed_model import build_unet_mifocat, mifocat_loss, mean_iou, dice_score
from transunet_model import build_transunet_mifocat, get_custom_objects


class KFoldTrainer:
    """
    Orchestrates k-fold cross-validation training and evaluation.
    
    Manages:
    - Fold iteration and data loading
    - Model training per fold
    - Metrics aggregation
    - Checkpoint and log management
    """
    
    def __init__(self, fold_metadata_path: str, base_data_dir: str,
                 output_dir: str = './kfold_results', seed: int = 42):
        """
        Initialize k-fold trainer.
        
        Args:
            fold_metadata_path: Path to kfold_metadata.json
            base_data_dir: Base directory with patient data (e.g., 'Data Test Resize 128 ED/')
            output_dir: Directory to save fold results, checkpoints, logs
            seed: Random seed for reproducibility
        """
        self.fold_metadata_path = Path(fold_metadata_path)
        self.base_data_dir = Path(base_data_dir)
        self.output_dir = Path(output_dir)
        self.seed = seed
        
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Load metadata
        if not self.fold_metadata_path.exists():
            raise FileNotFoundError(f"Metadata not found: {fold_metadata_path}")
        
        with open(self.fold_metadata_path, 'r') as f:
            self.metadata = json.load(f)
        
        self.n_splits = self.metadata['n_splits']
        print(f"[KFoldTrainer] Loaded metadata for {self.n_splits}-fold CV")
        print(f"[KFoldTrainer] Output directory: {self.output_dir}")
        
        # Initialize data loader
        print(f"[KFoldTrainer] Initializing data loader...")
        print(f"[KFoldTrainer] Base data directory: {base_data_dir}")
        print(f"[KFoldTrainer] Base data directory exists: {Path(base_data_dir).exists()}")
        
        if not Path(base_data_dir).exists():
            print(f"[KFoldTrainer] ✗ ERROR: Data directory does not exist!")
            print(f"[KFoldTrainer] Please check the path, especially for spaces in directory names.")
            raise FileNotFoundError(f"Data directory not found: {base_data_dir}")
        
        self.data_loader = FoldAwareDataLoader(str(base_data_dir), str(fold_metadata_path))
        
        # Metrics storage
        self.fold_results = {}
        self.current_model_type = None  # Track current model type for evaluation
    
    def get_model(self, model_type: str = 'unet') -> 'keras.Model':
        """
        Load or create a model for training.
        
        Args:
            model_type: Type of model ('unet', 'trans_unet', etc.)
            
        Returns:
            Compiled Keras model
            
        Note:
            This is a placeholder. In practice, load your model from
            predict_unet_2d.py or predict_trans_unet.py
        """
        if tf is None:
            raise ImportError("TensorFlow required for model training")
        
        if model_type == 'unet':
            print("[KFoldTrainer] Loading U-Net model with MIFOCAT loss...")
            # Build U-Net architecture
            model = build_unet_mifocat(input_shape=(256, 256, 1), num_classes=4)
            
            # Compile with MIFOCAT loss (MSE + Focal + Categorical Cross-Entropy)
            model.compile(
                optimizer=Adam(learning_rate=1e-3),
                loss=mifocat_loss(alpha=0.25, gamma=2.0, r1=1.0, r2=1.0, r3=1.0),
                metrics=['accuracy', mean_iou, dice_score]
            )
            
            print("[KFoldTrainer] Model compiled with MIFOCAT loss")
            return model
        elif model_type == 'transunet' or model_type == 'trans_unet':
            print("[KFoldTrainer] Loading TransUNet model with MIFOCAT loss...")
            # Build TransUNet architecture
            model = build_transunet_mifocat(input_shape=(256, 256, 1), num_classes=4)
            
            # Compile with MIFOCAT loss (MSE + Focal + Categorical Cross-Entropy)
            model.compile(
                optimizer=Adam(learning_rate=1e-3),
                loss=mifocat_loss(alpha=0.25, gamma=2.0, r1=1.0, r2=1.0, r3=1.0),
                metrics=['accuracy', mean_iou, dice_score]
            )
            
            print("[KFoldTrainer] TransUNet model compiled with MIFOCAT loss")
            return model
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
    
    def train_fold(self, fold_id: int, model: 'keras.Model', 
                   epochs: int = 50, batch_size: int = 8,
                   early_stop_patience: int = 10) -> Dict:
        """
        Train and evaluate model on a single fold.
        
        Args:
            fold_id: Fold index (0 to n_splits-1)
            model: Compiled Keras model
            epochs: Maximum number of training epochs
            batch_size: Batch size for training
            early_stop_patience: Early stopping patience (epochs with no improvement)
            
        Returns:
            Dictionary with fold metrics (loss, accuracy, val_loss, val_accuracy, etc.)
        """
        print(f"\n{'='*70}")
        print(f"[FOLD {fold_id}] Starting training")
        print(f"{'='*70}")
        
        fold_dir = self.output_dir / f"fold_{fold_id}"
        os.makedirs(fold_dir, exist_ok=True)
        
        # Get generators
        try:
            print(f"[FOLD {fold_id}] Loading data generators...")
            train_gen, val_gen, train_steps, val_steps = self.data_loader.get_generators(
                fold_id, batch_size=batch_size
            )
            if train_steps == 0 or val_steps == 0:
                print(f"[FOLD {fold_id}] ✗ WARNING: No training or validation steps found")
                print(f"[FOLD {fold_id}]   Training steps: {train_steps}, Validation steps: {val_steps}")
                print(f"[FOLD {fold_id}] This usually means the data directory is empty or structured incorrectly")
                return None
        except ValueError as ve:
            print(f"[FOLD {fold_id}] ✗ CRITICAL: {ve}")
            print(f"[FOLD {fold_id}] This usually means patient directories are not found in the data directory")
            print(f"[FOLD {fold_id}] Expected structure: <DATA_ROOT>/Pasien <patient_id>/images/")
            return None
        except FileNotFoundError as fe:
            print(f"[FOLD {fold_id}] ✗ Data directory error: {fe}")
            print(f"[FOLD {fold_id}] Check if the data path contains spaces and is correctly specified")
            return None
        except Exception as e:
            print(f"[FOLD {fold_id}] ✗ Error loading generators: {e}")
            import traceback
            traceback.print_exc()
            return None
        
        # Set up callbacks
        checkpoint_path = fold_dir / f"fold_{fold_id}_best_model.h5"
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=early_stop_patience,
                restore_best_weights=True,
                verbose=1
            ),
            ModelCheckpoint(
                str(checkpoint_path),
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Train
        print(f"[FOLD {fold_id}] Training steps per epoch: {train_steps}, "
              f"Validation steps: {val_steps}")
        
        try:
            history = model.fit(
                train_gen,
                steps_per_epoch=train_steps,
                validation_data=val_gen,
                validation_steps=val_steps,
                epochs=epochs,
                callbacks=callbacks,
                verbose=1
            )
            
            # Save history
            history_path = fold_dir / f"fold_{fold_id}_history.json"
            history_dict = {
                'loss': [float(v) for v in history.history.get('loss', [])],
                'val_loss': [float(v) for v in history.history.get('val_loss', [])],
                'accuracy': [float(v) for v in history.history.get('accuracy', [])],
                'val_accuracy': [float(v) for v in history.history.get('val_accuracy', [])],
            }
            with open(history_path, 'w') as f:
                json.dump(history_dict, f, indent=2)
            
            fold_metrics = {
                'fold_id': fold_id,
                'final_loss': float(history.history['loss'][-1]),
                'final_val_loss': float(history.history['val_loss'][-1]),
                'best_val_loss': float(np.min(history.history['val_loss'])),
                'epochs_trained': len(history.history['loss']),
                'checkpoint': str(checkpoint_path)
            }
            
            print(f"[FOLD {fold_id}] ✓ Training complete")
            print(f"[FOLD {fold_id}]   Final val_loss: {fold_metrics['final_val_loss']:.6f}")
            print(f"[FOLD {fold_id}]   Best val_loss: {fold_metrics['best_val_loss']:.6f}")
            
            return fold_metrics
            
        except Exception as e:
            print(f"[FOLD {fold_id}] ✗ Training failed: {e}")
            return None
    
    def evaluate_fold(self, fold_id: int, model_path: str, batch_size: int = 8) -> Dict:
        """
        Evaluate model on fold's test set.
        
        Args:
            fold_id: Fold index
            model_path: Path to saved model checkpoint
            batch_size: Batch size for evaluation
            
        Returns:
            Dictionary with evaluation metrics
        """
        if tf is None:
            print(f"[FOLD {fold_id}] Skipping evaluation (TensorFlow not available)")
            return None
        
        print(f"\n[FOLD {fold_id}] Evaluating on test set...")
        
        try:
            # Load model with appropriate custom objects
            if self.current_model_type in ['transunet', 'trans_unet']:
                custom_objs = get_custom_objects()
                # Also include MIFOCAT metrics
                custom_objs.update({
                    'mifocat_loss': mifocat_loss(alpha=0.25, gamma=2.0, r1=1.0, r2=1.0, r3=1.0),
                    'mean_iou': mean_iou,
                    'dice_score': dice_score
                })
                model = keras.models.load_model(model_path, custom_objects=custom_objs, compile=False)
            else:
                model = keras.models.load_model(model_path, compile=False)
            
            # Get test data
            test_patients = self.data_loader.get_fold_patients(fold_id, 'test')
            test_files, test_img_dir = self.data_loader.get_file_list(
                test_patients, image_subdir='images'
            )
            test_mask_dir = str(Path(test_img_dir).parent / 'masks')
            
            if not test_files:
                print(f"[FOLD {fold_id}] No test files found")
                return None
            
            print(f"[FOLD {fold_id}] Test files: {len(test_files)}")
            
            # Evaluate
            from custom_datagen import load_img
            test_X = load_img(test_img_dir, test_files)
            test_Y = load_img(test_mask_dir, test_files)
            
            eval_metrics = model.evaluate(test_X, test_Y, verbose=0)
            
            result = {
                'fold_id': fold_id,
                'test_loss': float(eval_metrics[0]) if isinstance(eval_metrics, list) else float(eval_metrics),
                'test_n_samples': len(test_files)
            }
            
            print(f"[FOLD {fold_id}] ✓ Evaluation complete - test_loss: {result['test_loss']:.6f}")
            
            return result
            
        except Exception as e:
            print(f"[FOLD {fold_id}] ✗ Evaluation failed: {e}")
            return None
    
    def run_all_folds(self, model_type: str = 'unet', epochs: int = 50, 
                     batch_size: int = 8, train_only: bool = False,
                     start_fold: int = 0) -> Dict:
        """
        Run k-fold cross-validation: train and evaluate all folds.
        
        Args:
            model_type: Type of model to use
            epochs: Max epochs per fold
            batch_size: Batch size for training
            train_only: If True, only train (skip evaluation)
            start_fold: Skip all folds with index < start_fold (useful to resume)
            
        Returns:
            Dictionary with aggregated metrics across all folds
        """
        print(f"\n{'#'*70}")
        print(f"# K-FOLD CROSS-VALIDATION: {self.n_splits} Folds")
        print(f"# Metadata: {self.fold_metadata_path}")
        print(f"# Data dir: {self.base_data_dir}")
        print(f"# Output dir: {self.output_dir}")
        print(f"# Model type: {model_type}")
        print(f"# Starting from fold: {start_fold}")
        print(f"# Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'#'*70}\n")
        
        # Store model type for later use in evaluation
        self.current_model_type = model_type
        
        fold_results_list = []
        
        for fold_id in range(self.n_splits):
            if fold_id < start_fold:
                print(f"[FOLD {fold_id}] Skipping (already completed / start_fold={start_fold})")
                fold_results_list.append({})
                continue
            fold_result = {}
            
            # Create fresh model for each fold
            if tf is not None:
                try:
                    model = self.get_model(model_type)
                except NotImplementedError:
                    print(f"[FOLD {fold_id}] Skipping training (model loading not implemented)")
                    continue
                
                # Train fold
                train_result = self.train_fold(fold_id, model, epochs=epochs, batch_size=batch_size)
                if train_result:
                    fold_result.update(train_result)
                    
                    # Evaluate fold
                    if not train_only:
                        eval_result = self.evaluate_fold(
                            fold_id, train_result['checkpoint'], batch_size=batch_size
                        )
                        if eval_result:
                            fold_result.update(eval_result)
            
            fold_results_list.append(fold_result)
        
        # Aggregate results
        aggregated = self._aggregate_results(fold_results_list)
        
        # Save results
        self._save_results(fold_results_list, aggregated)
        
        return aggregated
    
    def _aggregate_results(self, fold_results_list: List[Dict]) -> Dict:
        """
        Compute mean and std of metrics across folds.
        
        Args:
            fold_results_list: List of per-fold result dictionaries
            
        Returns:
            Dictionary with aggregated statistics
        """
        if not fold_results_list or all(not r for r in fold_results_list):
            return {'error': 'No fold results to aggregate'}
        
        # Extract metrics
        val_losses = [r.get('final_val_loss') for r in fold_results_list if r.get('final_val_loss')]
        test_losses = [r.get('test_loss') for r in fold_results_list if r.get('test_loss')]
        
        aggregated = {
            'n_folds_completed': len([r for r in fold_results_list if r]),
            'n_folds_total': self.n_splits,
        }
        
        if val_losses:
            aggregated['val_loss_mean'] = float(np.mean(val_losses))
            aggregated['val_loss_std'] = float(np.std(val_losses))
            aggregated['val_loss_min'] = float(np.min(val_losses))
            aggregated['val_loss_max'] = float(np.max(val_losses))
        
        if test_losses:
            aggregated['test_loss_mean'] = float(np.mean(test_losses))
            aggregated['test_loss_std'] = float(np.std(test_losses))
            aggregated['test_loss_min'] = float(np.min(test_losses))
            aggregated['test_loss_max'] = float(np.max(test_losses))
        
        return aggregated
    
    def _save_results(self, fold_results_list: List[Dict], aggregated: Dict) -> None:
        """
        Save detailed and aggregated results to JSON files.
        
        Args:
            fold_results_list: Per-fold results
            aggregated: Aggregated statistics
        """
        # Save fold results
        fold_results_path = self.output_dir / 'fold_results.json'
        with open(fold_results_path, 'w') as f:
            json.dump(fold_results_list, f, indent=2)
        print(f"\n[KFoldTrainer] Saved fold results: {fold_results_path}")
        
        # Save aggregated results
        aggregated_path = self.output_dir / 'aggregated_results.json'
        with open(aggregated_path, 'w') as f:
            json.dump(aggregated, f, indent=2)
        print(f"[KFoldTrainer] Saved aggregated results: {aggregated_path}")
        
        # Print summary
        print(f"\n{'='*70}")
        print(f"K-FOLD CROSS-VALIDATION SUMMARY")
        print(f"{'='*70}")
        for key, value in aggregated.items():
            if isinstance(value, float):
                print(f"{key:.<40} {value:.6f}")
            else:
                print(f"{key:.<40} {value}")
        print(f"{'='*70}\n")


def main():
    """Command-line interface for k-fold training."""
    parser = argparse.ArgumentParser(
        description="K-fold cross-validation training wrapper for cardiac segmentation"
    )
    parser.add_argument(
        '--fold-metadata',
        type=str,
        required=True,
        help='Path to kfold_metadata.json from split_data.py'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        required=True,
        help='Base data directory with patient folders (e.g., "Data Test Resize 128 ED/")'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./kfold_results',
        help='Output directory for fold results (default ./kfold_results)'
    )
    parser.add_argument(
        '--model',
        type=str,
        choices=['unet', 'trans_unet', 'resnet'],
        default='unet',
        help='Model architecture to use'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=50,
        help='Max epochs per fold'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=8,
        help='Batch size for training'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )
    parser.add_argument(
        '--train-only',
        action='store_true',
        help='Train only, skip evaluation on test set'
    )
    
    args = parser.parse_args()
    
    trainer = KFoldTrainer(
        fold_metadata_path=args.fold_metadata,
        base_data_dir=args.data_dir,
        output_dir=args.output_dir,
        seed=args.seed
    )
    
    results = trainer.run_all_folds(
        model_type=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        train_only=args.train_only
    )


if __name__ == '__main__':
    main()
