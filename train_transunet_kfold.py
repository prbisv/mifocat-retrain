#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TransUNet K-Fold Cross-Validation Training Script

This script runs k-fold cross-validation for TransUNet model with MIFOCAT loss.
It's a command-line alternative to the Jupyter notebook workflow.

Usage:
    python train_transunet_kfold.py --data-dir /path/to/data --fold-metadata /path/to/metadata.json
    
    # With custom parameters
    python train_transunet_kfold.py \
        --data-dir "Data Test Resize 128 ED/" \
        --fold-metadata kfold_metadata.json \
        --output-dir transunet_kfold_results \
        --epochs 50 \
        --batch-size 32

@author: ramad
"""

import os
import sys
import argparse
from pathlib import Path

# Ensure project modules can be imported
sys.path.insert(0, str(Path(__file__).parent))

try:
    import tensorflow as tf
    print(f"✓ TensorFlow {tf.__version__} loaded")
    
    # Check GPU availability
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"✓ GPU detected: {len(gpus)} device(s)")
        for gpu in gpus:
            print(f"  - {gpu}")
    else:
        print("⚠ No GPU detected. Training will use CPU (slower)")
except ImportError as e:
    print(f"✗ TensorFlow import failed: {e}")
    print("Install with: pip install tensorflow")
    sys.exit(1)

try:
    from train_kfold_wrapper import KFoldTrainer
    print("✓ Custom modules imported successfully")
except ImportError as e:
    print(f"✗ Custom module import failed: {e}")
    print("Ensure train_kfold_wrapper.py is in the same directory")
    sys.exit(1)


def main():
    """Main entry point for TransUNet k-fold training."""
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="K-fold cross-validation training for TransUNet with MIFOCAT loss",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument(
        '--data-dir',
        type=str,
        required=True,
        help='Base data directory with patient folders (e.g., "Data Test Resize 128 ED/")'
    )
    parser.add_argument(
        '--fold-metadata',
        type=str,
        required=True,
        help='Path to kfold_metadata.json from split_data.py'
    )
    
    # Optional arguments
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./transunet_kfold_results',
        help='Output directory for fold results'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=50,
        help='Maximum epochs per fold'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size for training'
    )
    parser.add_argument(
        '--early-stop-patience',
        type=int,
        default=10,
        help='Early stopping patience (epochs with no improvement)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    parser.add_argument(
        '--train-only',
        action='store_true',
        help='Train only, skip evaluation on test set'
    )
    parser.add_argument(
        '--start-fold',
        type=int,
        default=0,
        help='Start from this fold (useful for resuming)'
    )
    
    args = parser.parse_args()
    
    # Validate paths
    data_dir = Path(args.data_dir)
    fold_metadata = Path(args.fold_metadata)
    
    if not data_dir.exists():
        print(f"✗ ERROR: Data directory not found: {data_dir}")
        sys.exit(1)
    
    if not fold_metadata.exists():
        print(f"✗ ERROR: Fold metadata not found: {fold_metadata}")
        print("Run split_data.py first to generate kfold_metadata.json")
        sys.exit(1)
    
    # Print configuration
    print("\n" + "="*70)
    print("TRANSUNET K-FOLD TRAINING CONFIGURATION")
    print("="*70)
    print(f"Data directory:       {data_dir}")
    print(f"Fold metadata:        {fold_metadata}")
    print(f"Output directory:     {args.output_dir}")
    print(f"Model type:           TransUNet")
    print(f"Epochs per fold:      {args.epochs}")
    print(f"Batch size:           {args.batch_size}")
    print(f"Early stop patience:  {args.early_stop_patience}")
    print(f"Random seed:          {args.seed}")
    print(f"Train only:           {args.train_only}")
    print(f"Start from fold:      {args.start_fold}")
    print("="*70 + "\n")
    
    # Initialize trainer
    trainer = KFoldTrainer(
        fold_metadata_path=str(fold_metadata),
        base_data_dir=str(data_dir),
        output_dir=args.output_dir,
        seed=args.seed
    )
    
    # Run k-fold cross-validation
    print("\n🚀 Starting k-fold cross-validation training...\n")
    
    try:
        results = trainer.run_all_folds(
            model_type='transunet',
            epochs=args.epochs,
            batch_size=args.batch_size,
            train_only=args.train_only,
            start_fold=args.start_fold
        )
        
        print("\n✓ K-fold cross-validation completed successfully!")
        print(f"Results saved to: {args.output_dir}")
        
        # Print summary
        if results and 'val_loss_mean' in results:
            print("\n" + "="*70)
            print("FINAL RESULTS SUMMARY")
            print("="*70)
            print(f"Validation loss:  {results['val_loss_mean']:.6f} ± {results.get('val_loss_std', 0):.6f}")
            if 'test_loss_mean' in results:
                print(f"Test loss:        {results['test_loss_mean']:.6f} ± {results.get('test_loss_std', 0):.6f}")
            print(f"Folds completed:  {results.get('n_folds_completed', 0)}/{results.get('n_folds_total', 0)}")
            print("="*70)
        
    except KeyboardInterrupt:
        print("\n⚠ Training interrupted by user")
        print(f"Partial results saved to: {args.output_dir}")
        sys.exit(0)
    except Exception as e:
        print(f"\n✗ Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
