# -*- coding: utf-8 -*-
"""
Quick-Start Example: K-Fold Cross-Validation for MIFOCAT

This script demonstrates a complete k-fold CV workflow:
1. Generate fold metadata
2. (Optional) Visualize fold distribution
3. Run k-fold training
4. Analyze aggregated results

Usage:
    python example_kfold_quickstart.py

Adjust paths and parameters in the CONFIG section below.
"""

import os
import json
from pathlib import Path
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================================
# CONFIG: Modify these paths for your environment
# ============================================================================

CONFIG = {
    # Data paths (modify for your system)
    'input_data_dir': 'Data Test Resize 128 ED/',          # Raw 2D patient data
    'output_base_dir': './kfold_results',                  # Where to save fold metadata
    
    # K-fold parameters
    'n_splits': 5,                                         # Number of folds
    'val_ratio': 0.15,                                     # Validation % within training
    'seed': 42,                                            # For reproducibility
    
    # Training parameters
    'model_type': 'unet',                                  # 'unet', 'trans_unet', 'resnet'
    'epochs': 50,
    'batch_size': 8,
    'train_only': False,                                   # Skip test set evaluation if True
}


# ============================================================================
# STEP 1: Generate Fold Metadata
# ============================================================================

def step_generate_folds():
    """Generate k-fold splits and save metadata."""
    print("\n" + "="*70)
    print("STEP 1: Generate K-Fold Splits")
    print("="*70)
    
    from split_data import CardiacDataSplitter
    
    splitter = CardiacDataSplitter(
        input_folder=CONFIG['input_data_dir'],
        output_folder=CONFIG['output_base_dir'],
        seed=CONFIG['seed']
    )
    
    print(f"Input directory: {CONFIG['input_data_dir']}")
    print(f"Output directory: {CONFIG['output_base_dir']}")
    print(f"Number of folds: {CONFIG['n_splits']}")
    print(f"Validation ratio: {CONFIG['val_ratio']}")
    print(f"Seed: {CONFIG['seed']}")
    
    # Generate k-fold metadata
    splitter.kfold_split(
        n_splits=CONFIG['n_splits'],
        val_ratio=CONFIG['val_ratio']
    )
    
    print("\n✓ Fold metadata generated successfully!")
    
    # Return path to metadata
    metadata_path = Path(CONFIG['output_base_dir']) / 'kfold_metadata.json'
    return str(metadata_path)


# ============================================================================
# STEP 2: Visualize Fold Distribution (Optional)
# ============================================================================

def step_visualize_folds(metadata_path):
    """Display fold statistics."""
    print("\n" + "="*70)
    print("STEP 2: Fold Statistics")
    print("="*70)
    
    with open(metadata_path) as f:
        metadata = json.load(f)
    
    print(f"\nTotal patients: {metadata['total_patients']}")
    print(f"Number of folds: {len(metadata['folds'])}\n")
    
    for fold_info in metadata['folds']:
        fold_id = fold_info['fold_id']
        train_count = fold_info.get('train_count', len(fold_info.get('train', [])))
        val_count = fold_info.get('val_count', len(fold_info.get('val', [])))
        test_count = fold_info.get('test_count', len(fold_info.get('test', [])))
        
        print(f"Fold {fold_id}:")
        print(f"  Train patients: {train_count:3d} ({100*train_count/metadata['total_patients']:.1f}%)")
        print(f"  Val patients:   {val_count:3d} ({100*val_count/metadata['total_patients']:.1f}%)")
        print(f"  Test patients:  {test_count:3d} ({100*test_count/metadata['total_patients']:.1f}%)")
    
    print()


# ============================================================================
# STEP 3: Run K-Fold Training
# ============================================================================

def step_run_kfold_training(metadata_path):
    """Run k-fold cross-validation training."""
    print("\n" + "="*70)
    print("STEP 3: K-Fold Training")
    print("="*70)
    
    try:
        from train_kfold_wrapper import KFoldTrainer
    except ImportError:
        print("✗ Error: TensorFlow not installed or model loading not implemented")
        print("  Skipping training step. You can:")
        print("  1. Install TensorFlow: pip install tensorflow")
        print("  2. Implement model loading in train_kfold_wrapper.py::get_model()")
        return None
    
    trainer = KFoldTrainer(
        fold_metadata_path=metadata_path,
        base_data_dir=CONFIG['input_data_dir'],
        output_dir=CONFIG['output_base_dir'],
        seed=CONFIG['seed']
    )
    
    print(f"Model: {CONFIG['model_type']}")
    print(f"Epochs: {CONFIG['epochs']}")
    print(f"Batch size: {CONFIG['batch_size']}")
    print()
    
    results = trainer.run_all_folds(
        model_type=CONFIG['model_type'],
        epochs=CONFIG['epochs'],
        batch_size=CONFIG['batch_size'],
        train_only=CONFIG['train_only']
    )
    
    return results


# ============================================================================
# STEP 4: Analyze Results
# ============================================================================

def step_analyze_results():
    """Load and display aggregated results."""
    print("\n" + "="*70)
    print("STEP 4: Results Summary")
    print("="*70)
    
    results_path = Path(CONFIG['output_base_dir']) / 'aggregated_results.json'
    
    if not results_path.exists():
        print(f"✗ Results file not found: {results_path}")
        print("  Make sure training completed successfully.")
        return
    
    with open(results_path) as f:
        results = json.load(f)
    
    print("\nAggregated Metrics Across All Folds:")
    print("-" * 50)
    
    for key, value in results.items():
        if isinstance(value, float):
            print(f"{key:.<35} {value:.6f}")
        else:
            print(f"{key:.<35} {value}")
    
    # Calculate confidence interval if applicable
    if 'val_loss_mean' in results and 'val_loss_std' in results:
        mean = results['val_loss_mean']
        std = results['val_loss_std']
        ci_lower = mean - 1.96 * std
        ci_upper = mean + 1.96 * std
        print("-" * 50)
        print(f"{'95% CI for val_loss':<35} [{ci_lower:.6f}, {ci_upper:.6f}]")
    
    print()
    
    # Save summary to markdown
    summary_path = Path(CONFIG['output_base_dir']) / 'RESULTS_SUMMARY.md'
    with open(summary_path, 'w') as f:
        f.write(f"# K-Fold CV Results\n\n")
        f.write(f"**Configuration:**\n")
        f.write(f"- Folds: {results.get('n_folds_total', 'N/A')}\n")
        f.write(f"- Model: {CONFIG['model_type']}\n")
        f.write(f"- Epochs: {CONFIG['epochs']}\n")
        f.write(f"- Batch size: {CONFIG['batch_size']}\n\n")
        f.write(f"**Metrics:**\n")
        for key, value in results.items():
            if isinstance(value, float):
                f.write(f"- {key}: {value:.6f}\n")
            else:
                f.write(f"- {key}: {value}\n")
    
    print(f"✓ Summary saved to: {summary_path}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Run complete k-fold CV workflow."""
    print("\n" + "#"*70)
    print("# K-Fold Cross-Validation Quick-Start for MIFOCAT")
    print("#"*70)
    
    # Verify input directory exists
    if not os.path.exists(CONFIG['input_data_dir']):
        print(f"\n✗ Error: Input directory not found: {CONFIG['input_data_dir']}")
        print("  Modify CONFIG['input_data_dir'] to point to your data directory")
        return
    
    # Step 1: Generate folds
    metadata_path = step_generate_folds()
    
    # Step 2: Visualize fold distribution
    step_visualize_folds(metadata_path)
    
    # Step 3: Run k-fold training (optional if TensorFlow not available)
    try:
        step_run_kfold_training(metadata_path)
    except Exception as e:
        print(f"\n✗ Training step failed: {e}")
        print("  This is okay - you can still use the fold metadata manually")
    
    # Step 4: Analyze results (if available)
    step_analyze_results()
    
    print("\n" + "#"*70)
    print("# K-Fold CV Quick-Start Complete!")
    print("#"*70)
    print(f"\nResults saved to: {CONFIG['output_base_dir']}/")
    print("\nNext steps:")
    print("1. Check aggregated_results.json for cross-fold metrics")
    print("2. Review fold_*/fold_*_history.json for per-fold training curves")
    print("3. Use saved checkpoints (fold_*/fold_*_best_model.h5) for inference")
    print("\nFor more details, see: K_FOLD_CV_GUIDE.md")
    print()


if __name__ == '__main__':
    main()
