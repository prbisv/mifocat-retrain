#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to fill missing fold results from training history files.

Extracts metrics from fold_X_history.json and updates fold_results.json
with missing fold entries (e.g., fold 0).

Usage:
    python fill_fold_results.py --fold 0 --output-dir ./kfold_results
"""

import json
import argparse
from pathlib import Path
import numpy as np


def fill_fold_result(fold_id: int, output_dir: str) -> dict:
    """
    Extract metrics from fold history and create fold result entry.
    
    Args:
        fold_id: Fold index (0 to n_splits-1)
        output_dir: Directory containing fold_X/ subdirectories
        
    Returns:
        Dictionary with fold metrics, or empty dict if history not found
    """
    output_path = Path(output_dir)
    history_file = output_path / f"fold_{fold_id}" / f"fold_{fold_id}_history.json"
    checkpoint_file = output_path / f"fold_{fold_id}" / f"fold_{fold_id}_best_model.h5"
    
    if not history_file.exists():
        print(f"✗ History file not found: {history_file}")
        return {}
    
    # Load history
    with open(history_file, 'r') as f:
        history = json.load(f)
    
    # Extract metrics
    loss_list = history.get('loss', [])
    val_loss_list = history.get('val_loss', [])
    
    if not loss_list or not val_loss_list:
        print(f"✗ Missing loss or val_loss in history for fold {fold_id}")
        return {}
    
    fold_result = {
        'fold_id': fold_id,
        'final_loss': float(loss_list[-1]),
        'final_val_loss': float(val_loss_list[-1]),
        'best_val_loss': float(np.min(val_loss_list)),
        'epochs_trained': len(loss_list),
        'checkpoint': str(checkpoint_file)
    }
    
    print(f"✓ Fold {fold_id}:")
    print(f"  Final loss: {fold_result['final_loss']:.6f}")
    print(f"  Final val_loss: {fold_result['final_val_loss']:.6f}")
    print(f"  Best val_loss: {fold_result['best_val_loss']:.6f}")
    print(f"  Epochs trained: {fold_result['epochs_trained']}")
    
    return fold_result


def update_fold_results(fold_id: int, output_dir: str) -> None:
    """
    Update fold_results.json with metrics for a specific fold.
    
    Args:
        fold_id: Fold index to update
        output_dir: Directory containing fold_results.json
    """
    output_path = Path(output_dir)
    results_file = output_path / "fold_results.json"
    
    # Load existing results
    if results_file.exists():
        with open(results_file, 'r') as f:
            fold_results = json.load(f)
    else:
        print(f"✗ fold_results.json not found: {results_file}")
        return
    
    # Ensure list is large enough
    while len(fold_results) <= fold_id:
        fold_results.append({})
    
    # Get metrics from history
    fold_result = fill_fold_result(fold_id, output_dir)
    
    if fold_result:
        fold_results[fold_id] = fold_result
        
        # Save updated results
        with open(results_file, 'w') as f:
            json.dump(fold_results, f, indent=2)
        
        print(f"\n✓ Updated fold_results.json at index {fold_id}")
    else:
        print(f"\n✗ Failed to extract metrics for fold {fold_id}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Fill missing fold results from training history'
    )
    parser.add_argument('--fold', type=int, required=True,
                       help='Fold index to fill (e.g., 0)')
    parser.add_argument('--output-dir', type=str, default='./kfold_results',
                       help='Output directory containing fold_results.json')
    
    args = parser.parse_args()
    
    print(f"\n{'='*70}")
    print(f"Filling Fold {args.fold} Results")
    print(f"{'='*70}\n")
    
    update_fold_results(args.fold, args.output_dir)
    
    print(f"\n{'='*70}")
