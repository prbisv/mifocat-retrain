# K-Fold Cross-Validation Implementation for MIFOCAT

## Overview

This guide documents the k-fold cross-validation (CV) implementation for the MIFOCAT cardiac segmentation research. K-fold CV enables robust evaluation of model performance while maximizing use of limited training data.

## Why K-Fold Cross-Validation?

**Problem with single 90/10 split:**
- Only one train/val partition → high variance in reported metrics
- Wastes ~10% of data that could be used for training
- Risk of overoptimistic or pessimistic estimates due to random split
- Limited statistical confidence in results

**Benefits of k-fold CV (k=5 recommended):**
- All data used for training in different folds
- 5 independent train/val partitions → robust mean ± std metrics
- Better generalization estimates
- Aligns with research publication standards

## Architecture

### 1. Data Splitting: `split_data.py`

**Refactored to support two modes:**

#### Mode 1: Simple Split (Legacy)
```bash
python split_data.py --mode simple --input <data_dir> --output <output_dir>
```
Creates: `output_dir/train/`, `output_dir/val/`
Uses: 90/10 ratio, folder-based splitting

#### Mode 2: K-Fold Split (Recommended)
```bash
python split_data.py --mode kfold --input <data_dir> --output <output_dir> \
  --n-splits 5 --val-ratio 0.15 --create-dirs
```

**Output structure:**
```
output_dir/
├── kfold_metadata.json          # Metadata mapping patients to folds
├── fold_0/
│   ├── train/                   # Training patient folders
│   ├── val/                     # Validation patient folders
│   └── test/                    # Test patient folders (optional)
├── fold_1/
│   ├── train/
│   ├── val/
│   └── test/
└── ...
```

**Key features:**
- **Patient-level stratification**: All 2D slices of a patient stay in the same fold (prevents data leakage)
- **Stratified split**: Uses `StratifiedKFold` to preserve class balance across folds
- **Reproducible**: Seed=42 ensures same splits every run
- **Fold metadata JSON**: Maps patient IDs to train/val/test per fold

### 2. Fold-Aware Data Loading: `custom_datagen.py`

**New class: `FoldAwareDataLoader`**

Extends legacy `imageLoader()` with fold filtering:

```python
from custom_datagen import FoldAwareDataLoader

# Initialize
loader = FoldAwareDataLoader(
    base_dir='Data Test Resize 128 ED/',
    fold_metadata_path='kfold_results/kfold_metadata.json'
)

# Get train/val generators for fold 0
train_gen, val_gen, train_steps, val_steps = loader.get_generators(
    fold_id=0,
    batch_size=8,
    image_subdir='images',
    mask_subdir='masks'
)

# Use with Keras training
model.fit(
    train_gen,
    steps_per_epoch=train_steps,
    validation_data=val_gen,
    validation_steps=val_steps,
    epochs=50
)
```

**Methods:**
- `get_fold_patients(fold_id, split)` → List of patient IDs for train/val/test
- `get_file_list(patient_ids, image_subdir)` → List of .npy files to load
- `get_generators(fold_id, batch_size, ...)` → (train_gen, val_gen, train_steps, val_steps)

**Assumptions about data structure:**
```
base_dir/
├── Pasien 001/
│   ├── images/          # .npy files
│   └── masks/           # .npy files
├── Pasien 002/
│   ├── images/
│   └── masks/
└── ...
```

If your structure differs, edit `get_file_list()` accordingly.

### 3. K-Fold Training Orchestration: `train_kfold_wrapper.py`

**Main class: `KFoldTrainer`**

Automates the complete k-fold pipeline:

```bash
python train_kfold_wrapper.py \
  --fold-metadata kfold_results/kfold_metadata.json \
  --data-dir 'Data Test Resize 128 ED/' \
  --output-dir kfold_results \
  --model unet \
  --epochs 50 \
  --batch-size 8
```

**Workflow per fold:**
1. Load fold metadata
2. Get train/val generators via `FoldAwareDataLoader`
3. Create fresh model
4. Train with early stopping and checkpointing
5. Save history and best model
6. Evaluate on test set (optional)

**Output structure:**
```
kfold_results/
├── kfold_metadata.json           # From split_data.py
├── fold_0/
│   ├── fold_0_best_model.h5      # Saved checkpoint
│   ├── fold_0_history.json       # Training history (loss, accuracy, etc.)
│   └── fold_0_predictions.csv    # (Optional) Predictions on test set
├── fold_1/
│   └── ...
└── aggregated_results.json       # Mean ± std across all folds
```

**Key metrics saved:**

Per-fold (`fold_results.json`):
- `fold_id`: Fold index
- `final_loss`: Training loss at last epoch
- `final_val_loss`: Validation loss at last epoch
- `best_val_loss`: Best validation loss during training
- `epochs_trained`: Number of epochs before early stopping
- `test_loss`: Loss on test set (after training)
- `checkpoint`: Path to best saved model

Aggregated (`aggregated_results.json`):
```json
{
  "n_folds_completed": 5,
  "n_folds_total": 5,
  "val_loss_mean": 0.123456,
  "val_loss_std": 0.012345,
  "val_loss_min": 0.105,
  "val_loss_max": 0.142,
  "test_loss_mean": 0.125,
  "test_loss_std": 0.013
}
```

## Workflow: Step-by-Step

### Step 1: Generate Fold Metadata

```bash
cd /Users/iganarendra/Downloads/Code-Cardiac-Segmentation

python split_data.py \
  --mode kfold \
  --input "/path/to/raw/patient/data" \
  --output "./kfold_results" \
  --n-splits 5 \
  --val-ratio 0.15 \
  --seed 42
```

**Output:** `kfold_results/kfold_metadata.json`

Example metadata:
```json
{
  "n_splits": 5,
  "seed": 42,
  "val_ratio": 0.15,
  "total_patients": 150,
  "folds": [
    {
      "fold_id": 0,
      "train": ["001", "002", "003", ..., "120"],
      "val": ["121", "122", ..., "135"],
      "test": ["136", "137", ..., "150"],
      "train_count": 120,
      "val_count": 15,
      "test_count": 15
    },
    ...
  ]
}
```

### Step 2: (Optional) Create Fold Directories

If you want physical fold folders:
```bash
python split_data.py \
  --output "./kfold_results" \
  --create-dirs \
  --mode kfold
```

Creates: `kfold_results/fold_0/train/`, `fold_0/val/`, etc.

**Note:** This step is optional. `FoldAwareDataLoader` filters data in-memory from metadata.

### Step 3: Train Using K-Fold

```bash
python train_kfold_wrapper.py \
  --fold-metadata kfold_results/kfold_metadata.json \
  --data-dir 'Data Test Resize 128 ED/' \
  --output-dir kfold_results \
  --model unet \
  --epochs 50 \
  --batch-size 8
```

**Expected output:**
```
======================================================================
K-FOLD CROSS-VALIDATION: 5 Folds
Metadata: /path/to/kfold_metadata.json
Data dir: /path/to/Data Test Resize 128 ED/
Output dir: /path/to/kfold_results
Started: 2026-01-22 14:30:00
======================================================================

======================================================================
[FOLD 0] Starting training
======================================================================
[FoldAwareDataLoader] Fold 0: 120 train patients, 15 val patients
[FoldAwareDataLoader] Train files: 1200, Val files: 150
[FOLD 0] Training steps per epoch: 150, Validation steps: 19
Epoch 1/50
150/150 [==============================] - 45s - loss: 0.8231 - val_loss: 0.7821
...
Epoch 32/50 - EarlyStopping
[FOLD 0] ✓ Training complete
[FOLD 0]   Final val_loss: 0.1234
[FOLD 0]   Best val_loss: 0.1045

[FOLD 0] Evaluating on test set...
[FOLD 0] Test files: 150
[FOLD 0] ✓ Evaluation complete - test_loss: 0.1089

...

======================================================================
K-FOLD CROSS-VALIDATION SUMMARY
======================================================================
n_folds_completed............................ 5
n_folds_total.............................. 5
val_loss_mean.............................. 0.108967
val_loss_std............................... 0.005234
val_loss_min.............................. 0.102156
val_loss_max.............................. 0.115432
test_loss_mean............................ 0.110234
test_loss_std............................. 0.006123
======================================================================
```

### Step 4: Analyze Results

Results are saved to:
- **Per-fold details:** `kfold_results/fold_*/` (histories, checkpoints)
- **Summary:** `kfold_results/aggregated_results.json`

Example analysis script:
```python
import json
import numpy as np

# Load results
with open('kfold_results/aggregated_results.json') as f:
    results = json.load(f)

print(f"Mean validation loss: {results['val_loss_mean']:.6f} ± {results['val_loss_std']:.6f}")
print(f"Mean test loss: {results['test_loss_mean']:.6f} ± {results['test_loss_std']:.6f}")
print(f"Confidence interval (95%): [{results['val_loss_mean'] - 1.96*results['val_loss_std']:.6f}, "
      f"{results['val_loss_mean'] + 1.96*results['val_loss_std']:.6f}]")
```

## Integration with Existing Code

### Backward Compatibility

**Legacy single-split mode still works:**
```python
# Old way still supported
import splitfolders
splitfolders.ratio(input_folder, output=output_folder, seed=42, ratio=(.90, .10))
```

**Or use refactored API:**
```python
from split_data import CardiacDataSplitter
splitter = CardiacDataSplitter(input_folder, output_folder)
splitter.simple_split(ratio=(0.90, 0.10))
```

### Model Training Integration

The existing model files (`predict_unet_2d.py`, `predict_trans_unet.py`) can be adapted:

```python
# In train_kfold_wrapper.py, replace get_model():

def get_model(self, model_type: str = 'unet') -> 'keras.Model':
    if model_type == 'unet':
        from predict_unet_2d import create_unet_model  # Your model factory
        model = create_unet_model(input_shape=(128, 128, 1), num_classes=4)
        model.compile(
            optimizer=Adam(lr=1e-3),
            loss='categorical_crossentropy',  # Or MIFOCAT loss
            metrics=['accuracy']
        )
        return model
    elif model_type == 'trans_unet':
        from predict_trans_unet import create_trans_unet_model
        # ...
```

### Evaluation Integration

Post-training evaluation can use existing metric scripts:

```python
# After fold training, evaluate on test set with your metrics
from hitung_evaluasi_metrik_acdc2017 import hitung_evaluasi_metrik

for fold_id in range(5):
    checkpoint = f"kfold_results/fold_{fold_id}/fold_{fold_id}_best_model.h5"
    
    # Run inference on test set (prediction scripts)
    # Then compute metrics (evaluation scripts)
    hitung_evaluasi_metrik(predictions_dir, gt_dir)
```

## Customization

### Adjust Number of Folds

```bash
python split_data.py --mode kfold --n-splits 10  # 10-fold instead of 5
```

**Guidelines:**
- **k=5 or k=10**: Standard for moderate datasets (~150 patients)
- **k=3**: Quick prototyping (faster training)
- **k=Leave-One-Out (LOO)**: k=150 (exhaustive but very slow)

### Adjust Validation Ratio Within Folds

```bash
python split_data.py --mode kfold --val-ratio 0.20  # 20% validation
```

With 150 patients and k=5:
- `--val-ratio 0.15` (default): ~20 train, 3 val, 7 test per fold
- `--val-ratio 0.20`: ~20 train, 4 val, 6 test per fold
- `--val-ratio 0.25`: ~19 train, 5 val, 6 test per fold

### Custom Data Structure

If your patient folder naming differs, edit `FoldAwareDataLoader.extract_patient_ids()`:

```python
# Current: "Pasien 001", "Pasien 002", ...
# Custom: "Patient_001", "Cardiac_Data_001", etc.

def extract_patient_ids(self) -> List[str]:
    patient_ids = set()
    for item in self.base_dir.iterdir():
        if item.is_dir():
            # Extract ID from "Patient_001" or similar
            if item.name.startswith("Patient_"):
                patient_id = item.name.replace("Patient_", "")
                patient_ids.add(patient_id)
    return sorted(list(patient_ids))
```

## Key Design Decisions

### 1. **Patient-Level Stratification**
- All 2D slices of a patient remain in same fold
- Prevents leakage: model can't learn patient-specific quirks across folds
- Critical for 3D→2D sliced data

### 2. **Metadata-Driven**
- Fold assignments stored in JSON, not physical folders
- Enables flexibility (different train/val/test splits without reorganizing data)
- Reproducible: same metadata + seed = same splits

### 3. **Generator-Based Data Loading**
- Inherits Keras `ImageDataGenerator` pattern from legacy code
- Memory-efficient: doesn't load all data at once
- Supports on-the-fly augmentation in future versions

### 4. **Early Stopping Per Fold**
- Each fold trains until validation loss plateaus
- Prevents overfitting and wasted computation
- Folds may train different number of epochs (stored in results)

## Troubleshooting

### Issue: "No patients found in input folder"
**Solution:** Check folder naming matches extraction logic in `extract_patient_ids()`
```bash
ls Data\ Test\ Resize\ 128\ ED/ | head  # Verify "Pasien XXX" format
```

### Issue: File not found when loading fold data
**Solution:** Verify data structure matches expected layout
```bash
ls "Data Test Resize 128 ED/Pasien 001/"  # Should have 'images', 'masks', etc.
```

### Issue: Early stopping triggers immediately
**Solution:** Likely no improvement; check if learning rate is too high or loss function is nan

### Issue: "fold_id X exceeds n_splits"
**Solution:** Fold ID must be < n_splits. Check metadata file and fold ID argument

## Future Enhancements

1. **Ablation Studies Integration**
   - Add `--loss-component` flag to isolate $L_{MI}$, $L_{FO}$, $L_{CAT}$ per fold
   - Report per-component metrics across folds

2. **Automatic Data Augmentation**
   - Extend `FoldAwareDataLoader.get_generators()` to apply augmentation during loading
   - Preserve augmentation parameters in metadata for reproducibility

3. **Distributed Training**
   - Multi-GPU fold parallelization (train folds 0-2 on GPU0, folds 3-4 on GPU1)
   - Reduce total training time from k×fold_time to ~2×fold_time

4. **Hyperparameter Tuning Per Fold**
   - Grid search over batch_size, learning_rate, optimizer
   - Report best params for each fold + aggregated best

5. **Statistical Significance Testing**
   - Paired t-test between k-fold mean and other baselines
   - Report 95% confidence intervals in paper

## References

- Hastie, T., Tibshirani, R., & Friedman, J. (2009). The Elements of Statistical Learning (Chap. 7: Model Assessment and Selection)
- Scikit-learn StratifiedKFold: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html
- Keras fit() API: https://keras.io/api/models/model_training_api/

## Questions?

Refer to:
- [MIFOCAT Code Breakdown](docs/MIFOCAT%20code%20breakdown.md) for per-script details
- [split_data.py](split_data.py) for data splitting logic
- [custom_datagen.py](custom_datagen.py) for loading implementation
- [train_kfold_wrapper.py](train_kfold_wrapper.py) for orchestration
