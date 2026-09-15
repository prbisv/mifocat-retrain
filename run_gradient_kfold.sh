#!/bin/bash
set -x
cd "/Users/iganarendra/multica_workspaces_desktop-api.multica.ai/narenteam-a-ad0916b32d25/nar-4-d8d069bc6cdc/workdir/mifocat-retrain"
PY=~/miniconda3/envs/py310/bin/python
DATA_DIR="/Users/iganarendra/Downloads/mifocat-retrain/acdc2017/Data 2D/ED/Data Per Pasien Training 2D"

echo "=== U-Net k-fold run started: $(date) ==="
"$PY" -u train_kfold_wrapper.py \
  --fold-metadata kfold_results/kfold_metadata.json \
  --data-dir "$DATA_DIR" \
  --output-dir kfold_results_gradmon/unet \
  --model unet --epochs 50 --batch-size 8
echo "=== U-Net k-fold run finished: $(date) ==="

echo "=== TransUNet k-fold run started: $(date) ==="
"$PY" -u train_kfold_wrapper.py \
  --fold-metadata kfold_results/kfold_metadata.json \
  --data-dir "$DATA_DIR" \
  --output-dir kfold_results_gradmon/transunet \
  --model trans_unet --epochs 50 --batch-size 8
echo "=== TransUNet k-fold run finished: $(date) ==="

echo "=== ALL DONE: $(date) ==="
