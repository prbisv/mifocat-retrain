"""
Runs the TransUNet k-fold pass with the Metal GPU plugin disabled.

On this machine, TransUNet's k-fold run has twice crashed the whole
process (SIGABRT) with a Metal Performance Shaders Graph "MLIR pass
manager failed" assertion, partway through fold 0 (once at epoch ~13,
once at epoch ~8) — a tensorflow-metal/driver-level bug, not a bug in
this codebase. U-Net trains fine on the same GPU. Forcing CPU execution
avoids the Metal compiler path entirely, at the cost of speed.

start_fold=1: fold 0 already completed and saved
(kfold_results_gradmon/transunet/fold_0/) in the run that was
interrupted (machine needed to sleep) partway through fold 1. Fold 1 had
no saved history/gradient JSON when killed, so it's retrained from
scratch; fold 0's output is left untouched.
"""
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')

from train_kfold_wrapper import KFoldTrainer

DATA_DIR = "/Users/iganarendra/Downloads/mifocat-retrain/acdc2017/Data 2D/ED/Data Per Pasien Training 2D"

trainer = KFoldTrainer(
    fold_metadata_path="kfold_results/kfold_metadata.json",
    base_data_dir=DATA_DIR,
    output_dir="kfold_results_gradmon/transunet",
)
trainer.run_all_folds(model_type="trans_unet", epochs=50, batch_size=8, start_fold=1)
