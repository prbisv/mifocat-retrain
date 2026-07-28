import os
import pandas as pd

result_root = "kfold_results"
output_csv = "kfold_summary.csv"
data_to_output = []

def find_std(data):
    return pd.Series(data).std()
def find_mean(data):
    return pd.Series(data).mean()

for fold in os.listdir(result_root):
    fold_path = os.path.join(result_root, fold)
    if (fold == "checkpoints"):
        continue
    if os.path.isdir(fold_path):
        metrics_file = os.path.join(fold_path, "fold_" + fold.split("_")[-1] + "_history.json")
        if os.path.exists(metrics_file):
            df = pd.read_json(metrics_file)
            final_metrics = df.iloc[-1]
            data_to_output.append({
                "fold": fold,
                "final_loss": final_metrics["loss"],
                "final_val_loss": final_metrics["val_loss"],
                "final_accuracy": final_metrics["accuracy"],
                "final_val_accuracy": final_metrics["val_accuracy"],
                "mean_loss": find_mean(df["loss"]),
                "std_loss": find_std(df["loss"]),
                "mean_val_loss": find_mean(df["val_loss"]),
                "std_val_loss": find_std(df["val_loss"]),
                "mean_accuracy": find_mean(df["accuracy"]),
                "std_accuracy": find_std(df["accuracy"]),
                "mean_val_accuracy": find_mean(df["val_accuracy"]),
                "std_val_accuracy": find_std(df["val_accuracy"])
            })
        else: print(f"Metrics file not found for {fold}")
        

summary_df = pd.DataFrame(data_to_output)
summary_df.to_csv(output_csv, index=False)