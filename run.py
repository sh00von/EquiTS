"""
EquiTS: Official Multi-Dataset Benchmark & Training Runner.
Standard entrypoint to train and benchmark all models (Baselines vs. EquiTS)
across multiple benchmark datasets (ETTh1, ETTm1, Weather) under identical protocols.

Usage:
    python run.py --dataset ETTm1 --model all --epochs 3 --device cpu
    python run.py --dataset ETTh1 --model all --epochs 3 --device cpu
    python run.py --dataset all --epochs 3 --device cpu
"""

import os
import sys
import json
import yaml
import argparse
import torch

from experiments.exp_forecasting import Exp_Forecasting
from datasets.loader import DATASET_CONFIGS


def print_results_matrix(results_list, dataset_name):
    print("\n" + "=" * 94)
    print(f"               OFFICIAL PUBLICATION BENCHMARK MATRIX ({dataset_name})               ")
    print("=" * 94)
    header = f"{'Model':<24} | {'Parameters':<10} | {'MAE':<7} | {'MSE':<7} | {'Latency (ms)':<14} | {'Bypass %':<8}"
    print(header)
    print("-" * 94)
    for r in results_list:
        row = f"{r['Model']:<24} | {r['Parameters']:<10} | {r['MAE']:<7} | {r['MSE']:<7} | {r['Latency (ms)']:<14} | {r['Bypass (%)']}"
        print(row)
    print("=" * 94)


def save_results(results_list, dataset_name, out_dir="results"):
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, f"benchmark_{dataset_name.lower()}_results.json")
    md_path = os.path.join(out_dir, f"benchmark_{dataset_name.lower()}_table.md")
    
    with open(json_path, "w") as f:
        json.dump(results_list, f, indent=2)
        
    md_lines = [
        f"# Official Comparative Benchmark Matrix ({dataset_name})",
        "",
        f"Evaluated on {dataset_name} ($L=96, H=96$) under standardized chronological splits:",
        "",
        "| Model | Parameters | MAE (lower is better) | MSE (lower is better) | Latency (ms/sample) | Bypass / Hit Rate |",
        "| :--- | :--- | :--- | :--- | :--- | :--- |"
    ]
    for r in results_list:
        md_lines.append(f"| **{r['Model']}** | {r['Parameters']} | **{r['MAE']}** | **{r['MSE']}** | {r['Latency (ms)']} ms | {r['Bypass (%)']} |")
        
    with open(md_path, "w") as f:
        f.write("\n".join(md_lines) + "\n")
        
    print(f"\nSaved benchmark matrix for {dataset_name} to:")
    print(f"  - Markdown Table: {md_path}")
    print(f"  - JSON Results:   {json_path}")


def run_benchmark_for_dataset(dataset_name, config, args, device):
    cfg = yaml.safe_load(yaml.dump(config)) # deep copy
    cfg["dataset"]["name"] = dataset_name
    
    # Auto-populate dataset settings
    if dataset_name in DATASET_CONFIGS:
        ds_info = DATASET_CONFIGS[dataset_name]
        cfg["dataset"]["data_path"] = ds_info["file"]
        cfg["dataset"]["download_url"] = ds_info["url"]
        cfg["dataset"]["num_channels"] = ds_info["num_channels"]
        cfg["dataset"]["train_ratio"] = ds_info["train_ratio"]
        cfg["dataset"]["val_ratio"] = ds_info["val_ratio"]

    print(f"\n" + "#" * 70)
    print(f"   STARTING BENCHMARK: {dataset_name} (Channels: {cfg['dataset']['num_channels']}, Horizon: {cfg['dataset']['horizon']})")
    print("#" * 70)

    if args.model.lower() in ["our", "equits"]:
        models_to_run = ["EquiTS_NoMem", "EquiTS_RCPQ"]
    elif args.model.lower() == "all":
        models_to_run = ["DLinear", "PatchLinear", "LSTM", "TCN", "EquiTS_NoMem", "EquiTS_RCPQ"]
    else:
        models_to_run = [args.model]

    all_results = []
    for model_name in models_to_run:
        print(f"\n>>> Running Experiment for: {model_name} on {dataset_name}")
        exp = Exp_Forecasting(config=cfg, model_name=model_name, device=device)
        
        # 1. Train model from scratch on identical splits
        exp.train(epochs=args.epochs)
        
        # 2. Test best model checkpoint on held-out test split
        test_metrics = exp.test(use_memory_quotienting=("RCPQ" in model_name))
        all_results.append(test_metrics)

    print_results_matrix(all_results, dataset_name)
    save_results(all_results, dataset_name, out_dir="results")
    return all_results


def main():
    parser = argparse.ArgumentParser(description="Official Multi-Dataset Benchmark Runner")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config file")
    parser.add_argument("--dataset", type=str, default="ETTh1", help="Dataset to benchmark: ETTh1, ETTm1, Weather, or all")
    parser.add_argument("--model", type=str, default="EquiTS", help="Model to run: EquiTS (default: runs our model variants), all, or specific model")
    parser.add_argument("--horizon", type=int, default=None, help="Forecast horizon H (e.g. 96, 192, 336, 720)")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs per model")
    parser.add_argument("--device", type=str, default="cpu", help="Device: cpu or cuda")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    if args.horizon is not None:
        config["dataset"]["horizon"] = args.horizon

    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    print(f"Device: {device} | Lookback: {config['dataset']['lookback']} | Horizon: {config['dataset']['horizon']}")

    if args.dataset.lower() == "all":
        datasets = ["ETTh1", "ETTm1", "Weather"]
    else:
        datasets = [args.dataset]

    multi_results = {}
    for ds in datasets:
        multi_results[ds] = run_benchmark_for_dataset(ds, config, args, device)

    # If multiple datasets, generate combined cross-dataset markdown table
    if len(datasets) > 1:
        md_path = "results/benchmark_multidataset_summary.md"
        lines = [
            "# Multi-Dataset Comprehensive Benchmark Summary",
            "",
            "Comparative evaluation across distinct temporal sampling regimes (Hourly vs. 15-Minute):",
            ""
        ]
        for ds, res_list in multi_results.items():
            lines.append(f"### {ds} ($L=96, H=96$)")
            lines.append("| Model | Parameters | MAE | MSE | Latency (ms) | Bypass % |")
            lines.append("| :--- | :--- | :--- | :--- | :--- | :--- |")
            for r in res_list:
                lines.append(f"| **{r['Model']}** | {r['Parameters']} | **{r['MAE']}** | **{r['MSE']}** | {r['Latency (ms)']} ms | {r['Bypass (%)']} |")
            lines.append("")
        with open(md_path, "w") as f:
            f.write("\n".join(lines))
        print(f"\nMulti-dataset summary table saved to: {md_path}")


if __name__ == "__main__":
    main()
