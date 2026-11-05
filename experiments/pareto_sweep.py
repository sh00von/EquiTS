"""
EquiTS: Pareto Frontier and Sensitivity Analysis Suite.
Sweeps the quotienting merge threshold lambda_0 across [0.02, 0.35]
to map the exact empirical Accuracy-Latency Pareto Trade-off Curve.
"""

import os
import sys
# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import time
import yaml
import argparse
from typing import List, Dict, Any
import numpy as np
import torch

from datasets.loader import get_dataloaders
from equits.model import EquiTS
from equits.losses import compute_forecast_metrics


def run_pareto_sweep(
    config_path: str = "configs/default.yaml",
    checkpoint_path: str = "checkpoints/EquiTS_RCPQ_ETTh1_H96.pt",
    thresholds: List[float] = [2e-7, 5e-7, 1e-6, 3e-6, 7e-6, 1.5e-5, 3e-5],
    device_str: str = "cpu"
) -> List[Dict[str, Any]]:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    device = torch.device(device_str if torch.cuda.is_available() and device_str == "cuda" else "cpu")
    print(f"Running Pareto Frontier Sweep on: {device}")

    # Load Data
    _, _, test_loader, scaler = get_dataloaders(config)

    # Load Base Model
    model = EquiTS(config).to(device)
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"Loaded trained checkpoint: {checkpoint_path}")
    else:
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}. Train the model first!")

    # 1. First measure baseline with Memory Disabled (lambda -> 0 / no bypass)
    model.eval()
    model.memory_enabled = False
    
    t0 = time.perf_counter()
    all_preds_base, all_targets = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            mu, _, _, _ = model(x)
            all_preds_base.append(mu.cpu())
            all_targets.append(y.cpu())
            
    base_time = time.perf_counter() - t0
    all_preds_base = torch.cat(all_preds_base, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    base_metrics = compute_forecast_metrics(all_preds_base, all_targets)
    base_latency_ms = (base_time / len(all_preds_base)) * 1000.0

    print(f"\n[Baseline - No Memory Bypass]")
    print(f"  MAE: {base_metrics['MAE']:.4f} | MSE: {base_metrics['MSE']:.4f} | Latency: {base_latency_ms:.3f} ms")

    # 2. Sweep Thresholds
    sweep_results = []
    
    # Add baseline point (lambda = 0.0)
    sweep_results.append({
        "Threshold (lambda)": 0.0,
        "MAE": round(base_metrics["MAE"], 4),
        "MSE": round(base_metrics["MSE"], 4),
        "Latency (ms)": round(base_latency_ms, 3),
        "Speedup": "1.00x",
        "Bypass (%)": 0.0,
        "Active Prototypes": 0
    })

    model.memory_enabled = True
    
    for tau in thresholds:
        # Reset memory state for a clean streaming evaluation
        model.memory.occupied.zero_()
        model.memory.counts.zero_()
        model.memory.timestamps.zero_()
        model.memory.total_queries = 0
        model.memory.total_hits = 0
        model.memory.threshold.fill_(tau)

        t0 = time.perf_counter()
        all_preds = []
        total_bypassed = 0
        total_tokens = 0

        with torch.no_grad():
            for step, (x, y) in enumerate(test_loader):
                x = x.to(device)
                mu, _, _, aux = model(x, current_step=float(step), update_memory=True)
                all_preds.append(mu.cpu())
                total_bypassed += aux["bypassed_count"]
                total_tokens += aux["total_tokens"]

        elapsed_time = time.perf_counter() - t0
        all_preds = torch.cat(all_preds, dim=0)
        metrics = compute_forecast_metrics(all_preds, all_targets)
        latency_ms = (elapsed_time / len(all_preds)) * 1000.0
        speedup = base_latency_ms / max(1e-6, latency_ms)
        bypass_pct = (total_bypassed / max(1, total_tokens)) * 100.0

        record = {
            "Threshold (lambda)": round(tau, 3),
            "MAE": round(metrics["MAE"], 4),
            "MSE": round(metrics["MSE"], 4),
            "Latency (ms)": round(latency_ms, 3),
            "Speedup": f"{speedup:.2f}x",
            "Bypass (%)": round(bypass_pct, 2),
            "Active Prototypes": model.memory.num_active_prototypes
        }
        sweep_results.append(record)
        print(f"  lambda={tau:<5} | MAE={record['MAE']:<6} | MSE={record['MSE']:<6} | Latency={record['Latency (ms)']:<6} ms | Speedup={record['Speedup']:<5} | Bypass={record['Bypass (%)']}% | Prototypes={record['Active Prototypes']}")

    return sweep_results


def save_pareto_results(results: List[Dict[str, Any]], out_dir: str = "results"):
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, "pareto_frontier.json")
    md_path = os.path.join(out_dir, "pareto_frontier.md")

    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    md_lines = [
        "# EquiTS: Accuracy vs. Latency Pareto Frontier Analysis",
        "",
        "Empirical evaluation sweeping quotienting threshold $\\lambda_0$ on the test split:",
        "",
        "| Threshold ($\\lambda_0$) | MAE (lower is better) | MSE (lower is better) | Latency (ms/sample) | Speedup Gain | Bypass Hit Rate | Active Prototypes |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- |"
    ]
    for r in results:
        md_lines.append(
            f"| **{r['Threshold (lambda)']}** | **{r['MAE']}** | **{r['MSE']}** | {r['Latency (ms)']} ms | **{r['Speedup']}** | {r['Bypass (%)']}% | {r['Active Prototypes']} |"
        )

    with open(md_path, "w") as f:
        f.write("\n".join(md_lines) + "\n")

    print(f"\nPareto frontier table saved to {md_path}")


def main():
    parser = argparse.ArgumentParser(description="Sweep merge threshold lambda to map Pareto frontier")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/EquiTS_RCPQ_ETTh1_H96.pt")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    results = run_pareto_sweep(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        thresholds=[2e-7, 5e-7, 1e-6, 3e-6, 7e-6, 1.5e-5, 3e-5],
        device_str=args.device
    )
    save_pareto_results(results)


if __name__ == "__main__":
    main()
