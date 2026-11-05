"""
EquiTS: Multi-Dataset Benchmark & Training Runner.

Trains and evaluates baselines and EquiTS under one protocol (TSLib splits,
L=96, H in {96, 192, 336, 720}), so results can be placed next to numbers
published in TSLib-based papers.

Usage:
    python run.py --dataset ETTh1 --model EquiTS --horizon 96 --device cuda
    python run.py --dataset all --model all --horizon all --seeds 2021 2022 2023 --device cuda
"""

import os
import json
import argparse
from collections import defaultdict

import numpy as np
import torch
import yaml

from experiments.exp_forecasting import Exp_Forecasting
from datasets.loader import DATASET_CONFIGS

BASELINES = ["Persistence", "Linear", "DLinear", "DLinear-Gauss", "PatchLinear", "LSTM", "TCN"]
ALL_DATASETS = ["ETTh1", "ETTm1", "Weather"]
ALL_HORIZONS = [96, 192, 336, 720]

TABLE_COLS = ["Model", "Parameters", "MSE", "MAE", "Highway-only MSE", "Highway-only MAE",
              "NLL", "CRPS", "Cov90", "Width90", "IS90", "CP Cov90", "CP Width90", "CP IS90",
              "Latency (ms)", "Bypass (%)",
              "Audit overhead (%)", "FMR among audited bypasses", "Est. false-bypass risk", "Val bypass (%)",
              "Bypass threshold (cal -> final)"]


def make_config(base_cfg, dataset, horizon):
    cfg = yaml.safe_load(yaml.dump(base_cfg))  # deep copy
    cfg["dataset"]["name"] = dataset
    cfg["dataset"]["horizon"] = horizon
    if dataset in DATASET_CONFIGS:
        info = DATASET_CONFIGS[dataset]
        cfg["dataset"]["data_path"] = info["file"]
        cfg["dataset"]["download_url"] = info["url"]
        cfg["dataset"]["num_channels"] = info["num_channels"]
        cfg["dataset"]["train_ratio"] = info["train_ratio"]
        cfg["dataset"]["val_ratio"] = info["val_ratio"]
    return cfg


def run_one(cfg, model_names, args, device, seed):
    results = []
    for name in model_names:
        if name.lower().startswith("equits"):
            continue
        print(f"\n>>> {name} | {cfg['dataset']['name']} H={cfg['dataset']['horizon']} seed={seed}")
        exp = Exp_Forecasting(cfg, name, device, seed=seed)
        exp.train(args.epochs)
        results.append(exp.test())

    if any(n.lower().startswith("equits") for n in model_names):
        print(f"\n>>> EquiTS | {cfg['dataset']['name']} H={cfg['dataset']['horizon']} seed={seed}")
        exp = Exp_Forecasting(cfg, "EquiTS", device, seed=seed)
        exp.train(args.epochs)
        results.append(exp.test())
        if cfg.get("memory", {}).get("enabled", False):
            exp.prepare_memory()
            for a in args.alphas:
                results.append(exp.test_rcpq(alpha=a))   # optional memory bypass (memory.enabled)
    return results


def aggregate(records):
    """Mean +/- std of numeric metrics over seeds, grouped by (dataset, horizon, model)."""
    groups = defaultdict(list)
    for r in records:
        groups[(r["Dataset"], r["Horizon"], r["Model"])].append(r)
    rows = []
    for (ds, h, model), rs in groups.items():
        row = {"Dataset": ds, "Horizon": h, "Model": model, "Seeds": len(rs)}
        for k, v in rs[0].items():
            if k in row or k == "Seed":
                continue
            vals = [x[k] for x in rs]
            if all(isinstance(x, int) for x in vals) and len(set(vals)) == 1:
                row[k] = vals[0]
            elif all(isinstance(x, (int, float)) for x in vals):
                mean = float(np.mean(vals))
                row[k] = f"{mean:.4f} ± {np.std(vals):.4f}" if len(vals) > 1 and k in ("MSE", "MAE") else round(mean, 4)
            else:
                row[k] = vals[0]
        rows.append(row)
    return rows


def write_tables(rows, out_dir, tag):
    os.makedirs(out_dir, exist_ok=True)
    lines = [f"# EquiTS Benchmark ({tag})", "",
             "TSLib splits, L=96, metrics on the standardized test split. "
             "MSE/MAE are mean ± std over seeds when several seeds were run.", ""]
    by_block = defaultdict(list)
    for r in rows:
        by_block[(r["Dataset"], r["Horizon"])].append(r)
    for (ds, h), rs in sorted(by_block.items()):
        cols = [c for c in TABLE_COLS if any(c in r for r in rs)]
        lines += [f"### {ds}, H={h}", "", "| " + " | ".join(cols) + " |", "|" + " :--- |" * len(cols)]
        for r in rs:
            lines.append("| " + " | ".join(str(r.get(c, "-")) for c in cols) + " |")
        lines.append("")
    md = os.path.join(out_dir, f"benchmark_{tag}.md")
    with open(md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\nSaved table: {md}")


def main():
    p = argparse.ArgumentParser(description="EquiTS benchmark runner")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--dataset", default="ETTh1", help="ETTh1, ETTm1, Weather, or all")
    p.add_argument("--model", default="EquiTS", help="EquiTS, all, or a single baseline name")
    p.add_argument("--horizon", default="96", help="96, 192, 336, 720, or all")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--seeds", type=int, nargs="+", default=None, help="Default: training.seed from config")
    p.add_argument("--alphas", type=float, nargs="+", default=None,
                   help="Target false-bypass risk levels for RCPQ (default: certification.target_risk_alpha)")
    p.add_argument("--device", default="cpu")
    p.add_argument("--out_dir", default="results")
    args = p.parse_args()

    with open(args.config) as f:
        base_cfg = yaml.safe_load(f)
    seeds = args.seeds or [base_cfg["training"].get("seed", 42)]
    if args.alphas is None:
        args.alphas = [base_cfg.get("certification", {}).get("target_risk_alpha", 0.05)]

    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    datasets = ALL_DATASETS if args.dataset.lower() == "all" else [args.dataset]
    horizons = ALL_HORIZONS if args.horizon.lower() == "all" else [int(args.horizon)]
    if args.model.lower() == "all":
        models = BASELINES + ["EquiTS"]
    elif args.model.lower() in ("equits", "our"):
        models = ["EquiTS"]
    else:
        models = [args.model]

    print(f"Device: {device} | datasets={datasets} horizons={horizons} models={models} seeds={seeds}")
    records = []
    for ds in datasets:
        for h in horizons:
            cfg = make_config(base_cfg, ds, h)
            for s in seeds:
                records += run_one(cfg, models, args, device, s)

    tag = "_".join(datasets).lower() + "_H" + "-".join(map(str, horizons))
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, f"benchmark_{tag}_raw.json"), "w") as f:
        json.dump(records, f, indent=2)
    write_tables(aggregate(records), args.out_dir, tag)


if __name__ == "__main__":
    main()
