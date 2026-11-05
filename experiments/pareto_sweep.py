"""
EquiTS accuracy / latency / risk trade-off.

Trains (or loads) one EquiTS checkpoint, then evaluates the memory bypass at
several target risk levels alpha. Each point is conformally calibrated on the
validation split and adapted online on the test stream, so the x-axis is a
controlled risk level rather than a hand-tuned threshold.

Usage:
    python experiments/pareto_sweep.py --dataset ETTh1 --device cuda
"""

import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
os.chdir(ROOT_DIR)

import json
import argparse
import torch
import yaml

from experiments.exp_forecasting import Exp_Forecasting
from run import make_config


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--dataset", default="ETTh1")
    p.add_argument("--horizon", type=int, default=96)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--alphas", type=float, nargs="+", default=[0.01, 0.02, 0.05, 0.1, 0.2, 0.3])
    p.add_argument("--retrain", action="store_true", help="Retrain even if a checkpoint exists")
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    with open(args.config) as f:
        cfg = make_config(yaml.safe_load(f), args.dataset, args.horizon)
    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")

    exp = Exp_Forecasting(cfg, "EquiTS", device, seed=args.seed)
    if args.retrain or not os.path.exists(exp.ckpt_path):
        exp.train(args.epochs)
    results = [exp.test()]
    exp.prepare_memory()
    for a in args.alphas:
        r = exp.test_rcpq(alpha=a)
        print(f"  alpha={a:<5} MSE={r['MSE']:.4f} MAE={r['MAE']:.4f} latency={r['Latency (ms)']:.4f}ms "
              f"bypass={r['Bypass (%)']}% est.risk={r['Est. false-bypass risk']}")
        results.append(r)

    base_lat = results[0]["Latency (ms)"]
    for r in results:
        r["Speedup"] = round(base_lat / max(1e-9, r["Latency (ms)"]), 3)

    os.makedirs("results", exist_ok=True)
    tag = f"{args.dataset.lower()}_H{args.horizon}"
    with open(f"results/pareto_{tag}.json", "w") as f:
        json.dump(results, f, indent=2)
    cols = ["Model", "MSE", "MAE", "Latency (ms)", "Speedup", "Bypass (%)", "Audit overhead (%)",
            "FMR among audited bypasses", "Est. false-bypass risk", "Val false-bypass risk",
            "Bypass threshold (cal -> final)"]
    lines = [f"# EquiTS risk / accuracy / latency trade-off ({args.dataset}, H={args.horizon})", "",
             "| " + " | ".join(cols) + " |", "|" + " :--- |" * len(cols)]
    for r in results:
        lines.append("| " + " | ".join(str(r.get(c, "-")) for c in cols) + " |")
    with open(f"results/pareto_{tag}.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Saved results/pareto_{tag}.md")


if __name__ == "__main__":
    main()
