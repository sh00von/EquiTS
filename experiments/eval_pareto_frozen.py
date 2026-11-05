import os
import sys
import time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import yaml
from equits.model import EquiTS
from datasets.loader import get_dataloaders
from equits.losses import compute_forecast_metrics

with open('configs/default.yaml') as f:
    cfg = yaml.safe_load(f)

train_loader, _, test_loader, _ = get_dataloaders(cfg)
model = EquiTS(cfg)
model.load_state_dict(torch.load('checkpoints/EquiTS_RCPQ_ETTh1_H96.pt', map_location='cpu'))
model.eval()

# 1. Pre-populate on training data
model.memory_enabled = True
model.memory.threshold.fill_(1e-5)
model.memory.occupied.zero_()
with torch.no_grad():
    for x, _ in train_loader:
        _ = model(x, current_step=0.0, update_memory=True)
        if model.memory.occupied.all():
            break

print(f"Populated {model.memory.num_active_prototypes} prototypes from train set.")

# Baseline without memory
model.memory_enabled = False
t0 = time.perf_counter()
all_preds, all_y = [], []
with torch.no_grad():
    for x, y in test_loader:
        mu, _, _, _ = model(x)
        all_preds.append(mu)
        all_y.append(y)
base_time = (time.perf_counter() - t0) / len(test_loader.dataset) * 1000.0
base_m = compute_forecast_metrics(torch.cat(all_preds), torch.cat(all_y))
print(f"Baseline (No Memory) -> MAE: {base_m['MAE']:.4f} | MSE: {base_m['MSE']:.4f} | Latency: {base_time:.3f} ms")

# 2. Frozen memory inference across thresholds
model.memory_enabled = True
for tau in [1e-7, 3e-7, 7e-7, 1.5e-6, 3e-6, 7e-6, 1.5e-5]:
    model.memory.threshold.fill_(tau * 2.0)
    all_preds, all_y = [], []
    tot_bp, tot_tok = 0, 0
    t0 = time.perf_counter()
    with torch.no_grad():
        for s, (x, y) in enumerate(test_loader):
            mu, _, _, aux = model(x, current_step=float(s), update_memory=False)
            all_preds.append(mu)
            all_y.append(y)
            tot_bp += aux["bypassed_count"]
            tot_tok += aux["total_tokens"]
    elapsed = time.perf_counter() - t0
    preds = torch.cat(all_preds, dim=0)
    ys = torch.cat(all_y, dim=0)
    m = compute_forecast_metrics(preds, ys)
    lat = (elapsed / len(preds)) * 1000.0
    bp = (tot_bp / tot_tok) * 100.0
    speedup = base_time / max(1e-6, lat)
    print(f"tau: {tau:8.2e} | MAE: {m['MAE']:.4f} | MSE: {m['MSE']:.4f} | Latency: {lat:.3f} ms | Speedup: {speedup:.2f}x | Bypass: {bp:5.1f}%")

