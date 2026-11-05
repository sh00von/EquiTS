import os
import sys
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

# 1. Pre-populate memory on training set
print('Pre-populating memory prototypes on training split...')
model.memory_enabled = True
model.memory.occupied.zero_()
model.memory.threshold.fill_(0.001)

step = 0
with torch.no_grad():
    for x, _ in train_loader:
        _ = model(x, current_step=float(step), update_memory=True)
        step += 1
        if model.memory.occupied.all():
            break

print(f'Prototypes populated: {model.memory.num_active_prototypes}/{model.memory.max_prototypes}')

# 2. Evaluate on test set with frozen memory
all_preds, all_y = [], []
tot_bp, tot_tok = 0, 0
with torch.no_grad():
    for step, (x, y) in enumerate(test_loader):
        mu, _, _, aux = model(x, current_step=float(step), update_memory=False)
        all_preds.append(mu)
        all_y.append(y)
        tot_bp += aux['bypassed_count']
        tot_tok += aux['total_tokens']

preds = torch.cat(all_preds, dim=0)
ys = torch.cat(all_y, dim=0)
m = compute_forecast_metrics(preds, ys)
bp_rate = (tot_bp / tot_tok) * 100.0
print(f"Pre-populated Memory Evaluation -> MAE: {m['MAE']:.4f} | MSE: {m['MSE']:.4f} | Bypass: {bp_rate:.1f}%")

