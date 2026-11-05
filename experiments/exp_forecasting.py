"""
Standardized Time-Series Forecasting Experiment Engine.

Protocol (shared by every model, following Time-Series-Library conventions):
  - TSLib chronological splits (see datasets/loader.py), train-only scaling
  - AdamW, lr halved every epoch ("type1" schedule), gradient clipping
  - early stopping on validation MSE (patience from config), best checkpoint kept
  - MSE / MAE reported on the standardized test split

EquiTS-specific:
  - trained with Gaussian NLL (+ EPS key alignment loss when memory is configured)
  - one checkpoint serves both the no-memory variant and the memory (RCPQ) variant
  - RCPQ evaluation: memory prepopulated from the training split (untimed), key
    threshold calibrated by split conformal risk control on the validation split
    (untimed), then a timed streaming pass over the test split with audited
    bypasses, H-step delayed ground truth, and online adaptive risk control.
"""

import os
import copy
import random
import time
from collections import deque
from typing import Dict, Any, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

from datasets.loader import get_dataloaders
from equits.model import EquiTS
from equits.losses import (
    GaussianNLLLoss, EPSMetricAlignmentLoss, compute_forecast_metrics, gaussian_interval_coverage
)
from equits.divergence import compute_prototype_distances
from equits.certification import false_merge_indicator, split_crc_threshold, OnlineAdaptiveCRC
from baselines import DLinear, PatchLinear, LSTM, TCN, Persistence, Linear


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _sync(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def _rows(t: torch.Tensor) -> torch.Tensor:
    """[B, H, D] -> [B*D, H]"""
    B, H, D = t.shape
    return t.permute(0, 2, 1).reshape(B * D, H)


class Exp_Forecasting:
    """
    Trains any model on identical splits, keeps the best checkpoint by validation
    MSE, and evaluates on the held-out test split with wall-clock timing.
    """
    def __init__(self, config: Dict[str, Any], model_name: str, device: torch.device, seed: int = None):
        self.config = config
        self.model_name = model_name
        self.device = device
        self.seed = config["training"].get("seed", 42) if seed is None else seed
        set_seed(self.seed)

        self.lookback = config["dataset"]["lookback"]
        self.horizon = config["dataset"]["horizon"]

        self.train_loader, self.val_loader, self.test_loader, self.scaler = get_dataloaders(config)
        self.num_channels = config["dataset"]["num_channels"]

        self.is_equits = "equits" in model_name.lower()
        self.model = self._build_model().to(self.device)
        self.trainable = any(p.requires_grad for p in self.model.parameters())

        tr = config["training"]
        self.criterion_nll = GaussianNLLLoss(beta=tr.get("nll_beta", 0.0))
        self.criterion_mse = nn.MSELoss()
        self.base_lr = tr.get("lr", 0.001)
        self.grad_clip = tr.get("grad_clip", 1.0)
        self.patience = tr.get("patience", 3)
        self.lr_schedule = tr.get("lr_schedule", "type1")
        if self.trainable:
            self.optimizer = optim.AdamW(self.model.parameters(), lr=self.base_lr,
                                         weight_decay=tr.get("weight_decay", 1e-4))

        mem_cfg = config.get("memory", {})
        self.align_weight = tr.get("align_weight", 0.0) if self.is_equits and mem_cfg.get("enabled", False) else 0.0
        self.align_max_rows = tr.get("align_max_rows", 256)
        if self.align_weight > 0:
            self.criterion_align = EPSMetricAlignmentLoss(
                equiv_threshold=mem_cfg.get("merge_threshold", 0.25),
                key_margin=tr.get("align_key_margin", 0.3)
            )
            self.align_metric = mem_cfg.get("metric", "wasserstein_normalized")

        os.makedirs("checkpoints", exist_ok=True)
        ckpt_name = "EquiTS" if self.is_equits else self.model_name
        self.ckpt_path = f"checkpoints/{ckpt_name}_{config['dataset']['name']}_H{self.horizon}_s{self.seed}.pt"
        self._memory_snapshot = None

    def _build_model(self) -> nn.Module:
        name = self.model_name.lower()
        kw = dict(lookback=self.lookback, horizon=self.horizon, num_channels=self.num_channels)
        if name == "persistence":
            return Persistence(**kw)
        if name == "linear":
            return Linear(**kw)
        if "dlinear" in name:
            return DLinear(**kw)
        if "patch" in name:
            return PatchLinear(**kw)
        if "lstm" in name:
            return LSTM(**kw, hidden_dim=64)
        if "tcn" in name:
            return TCN(**kw, hidden_dim=64)
        if "equits" in name:
            model = EquiTS(self.config)
            model.memory_enabled = False   # memory is only switched on for RCPQ evaluation
            return model
        raise ValueError(f"Unknown model name: {self.model_name}")

    # --------------------------------------------------------------- training
    def _set_lr(self, epoch: int):
        if self.lr_schedule == "type1":
            lr = self.base_lr * (0.5 ** (epoch - 1))
        else:
            lr = self.base_lr
        for g in self.optimizer.param_groups:
            g["lr"] = lr

    def _equits_loss(self, x, y):
        mu, logvar, eps_keys, aux = self.model(x)
        loss = self.criterion_nll(mu, logvar, y)
        if self.align_weight > 0:
            N = eps_keys.shape[0]
            r = torch.randperm(N, device=eps_keys.device)[: self.align_max_rows]
            with torch.no_grad():
                rm, rl = aux["res_mu"][r].detach(), aux["res_logvar"][r].detach()
                dist = compute_prototype_distances(rm, rl, rm, rl, metric=self.align_metric)
            loss = loss + self.align_weight * self.criterion_align(eps_keys[r], dist)
        return loss

    def train_epoch(self) -> float:
        self.model.train()
        total = 0.0
        for x, y in self.train_loader:
            x, y = x.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()
            if self.is_equits:
                loss = self._equits_loss(x, y)
            else:
                loss = self.criterion_mse(self.model(x), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip)
            self.optimizer.step()
            total += loss.item()
        return total / max(1, len(self.train_loader))

    @torch.no_grad()
    def _predict(self, loader) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        self.model.eval()
        preds, logvars, targets = [], [], []
        for x, y in loader:
            x = x.to(self.device)
            if self.is_equits:
                mu, lv, _, _ = self.model(x)
                logvars.append(lv.cpu())
            else:
                mu = self.model(x)
            preds.append(mu.cpu())
            targets.append(y)
        lv = torch.cat(logvars) if logvars else None
        return torch.cat(preds), lv, torch.cat(targets)

    def validate(self) -> Dict[str, float]:
        pred, _, y = self._predict(self.val_loader)
        return compute_forecast_metrics(pred, y)

    def train(self, epochs: int) -> float:
        if not self.trainable:
            print(f"\n[{self.model_name}] has no trainable parameters; skipping training.")
            return float("nan")
        best, bad = float("inf"), 0
        print(f"\nTraining [{self.model_name}] (seed {self.seed}) for up to {epochs} epochs...")
        for epoch in range(1, epochs + 1):
            self._set_lr(epoch)
            t0 = time.perf_counter()
            train_loss = self.train_epoch()
            val = self.validate()
            dt = time.perf_counter() - t0
            print(f"  Epoch [{epoch:02d}/{epochs:02d}] ({dt:.1f}s) | Train Loss: {train_loss:.4f} | "
                  f"Val MSE: {val['MSE']:.4f} | Val MAE: {val['MAE']:.4f}")
            if val["MSE"] < best:
                best, bad = val["MSE"], 0
                # Learned weights only: the memory bank is rebuilt from training data at test time.
                weights = {k: v for k, v in self.model.state_dict().items() if not k.startswith("memory.")}
                torch.save(weights, self.ckpt_path)
            else:
                bad += 1
                if bad >= self.patience:
                    print(f"  Early stopping (patience {self.patience}).")
                    break
        print(f"  Best checkpoint (val MSE {best:.4f}): {self.ckpt_path}")
        return best

    def load_best(self):
        if self.trainable and os.path.exists(self.ckpt_path):
            state = torch.load(self.ckpt_path, map_location=self.device, weights_only=True)
            state = {k: v for k, v in state.items() if not k.startswith("memory.")}
            missing, unexpected = self.model.load_state_dict(state, strict=False)
            missing = [k for k in missing if not k.startswith("memory.")]
            if missing or unexpected:
                raise RuntimeError(f"Checkpoint mismatch. Missing: {missing}; unexpected: {unexpected}")
        self.model.eval()

    # ---------------------------------------------------------- point testing
    @torch.no_grad()
    def test(self) -> Dict[str, Any]:
        """Standard evaluation (baselines and EquiTS without memory)."""
        self.load_best()
        if self.is_equits:
            self.model.memory_enabled = False
        self._warmup()

        preds, logvars, targets = [], [], []
        _sync(self.device)
        t0 = time.perf_counter()
        for x, y in self.test_loader:
            x = x.to(self.device)
            if self.is_equits:
                mu, lv, _, _ = self.model(x)
                logvars.append(lv)
            else:
                mu = self.model(x)
            preds.append(mu)
            targets.append(y)
        _sync(self.device)
        elapsed = time.perf_counter() - t0

        pred = torch.cat(preds).cpu()
        y = torch.cat(targets)
        name = "EquiTS_NoMem" if self.is_equits else self.model_name
        res = self._base_record(name, pred, y, elapsed)
        if self.is_equits:
            res.update(self._prob_metrics(pred, torch.cat(logvars).cpu(), y))
            # Ablation: does the deep trunk improve on the linear highway alone?
            hw = torch.cat([self.model.forward_highway_only(x.to(self.device)).cpu()
                            for x, _ in self.test_loader])
            hm = compute_forecast_metrics(hw, y)
            res["Highway-only MSE"] = round(hm["MSE"], 4)
            res["Highway-only MAE"] = round(hm["MAE"], 4)
        return res

    def _warmup(self):
        with torch.no_grad():
            for i, (x, _) in enumerate(self.test_loader):
                self.model(x.to(self.device))
                if i >= 2:
                    break

    def _base_record(self, name, pred, y, elapsed) -> Dict[str, Any]:
        m = compute_forecast_metrics(pred, y)
        n = len(pred)
        return {
            "Model": name,
            "Dataset": self.config["dataset"]["name"],
            "Horizon": self.horizon,
            "Seed": self.seed,
            "Parameters": sum(p.numel() for p in self.model.parameters() if p.requires_grad),
            "MSE": round(m["MSE"], 4),
            "MAE": round(m["MAE"], 4),
            "Latency (ms)": round(elapsed / n * 1000.0, 4),
            "Throughput": round(n / elapsed, 1),
        }

    def _prob_metrics(self, mu, logvar, y) -> Dict[str, Any]:
        nll = GaussianNLLLoss(include_constant=True)(mu, logvar, y).item()
        return {"NLL": round(nll, 4), "Cov90": round(gaussian_interval_coverage(mu, logvar, y), 4)}

    # ------------------------------------------------------- RCPQ evaluation
    @torch.no_grad()
    def prepare_memory(self):
        """Prepopulates the memory bank from (evenly spaced) training windows. Untimed."""
        self.load_best()
        mem = self.model.memory
        mem.reset()
        ds = self.train_loader.dataset
        max_w = self.config["memory"].get("prepopulate_max_windows", 4000)
        idx = np.linspace(0, len(ds) - 1, num=min(max_w, len(ds))).astype(int).tolist()
        loader = DataLoader(Subset(ds, idx), batch_size=self.train_loader.batch_size, shuffle=False)
        for x, _ in loader:
            self.model.populate_memory(x.to(self.device))
        print(f"  Memory prepopulated from {len(idx)} training windows: "
              f"{mem.num_active_prototypes}/{mem.max_prototypes} prototypes, "
              f"{mem.total_merges} merges, {mem.total_evictions} evictions")
        self._memory_snapshot = copy.deepcopy(mem.state_dict())

    @torch.no_grad()
    def calibrate(self, alpha: float, tolerance: float) -> Dict[str, float]:
        """Split conformal risk control of the bypass threshold on the validation split. Untimed."""
        d_all, e_all = [], []
        for x, y in self.val_loader:
            d_star, mu_full, mu_bp = self.model.calibration_forward(x.to(self.device))
            e = false_merge_indicator(mu_bp, mu_full, _rows(y.to(self.device)), tolerance)
            e = torch.where(torch.isfinite(d_star), e, torch.zeros_like(e))
            d_all.append(d_star.cpu())
            e_all.append(e.cpu())
        d, e = torch.cat(d_all), torch.cat(e_all)
        lam = split_crc_threshold(d, e, alpha)
        hit = d <= lam
        return {
            "lambda_cal": lam,
            "val_bypass": hit.float().mean().item(),
            "val_false_bypass_risk": (hit.float() * e).mean().item(),
        }

    @torch.no_grad()
    def test_rcpq(self, alpha: Optional[float] = None) -> Dict[str, Any]:
        """Timed streaming evaluation with the memory bypass and online risk control."""
        cert = self.config.get("certification", {})
        mem_cfg = self.config.get("memory", {})
        alpha = cert.get("target_risk_alpha", 0.05) if alpha is None else alpha
        tol = cert.get("false_merge_tolerance", 0.1)
        audit_p = cert.get("audit_prob", 0.1)
        online_update = mem_cfg.get("online_update", False)

        if self._memory_snapshot is None:
            self.prepare_memory()
        mem = self.model.memory
        mem.load_state_dict(self._memory_snapshot)
        mem.reset_stats()

        self.model.memory_enabled = True
        cal = self.calibrate(alpha, tol)
        crc = OnlineAdaptiveCRC(
            target_risk_alpha=alpha,
            eta=cert.get("learning_rate_conformal", 0.001),
            initial_threshold=cal["lambda_cal"],
            min_threshold=cert.get("min_bypass_threshold", 0.0),
            max_threshold=cert.get("max_bypass_threshold", 2.0),
        )
        mem.bypass_threshold.fill_(crc.get_threshold())

        D = self.num_channels
        H = self.horizon
        pending = deque()          # (resolve_window_index, num_decisions, weighted_false_merges)
        n_bypass = n_audit = n_tokens = 0
        audited_fm = 0.0
        preds, logvars, targets = [], [], []
        window = 0

        _sync(self.device)
        t0 = time.perf_counter()
        for x, y in self.test_loader:
            x, y = x.to(self.device), y.to(self.device)
            B = x.shape[0]
            mu, lv, _, aux = self.model(x, update_memory=online_update, audit_prob=audit_p)
            n_bypass += aux["bypassed_count"]
            n_audit += aux["audited_count"]
            n_tokens += aux["total_tokens"]

            # False-merge labels for audited bypasses. They only become usable once the
            # target window has been observed, i.e. H windows later.
            weighted = torch.zeros(B, device=self.device)
            if "audit_mask" in aux:
                am = aux["audit_mask"].view(B, D)
                fm = false_merge_indicator(_rows(mu), _rows(aux["mu_full"]), _rows(y), tol).view(B, D)
                fm = fm * am.float()
                audited_fm += fm.sum().item()
                weighted = fm.sum(dim=1) / audit_p
            for b, w in enumerate(weighted.tolist()):
                pending.append((window + b + H, D, w))
            window += B

            # Resolve feedback whose ground truth is now available
            n_res, w_res = 0, 0.0
            while pending and pending[0][0] <= window - 1:
                _, n_d, w = pending.popleft()
                n_res += n_d
                w_res += w
            if n_res:
                mem.bypass_threshold.fill_(crc.update(n_res, w_res))

            preds.append(mu)
            logvars.append(lv)
            targets.append(y)
        _sync(self.device)
        elapsed = time.perf_counter() - t0

        pred = torch.cat(preds).cpu()
        y_all = torch.cat(targets).cpu()
        res = self._base_record(f"EquiTS_RCPQ(a={alpha})", pred, y_all, elapsed)
        res.update(self._prob_metrics(pred, torch.cat(logvars).cpu(), y_all))
        res.update({
            "Alpha": alpha,
            "Bypass (%)": round(100.0 * n_bypass / max(1, n_tokens), 2),
            "Audit overhead (%)": round(100.0 * n_audit / max(1, n_tokens), 2),
            "FMR among audited bypasses": round(audited_fm / max(1, n_audit), 4),
            "Est. false-bypass risk": round(crc.estimated_risk(), 4),
            "Val false-bypass risk": round(cal["val_false_bypass_risk"], 4),
            "Val bypass (%)": round(100.0 * cal["val_bypass"], 2),
            "Bypass threshold (cal -> final)": f"{cal['lambda_cal']:.4g} -> {crc.get_threshold():.4g}",
            "Prototypes": mem.num_active_prototypes,
        })
        self.model.memory_enabled = False
        return res
