"""
Standardized Time-Series Forecasting Experiment Engine.
Provides fair, identical training budgets, loss functions, and evaluation metrics
across all baseline architectures and EquiTS, following modern benchmark standards (e.g. THUML / TSLib).
"""

import os
import time
from typing import Dict, Any, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from datasets.loader import get_dataloaders
from equits.model import EquiTS
from equits.losses import GaussianNLLLoss, compute_forecast_metrics
from baselines import DLinear, PatchLinear, LSTM, TCN


class Exp_Forecasting:
    """
    Standard Time-Series Forecasting Experiment Harness.
    Trains any model on identical train splits, saves best checkpoint based on validation loss,
    and runs official evaluation on held-out test splits with high-precision wall-clock timing.
    """
    def __init__(self, config: Dict[str, Any], model_name: str, device: torch.device):
        self.config = config
        self.model_name = model_name
        self.device = device
        
        self.lookback = config["dataset"]["lookback"]
        self.horizon = config["dataset"]["horizon"]
        self.num_channels = config["dataset"]["num_channels"]
        
        # Build DataLoaders
        self.train_loader, self.val_loader, self.test_loader, self.scaler = get_dataloaders(config)
        
        # Build Model
        self.model = self._build_model().to(self.device)
        self.is_equits = "EquiTS" in self.model_name
        self.is_probabilistic = self.is_equits

        # Loss and Optimizer
        self.criterion_nll = GaussianNLLLoss()
        self.criterion_mse = nn.MSELoss()
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config["training"].get("lr", 0.001),
            weight_decay=config["training"].get("weight_decay", 1e-4)
        )
        
        os.makedirs("checkpoints", exist_ok=True)
        self.ckpt_path = f"checkpoints/{self.model_name}_{config['dataset']['name']}_H{self.horizon}.pt"

    def _build_model(self) -> nn.Module:
        name = self.model_name.lower()
        if "dlinear" in name:
            return DLinear(lookback=self.lookback, horizon=self.horizon, num_channels=self.num_channels)
        elif "patch" in name:
            return PatchLinear(lookback=self.lookback, horizon=self.horizon, num_channels=self.num_channels)
        elif "lstm" in name:
            return LSTM(lookback=self.lookback, horizon=self.horizon, num_channels=self.num_channels, hidden_dim=64)
        elif "tcn" in name:
            return TCN(lookback=self.lookback, horizon=self.horizon, num_channels=self.num_channels, hidden_dim=64)
        elif "equits" in name:
            model = EquiTS(self.config)
            if "nomem" in name:
                model.memory_enabled = False
            else:
                model.memory_enabled = True
            return model
        else:
            raise ValueError(f"Unknown model name: {self.model_name}")

    def train_epoch(self, epoch: int, epochs: int) -> float:
        self.model.train()
        total_loss = 0.0
        
        for batch_idx, (x, y) in enumerate(self.train_loader):
            x, y = x.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()
            
            if self.is_equits:
                mu, logvar, _, _ = self.model(x)
                loss = self.criterion_nll(mu, logvar, y)
            else:
                pred = self.model(x)
                loss = self.criterion_mse(pred, y)
                
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            total_loss += loss.item()
            
        return total_loss / max(1, len(self.train_loader))

    @torch.no_grad()
    def validate(self) -> Tuple[float, float, float]:
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_targets = []
        
        for step, (x, y) in enumerate(self.val_loader):
            x, y = x.to(self.device), y.to(self.device)
            
            if self.is_equits:
                mu, logvar, _, _ = self.model(x, current_step=float(step), update_memory=False)
                pred = mu
                loss = self.criterion_nll(mu, logvar, y)
            else:
                pred = self.model(x)
                loss = self.criterion_mse(pred, y)
                
            total_loss += loss.item()
            all_preds.append(pred.cpu())
            all_targets.append(y.cpu())
            
        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        metrics = compute_forecast_metrics(all_preds, all_targets)
        return total_loss / max(1, len(self.val_loader)), metrics["MAE"], metrics["MSE"]

    def train(self, epochs: int) -> float:
        best_val_loss = float("inf")
        print(f"\nTraining [{self.model_name}] for {epochs} epochs...")
        
        for epoch in range(1, epochs + 1):
            t0 = time.perf_counter()
            train_loss = self.train_epoch(epoch, epochs)
            val_loss, val_mae, val_mse = self.validate()
            dt = time.perf_counter() - t0
            
            print(f"  Epoch [{epoch:02d}/{epochs:02d}] ({dt:.2f}s) | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val MAE: {val_mae:.4f} | Val MSE: {val_mse:.4f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), self.ckpt_path)
                
        print(f"  Best checkpoint saved: {self.ckpt_path}")
        return best_val_loss

    @torch.no_grad()
    def test(self, use_memory_quotienting: bool = True) -> Dict[str, Any]:
        # Load best trained model weights
        if os.path.exists(self.ckpt_path):
            self.model.load_state_dict(torch.load(self.ckpt_path, map_location=self.device))
            
        self.model.eval()
        if self.is_equits:
            self.model.memory_enabled = use_memory_quotienting
            
        # Warmup
        for i, (x, y) in enumerate(self.test_loader):
            x = x.to(self.device)
            _ = self.model(x)
            if i >= 2:
                break

        # Benchmarked evaluation pass with timer
        all_preds = []
        all_targets = []
        total_bypassed = 0
        total_tokens = 0
        
        t0 = time.perf_counter()
        for step, (x, y) in enumerate(self.test_loader):
            x, y = x.to(self.device), y.to(self.device)
            
            if self.is_equits:
                mu, logvar, _, aux = self.model(
                    x,
                    current_step=float(step),
                    update_memory=use_memory_quotienting
                )
                pred = mu
                total_bypassed += aux.get("bypassed_count", 0)
                total_tokens += aux.get("total_tokens", 0)
            else:
                pred = self.model(x)
                
            all_preds.append(pred.cpu())
            all_targets.append(y.cpu())
            
        test_time = time.perf_counter() - t0
        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        metrics = compute_forecast_metrics(all_preds, all_targets)
        num_samples = len(all_preds)
        latency_ms = (test_time / num_samples) * 1000.0
        param_count = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        bypass_pct = (total_bypassed / max(1, total_tokens)) * 100 if self.is_equits and use_memory_quotienting else 0.0
        
        return {
            "Model": self.model_name,
            "Parameters": f"{param_count:,}",
            "MAE": round(metrics["MAE"], 4),
            "MSE": round(metrics["MSE"], 4),
            "RMSE": round(metrics["RMSE"], 4),
            "Latency (ms)": round(latency_ms, 3),
            "Throughput": round(num_samples / test_time, 1),
            "Bypass (%)": round(bypass_pct, 2) if bypass_pct > 0 else "-"
        }

