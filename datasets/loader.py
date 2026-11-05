"""
Standardized Data Loader and Chronological Train/Val/Test Splitter.
Supports ETTh1, ETTm1, and Weather benchmark datasets with zero target leakage.
"""

import os
import urllib.request
from typing import Tuple, Dict, Any, List
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from datasets.preprocessing import TimeSeriesStandardScaler
from datasets.windows import SlidingWindowDataset

DATASET_CONFIGS = {
    "ETTh1": {
        "url": "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv",
        "file": "data/ETTh1.csv",
        "num_channels": 7,
        "train_ratio": 0.6,
        "val_ratio": 0.2
    },
    "ETTm1": {
        "url": "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTm1.csv",
        "file": "data/ETTm1.csv",
        "num_channels": 7,
        "train_ratio": 0.6,
        "val_ratio": 0.2
    },
    "Weather": {
        "url": "https://huggingface.co/datasets/salmanali83/High_Dimensional_Time_Series/resolve/main/weather.csv",
        "file": "data/weather.csv",
        "num_channels": 21,
        "train_ratio": 0.7,
        "val_ratio": 0.1
    }
}


def download_if_missing(url: str, dest_path: str):
    """Download benchmark dataset if not already present."""
    if not os.path.exists(dest_path):
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        print(f"Downloading dataset from {url} to {dest_path}...")
        try:
            # Set User-Agent header for repositories like Hugging Face
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req) as response, open(dest_path, "wb") as out_file:
                out_file.write(response.read())
            print("Download complete.")
        except Exception as e:
            raise RuntimeError(f"Failed to download dataset from {url}: {e}")


def load_dataset_splits(
    csv_path: str,
    lookback: int = 96,
    horizon: int = 96,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    download_url: str = None
) -> Tuple[SlidingWindowDataset, SlidingWindowDataset, SlidingWindowDataset, TimeSeriesStandardScaler, List[str]]:
    """
    Chronological partition of multivariate time-series without future leakage.
    Scalers are strictly fitted on the training split only.
    """
    if not os.path.exists(csv_path) and download_url:
        download_if_missing(download_url, csv_path)
        
    df = pd.read_csv(csv_path)
    
    # Drop date/timestamp column for array processing
    date_cols = [c for c in df.columns if "date" in c.lower() or "time" in c.lower()]
    feature_cols = [c for c in df.columns if c not in date_cols]
    
    raw_data = df[feature_cols].values.astype(np.float32) # [T, D]
    total_len = len(raw_data)
    
    train_end = int(total_len * train_ratio)
    val_end = int(total_len * (train_ratio + val_ratio))
    
    train_raw = raw_data[:train_end]
    val_raw = raw_data[train_end:val_end]
    test_raw = raw_data[val_end:]
    
    # Fit scaler strictly on training split
    scaler = TimeSeriesStandardScaler()
    scaler.fit(train_raw)
    
    train_norm = scaler.transform(train_raw)
    val_norm = scaler.transform(val_raw)
    test_norm = scaler.transform(test_raw)
    
    train_ds = SlidingWindowDataset(train_norm, lookback=lookback, horizon=horizon, stride=1)
    val_ds = SlidingWindowDataset(val_norm, lookback=lookback, horizon=horizon, stride=1)
    test_ds = SlidingWindowDataset(test_norm, lookback=lookback, horizon=horizon, stride=1)
    
    return train_ds, val_ds, test_ds, scaler, feature_cols


def get_dataloaders(
    config: Dict[str, Any]
) -> Tuple[DataLoader, DataLoader, DataLoader, TimeSeriesStandardScaler]:
    cfg_data = config["dataset"]
    ds_name = cfg_data.get("name", "ETTh1")
    
    # Auto-resolve defaults if known dataset
    if ds_name in DATASET_CONFIGS:
        ds_info = DATASET_CONFIGS[ds_name]
        data_path = cfg_data.get("data_path", ds_info["file"])
        download_url = cfg_data.get("download_url", ds_info["url"])
        train_ratio = cfg_data.get("train_ratio", ds_info["train_ratio"])
        val_ratio = cfg_data.get("val_ratio", ds_info["val_ratio"])
        config["dataset"]["num_channels"] = ds_info["num_channels"]
    else:
        data_path = cfg_data["data_path"]
        download_url = cfg_data.get("download_url")
        train_ratio = cfg_data.get("train_ratio", 0.6)
        val_ratio = cfg_data.get("val_ratio", 0.2)
        
    train_ds, val_ds, test_ds, scaler, _ = load_dataset_splits(
        csv_path=data_path,
        lookback=cfg_data["lookback"],
        horizon=cfg_data["horizon"],
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        download_url=download_url
    )
    
    batch_size = cfg_data.get("batch_size", 32)
    num_workers = cfg_data.get("num_workers", 0)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader, test_loader, scaler
