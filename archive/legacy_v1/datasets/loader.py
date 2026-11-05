"""
Standardized Data Loader and Chronological Train/Val/Test Splitter.
Supports ETTh1, ETTm1, and Weather benchmark datasets with zero target leakage.

Split protocols:
  "tslib" (default): identical borders to the Time-Series-Library / Autoformer /
      DLinear / PatchTST / iTransformer code bases, so results are directly
      comparable to numbers published in those papers.
        ETTh*: 12 / 4 / 4 months of hourly data (8640 / 2880 / 2880 rows)
        ETTm*: same months at 15-min resolution (x4)
        other: 70% / 10% / 20%
      The validation and test segments start `lookback` rows early so the first
      forecast target begins exactly at the split border. The scaler is fit on the
      training segment only.
  "ratio": plain train_ratio / val_ratio / remainder split without the lookback
      overlap (the original protocol of this repo; NOT comparable to TSLib).
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
        "val_ratio": 0.2,
        "tslib_type": "ETTh"
    },
    "ETTm1": {
        "url": "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTm1.csv",
        "file": "data/ETTm1.csv",
        "num_channels": 7,
        "train_ratio": 0.6,
        "val_ratio": 0.2,
        "tslib_type": "ETTm"
    },
    "Weather": {
        "url": "https://huggingface.co/datasets/salmanali83/High_Dimensional_Time_Series/resolve/main/weather.csv",
        "file": "data/weather.csv",
        "num_channels": 21,
        "train_ratio": 0.7,
        "val_ratio": 0.1,
        "tslib_type": "custom"
    }
}


def compute_borders(total_len: int, lookback: int, protocol: str, tslib_type: str = "custom",
                    train_ratio: float = 0.7, val_ratio: float = 0.1):
    """Returns ([b1_train, b1_val, b1_test], [b2_train, b2_val, b2_test]) row borders."""
    if protocol == "tslib":
        if tslib_type in ("ETTh", "ETTm"):
            m = 30 * 24 * (4 if tslib_type == "ETTm" else 1)   # rows per month
            b1 = [0, 12 * m - lookback, 12 * m + 4 * m - lookback]
            b2 = [12 * m, 12 * m + 4 * m, 12 * m + 8 * m]
        else:
            num_train = int(total_len * 0.7)
            num_test = int(total_len * 0.2)
            num_val = total_len - num_train - num_test
            b1 = [0, num_train - lookback, total_len - num_test - lookback]
            b2 = [num_train, num_train + num_val, total_len]
        return b1, b2
    if protocol == "ratio":
        train_end = int(total_len * train_ratio)
        val_end = int(total_len * (train_ratio + val_ratio))
        return [0, train_end, val_end], [train_end, val_end, total_len]
    raise ValueError(f"Unknown split protocol: {protocol}")


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
    download_url: str = None,
    protocol: str = "tslib",
    tslib_type: str = "custom"
) -> Tuple[SlidingWindowDataset, SlidingWindowDataset, SlidingWindowDataset, TimeSeriesStandardScaler, List[str]]:
    """
    Chronological partition of multivariate time-series without future leakage.
    Scalers are strictly fitted on the training split only.
    """
    if not os.path.exists(csv_path) and download_url:
        download_if_missing(download_url, csv_path)

    df = pd.read_csv(csv_path)

    # Drop the timestamp column for array processing
    date_cols = [c for c in df.columns if c.lower() == "date"]
    feature_cols = [c for c in df.columns if c not in date_cols]

    raw_data = df[feature_cols].values.astype(np.float32)  # [T, D]
    b1, b2 = compute_borders(len(raw_data), lookback, protocol, tslib_type, train_ratio, val_ratio)

    scaler = TimeSeriesStandardScaler()
    scaler.fit(raw_data[b1[0]:b2[0]])

    splits = [scaler.transform(raw_data[b1[i]:b2[i]]) for i in range(3)]
    train_ds, val_ds, test_ds = [
        SlidingWindowDataset(seg, lookback=lookback, horizon=horizon, stride=1) for seg in splits
    ]
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
        tslib_type = ds_info["tslib_type"]
        config["dataset"]["num_channels"] = ds_info["num_channels"]
    else:
        data_path = cfg_data["data_path"]
        download_url = cfg_data.get("download_url")
        train_ratio = cfg_data.get("train_ratio", 0.6)
        val_ratio = cfg_data.get("val_ratio", 0.2)
        tslib_type = cfg_data.get("tslib_type", "custom")
        
    train_ds, val_ds, test_ds, scaler, _ = load_dataset_splits(
        csv_path=data_path,
        lookback=cfg_data["lookback"],
        horizon=cfg_data["horizon"],
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        download_url=download_url,
        protocol=cfg_data.get("split_protocol", "tslib"),
        tslib_type=tslib_type
    )
    
    batch_size = cfg_data.get("batch_size", 32)
    num_workers = cfg_data.get("num_workers", 0)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader, test_loader, scaler
