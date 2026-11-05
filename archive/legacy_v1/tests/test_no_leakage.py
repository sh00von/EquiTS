"""
Unit tests for data pipeline and zero-leakage guarantee.
"""

import pytest
import numpy as np
import torch
from datasets.preprocessing import TimeSeriesStandardScaler
from datasets.windows import SlidingWindowDataset


def test_scaler_no_leakage():
    # Synthetic time series with distinct means in train vs test
    train_data = np.ones((100, 3)) * 10.0
    test_data = np.ones((50, 3)) * 500.0
    
    scaler = TimeSeriesStandardScaler()
    scaler.fit(train_data)
    
    assert np.allclose(scaler.mean, 10.0), "Scaler mean was not fitted accurately on train data"
    
    # Transform test data using train parameters
    norm_test = scaler.transform(test_data)
    # Since std of constant train_data is clamped to 1.0, (500 - 10) / 1.0 = 490.0
    assert np.allclose(norm_test, 490.0), "Scaler transform leaked or miscalculated"
    
    # Ensure train transform has zero mean
    norm_train = scaler.transform(train_data)
    assert np.allclose(norm_train, 0.0), "Normalized train data is not zero-centered"


def test_sliding_window_separation_and_indexing():
    # Array with consecutive integer timestamps
    T = 200
    D = 4
    lookback = 96
    horizon = 96
    data = np.arange(T)[:, None].repeat(D, axis=1) # [T, D]
    
    ds = SlidingWindowDataset(data, lookback=lookback, horizon=horizon, stride=1)
    
    assert len(ds) == (T - lookback - horizon) + 1
    
    # Check first window
    x0, y0 = ds[0]
    assert x0.shape == (lookback, D)
    assert y0.shape == (horizon, D)
    
    # The last timestamp in x0 must be 95
    assert x0[-1, 0].item() == lookback - 1
    # The first timestamp in y0 must be 96 (strictly future, no overlap)
    assert y0[0, 0].item() == lookback
    assert y0[-1, 0].item() == lookback + horizon - 1
    
    # Zero overlap check
    assert x0[-1, 0].item() < y0[0, 0].item(), "Future target values leaked into input window!"

