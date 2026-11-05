"""
EquiTS: Equivariant Time Series normalization.

A plug-in layer that turns any point forecaster into a calibrated probabilistic
forecaster: affine-equivariant canonicalization with a relaxed scale (safe on flat
windows) for the mean path, and a decoupled scale head for the predictive variance.
"""

from equits.canonicalizer import Canonicalizer, relaxed_scale, MODES
from equits.layer import EquiTS, ScaleHead
from equits.metrics import (
    gaussian_nll, gaussian_crps, point_metrics, interval_metrics,
    conformal_interval, conformal_quantile, normal_z,
)

__version__ = "0.2.0"

__all__ = [
    "EquiTS", "ScaleHead", "Canonicalizer", "relaxed_scale", "MODES",
    "gaussian_nll", "gaussian_crps", "point_metrics", "interval_metrics",
    "conformal_interval", "conformal_quantile", "normal_z",
]
