# EquiTS: Equivalence-Quotiented Time-Series Network

Official implementation and standardized benchmarking suite for **EquiTS** (**Equi**valence-Quotiented **T**ime-**S**eries Network), an uncertainty-aware, high-speed deep learning architecture for general-purpose multivariate time-series forecasting.

---

## 1. Project Organization (Standard Benchmark Layout)

```
.
├── configs/
│   └── default.yaml             # Dataset and training hyperparameters
├── datasets/
│   ├── loader.py                # Automated download, caching & chronological splitting
│   ├── preprocessing.py         # Train-fitted standard scaler (strictly zero leakage)
│   └── windows.py               # Stride-1 sliding window generator
├── equits/
│   ├── revin.py                 # Reversible Instance Normalization
│   ├── encoder.py               # CI Large-Kernel Depthwise 1D Conv (K=31) + EPS projection
│   ├── predictor.py             # Dual probabilistic head (mu & logvar)
│   ├── divergence.py            # Exact Gaussian KL & normalized Wasserstein-2 (W2_bar)
│   ├── memory.py                # Two-Stage Memory Bank with Computational Bypass & Welford updates
│   ├── certification.py         # Online Adaptive Conformal Risk Control (O-CRC)
│   ├── losses.py                # Gaussian NLL & EPS alignment loss
│   └── model.py                 # Unified EquiTS model
├── baselines/                   # Standard Competitive Baselines (Trained under identical protocol)
│   ├── dlinear.py               # DLinear (AAAI 2023)
│   ├── patch_linear.py          # Patch-based Linear Forecaster (ICLR 2023)
│   ├── lstm.py                  # Standard Multi-layer LSTM Seq2Seq
│   └── tcn.py                   # Dilated Causal Convolutional Network
├── experiments/
│   └── exp_forecasting.py       # Unified trainer, early-stopping checkpointing, and evaluation
├── tests/                       # Unit tests (100% pass on CPU)
├── checkpoints/                 # Saved best model weights (.pt)
├── results/                     # Official benchmark matrix (markdown & json)
├── run.py                       # Unified publication runner (Trains & benchmarks all models)
└── README.md
```

---

## 2. Running the Benchmark

All models are trained and evaluated under the Time-Series-Library (TSLib) protocol:
the same chronological split borders as Autoformer / DLinear / PatchTST / iTransformer
(ETT: 12/4/4 months; Weather: 70/10/20), train-only scaling, lookback L=96, and
MSE/MAE on the standardized test split. Numbers are therefore directly comparable to
results published in TSLib-based papers. Use `dataset.split_protocol: "ratio"` in the
config only to reproduce the legacy 60/20/20 runs.

```bash
# Main table: every baseline + EquiTS, all datasets, H=96, 3 seeds (GPU recommended)
python run.py --dataset all --model all --horizon 96 --epochs 10 --seeds 2021 2022 2023 --device cuda

# EquiTS over all standard horizons (96, 192, 336, 720)
python run.py --dataset all --model EquiTS --horizon all --seeds 2021 2022 2023 --device cuda

# Risk / accuracy / latency trade-off over target risk levels alpha
python experiments/pareto_sweep.py --dataset ETTh1 --horizon 96 --device cuda
```

For each EquiTS run the table reports:
* `EquiTS_NoMem`: the full network, plus a highway-only ablation (deep residual removed)
  and probabilistic metrics (Gaussian NLL, empirical coverage of the 90% interval).
* `EquiTS_RCPQ(a=alpha)`: the memory bypass. The memory is prepopulated from training
  windows, the bypass threshold is calibrated on the validation split by conformal risk
  control at level alpha, then adapted online on the test stream from audited bypasses
  (ground truth arrives H steps late). Reported: bypass %, audit overhead %, observed
  false-merge rate among audits, and estimated per-query false-bypass risk.

### Running Unit Tests
```bash
python -m pytest tests
```
