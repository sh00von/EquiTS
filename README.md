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

## 2. Running the Full Publication Benchmark

To train **every model from scratch** on identical ETTh1 training splits, evaluate each model on the held-out test split, and generate the official paper-ready comparison table:

```bash
python run.py --model all --epochs 5 --device cpu
```

### Running on Kaggle or Google Colab (GPU)
```bash
python run.py --model all --epochs 20 --device cuda
```

### Running a Specific Model
```bash
python run.py --model EquiTS_RCPQ --epochs 10 --device cpu
```

### Running Unit Tests
```bash
python -m pytest tests
```
