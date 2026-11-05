# EquiTS: Equivariant Time Series normalization

A plug-in layer that turns **any point forecaster** (DLinear, PatchTST, iTransformer, ...)
into a **calibrated probabilistic forecaster**, without changing the backbone and
without changing its point forecast.

```python
from equits import EquiTS

layer = EquiTS(seq_len=96, pred_len=96, num_channels=7)        # normalizer="learned"
mu, logvar = layer(x, lambda z: backbone(z))                     # x: [B, L, D]
```

## Method

* **Affine-equivariant canonicalization.** The window is mapped to a canonical
  location/scale `(m, s)`, the backbone forecasts in normalized units, and the
  forecast is mapped back: `mu = m + s * backbone((x - m) / s)`.
* **Relaxed scale for flat windows.** No continuous canonicalizer can be exactly
  scale-equivariant on constant windows (they force `s = 0`). EquiTS uses
  `s_eps = sqrt(s^2 + (eps * sigma0)^2)`: exactly shift-equivariant, scale-equivariant
  up to an explicit bound `|a - 1| * eps * sigma0`, and never collapses. RevIN's
  `+1e-5` is an uncontrolled version of this and fails on flat windows.
* **Learned canonicalizer.** Location and scale are softmax-weighted statistics whose
  weights depend only on the window's normalized shape, so equivariance holds by
  construction. It is initialized exactly at the plain mean/std.
* **Decoupled scale head.** A small head predicts the normalized variance from the
  normalized window and forecast; `Var = s^2 (sigma_h^2 + tau^2)`, where `tau^2` is
  a level-shift variance shared across the horizon. It is trained with Gaussian NLL
  on detached inputs, so the mean forecast is exactly "backbone + normalization"
  trained with MSE.

## Repository

| Path | Contents |
| :--- | :--- |
| `equits/canonicalizer.py` | RevIN / relaxed / learned canonicalizers |
| `equits/layer.py` | `EquiTS` layer and scale head |
| `equits/metrics.py` | MSE/MAE, NLL, CRPS, interval score, split conformal intervals |
| `tests/` | Property tests (equivariance, error bound, flat windows, mean/variance decoupling) |
| `experiments/tslib_runner.py` | Experiments on the official Time-Series-Library at a pinned commit |
| `demo_colab.ipynb` | Colab notebook for the pilot |
| `archive/legacy_v1/` | Previous EquiTS model (prototype memory and bypass); not maintained |

## Experiments

Backbones, data loaders, splits and hyperparameters come unmodified from
[Time-Series-Library](https://github.com/thuml/Time-Series-Library) at commit
`4e938a1767106324dd753b2a44832bf870a0252e`.

```bash
pip install -r requirements.txt
git clone https://github.com/thuml/Time-Series-Library.git TSL
git -C TSL checkout 4e938a1767106324dd753b2a44832bf870a0252e
python -m pytest tests -q
python experiments/tslib_runner.py --tslib_path TSL --pilot --device cuda
```

Results are appended to `results/pilot.jsonl`, and finished runs are skipped on restart.

### Pilot and pre-registered decision rule

DLinear and PatchTST on ETTh1 and Weather, H = 96, seed 2021. Each backbone/dataset pair is
trained with three normalizers:

| Run | Reported as |
| :--- | :--- |
| RevIN, no shift term | (1) RevIN + conformal interval, (2) RevIN + scale head (**the rival**) |
| learned, shift term | (3) **EquiTS** |
| relaxed, shift term | (4) EquiTS without learned weights (ablation) |

The rule, fixed before any results:
* **GO:** (3) improves CRPS over (2) by at least 3% on average, its coverage is closer to
  90%, and no pair is worse by more than 1%.
* **WEAK:** a 1–3% gain.
* **STOP:** below 1%.

Runs (1) and (2) share the same trained mean, so their MSE is identical. (3) and (4) change
the normalizer, so their MSE can differ.
