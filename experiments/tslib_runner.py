"""
EquiTS experiments on top of the official Time-Series-Library (TSLib).

TSLib is used unmodified, cloned at a pinned commit: its data loaders (exact
published splits and scaling), its backbones, its default hyperparameters and
the hyperparameters of its official scripts. EquiTS wraps the backbone:

    mu, logvar = EquiTS(x, lambda z: backbone(z, x_mark, dec_inp, y_mark))

Training (two stages, so the variance can never affect the mean):
  A. backbone + canonicalizer, MSE, Adam, TSLib lr / "type1" schedule / early stopping
  B. scale head only (mean path frozen, eval mode), Gaussian NLL, early stopping on val NLL

Pilot (pre-registered decision rule, see README):
    python experiments/tslib_runner.py --tslib_path TSL --pilot --device cuda
Single run:
    python experiments/tslib_runner.py --tslib_path TSL --model DLinear --data ETTh1 --normalizer learned
"""

import argparse
import copy
import json
import os
import random
import sys
import textwrap
import time
import types
import urllib.request

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from equits import EquiTS, gaussian_nll, gaussian_crps, point_metrics, interval_metrics, conformal_interval, normal_z

TSLIB_COMMIT = "4e938a1767106324dd753b2a44832bf870a0252e"

DATA = {
    "ETTh1": dict(data="ETTh1", data_path="ETTh1.csv", enc_in=7,
                  url="https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv"),
    "Weather": dict(data="custom", data_path="weather.csv", enc_in=21,
                    url="https://huggingface.co/datasets/salmanali83/High_Dimensional_Time_Series/resolve/main/weather.csv"),
}

# Hyperparameters from TSLib's official scripts (scripts/long_term_forecast/*), H=96.
OFFICIAL = {
    ("DLinear", "ETTh1"): dict(e_layers=2, d_layers=1, factor=3),
    ("DLinear", "Weather"): dict(e_layers=2, d_layers=1, factor=3),          # no official script; defaults
    ("PatchTST", "ETTh1"): dict(e_layers=1, d_layers=1, factor=3, n_heads=2),
    ("PatchTST", "Weather"): dict(e_layers=2, d_layers=1, factor=3, n_heads=4, train_epochs=3),
}

# Pilot variants: (normalizer, shift_term). Each run reports both the conformal interval
# around its mean and the scale-head Gaussian interval.
#   revin   + no shift : variants 1 (RevIN + conformal) and 2 (RevIN + scale head, the honest rival)
#   learned + shift    : variant 3 (ours)
#   relaxed + shift    : variant 4 (ours without learned weights)
PILOT_VARIANTS = [("revin", False), ("learned", True), ("relaxed", True)]
PILOT_MODELS = ["DLinear", "PatchTST"]
PILOT_DATA = ["ETTh1", "Weather"]

OPTIONAL_MODULES = {
    "sktime": [], "sktime.datasets": ["load_from_tsfile_to_dataframe"],
    "reformer_pytorch": ["LSHSelfAttention"], "patoolib": [],
    "datasets": ["load_dataset"], "huggingface_hub": ["hf_hub_download"],
}


# --------------------------------------------------------------------------- setup
def _stub_optional_modules():
    """TSLib imports packages only needed by models/tasks we do not use; stub them if absent."""
    def unavailable(*_, **__):
        raise ImportError("optional dependency not installed")
    for name, attrs in OPTIONAL_MODULES.items():
        try:
            __import__(name)
        except Exception:
            mod = types.ModuleType(name)
            for a in attrs:
                setattr(mod, a, unavailable)
            sys.modules[name] = mod


def import_tslib(tslib_path):
    tslib_path = os.path.abspath(tslib_path)
    if not os.path.isdir(os.path.join(tslib_path, "models")):
        raise FileNotFoundError(f"TSLib not found at {tslib_path}")
    _stub_optional_modules()
    if tslib_path not in sys.path:
        sys.path.insert(1, tslib_path)
    from data_provider.data_factory import data_provider
    return data_provider


def tslib_args(tslib_path, model, data_name, data_root, **overrides):
    """Builds TSLib's argparse namespace with the defaults of its own run.py."""
    import argparse as _argparse
    src = open(os.path.join(tslib_path, "run.py"), encoding="utf-8").read()
    start = src.rfind("\n", 0, src.index("parser = argparse.ArgumentParser")) + 1
    end = src.rfind("\n", 0, src.index("args = parser.parse_args()")) + 1
    block = textwrap.dedent("\n".join(l for l in src[start:end].splitlines()))
    ns = {"argparse": _argparse}
    exec(block, ns)
    d = DATA[data_name]
    args = ns["parser"].parse_args([
        "--task_name", "long_term_forecast", "--is_training", "1",
        "--model_id", f"{data_name}_96_96", "--model", model, "--data", d["data"],
    ])
    args.data, args.root_path, args.data_path = d["data"], data_root, d["data_path"]
    args.features, args.seq_len, args.label_len, args.pred_len = "M", 96, 48, 96
    args.enc_in = args.dec_in = args.c_out = d["enc_in"]
    args.num_workers = 0   # data is in memory; worker processes would not inherit the module stubs on spawn
    for k, v in {**OFFICIAL[(model, data_name)], **overrides}.items():
        setattr(args, k, v)
    return args


def ensure_data(data_name, data_root):
    os.makedirs(data_root, exist_ok=True)
    path = os.path.join(data_root, DATA[data_name]["data_path"])
    if not os.path.exists(path):
        print(f"Downloading {data_name} -> {path}")
        req = urllib.request.Request(DATA[data_name]["url"], headers={"User-Agent": "equits-runner"})
        with urllib.request.urlopen(req) as r, open(path, "wb") as f:
            f.write(r.read())


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ------------------------------------------------------------------------ one run
class Runner:
    def __init__(self, a, model_name, data_name, normalizer, shift_term, seed):
        self.a, self.model_name, self.data_name = a, model_name, data_name
        self.normalizer, self.shift_term, self.seed = normalizer, shift_term, seed
        self.device = torch.device("cuda" if a.device == "cuda" and torch.cuda.is_available() else "cpu")
        data_provider = import_tslib(a.tslib_path)
        ensure_data(data_name, a.data_root)
        extra = {"train_epochs": a.train_epochs} if a.train_epochs else {}
        self.args = tslib_args(a.tslib_path, model_name, data_name, a.data_root, **extra)
        set_seed(seed)
        self.loaders = {f: data_provider(self.args, f)[1] for f in ("train", "val", "test")}
        import importlib
        self.backbone = importlib.import_module(f"models.{model_name}").Model(self.args).float().to(self.device)
        self.layer = EquiTS(96, 96, self.args.enc_in, normalizer=normalizer, shift_term=shift_term,
                            eps=a.eps).to(self.device)

    def _batches(self, flag):
        for bx, by, bxm, bym in self.loaders[flag]:
            bx = bx.float().to(self.device)
            by = by.float()[:, -96:, :].to(self.device)
            bxm, bym = bxm.float().to(self.device), bym.float().to(self.device)
            dec = torch.zeros(bx.shape[0], self.args.label_len + 96, bx.shape[2], device=self.device)
            yield bx, by, (lambda z, bxm=bxm, dec=dec, bym=bym: self.backbone(z, bxm, dec, bym))

    @torch.no_grad()
    def _collect(self, flag):
        self.backbone.eval(); self.layer.eval()
        mus, lvs, ys = [], [], []
        for bx, by, fn in self._batches(flag):
            mu, lv = self.layer(bx, fn)
            mus.append(mu.cpu()); lvs.append(lv.cpu()); ys.append(by.cpu())
        return torch.cat(mus), torch.cat(lvs), torch.cat(ys)

    def _stage(self, params, loss_fn, metric_fn, lr, epochs, patience, train_mean):
        opt = torch.optim.Adam(params, lr=lr)
        best, bad, best_state = float("inf"), 0, None
        for ep in range(1, epochs + 1):
            for g in opt.param_groups:                       # TSLib "type1"
                g["lr"] = lr * 0.5 ** (ep - 1)
            self.backbone.train(train_mean); self.layer.train()
            if not train_mean:
                self.layer.canonicalizer.eval()
            t0, tot, n = time.time(), 0.0, 0
            for bx, by, fn in self._batches("train"):
                opt.zero_grad()
                mu, lv = self.layer(bx, fn)
                loss = loss_fn(mu, lv, by)
                loss.backward()
                opt.step()
                tot += loss.item(); n += 1
            val = metric_fn(*self._collect("val"))
            print(f"    epoch {ep:2d} ({time.time() - t0:.1f}s) train {tot / max(1, n):.4f} val {val:.4f}")
            if val < best:
                best, bad = val, 0
                best_state = (copy.deepcopy(self.backbone.state_dict()), copy.deepcopy(self.layer.state_dict()))
            else:
                bad += 1
                if bad >= patience:
                    break
        self.backbone.load_state_dict(best_state[0]); self.layer.load_state_dict(best_state[1])
        return best

    def run(self):
        a, args = self.a, self.args
        print(f"\n=== {self.model_name} | {self.data_name} | normalizer={self.normalizer} "
              f"shift_term={self.shift_term} seed={self.seed}")
        # Stage A: mean path
        print("  Stage A (mean, MSE)")
        for p in self.layer.variance_parameters():
            p.requires_grad_(False)
        self._stage(list(self.backbone.parameters()) + list(self.layer.mean_parameters()),
                    lambda mu, lv, y: torch.mean((mu - y) ** 2),
                    lambda mu, lv, y: point_metrics(mu, y)["MSE"],
                    args.learning_rate, args.train_epochs, args.patience, train_mean=True)
        # Stage B: scale head
        print("  Stage B (variance, NLL)")
        for p in list(self.backbone.parameters()) + list(self.layer.mean_parameters()):
            p.requires_grad_(False)
        for p in self.layer.variance_parameters():
            p.requires_grad_(True)
        self._stage(list(self.layer.variance_parameters()),
                    lambda mu, lv, y: gaussian_nll(mu.detach(), lv, y),
                    lambda mu, lv, y: gaussian_nll(mu, lv, y).item(),
                    a.scale_lr, a.scale_epochs, args.patience, train_mean=False)
        return self.evaluate()

    def evaluate(self, alpha=0.1):
        mu, lv, y = self._collect("test")
        vmu, vlv, vy = self._collect("val")
        r = lambda v: round(float(v), 4)
        sigma = torch.exp(0.5 * lv)
        z = normal_z(alpha)
        g = interval_metrics(mu - z * sigma, mu + z * sigma, y, alpha)
        ca = interval_metrics(*conformal_interval(vmu, vy, mu, alpha=alpha), y, alpha)
        cn = interval_metrics(*conformal_interval(vmu, vy, mu, torch.exp(0.5 * vlv), sigma, alpha), y, alpha)
        pm = point_metrics(mu, y)
        return {
            "model": self.model_name, "data": self.data_name, "normalizer": self.normalizer,
            "shift_term": self.shift_term, "seed": self.seed, "eps": self.a.eps, "tslib_commit": TSLIB_COMMIT,
            "MSE": r(pm["MSE"]), "MAE": r(pm["MAE"]),
            "NLL": r(gaussian_nll(mu, lv, y).item()), "CRPS": r(gaussian_crps(mu, lv, y)),
            "Cov90": r(g["coverage"]), "Width90": r(g["width"]), "IS90": r(g["interval_score"]),
            "CPabs_Cov90": r(ca["coverage"]), "CPabs_Width90": r(ca["width"]), "CPabs_IS90": r(ca["interval_score"]),
            "CPnorm_Cov90": r(cn["coverage"]), "CPnorm_IS90": r(cn["interval_score"]),
        }


# -------------------------------------------------------------------------- pilot
def _key(r):
    return (r["model"], r["data"], r["normalizer"], bool(r["shift_term"]), r["seed"])


def load_results(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def pilot_decision(results, seed):
    """Pre-registered rule: ours (learned+shift) vs honest rival (revin + scale head)."""
    idx = {_key(r): r for r in results}
    rows, imps, dev_ours, dev_rival = [], [], [], []
    for m in PILOT_MODELS:
        for d in PILOT_DATA:
            rival = idx.get((m, d, "revin", False, seed))
            ours = idx.get((m, d, "learned", True, seed))
            abl = idx.get((m, d, "relaxed", True, seed))
            if not (rival and ours):
                continue
            imp = (rival["CRPS"] - ours["CRPS"]) / rival["CRPS"]
            imps.append(imp)
            dev_ours.append(abs(ours["Cov90"] - 0.9)); dev_rival.append(abs(rival["Cov90"] - 0.9))
            rows.append((m, d, rival, ours, abl, imp))

    print("\n" + "=" * 110)
    print("PILOT: CRPS / Cov90 / IS90   (rival = RevIN + scale head; ours = learned canonicalizer + shift term)")
    print("=" * 110)
    print(f"{'model':9} {'data':8} | {'RevIN+CP IS90':>13} | {'rival CRPS':>10} {'cov':>6} {'IS90':>6} | "
          f"{'ours CRPS':>9} {'cov':>6} {'IS90':>6} | {'no-learn CRPS':>13} | {'CRPS gain':>9} | MSE rival/ours")
    for m, d, rv, ou, ab, imp in rows:
        abl = f"{ab['CRPS']:.4f}" if ab else "-"
        print(f"{m:9} {d:8} | {rv['CPabs_IS90']:13.4f} | {rv['CRPS']:10.4f} {rv['Cov90']:6.3f} {rv['IS90']:6.3f} | "
              f"{ou['CRPS']:9.4f} {ou['Cov90']:6.3f} {ou['IS90']:6.3f} | {abl:>13} | {100 * imp:8.2f}% | "
              f"{rv['MSE']:.4f}/{ou['MSE']:.4f}")
    if len(rows) < len(PILOT_MODELS) * len(PILOT_DATA):
        print(f"\nIncomplete: {len(rows)}/{len(PILOT_MODELS) * len(PILOT_DATA)} combinations finished.")
        return
    avg = float(np.mean(imps))
    cov_closer = float(np.mean(dev_ours)) < float(np.mean(dev_rival))
    no_big_loss = min(imps) >= -0.01
    if avg >= 0.03 and cov_closer and no_big_loss:
        verdict = "GO"
    elif avg >= 0.01:
        verdict = "WEAK"
    else:
        verdict = "STOP"
    print(f"\nAverage CRPS gain: {100 * avg:.2f}% | coverage closer to 0.90: {cov_closer} | "
          f"worst combination: {100 * min(imps):.2f}%")
    print(f"DECISION (pre-registered: GO >=3% & closer coverage & none worse than -1%; WEAK 1-3%; STOP <1%): {verdict}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tslib_path", required=True)
    p.add_argument("--data_root", default=os.path.join(ROOT, "data"))
    p.add_argument("--pilot", action="store_true")
    p.add_argument("--model", default="DLinear")
    p.add_argument("--data", default="ETTh1", choices=list(DATA))
    p.add_argument("--normalizer", default="learned", choices=["revin", "relaxed", "learned"])
    p.add_argument("--shift_term", type=int, default=1)
    p.add_argument("--seed", type=int, default=2021)
    p.add_argument("--eps", type=float, default=0.1)
    p.add_argument("--scale_lr", type=float, default=1e-3)
    p.add_argument("--scale_epochs", type=int, default=10)
    p.add_argument("--train_epochs", type=int, default=None, help="Override (smoke tests only); default = TSLib/official")
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", default=os.path.join(ROOT, "results", "pilot.jsonl"))
    a = p.parse_args()

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    done = {_key(r) for r in load_results(a.out)}
    if a.pilot:
        jobs = [(m, d, n, s) for m in PILOT_MODELS for d in PILOT_DATA for n, s in PILOT_VARIANTS]
    else:
        jobs = [(a.model, a.data, a.normalizer, bool(a.shift_term))]
    for m, d, n, s in jobs:
        if (m, d, n, s, a.seed) in done:
            print(f"skip (done): {m} {d} {n} shift={s}")
            continue
        res = Runner(a, m, d, n, s, a.seed).run()
        print("  " + json.dumps(res))
        with open(a.out, "a") as f:
            f.write(json.dumps(res) + "\n")
    pilot_decision(load_results(a.out), a.seed)


if __name__ == "__main__":
    main()
