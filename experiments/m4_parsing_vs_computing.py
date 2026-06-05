"""M4 headline: parsing vs computing — Transformer (path C) vs Tree-LSTM (path D).

This runs ONE (model, seed) per process and writes a JSON result, so many runs can
be launched in parallel across GPUs (both models x several seeds). Aggregate with
experiments/m4_plot.py.

Matched protocol: identical optimizer (AdamW), epochs, batch size, masked
multi-head loss, output space (signed-log regression + 3-class error head), and
metrics. The ONLY intentional asymmetry is the input — C reads the string (must
parse), D is handed the tree (need only compute).

Path C is pre-tokenized once (GPU-bound, fast). Path D uses a recursive forward
(Python-bound; the per-run cost is why we parallelize across seeds/GPUs).

Env: PROVENAS_MODEL=C|D, PROVENAS_SEED, M4_EPOCHS, M4_BATCH.
Writes: artifacts/m4_runs/{model}_seed{seed}.json
"""
from __future__ import annotations

import json
import os
import pickle
import time

import numpy as np
import torch

from provenas.calculator import ERR_CLASSES
from provenas.losses import masked_multihead_loss
from provenas.metrics import relative_error
from provenas.models_transformer import TinyTransformer
from provenas.models_tree import BinaryTreeLSTM
from provenas.tokenize_expr import batch_encode

DATA = "data/m4_expressions.pkl"
EPOCHS = int(os.environ.get("M4_EPOCHS", "25"))
BATCH = int(os.environ.get("M4_BATCH", "128"))
LR = 3e-4
OUTDIR = "artifacts/m4_runs"

# M4 signed-log regression target. With _SCALE = 1 the decode is the exact inverse,
# so a model's absolute error in log-space ~= relative error in value-space (no
# amplification). (M3's encoder divided by ~43.75, which AMPLIFIED errors ~44x and
# collapsed M4 to ~100% error; any scale > 1 hurts.)
_SCALE = 1.0


def _enc(vals):
    return (np.sign(vals) * np.log1p(np.abs(vals)) / _SCALE).astype(np.float32)[:, None]


def _dec(t):
    sl = np.asarray(t, dtype=np.float64).reshape(-1) * _SCALE
    return np.sign(sl) * np.expm1(np.abs(sl))


def prep_targets(samples, device):
    vals = np.array([s.value for s in samples], dtype=np.float64)
    y = torch.nan_to_num(torch.from_numpy(_enc(vals)), nan=0.0).to(device)
    e = torch.tensor([int(s.error) for s in samples], dtype=torch.long, device=device)
    return y, e


def n_params(m):
    return sum(p.numel() for p in m.parameters())


def _opt(model):
    return torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2)


def train_C(model, samples, device, cw, seed):
    """Pre-tokenize once, then a pure-GPU loop (no per-step Python tokenization)."""
    model.to(device)
    ids, mask = batch_encode([s.string for s in samples], device)
    y, e = prep_targets(samples, device)
    n = ids.shape[0]
    opt = _opt(model)
    g = torch.Generator(device=device).manual_seed(seed)
    for _ in range(EPOCHS):
        perm = torch.randperm(n, device=device, generator=g)
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            reg, errl = model(ids[idx], mask[idx])
            loss, _, _ = masked_multihead_loss(reg, y[idx], errl, e[idx], 1.0, 1.0, cw)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
    return model


def train_D(model, samples, device, cw, seed):
    model.to(device)
    trees = [s.tree for s in samples]
    y, e = prep_targets(samples, device)
    n = len(trees)
    opt = _opt(model)
    rng = np.random.default_rng(seed)
    for _ in range(EPOCHS):
        perm = rng.permutation(n)
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            ti = torch.as_tensor(idx, device=device)
            reg, errl = model([trees[j] for j in idx], device)
            loss, _, _ = masked_multihead_loss(reg, y[ti], errl, e[ti], 1.0, 1.0, cw)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
    return model


def fwd_C(model, batch, device):
    ids, mask = batch_encode([s.string for s in batch], device)
    return model(ids, mask)


def fwd_D(model, batch, device):
    return model([s.tree for s in batch], device)


@torch.no_grad()
def evaluate(model, samples, fwd, device, batch_size=256):
    model.eval()
    preds, perr = [], []
    for i in range(0, len(samples), batch_size):
        reg, errl = fwd(model, samples[i:i + batch_size], device)
        preds.append(reg.cpu().numpy().reshape(-1))
        perr.append(errl.argmax(1).cpu().numpy())
    pv = _dec(np.concatenate(preds))
    pe = np.concatenate(perr)
    tv = np.array([s.value for s in samples], dtype=np.float64)
    te = np.array([int(s.error) for s in samples])
    dp = np.array([s.depth for s in samples])
    return pv, pe, tv, te, dp


def fidelity(pv, pe, tv, te):
    ok = te == 0
    rel = relative_error(pv[ok], tv[ok])
    return dict(median_rel=float(np.median(rel)), p90_rel=float(np.percentile(rel, 90)),
                exact_pct=float(np.mean(rel < 1e-3)), err_acc=float(np.mean(pe == te)))


def depth_curve(pv, pe, tv, te, dp):
    out = {}
    for d in range(1, 8):
        m = (dp == d) & (te == 0)
        if m.any():
            out[d] = float(np.median(relative_error(pv[m], tv[m])))
    return out


def main():
    kind = os.environ.get("PROVENAS_MODEL", "C").upper()
    seed = int(os.environ.get("PROVENAS_SEED", "0"))
    with open(DATA, "rb") as f:
        data = pickle.load(f)
    train = data["train"]
    full_test = data["test_indist"] + data["test_depth"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    counts = np.bincount([int(s.error) for s in train], minlength=len(ERR_CLASSES))
    cw = torch.tensor((counts.sum() / (len(ERR_CLASSES) * np.maximum(counts, 1))).astype(np.float32),
                      device=device)

    torch.manual_seed(seed)
    if kind == "C":
        model, fwd, trainer, label = TinyTransformer(), fwd_C, train_C, "C (Transformer)"
    else:
        model, fwd, trainer, label = BinaryTreeLSTM(h_dim=144), fwd_D, train_D, "D (Tree-LSTM)"

    t0 = time.time()
    trainer(model, train, device, cw, seed)
    dt = time.time() - t0

    pv, pe, tv, te, _ = evaluate(model, data["test_indist"], fwd, device)
    fid = fidelity(pv, pe, tv, te)
    fpv, fpe, ftv, fte, fdp = evaluate(model, full_test, fwd, device)
    dc = depth_curve(fpv, fpe, ftv, fte, fdp)

    os.makedirs(OUTDIR, exist_ok=True)
    out = dict(model=kind, label=label, seed=seed, params=n_params(model),
               train_time=dt, epochs=EPOCHS, fidelity=fid, depth_curve=dc)
    path = f"{OUTDIR}/{kind}_seed{seed}.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[{label} seed{seed}] params={n_params(model):,} t={dt:.1f}s "
          f"median_rel={fid['median_rel']:.3e} err_acc={fid['err_acc']:.3f} -> {path}")


if __name__ == "__main__":
    main()
