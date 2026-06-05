"""Item 1 (flow control): iteration-count generalization on a neural-controlled loop.

Open challenge (literature): monolithic nets fail to generalize loop iteration count /
input length — "all current approaches to neural programming fare poorly on this
generalization issue" (Cai et al. 2017; neural-execution work 2024).

Hybrid: a tiny controller learns the loop CONDITION (continue vs halt) from the state;
an exact VM runs the body and holds state; the loop runs a data-dependent number of
times. Programs: sum-to-n (`while n>0: acc+=n; n-=1`) and gcd/Euclid
(`while b>0: a,b=b,a%b`). Train conditions on inputs <= 100; test on inputs to 20000.
A monolithic MLP (input->output) cannot produce exact answers at all; the hybrid is exact
at any magnitude — the iteration is the VM's job, the per-step decision is a local
threshold that extrapolates.

Honest scope: the controller's learned part (halt iff state==0) is simple; the point is
that decomposition into local control + an exact-state loop generalizes iteration count,
where the monolith can't. (The win is architectural, mirroring an interpreter's thin
control logic.)

  artifacts/item1_loops.png
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from provenas import plotting as P

SEED = 0
TRAIN_MAX = 100
CONT, HALT = 0, 1


class CondCtrl(nn.Module):
    def __init__(self, hid=16):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(1, hid), nn.ReLU(), nn.Linear(hid, 2))

    def forward(self, x):
        return self.net(x)


def _feat(v):
    # clamp so the decision is cleanly "v==0 vs v>=1" (bounded; large v -> CONT extrapolates)
    return np.minimum(np.asarray(v, dtype=np.float32), 3.0)[:, None]


def train_cond(device, rng):
    ctrl = CondCtrl().to(device)
    opt = torch.optim.Adam(ctrl.parameters(), lr=1e-2)
    for _ in range(600):
        half = 2048
        v = np.concatenate([np.zeros(half), rng.integers(1, TRAIN_MAX + 1, half)])
        y = np.concatenate([np.full(half, HALT), np.full(half, CONT)]).astype(np.int64)
        loss = F.cross_entropy(ctrl(torch.from_numpy(_feat(v)).to(device)),
                               torch.from_numpy(y).to(device))
        opt.zero_grad()
        loss.backward()
        opt.step()
    return ctrl


@torch.no_grad()
def _decide(ctrl, vvec, device):
    return ctrl(torch.from_numpy(_feat(vvec)).to(device)).argmax(1).cpu().numpy()


def eval_hyb_sum(ctrl, vals, device):
    n = vals.astype(np.int64).copy()
    acc = np.zeros(len(vals), dtype=np.int64)
    active = np.ones(len(vals), dtype=bool)
    truth = n * (n + 1) // 2
    for _ in range(int(vals.max()) + 2):
        if not active.any():
            break
        idx = np.where(active)[0]
        c = _decide(ctrl, n[idx], device)
        cont = idx[c == CONT]
        acc[cont] += n[cont]
        n[cont] -= 1
        active[idx[c == HALT]] = False
    return float(np.mean(acc == truth))


def eval_hyb_gcd(ctrl, A, B, device):
    a, b = A.astype(np.int64).copy(), B.astype(np.int64).copy()
    active = np.ones(len(A), dtype=bool)
    truth = np.array([math.gcd(int(x), int(y)) for x, y in zip(A, B)])
    for _ in range(200):                       # Euclid: O(log) iterations
        if not active.any():
            break
        idx = np.where(active)[0]
        c = _decide(ctrl, b[idx], device)
        halt = (c == HALT) | (b[idx] == 0)
        go = idx[~halt]
        newb = a[go] % b[go]
        a[go], b[go] = b[go], newb
        active[idx[halt]] = False
    return float(np.mean(a == truth))


class MonoMLP(nn.Module):
    def __init__(self, in_dim, hid=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hid), nn.ReLU(),
                                 nn.Linear(hid, hid), nn.ReLU(), nn.Linear(hid, 1))

    def forward(self, x):
        return self.net(x)


def train_mono(task, device, rng):
    out_scale = {"sum": TRAIN_MAX * TRAIN_MAX, "gcd": float(TRAIN_MAX)}[task]
    model = MonoMLP(1 if task == "sum" else 2).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(3000):
        if task == "sum":
            n = rng.integers(1, TRAIN_MAX + 1, 512)
            X = (n[:, None] / TRAIN_MAX).astype(np.float32)
            Y = n * (n + 1) // 2
        else:
            a = rng.integers(1, TRAIN_MAX + 1, 512)
            b = rng.integers(1, TRAIN_MAX + 1, 512)
            X = np.stack([a, b], 1).astype(np.float32) / TRAIN_MAX
            Y = np.array([math.gcd(int(x), int(y)) for x, y in zip(a, b)])
        yhat = model(torch.from_numpy(X).to(device)).squeeze(-1)
        loss = F.mse_loss(yhat, torch.from_numpy((Y / out_scale).astype(np.float32)).to(device))
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model, out_scale


@torch.no_grad()
def eval_mono(model, out_scale, task, A, B, device):
    if task == "sum":
        X = (A[:, None] / TRAIN_MAX).astype(np.float32)
        Y = A * (A + 1) // 2
    else:
        X = np.stack([A, B], 1).astype(np.float32) / TRAIN_MAX
        Y = np.array([math.gcd(int(x), int(y)) for x, y in zip(A, B)])
    pred = model(torch.from_numpy(X).to(device)).squeeze(-1).cpu().numpy() * out_scale
    return float(np.mean(np.round(pred) == Y))


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ctrl = train_cond(device, np.random.default_rng(SEED))
    mags = [50, 100, 200, 500, 1000, 5000, 20000]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for ax, task in zip(axes, ["sum", "gcd"]):
        mono, out_scale = train_mono(task, device, np.random.default_rng(SEED + 1))
        mono_acc, hyb_acc = [], []
        for mag in mags:
            re = np.random.default_rng(7 + mag)
            A = re.integers(max(1, mag // 2), mag + 1, 120)
            B = re.integers(max(1, mag // 2), mag + 1, 120)
            mono_acc.append(eval_mono(mono, out_scale, task, A, B, device))
            hyb_acc.append(eval_hyb_sum(ctrl, A, device) if task == "sum"
                           else eval_hyb_gcd(ctrl, A, B, device))
            print(f"  {task} mag={mag:6d}: monolithic={mono_acc[-1]:.3f}  hybrid={hyb_acc[-1]:.3f}")
        ax.plot(mags, hyb_acc, marker="o", color="#1f77b4", label="hybrid (controller + exact-loop VM)")
        ax.plot(mags, mono_acc, marker="x", ls="--", color="#d62728", label="monolithic MLP")
        ax.axvspan(TRAIN_MAX, mags[-1] * 1.3, alpha=0.06, color="red")
        ax.axvline(TRAIN_MAX, ls="--", color="gray")
        ax.set_xscale("log")
        ax.set_ylim(-0.03, 1.05)
        ax.set_xlabel("input magnitude  (-> iteration count)")
        ax.set_ylabel("EXACT-match rate")
        ax.set_title(f"{'sum 1..n' if task == 'sum' else 'gcd (Euclid)'}  (cond trained <= {TRAIN_MAX})")
        ax.legend(loc="center left", fontsize=8)
        ax.grid(True, which="both", alpha=0.25)
    fig.suptitle("Item 1: iteration-count generalization — neural-controlled loop vs monolithic")
    P.save(fig, "item1_loops.png")


if __name__ == "__main__":
    main()
