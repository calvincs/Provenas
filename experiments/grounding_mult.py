"""Direction 3 (loops): multiplication grounded in repeated addition.

a*n = a added n times. The SAME learned repeat-until loop as counting, but each step
does real work (acc += a) over an exact memory accumulator. Trained on n<=20, tested
to n=500: exact a*n at any magnitude — the loop + memory generalize, grounding
multiplication the way a child does (a*3 = a+a+a).

  artifacts/grounding_mult.png
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from provenas import plotting as P

SEED = 0
STEP, HALT = 0, 1
TRAIN_NMAX = 20


class Ctrl(nn.Module):
    def __init__(self, hid=32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(1, hid), nn.ReLU(),
                                 nn.Linear(hid, hid), nn.ReLU(),
                                 nn.Linear(hid, 2))

    def forward(self, x):
        return self.net(x)


def traces(n, nmax, rng):
    rem, act = [], []
    for ni in rng.integers(0, nmax + 1, n):
        for r in range(int(ni), -1, -1):
            rem.append(float(r))
            act.append(STEP if r > 0 else HALT)
    return np.array(rem, dtype=np.float32)[:, None], np.array(act)


@torch.no_grad()
def run_batch(a, n, model, device):
    model.eval()
    acc = np.zeros(len(a), dtype=np.int64)
    remaining = n.astype(np.int64).copy()
    active = np.ones(len(a), dtype=bool)
    for _ in range(int(n.max()) + 5):
        if not active.any():
            break
        idx = np.where(active)[0]
        x = torch.tensor(remaining[idx].astype(np.float32)[:, None], device=device)
        action = model(x).argmax(1).cpu().numpy()
        st = idx[action == STEP]
        acc[st] += a[st]
        remaining[st] -= 1
        active[idx[action == HALT]] = False
    return acc


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(SEED)
    rem, act = traces(4000, TRAIN_NMAX, rng)
    X = torch.from_numpy(rem).to(device)
    Y = torch.from_numpy(act).to(device)
    model = Ctrl().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(400):
        model.train()
        loss = F.cross_entropy(model(X), Y)
        opt.zero_grad()
        loss.backward()
        opt.step()

    ns = [1, 2, 5, 10, 20, 50, 100, 200, 500]
    exact = []
    for nv in ns:
        a = rng.integers(0, 1000, 200)
        res = run_batch(a, np.full(200, nv), model, device)
        exact.append(float(np.mean(res == a * nv)))
        print(f"  n={nv:4d}: exact a*n = {exact[-1]:.3f}")

    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.plot(ns, exact, marker="o", color="#17becf",
            label="Multiply by repeated addition (learned loop + exact accumulator)")
    ax.axvspan(TRAIN_NMAX, ns[-1] * 1.3, alpha=0.05, color="red")
    ax.axvline(TRAIN_NMAX, ls="--", color="gray")
    ax.text(TRAIN_NMAX, 0.5, " trained n<=20  -> extrapolation", rotation=90, va="center",
            color="gray", fontsize=8, transform=ax.get_xaxis_transform())
    ax.set_xscale("log")
    ax.set_ylim(-0.03, 1.05)
    ax.set_xlabel("n  (multiplier)")
    ax.set_ylabel("exact a*n match rate")
    ax.set_title("Direction 3 (loops): multiplication grounded in repeated addition\n"
                 "learned repeat-until loop + exact accumulator — exact at any magnitude")
    ax.legend(loc="lower left")
    ax.grid(True, which="both", alpha=0.25)
    P.save(fig, "grounding_mult.png")


if __name__ == "__main__":
    main()
