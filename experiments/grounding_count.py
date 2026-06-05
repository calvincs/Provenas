"""Grounding facet (3 = 1+1+1): addition by COUNTING over an exact memory counter.

Instead of approximating a+b (M1 showed that cliffs with magnitude), compute it the
way a child does — repeated successor: set count=a, increment b times. A tiny
controller learns the LOOP policy (INC while work remains, else HALT); an exact
Python counter (the memory) holds the quantity. Trained on b<=20, tested out to
b=2000: if the net learned the loop (a threshold on `remaining`, not memorized step
counts), addition is EXACT at ANY magnitude — the inverse of M1's cliff, because the
system counts instead of approximating.

  artifacts/grounding_count.png
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from provenas import plotting as P

SEED = 0
INC, HALT = 0, 1
TRAIN_BMAX = 20


class Controller(nn.Module):
    """Decides INC vs HALT from the raw `remaining` count (1 input)."""

    def __init__(self, hid=32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(1, hid), nn.ReLU(),
                                 nn.Linear(hid, hid), nn.ReLU(),
                                 nn.Linear(hid, 2))

    def forward(self, remaining):
        return self.net(remaining)


def make_traces(n, bmax, rng):
    """Every state along the count-down: remaining r -> action (INC if r>0 else HALT)."""
    rem, act = [], []
    for bi in rng.integers(0, bmax + 1, n):
        for r in range(int(bi), -1, -1):
            rem.append(float(r))
            act.append(INC if r > 0 else HALT)
    return np.array(rem, dtype=np.float32)[:, None], np.array(act)


@torch.no_grad()
def run_batch(a_arr, b_arr, model, device):
    """Run the learned loop for a batch of (a,b): step all trials together until each
    halts; the memory counter does the exact increment."""
    model.eval()
    count = a_arr.astype(np.int64).copy()
    remaining = b_arr.astype(np.int64).copy()
    active = np.ones(len(a_arr), dtype=bool)
    for _ in range(int(b_arr.max()) + 5):
        if not active.any():
            break
        idx = np.where(active)[0]
        x = torch.tensor(remaining[idx].astype(np.float32)[:, None], device=device)
        action = model(x).argmax(1).cpu().numpy()
        inc = idx[action == INC]
        count[inc] += 1
        remaining[inc] -= 1
        active[idx[action == HALT]] = False
    return count


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(SEED)
    rem, act = make_traces(4000, TRAIN_BMAX, rng)
    X = torch.from_numpy(rem).to(device)
    Y = torch.from_numpy(act).to(device)

    model = Controller().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(400):
        model.train()
        loss = F.cross_entropy(model(X), Y)
        opt.zero_grad()
        loss.backward()
        opt.step()

    bs = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000]
    exact = []
    for bv in bs:
        a_arr = rng.integers(0, 10000, 200)
        b_arr = np.full(200, bv)
        res = run_batch(a_arr, b_arr, model, device)
        exact.append(float(np.mean(res == a_arr + b_arr)))
        print(f"  b={bv:5d}: exact a+b = {exact[-1]:.3f}")

    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.plot(bs, exact, marker="o", color="#2ca02c",
            label="Addition by counting (learned loop + exact memory)")
    ax.axhline(1.0, ls=":", color="gray", lw=1)
    ax.axvspan(TRAIN_BMAX, bs[-1] * 1.3, alpha=0.05, color="red")
    ax.axvline(TRAIN_BMAX, ls="--", color="gray")
    ax.text(TRAIN_BMAX, 0.5, " trained b<=20  -> extrapolation", rotation=90, va="center",
            color="gray", fontsize=8, transform=ax.get_xaxis_transform())
    ax.set_xscale("log")
    ax.set_ylim(-0.03, 1.05)
    ax.set_xlabel("b  (number of increments)")
    ax.set_ylabel("exact a+b match rate")
    ax.set_title("Grounding: addition by COUNTING is exact at any magnitude\n"
                 "(net learns the loop; memory holds the count) — vs M1's approximation cliff")
    ax.legend(loc="lower left")
    ax.grid(True, which="both", alpha=0.25)
    P.save(fig, "grounding_count.png")


if __name__ == "__main__":
    main()
