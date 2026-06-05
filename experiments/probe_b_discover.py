"""Probe B (pilot): can the controller DISCOVER a length-generalizing policy from
OUTCOME reward alone — no per-step supervision — using the symbolic VM as its exact
environment?

Easy case: counting (compute a+b by incrementing). The policy sees only `remaining`
and picks STEP/HALT; reward = +1 iff the final count == a+b (sparse, end-of-episode).
Trained by REINFORCE on b<=20; tested to b=500. If it discovers the local
"step while remaining>0" rule from reward alone, it length-generalizes — a clean
(if small) demonstration of outcome-only algorithm discovery. The genuinely-hard
frontier is discovering a symbolic REDUCTION algorithm this way.

  artifacts/probe_b_discover.png
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from provenas import plotting as P

SEED = 0
STEP, HALT = 0, 1
TRAIN_BMAX = 20


class Policy(nn.Module):
    def __init__(self, hid=32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(1, hid), nn.ReLU(),
                                 nn.Linear(hid, hid), nn.ReLU(),
                                 nn.Linear(hid, 2))

    def forward(self, x):
        return self.net(x)


def train(device, rng, iters=3000, bs=256, max_steps=40):
    pol = Policy().to(device)
    opt = torch.optim.Adam(pol.parameters(), lr=3e-3)
    for _ in range(iters):
        a = rng.integers(0, 50, bs)
        b = rng.integers(0, TRAIN_BMAX + 1, bs)
        count = a.astype(np.int64).copy()
        remaining = b.astype(np.int64).copy()
        active = np.ones(bs, dtype=bool)
        logp_terms = []
        for _ in range(max_steps + 1):
            if not active.any():
                break
            idx = np.where(active)[0]
            x = torch.tensor(remaining[idx].astype(np.float32)[:, None], device=device)
            dist = torch.distributions.Categorical(logits=pol(x))
            acts = dist.sample()
            logp_terms.append((idx, dist.log_prob(acts)))
            acts_np = acts.cpu().numpy()
            stp = idx[acts_np == STEP]
            count[stp] += 1
            remaining[stp] -= 1
            active[idx[acts_np == HALT]] = False
        reward = (count == (a + b)).astype(np.float32)         # sparse, end-of-episode
        adv = reward - reward.mean()                           # baseline
        loss = sum(-(lp * torch.tensor(adv[idx], device=device)).sum() for idx, lp in logp_terms) / bs
        opt.zero_grad()
        loss.backward()
        opt.step()
    return pol


@torch.no_grad()
def eval_policy(pol, b, n, rng, device):
    a = rng.integers(0, 1000, n)
    count = a.astype(np.int64).copy()
    remaining = np.full(n, b, dtype=np.int64)
    active = np.ones(n, dtype=bool)
    pol.eval()
    for _ in range(b * 2 + 5):
        if not active.any():
            break
        idx = np.where(active)[0]
        x = torch.tensor(remaining[idx].astype(np.float32)[:, None], device=device)
        acts = pol(x).argmax(1).cpu().numpy()                  # greedy at eval
        stp = idx[acts == STEP]
        count[stp] += 1
        remaining[stp] -= 1
        active[idx[acts == HALT]] = False
    return float(np.mean(count == (a + b)))


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pol = train(device, np.random.default_rng(SEED))
    bs = [1, 2, 5, 10, 20, 50, 100, 200, 500]
    ex = []
    for bv in bs:
        ex.append(eval_policy(pol, bv, 200, np.random.default_rng(300 + bv), device))
        print(f"  b={bv:4d}: exact a+b (DISCOVERED policy) = {ex[-1]:.3f}")

    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.plot(bs, ex, marker="o", color="#9467bd",
            label="Policy DISCOVERED from outcome reward (REINFORCE, no step labels)")
    ax.axvspan(TRAIN_BMAX, bs[-1] * 1.3, alpha=0.05, color="red")
    ax.axvline(TRAIN_BMAX, ls="--", color="gray")
    ax.text(TRAIN_BMAX, 0.5, " trained b<=20  -> extrapolation", rotation=90, va="center",
            color="gray", fontsize=8, transform=ax.get_xaxis_transform())
    ax.set_xscale("log")
    ax.set_ylim(-0.03, 1.05)
    ax.set_xlabel("b  (increments)")
    ax.set_ylabel("exact a+b match rate")
    ax.set_title("Probe B (pilot): discovering a length-generalizing policy from OUTCOME reward\n"
                 "(no per-step supervision; symbolic VM as the exact environment)")
    ax.legend(loc="lower left")
    ax.grid(True, which="both", alpha=0.25)
    P.save(fig, "probe_b_discover.png")


if __name__ == "__main__":
    main()
