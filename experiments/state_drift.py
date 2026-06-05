"""Stateful primitive: state drift (M1 recurrent) vs state-as-input (M0).

A running balance over a deposit/withdraw sequence. Both train on sequences of
length <=20 and are evaluated out to length 100. The headline: the recurrent net
(M1), which must hold the balance in an opaque hidden vector, DRIFTS — per-step
error compounds with sequence length. The state-as-input net (M0), whose update
is a simple affine step the net learns near-exactly, stays flat (the *wrapper*
holds the state, not the net).

  artifacts/state_drift.png
"""
from __future__ import annotations

import numpy as np
import torch
import matplotlib.pyplot as plt

from provenas import plotting as P
from provenas.bankaccount import make_sequences
from provenas.models_seq import BalanceGRU, StepMLP

SEED = 0
BAL_SCALE = 1000.0
AMT_SCALE = 100.0
TRAIN_LEN = 20
TEST_LEN = 100


def seq_features(ops, amts):
    isd = (ops > 0).astype(np.float32)
    isw = (ops < 0).astype(np.float32)
    return np.stack([isd, isw, (amts / AMT_SCALE).astype(np.float32)], axis=-1)


def train_gru(rng, device, epochs=80):
    model = BalanceGRU().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(epochs):
        ops, amts, bal = make_sequences(2048, TRAIN_LEN, rng)
        X = torch.from_numpy(seq_features(ops, amts)).to(device)
        Y = torch.from_numpy((bal / BAL_SCALE).astype(np.float32)[..., None]).to(device)
        loss = ((model(X) - Y) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model


def train_m0(rng, device, epochs=80):
    model = StepMLP().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(epochs):
        n = 8192
        bin_ = rng.uniform(-2500, 2500, n)
        ops = rng.choice(np.array([1.0, -1.0]), n)
        amts = rng.uniform(0, AMT_SCALE, n)
        bout = bin_ + ops * amts
        X = np.stack([bin_ / BAL_SCALE, ops, amts / AMT_SCALE], 1).astype(np.float32)
        Y = (bout / BAL_SCALE).astype(np.float32)[:, None]
        loss = ((model(torch.from_numpy(X).to(device)) - torch.from_numpy(Y).to(device)) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model


@torch.no_grad()
def drift_gru(model, ops, amts, bal, device):
    X = torch.from_numpy(seq_features(ops, amts)).to(device)
    pred = model(X).cpu().numpy().squeeze(-1) * BAL_SCALE
    return np.abs(pred - bal).mean(0)


@torch.no_grad()
def drift_m0(model, ops, amts, bal, device):
    n, T = ops.shape
    cur = np.zeros(n)
    errs = np.zeros((n, T))
    for t in range(T):
        X = np.stack([cur / BAL_SCALE, ops[:, t], amts[:, t] / AMT_SCALE], 1).astype(np.float32)
        cur = model(torch.from_numpy(X).to(device)).cpu().numpy().squeeze(-1) * BAL_SCALE
        errs[:, t] = np.abs(cur - bal[:, t])
    return errs.mean(0)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gru = train_gru(np.random.default_rng(SEED), device)
    m0 = train_m0(np.random.default_rng(SEED + 1), device)

    ops, amts, bal = make_sequences(4000, TEST_LEN, np.random.default_rng(SEED + 9))
    d_gru = drift_gru(gru, ops, amts, bal, device)
    d_m0 = drift_m0(m0, ops, amts, bal, device)
    steps = np.arange(1, TEST_LEN + 1)
    print(f"  M1 recurrent: abs balance err  step20={d_gru[19]:.2f}  step100={d_gru[-1]:.2f}")
    print(f"  M0 state-in : abs balance err  step20={d_m0[19]:.2f}  step100={d_m0[-1]:.2f}")

    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.plot(steps, d_m0, label="M0 state-as-input (net threads the balance)", color="#2ca02c")
    ax.plot(steps, d_gru, label="M1 recurrent (balance in hidden state)", color="#d62728")
    ax.plot(steps, np.zeros_like(steps, dtype=float), lw=2, color="#1f77b4",
            label="M2 external store (a Python variable) — exact")
    ax.axvspan(TRAIN_LEN, TEST_LEN, alpha=0.05, color="red")
    ax.axvline(TRAIN_LEN, ls="--", color="gray")
    ax.text(TRAIN_LEN, 0.95, " trained length 20 -> extrapolation", rotation=90, va="top",
            color="gray", fontsize=8, transform=ax.get_xaxis_transform())
    ax.set_xlabel("number of operations in the sequence")
    ax.set_ylabel("mean |balance error|")
    ax.set_title("Stateful: both neural state-trackers DRIFT;\nonly an external classical store stays exact")
    ax.legend()
    ax.grid(True, alpha=0.25)
    P.save(fig, "state_drift.png")


if __name__ == "__main__":
    main()
