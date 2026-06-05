"""Probe A: does hybrid decomposition length-generalize on a task that is PROVABLY
hard for monolithic transformers (RASP-L)?

Task: PARITY (XOR of a bit string) — a canonical RASP-L-hard task. We compare, on the
SAME task:
  - Monolithic: a transformer reads the whole string -> parity (one shot).
  - Hybrid: a tiny controller learns the LOCAL step (bit, accumulator) -> accumulator;
    an exact 1-bit memory threads the accumulator. This is a learned finite-state
    transducer, which processes arbitrary-length input by construction.

Train on lengths <= 20, test to 80. Claim: the monolith cliffs to chance (~0.5) past
the training length; the hybrid generalizes to any length.

  artifacts/probe_a_parity.png
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from provenas import plotting as P

SEED = 0
TRAIN_LENS = list(range(2, 21))
TEST_LENS = [5, 10, 20, 30, 40, 50, 60, 80]


class ParityTransformer(nn.Module):
    def __init__(self, d=64, nhead=4, layers=3, max_len=160):
        super().__init__()
        self.emb = nn.Embedding(3, d, padding_idx=2)        # 0, 1, pad=2
        self.cls = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.pos = nn.Embedding(max_len, d)
        layer = nn.TransformerEncoderLayer(d, nhead, 4 * d, 0.1, batch_first=True,
                                           norm_first=True, activation="gelu")
        self.enc = nn.TransformerEncoder(layer, layers)
        self.head = nn.Linear(d, 2)

    def forward(self, ids, key_pad):
        B = ids.size(0)
        x = torch.cat([self.cls.expand(B, 1, -1), self.emb(ids)], dim=1)
        pos = torch.arange(x.size(1), device=ids.device).unsqueeze(0)
        x = x + self.pos(pos)
        kp = torch.cat([torch.zeros(B, 1, dtype=torch.bool, device=ids.device), key_pad], dim=1)
        return self.head(self.enc(x, src_key_padding_mask=kp)[:, 0])


class StepCtrl(nn.Module):
    """Learned local step: (bit, accumulator) -> new accumulator."""

    def __init__(self, hid=16):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2, hid), nn.ReLU(), nn.Linear(hid, 2))

    def forward(self, x):
        return self.net(x)


def gen_parity(n, lengths, rng, maxlen):
    ids = np.full((n, maxlen), 2, dtype=np.int64)        # pad
    y = np.zeros(n, dtype=np.int64)
    for i in range(n):
        L = int(rng.choice(lengths))
        bits = rng.integers(0, 2, L)
        ids[i, :L] = bits
        y[i] = int(bits.sum() % 2)
    return ids, y


def train_monolithic(device, rng):
    model = ParityTransformer().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)
    maxlen = max(TRAIN_LENS)
    for _ in range(400):
        ids, y = gen_parity(512, TRAIN_LENS, rng, maxlen)
        ids = torch.from_numpy(ids).to(device)
        kp = (ids == 2)
        yt = torch.from_numpy(y).to(device)
        loss = F.cross_entropy(model(ids, kp), yt)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model


@torch.no_grad()
def eval_monolithic(model, L, n, rng, device):
    ids = np.full((n, L), 2, dtype=np.int64)
    bits = rng.integers(0, 2, (n, L))
    ids[:, :] = bits
    true = bits.sum(1) % 2
    ids = torch.from_numpy(ids).to(device)
    pred = model(ids, ids == 2).argmax(1).cpu().numpy()
    return float(np.mean(pred == true))


def train_ctrl(device, rng):
    ctrl = StepCtrl().to(device)
    opt = torch.optim.Adam(ctrl.parameters(), lr=1e-2)
    for _ in range(300):
        xb = rng.integers(0, 2, (4096, 2)).astype(np.float32)
        yb = (xb[:, 0].astype(int) ^ xb[:, 1].astype(int))
        loss = F.cross_entropy(ctrl(torch.from_numpy(xb).to(device)),
                               torch.from_numpy(yb).to(device))
        opt.zero_grad()
        loss.backward()
        opt.step()
    return ctrl


@torch.no_grad()
def eval_hybrid(ctrl, L, n, rng, device):
    bits = rng.integers(0, 2, (n, L)).astype(np.float32)
    acc = np.zeros(n, dtype=np.float32)
    ctrl.eval()
    for t in range(L):
        x = np.stack([bits[:, t], acc], axis=1)
        acc = ctrl(torch.from_numpy(x).to(device)).argmax(1).cpu().numpy().astype(np.float32)
    return float(np.mean(acc == (bits.sum(1) % 2)))


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(SEED)
    mono = train_monolithic(device, rng)
    ctrl = train_ctrl(device, rng)

    mono_acc, hyb_acc = [], []
    for L in TEST_LENS:
        mono_acc.append(eval_monolithic(mono, L, 2000, np.random.default_rng(100 + L), device))
        hyb_acc.append(eval_hybrid(ctrl, L, 2000, np.random.default_rng(200 + L), device))
        print(f"  L={L:3d}: monolithic transformer={mono_acc[-1]:.3f}   hybrid (controller+VM)={hyb_acc[-1]:.3f}")

    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.plot(TEST_LENS, hyb_acc, marker="o", color="#1f77b4",
            label="Hybrid: controller + exact 1-bit memory (learned FST)")
    ax.plot(TEST_LENS, mono_acc, marker="x", ls="--", color="#d62728",
            label="Monolithic transformer (RASP-L-hard: parity)")
    ax.axhline(0.5, ls=":", color="gray", lw=1)
    ax.text(TEST_LENS[-1], 0.52, "chance", color="gray", fontsize=8, ha="right")
    ax.axvspan(20, TEST_LENS[-1] + 3, alpha=0.06, color="red")
    ax.axvline(20, ls="--", color="gray")
    ax.text(21, 0.7, "trained <=20\n-> extrapolation", color="gray", fontsize=8)
    ax.set_ylim(0.4, 1.05)
    ax.set_xlabel("bit-string length")
    ax.set_ylabel("parity accuracy")
    ax.set_title("Probe A: hybrid decomposition length-generalizes on PARITY\n"
                 "(RASP-L-hard for monolithic transformers)")
    ax.legend(loc="center left")
    ax.grid(True, alpha=0.25)
    P.save(fig, "probe_a_parity.png")


if __name__ == "__main__":
    main()
