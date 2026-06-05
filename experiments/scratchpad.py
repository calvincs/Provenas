"""Direction 4+ : scratchpad for UNBOUNDED length generalization.

A pointer model learns the LOCAL reduction step (which operator to reduce next, from
structure only); the symbolic VM reduces it exactly and iterates. Because the per-step
decision is local, learning it should generalize to depths far beyond training. Train
on depths 1-4, test to depth 10 -> does exact-match hold where the one-shot dispatcher
cliffed (~2% at depth 7)?

  artifacts/scratchpad_depth.png

Env: SP_EPOCHS (default 12).
"""
from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from provenas import plotting as P
from provenas import exprgen as G
from provenas.reducer import (TYPES, TIDX, tree_to_state, reducible_positions,
                              apply_reduction, local_next, valid_reductions,
                              reduce_to_value)

SEED = 0
EPOCHS = int(os.environ.get("SP_EPOCHS", "12"))
BATCH = 256
PAD = 0


class PointerNet(nn.Module):
    def __init__(self, ntypes, d=128, nhead=4, layers=3, ff=256, max_len=1024):
        super().__init__()
        self.emb = nn.Embedding(ntypes, d, padding_idx=0)
        self.pos = nn.Embedding(max_len, d)
        layer = nn.TransformerEncoderLayer(d, nhead, ff, 0.1, batch_first=True,
                                           norm_first=True, activation="gelu")
        self.enc = nn.TransformerEncoder(layer, layers)
        self.score = nn.Linear(d, 1)

    def forward(self, ids, key_pad):
        pos = torch.arange(ids.size(1), device=ids.device).unsqueeze(0)
        h = self.enc(self.emb(ids) + self.pos(pos), src_key_padding_mask=key_pad)
        return self.score(h).squeeze(-1)        # (B, T)


class ConvPointer(nn.Module):
    """Translation-equivariant LOCAL model (no positional encoding): applies the SAME
    fixed-radius function at every position, so a local rule generalizes to any length
    by construction."""

    def __init__(self, ntypes, d=64, layers=5, k=5):
        super().__init__()
        self.emb = nn.Embedding(ntypes, d, padding_idx=0)
        self.convs = nn.ModuleList([nn.Conv1d(d, d, k, padding=k // 2) for _ in range(layers)])
        self.score = nn.Linear(d, 1)

    def forward(self, ids, key_pad):
        x = self.emb(ids).transpose(1, 2)
        for c in self.convs:
            x = torch.relu(c(x))
        return self.score(x.transpose(1, 2)).squeeze(-1)


def pad_states(state_types, device):
    T = max(len(s) for s in state_types)
    ids = np.zeros((len(state_types), T), dtype=np.int64)
    red = np.zeros((len(state_types), T), dtype=bool)
    for i, s in enumerate(state_types):
        ids[i, :len(s)] = [TIDX[t] for t in s]
        for p in reducible_positions(s):
            red[i, p] = True
    return (torch.from_numpy(ids).to(device), torch.from_numpy(ids == PAD).to(device),
            torch.from_numpy(red).to(device))


def gen_steps(depth_counts, rng):
    """Collect (state_types, target_pos) training pairs from oracle reductions."""
    states, targets = [], []
    for s in G.build(depth_counts, rng, seen=set()):
        if s.error != 0:
            continue
        types, vals = tree_to_state(s.tree)
        _, st, steps = reduce_to_value(types, vals, local_next)
        if st != "ok":
            continue
        for stp_types, pos in steps:
            states.append(stp_types)
            targets.append(pos)
    return states, np.array(targets)


def test_exprs(depth_counts, rng):
    out = []
    for s in G.build(depth_counts, rng, seen=set()):
        if s.error != 0:
            continue
        out.append((tree_to_state(s.tree), s.value, s.depth))
    return out


@torch.no_grad()
def eval_exact(items, model, device, max_steps=120):
    """Run model-driven reduction on a batch of expressions; return exact-match bool."""
    states = [(list(t), list(v)) for (t, v), _, _ in items]
    truth = [val for _, val, _ in items]
    done = [False] * len(items)
    ok = [False] * len(items)
    model.eval()
    for _ in range(max_steps):
        act = [i for i in range(len(states)) if not done[i]]
        if not act:
            break
        ids, kp, red = pad_states([states[i][0] for i in act], device)
        scores = model(ids, kp)
        scores = scores.masked_fill(~red, -1e9).cpu().numpy()
        for bi, gi in enumerate(act):
            types, vals = states[gi]
            rpos = reducible_positions(types)
            if not rpos:
                done[gi] = True
                continue
            p = int(np.argmax(scores[bi]))
            if p not in rpos:
                done[gi] = True
                continue
            nt, nv, e = apply_reduction(types, vals, p)
            if e != "ok":
                done[gi] = True
                continue
            states[gi] = (nt, nv)
            if len(nt) == 1:
                done[gi] = True
                tv = truth[gi]
                ok[gi] = (nt[0] == "VAL" and abs(nv[0] - tv) <= 1e-9 * max(1.0, abs(tv)))
    return ok


@torch.no_grad()
def per_step_acc(items, model, device):
    """Fraction of steps where the model picks a VALID reduction (local rule). Should
    stay ~flat with depth if the local decision genuinely generalizes."""
    states = []
    for (t, v), _, _ in items:
        _, st, steps = reduce_to_value(list(t), list(v), local_next)
        for stp_types, _ in steps:
            states.append(stp_types)
    if not states:
        return float("nan")
    model.eval()
    hits = 0
    for i in range(0, len(states), 512):
        chunk = states[i:i + 512]
        ids, kp, red = pad_states(chunk, device)
        picks = model(ids, kp).masked_fill(~red, -1e9).argmax(1).cpu().numpy()
        hits += sum(int(picks[j]) in valid_reductions(s) for j, s in enumerate(chunk))
    return hits / len(states)


def main():
    rng = np.random.default_rng(SEED)
    states, targets = gen_steps({1: 4000, 2: 4000, 3: 4000, 4: 4000}, rng)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  training pairs={len(states)}")

    model = (ConvPointer if os.environ.get("SP_MODEL", "cnn") == "cnn"
             else PointerNet)(len(TYPES)).to(device)
    print(f"  model = {type(model).__name__}")
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)
    n = len(states)
    for ep in range(EPOCHS):
        model.train()
        perm = rng.permutation(n)
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            ids, kp, red = pad_states([states[j] for j in idx], device)
            tgt = torch.tensor(targets[idx], dtype=torch.long, device=device)
            logits = model(ids, kp).masked_fill(~red, -1e9)
            loss = F.cross_entropy(logits, tgt)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

    by_depth, by_pstep = {}, {}
    for d in range(1, 11):
        items = test_exprs({d: 600}, np.random.default_rng(100 + d))
        if not items:
            continue
        by_depth[d] = float(np.mean(eval_exact(items, model, device)))
        by_pstep[d] = per_step_acc(items, model, device)
        print(f"  depth {d:2d}: exact-match = {by_depth[d]:.3f}  per-step acc = {by_pstep[d]:.4f}  (n={len(items)})")

    fig, ax = plt.subplots(figsize=(8.5, 5))
    ds = sorted(by_depth)
    ax.plot(ds, [by_pstep[d] for d in ds], marker="s", color="#2ca02c",
            label="per-step decision accuracy (the LOCAL rule)")
    ax.plot(ds, [by_depth[d] for d in ds], marker="o", color="#1f77b4",
            label="end-to-end exact-match (scratchpad: pointer + symbolic VM)")
    ax.plot([1, 2, 3, 4, 5, 6, 7], [1, 1, 1, .98, .34, .09, .02], marker="x", ls="--",
            color="#ff7f0e", label="one-shot dispatcher (trained 1-4)")
    ax.axvspan(4.5, 10.5, alpha=0.06, color="red")
    ax.axvline(4.5, ls="--", color="gray")
    ax.text(4.6, 0.5, "trained 1-4\n-> extrapolation", color="gray", fontsize=8, va="center")
    ax.set_ylim(-0.03, 1.05)
    ax.set_xlabel("expression nesting depth")
    ax.set_ylabel("EXACT-value match rate")
    ax.set_title("Scratchpad: a learned LOCAL reduction step generalizes to unbounded depth")
    ax.legend(loc="lower left")
    ax.grid(True, alpha=0.25)
    P.save(fig, "scratchpad_depth.png")


if __name__ == "__main__":
    main()
