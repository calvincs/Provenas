"""Probe B (hard): can a controller DISCOVER a length-generalizing symbolic-reduction
policy from OUTCOME reward alone — no per-step supervision?

Environment: the expression-reduction VM. State = token-type sequence (VAL, + - * /,
parens); action = pick a reducible operator position; the VM reduces it EXACTLY; episode
ends when one VAL remains. Reward = +1 iff the final value is exactly correct, else 0.
The model is NEVER told the right reduction order — only the terminal reward.

Trained by REINFORCE on depths 1-4; tested to depth 10. A conv (local, translation-
equivariant) policy *should* length-generalize IF it discovers the local 'safe reduction'
rule from reward. Honest: this is the hard, high-risk case — a clean negative is a fine
outcome. Compares the DISCOVERED policy to the supervised scratchpad (which got 100%).

  artifacts/probe_b_hard.png
"""
from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from provenas import plotting as P
from provenas import exprgen as G
from provenas.reducer import (TYPES, TIDX, tree_to_state, reducible_positions,
                              apply_reduction, valid_reductions)

SEED = 0
ITERS = int(os.environ.get("PBH_ITERS", "4000"))
BATCH = 256
MAX_STEPS = 140
TRAIN_DMAX = int(os.environ.get("PBH_DMAX", "4"))


class ConvPolicy(nn.Module):
    def __init__(self, ntypes, d=64, layers=5, k=5):
        super().__init__()
        self.emb = nn.Embedding(ntypes, d, padding_idx=0)
        self.convs = nn.ModuleList([nn.Conv1d(d, d, k, padding=k // 2) for _ in range(layers)])
        self.score = nn.Linear(d, 1)

    def forward(self, ids):
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
    return torch.from_numpy(ids).to(device), torch.from_numpy(red).to(device)


def gen_pool(depth_counts, rng):
    pool = []
    for s in G.build(depth_counts, rng, seen=set()):
        if s.error != 0:
            continue
        types, vals = tree_to_state(s.tree)
        pool.append((types, vals, s.value, s.depth))
    return pool


def rollout(policy, items, device, sample):
    """One batched rollout. Returns (rewards, logp_terms, ent_terms, final_states)."""
    states = [(list(t), list(v)) for t, v, _, _ in items]
    truth = [tv for _, _, tv, _ in items]
    n = len(states)
    done = [False] * n
    reward = np.zeros(n, dtype=np.float32)
    logp_terms, ent_terms = [], []
    for _ in range(MAX_STEPS):
        active = [i for i in range(n) if not done[i]]
        if not active:
            break
        ids, red = pad_states([states[i][0] for i in active], device)
        scores = policy(ids).masked_fill(~red, -1e9)
        dist = torch.distributions.Categorical(logits=scores)
        picks = dist.sample() if sample else scores.argmax(1)
        if sample:
            logp_terms.append((active, dist.log_prob(picks)))
            ent_terms.append(dist.entropy())
        pk = picks.cpu().numpy()
        for bi, gi in enumerate(active):
            types, vals = states[gi]
            p = int(pk[bi])
            if p not in reducible_positions(types):
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
                reward[gi] = 1.0 if (nt[0] == "VAL" and abs(nv[0] - tv) <= 1e-9 * max(1.0, abs(tv))) else 0.0
    return reward, logp_terms, ent_terms


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(SEED)
    pool = gen_pool({d: 8000 for d in range(1, TRAIN_DMAX + 1)}, rng)
    print(f"  train pool={len(pool)}  device={device}")

    policy = ConvPolicy(len(TYPES)).to(device)
    opt = torch.optim.Adam(policy.parameters(), lr=1e-3)

    for it in range(ITERS):
        idx = rng.integers(0, len(pool), BATCH)
        items = [pool[i] for i in idx]
        reward, logp_terms, ent_terms = rollout(policy, items, device, sample=True)
        adv = reward - reward.mean()
        loss = torch.zeros((), device=device)
        for active, lp in logp_terms:
            loss = loss - (lp * torch.as_tensor(adv[active], device=device)).sum()
        ent = sum(e.sum() for e in ent_terms)
        ent_coef = 0.02 * max(0.0, 1.0 - it / (0.6 * ITERS))     # decay exploration
        loss = (loss - ent_coef * ent) / BATCH
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt.step()
        if it % 250 == 0 or it == ITERS - 1:
            print(f"    iter {it:4d}  mean reward {float(reward.mean()):.3f}")

    # eval (greedy) by depth, including beyond training
    by_depth = {}
    for d in range(1, 11):
        items = gen_pool({d: 600}, np.random.default_rng(500 + d))
        if not items:
            continue
        r, _, _ = rollout(policy, items, device, sample=False)
        by_depth[d] = float(r.mean())
        print(f"  depth {d:2d}: exact-match (DISCOVERED policy) = {by_depth[d]:.3f}  (n={len(items)})")

    fig, ax = plt.subplots(figsize=(8.5, 5))
    ds = sorted(by_depth)
    ax.plot(ds, [by_depth[d] for d in ds], marker="o", color="#9467bd",
            label="DISCOVERED from outcome reward (REINFORCE, no step labels)")
    ax.plot(ds, [1.0] * len(ds), marker="s", ls=":", color="#2ca02c",
            label="supervised scratchpad reference (100%)")
    ax.axvspan(TRAIN_DMAX + 0.5, 10.5, alpha=0.06, color="red")
    ax.axvline(TRAIN_DMAX + 0.5, ls="--", color="gray")
    ax.text(TRAIN_DMAX + 0.6, 0.5, "trained 1-%d\n-> extrapolation" % TRAIN_DMAX,
            color="gray", fontsize=8, va="center")
    ax.set_ylim(-0.03, 1.05)
    ax.set_xlabel("expression nesting depth")
    ax.set_ylabel("EXACT-value match rate")
    ax.set_title("Probe B (hard): discovering a length-generalizing reduction policy\n"
                 "from OUTCOME reward alone (no per-step supervision)")
    ax.legend(loc="lower left")
    ax.grid(True, alpha=0.25)
    P.save(fig, "probe_b_hard.png")


if __name__ == "__main__":
    main()
