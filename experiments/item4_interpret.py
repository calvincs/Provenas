"""Item 4: interpretability — what RULE did the learned reduction controllers discover?

Two conv reduction policies on the SAME structure-only task:
  - SUPERVISED: trained to imitate local_next (leftmost-valid reduction).
  - RL-DISCOVERED: trained by REINFORCE from terminal reward only (no rule given).

(A) Agreement: how often does each pick exactly local_next, vs any VALID reduction?
(B) Rule extraction (the conv is local, so we can reverse-engineer it):
    - precedence: on `V o1 V o2 V`, which op does it reduce first? (reveals precedence)
    - tie-break: on `(V+V) o (V+V)` (two INDEPENDENT valid reductions), left or right?
      Supervised was trained leftmost; does RL pick the same, or a different valid order?

  artifacts/item4_interpret.png
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from provenas import plotting as P
from provenas import exprgen as G
from provenas.reducer import (TYPES, TIDX, PREC, reducible_positions, valid_reductions,
                              local_next, tree_to_state, apply_reduction, reduce_to_value)

SEED = 0
OPS = ["+", "-", "*", "/"]


class ConvPointer(nn.Module):
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


def gen_pool(dc, rng):
    pool = []
    for s in G.build(dc, rng, seen=set()):
        if s.error != 0:
            continue
        t, v = tree_to_state(s.tree)
        pool.append((t, v, s.value))
    return pool


def train_supervised(pool, device, rng, epochs=15):
    states, targets = [], []
    for types, vals, _ in pool:
        _, st, steps = reduce_to_value(types, vals, local_next)
        if st != "ok":
            continue
        for stp, pos in steps:
            states.append(stp)
            targets.append(pos)
    targets = np.array(targets)
    model = ConvPointer(len(TYPES)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)
    n = len(states)
    for _ in range(epochs):
        perm = rng.permutation(n)
        for i in range(0, n, 256):
            idx = perm[i:i + 256]
            ids, red = pad_states([states[j] for j in idx], device)
            loss = F.cross_entropy(model(ids).masked_fill(~red, -1e9),
                                   torch.tensor(targets[idx], device=device))
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model


def train_rl(pool, device, rng, iters=2500, bs=256):
    model = ConvPointer(len(TYPES)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for it in range(iters):
        items = [pool[i] for i in rng.integers(0, len(pool), bs)]
        states = [(list(t), list(v)) for t, v, _ in items]
        truth = [tv for _, _, tv in items]
        done = [False] * bs
        reward = np.zeros(bs, dtype=np.float32)
        logps, ents = [], []
        for _ in range(140):
            active = [i for i in range(bs) if not done[i]]
            if not active:
                break
            ids, red = pad_states([states[i][0] for i in active], device)
            dist = torch.distributions.Categorical(logits=model(ids).masked_fill(~red, -1e9))
            picks = dist.sample()
            logps.append((active, dist.log_prob(picks)))
            ents.append(dist.entropy())
            pk = picks.cpu().numpy()
            for bi, gi in enumerate(active):
                tt, vv = states[gi]
                p = int(pk[bi])
                if p not in reducible_positions(tt):
                    done[gi] = True
                    continue
                nt, nv, e = apply_reduction(tt, vv, p)
                if e != "ok":
                    done[gi] = True
                    continue
                states[gi] = (nt, nv)
                if len(nt) == 1:
                    done[gi] = True
                    reward[gi] = 1.0 if abs(nv[0] - truth[gi]) <= 1e-9 * max(1.0, abs(truth[gi])) else 0.0
        adv = reward - reward.mean()
        loss = torch.zeros((), device=device)
        for active, lp in logps:
            loss = loss - (lp * torch.as_tensor(adv[active], device=device)).sum()
        ec = 0.02 * max(0.0, 1 - it / (0.6 * iters))
        loss = (loss - ec * sum(e.sum() for e in ents)) / bs
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    return model


@torch.no_grad()
def pick(model, types, device):
    red = reducible_positions(types)
    ids, mask = pad_states([types], device)
    sc = model(ids).masked_fill(~mask, -1e9)[0].cpu().numpy()
    return int(np.argmax(sc)) if red else None


@torch.no_grad()
def agreement(model, pool, device):
    same, valid, total = 0, 0, 0
    multi_left = multi_right = multi_total = 0
    for types, vals, _ in pool:
        _, st, steps = reduce_to_value(types, vals, local_next)
        for stp, _ in steps:
            p = pick(model, stp, device)
            vr = valid_reductions(stp)
            same += (p == local_next(stp))
            valid += (p in vr)
            total += 1
            if len(vr) > 1:                       # multiple independent valid reductions
                multi_total += 1
                multi_left += (p == min(vr))      # leftmost?
                multi_right += (p == max(vr))     # rightmost?
    return (same / total, valid / total,
            (multi_left / multi_total) if multi_total else float("nan"),
            (multi_right / multi_total) if multi_total else float("nan"), multi_total)


@torch.no_grad()
def precedence_table(model, device):
    """On `V o1 V o2 V`, does it pick o1 (left) or o2 (right)? vs the precedence-correct pick."""
    tab = np.zeros((4, 4), dtype=int)          # 1 = matches precedence rule
    for i, o1 in enumerate(OPS):
        for j, o2 in enumerate(OPS):
            types = ["VAL", o1, "VAL", o2, "VAL"]
            p = pick(model, types, device)
            correct = 1 if PREC[o1] >= PREC[o2] else 3      # pos of higher-prec (tie -> left)
            tab[i, j] = int(p == correct)
    return tab


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(SEED)
    pool = gen_pool({1: 5000, 2: 5000, 3: 5000, 4: 5000}, rng)
    test = gen_pool({2: 800, 3: 800, 4: 800, 5: 800, 6: 800}, np.random.default_rng(99))

    sup = train_supervised(pool, device, np.random.default_rng(1))
    rl = train_rl(pool, device, np.random.default_rng(2))

    out = {}
    for name, m in [("supervised", sup), ("RL-discovered", rl)]:
        s, v, ml, mr, mt = agreement(m, test, device)
        pt = precedence_table(m, device)
        out[name] = dict(same=s, valid=v, multi_left=ml, multi_right=mr, multi_total=mt, prec=float(pt.mean()))
        print(f"  {name:14s}  ==local_next={s:.3f}  valid={v:.3f}  precedence-correct={pt.mean():.3f}  "
              f"tie-break: leftmost={ml:.3f} rightmost={mr:.3f} (n_multi={mt})")

    fig, ax = plt.subplots(figsize=(9, 5))
    labels = ["== local_next\n(our rule)", "valid\n(any correct)", "precedence\ncorrect", "leftmost\ntie-break"]
    x = np.arange(len(labels))
    w = 0.38
    for k, (name, color) in enumerate([("supervised", "#1f77b4"), ("RL-discovered", "#9467bd")]):
        vals = [out[name]["same"], out[name]["valid"], out[name]["prec"], out[name]["multi_left"]]
        ax.bar(x + (k - 0.5) * w, vals, w, label=name, color=color)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("fraction")
    ax.set_title("Item 4: what rule did each controller learn?\n"
                 "(both correct & precedence-aware; tie-break reveals if RL found a DIFFERENT valid order)")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.25)
    P.save(fig, "item4_interpret.png")


if __name__ == "__main__":
    main()
