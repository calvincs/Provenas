"""Item 5: self-improving curriculum — does the system bootstrap to a task that direct
training can't crack?

Setup: RL-discover the reduction policy (Probe B style) but on DEEP expressions with sparse
terminal reward. Direct training only on depth-8 should fail (a random policy almost never
produces a fully-correct depth-8 reduction -> no reward -> stuck). A SELF-CURRICULUM that
starts shallow and auto-ratchets depth once it masters the current level should bootstrap to
depth 8. We track exact-match at depth 8 over training for both.

  artifacts/item5_curriculum.png
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from provenas import plotting as P
from provenas import exprgen as G
from provenas.reducer import (TYPES, TIDX, reducible_positions, apply_reduction, tree_to_state)

SEED = 0
ITERS = 3000
BATCH = 256
TARGET_DEPTH = 8


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


def pools_by_depth(maxd, per, rng):
    pools = {}
    for d in range(1, maxd + 1):
        pools[d] = [(tree_to_state(s.tree)[0], tree_to_state(s.tree)[1], s.value)
                    for s in G.build({d: per}, rng, seen=set()) if s.error == 0]
    return pools


def rollout(policy, items, device, sample):
    states = [(list(t), list(v)) for t, v, _ in items]
    truth = [tv for _, _, tv in items]
    n = len(states)
    done = [False] * n
    reward = np.zeros(n, dtype=np.float32)
    logps, ents = [], []
    for _ in range(140):
        active = [i for i in range(n) if not done[i]]
        if not active:
            break
        ids, red = pad_states([states[i][0] for i in active], device)
        scores = policy(ids).masked_fill(~red, -1e9)
        dist = torch.distributions.Categorical(logits=scores)
        picks = dist.sample() if sample else scores.argmax(1)
        if sample:
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
    return reward, logps, ents


def update(policy, opt, items, device, ent_coef):
    reward, logps, ents = rollout(policy, items, device, sample=True)
    adv = reward - reward.mean()
    loss = torch.zeros((), device=device)
    for active, lp in logps:
        loss = loss - (lp * torch.as_tensor(adv[active], device=device)).sum()
    loss = (loss - ent_coef * sum(e.sum() for e in ents)) / len(items)
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
    opt.step()
    return float(reward.mean())


@torch.no_grad()
def eval_depth(policy, pool_d, device):
    r, _, _ = rollout(policy, pool_d, device, sample=False)
    return float(r.mean())


def run(mode, pools, device, rng):
    policy = ConvPolicy(len(TYPES)).to(device)
    opt = torch.optim.Adam(policy.parameters(), lr=1e-3)
    cur = TARGET_DEPTH if mode == "direct" else 1
    recent, trace = [], []
    test8 = pools[TARGET_DEPTH][:400]
    for it in range(ITERS):
        d = TARGET_DEPTH if mode == "direct" else int(rng.integers(1, cur + 1))
        items = [pools[d][i] for i in rng.integers(0, len(pools[d]), BATCH)]
        ec = 0.005 + 0.015 * max(0.0, 1 - it / (0.6 * ITERS))   # entropy FLOOR avoids collapse
        r = update(policy, opt, items, device, ec)
        recent.append(r)
        if mode == "curriculum" and len(recent) >= 60 and np.mean(recent[-60:]) > 0.9 and cur < TARGET_DEPTH:
            cur += 1
            recent = []
        if it % 150 == 0 or it == ITERS - 1:
            e8 = eval_depth(policy, test8, device)
            trace.append((it, e8, cur))
            print(f"  {mode:10s} it={it:4d} depth-curr={cur} depth8-exact={e8:.3f}")
    return trace


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pools = pools_by_depth(TARGET_DEPTH, 6000, np.random.default_rng(SEED))
    direct = run("direct", pools, device, np.random.default_rng(1))
    curr = run("curriculum", pools, device, np.random.default_rng(2))

    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.plot([t for t, _, _ in direct], [e for _, e, _ in direct], marker="x", ls="--",
            color="#d62728", label="direct RL on depth-8 (sparse reward)")
    ax.plot([t for t, _, _ in curr], [e for _, e, _ in curr], marker="o", color="#2ca02c",
            label="self-curriculum (auto-ratchets depth)")
    ax.set_ylim(-0.03, 1.05)
    ax.set_xlabel("training iteration")
    ax.set_ylabel("exact-match at depth 8")
    ax.set_title("Item 5 (honest negative): the reduction task is directly RL-learnable even at depth 8\n"
                 "curriculum is unnecessary here; the earlier 'direct collapse' was an entropy-schedule artifact")
    ax.legend(loc="center left")
    ax.grid(True, alpha=0.25)
    P.save(fig, "item5_curriculum.png")


if __name__ == "__main__":
    main()
