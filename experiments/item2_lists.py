"""Item 2: does the hybrid hold up BEYOND arithmetic?

A non-arithmetic domain — nested LIST programs over concat (`+`) and interleave (`*`) —
evaluated by the SAME local-reduction machinery as the scratchpad. Key point: the conv
reduction controller reasons about STRUCTURE only (token types; it never sees values), so
it should be DOMAIN-AGNOSTIC. We swap the VM's ops to list ops, train the conv policy on
list-reduction traces, and check exact list-match generalizing depth (trained 1-4, tested
to 8). If it works, "neural dispatch + symbolic VM" holds on symbolic/list values, not just
numbers.

  artifacts/item2_lists.png
"""
from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from provenas import plotting as P
from provenas.reducer import TYPES, TIDX, reducible_positions, local_next

SEED = 0
EPOCHS = int(os.environ.get("IT2_EPOCHS", "15"))
PRECL = {"+": 1, "*": 2}            # same precedences as reducer.PREC, so local_next transfers
SYMS = ["+", "*"]


class LNode:
    __slots__ = ("kind", "value", "op", "left", "right", "depth")

    def __init__(self, kind, value=None, op=None, left=None, right=None):
        self.kind, self.value, self.op, self.left, self.right = kind, value, op, left, right
        self.depth = 0 if kind == "leaf" else 1 + max(left.depth, right.depth)


def rand_list(rng):
    return [int(rng.integers(0, 10)) for _ in range(int(rng.integers(1, 4)))]


def gen(depth, rng):
    if depth == 0:
        return LNode("leaf", value=rand_list(rng))
    op = SYMS[int(rng.integers(2))]
    deep, shallow = gen(depth - 1, rng), gen(int(rng.integers(0, depth)), rng)
    l, r = (deep, shallow) if rng.random() < 0.5 else (shallow, deep)
    return LNode("op", op=op, left=l, right=r)


def interleave(a, b):
    out = []
    for i in range(max(len(a), len(b))):
        if i < len(a):
            out.append(a[i])
        if i < len(b):
            out.append(b[i])
    return out


def list_op(op, a, b):
    return (a + b) if op == "+" else interleave(a, b)


def tree_to_state(node):
    types, vals = [], []

    def emit(n):
        if n.kind == "leaf":
            types.append("VAL")
            vals.append(list(n.value))
            return
        p = PRECL[n.op]
        lpar = (n.left.kind == "op" and PRECL[n.left.op] < p)
        rpar = (n.right.kind == "op" and PRECL[n.right.op] <= p)
        if lpar:
            types.append("("); vals.append(None)
        emit(n.left)
        if lpar:
            types.append(")"); vals.append(None)
        types.append(n.op); vals.append(None)
        if rpar:
            types.append("("); vals.append(None)
        emit(n.right)
        if rpar:
            types.append(")"); vals.append(None)

    emit(node)
    return types, vals


def eval_tree(node):
    if node.kind == "leaf":
        return list(node.value)
    return list_op(node.op, eval_tree(node.left), eval_tree(node.right))


def apply_reduction(types, vals, p):
    v = list_op(types[p], vals[p - 1], vals[p + 1])
    types = types[:p - 1] + ["VAL"] + types[p + 2:]
    vals = vals[:p - 1] + [v] + vals[p + 2:]
    q = p - 1
    while q - 1 >= 0 and q + 1 < len(types) and types[q - 1] == "(" and types[q + 1] == ")":
        types = types[:q - 1] + ["VAL"] + types[q + 2:]
        vals = vals[:q - 1] + [vals[q]] + vals[q + 2:]
        q -= 1
    return types, vals


def reduce_run(types, vals, policy, max_steps=200):
    types, vals = list(types), list(vals)
    steps = []
    for _ in range(max_steps):
        if len(types) <= 1:
            break
        pp = policy(types)
        if pp is None:
            return None, steps
        steps.append((list(types), pp))
        types, vals = apply_reduction(types, vals, pp)
    return (vals[0] if len(types) == 1 and types[0] == "VAL" else None), steps


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


def make_trees(depth_counts, rng):
    out = []
    for d, n in depth_counts.items():
        for _ in range(n):
            t = gen(d, rng)
            out.append((t, eval_tree(t), d))
    return out


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(SEED)
    train = make_trees({1: 4000, 2: 4000, 3: 4000, 4: 4000}, rng)

    states, targets = [], []
    for t, _, _ in train:
        types, vals = tree_to_state(t)
        _, steps = reduce_run(types, vals, local_next)
        for st, pos in steps:
            states.append(st)
            targets.append(pos)
    targets = np.array(targets)
    print(f"  training pairs={len(states)}")

    model = ConvPointer(len(TYPES)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)
    n = len(states)
    for _ in range(EPOCHS):
        perm = rng.permutation(n)
        for i in range(0, n, 256):
            idx = perm[i:i + 256]
            ids, red = pad_states([states[j] for j in idx], device)
            logits = model(ids).masked_fill(~red, -1e9)
            loss = F.cross_entropy(logits, torch.tensor(targets[idx], device=device))
            opt.zero_grad()
            loss.backward()
            opt.step()

    @torch.no_grad()
    def model_policy(types):
        red = reducible_positions(types)
        if not red:
            return None
        ids, mask = pad_states([types], device)
        sc = model(ids).masked_fill(~mask, -1e9)[0].cpu().numpy()
        return int(np.argmax(sc))

    by_depth = {}
    for d in range(1, 9):
        items = make_trees({d: 400}, np.random.default_rng(50 + d))
        ok = 0
        for t, truth, _ in items:
            types, vals = tree_to_state(t)
            pv, _ = reduce_run(types, vals, model_policy)
            ok += (pv == truth)
        by_depth[d] = ok / len(items)
        print(f"  depth {d}: exact list-match = {by_depth[d]:.3f}")

    fig, ax = plt.subplots(figsize=(8.5, 5))
    ds = sorted(by_depth)
    ax.plot(ds, [by_depth[d] for d in ds], marker="o", color="#8c564b",
            label="conv reduction controller on LIST programs (concat/interleave)")
    ax.axvspan(4.5, 8.5, alpha=0.06, color="red")
    ax.axvline(4.5, ls="--", color="gray")
    ax.text(4.6, 0.5, "trained 1-4\n-> extrapolation", color="gray", fontsize=8, va="center")
    ax.set_ylim(-0.03, 1.05)
    ax.set_xlabel("expression nesting depth")
    ax.set_ylabel("EXACT list-match rate")
    ax.set_title("Item 2: the hybrid holds beyond arithmetic\n"
                 "same structure-only controller, non-arithmetic (list) VM")
    ax.legend(loc="lower left")
    ax.grid(True, alpha=0.25)
    P.save(fig, "item2_lists.png")


if __name__ == "__main__":
    main()
