"""provenas.engine — a neural-controlled exact-evaluation engine.

The realization of the whole project: a single **structure-only** controller (a conv
reduction policy) decides WHICH operation to reduce next — purely from token structure,
never from values — while a per-domain **tool table** supplies the EXACT computation.

Because control is structural and computation is offloaded, the SAME controller evaluates
programs in ANY binary-op domain (arithmetic, lists, boolean, …) exactly and at unbounded
depth — cross-domain pattern routing. Register a domain (a dict of op-symbol -> callable)
and call `evaluate`.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .reducer import TYPES, TIDX, reducible_positions


class ConvController(nn.Module):
    """Translation-equivariant local policy over token TYPES (never sees values)."""

    def __init__(self, ntypes=len(TYPES), d=64, layers=5, k=5):
        super().__init__()
        self.emb = nn.Embedding(ntypes, d, padding_idx=0)
        self.convs = nn.ModuleList([nn.Conv1d(d, d, k, padding=k // 2) for _ in range(layers)])
        self.score = nn.Linear(d, 1)

    def forward(self, ids):
        x = self.emb(ids).transpose(1, 2)
        for c in self.convs:
            x = torch.relu(c(x))
        return self.score(x.transpose(1, 2)).squeeze(-1)


def encode(state_types, device):
    T = max(len(s) for s in state_types)
    ids = np.zeros((len(state_types), T), dtype=np.int64)
    red = np.zeros((len(state_types), T), dtype=bool)
    for i, s in enumerate(state_types):
        ids[i, :len(s)] = [TIDX[t] for t in s]
        for p in reducible_positions(s):
            red[i, p] = True
    return torch.from_numpy(ids).to(device), torch.from_numpy(red).to(device)


@torch.no_grad()
def _pick(model, types, device):
    red = reducible_positions(types)
    if not red:
        return None
    ids, mask = encode([types], device)
    sc = model(ids).masked_fill(~mask, -1e9)[0].cpu().numpy()
    return int(np.argmax(sc))


def reduce_exact(types, vals, ops, model, device, max_steps=600):
    """Reduce a (types, vals) program to one value: `model` chooses the reduction (control),
    `ops` (symbol -> callable) computes it (exactness). Strips redundant parens as it goes."""
    types, vals = list(types), list(vals)
    model.eval()
    for _ in range(max_steps):
        if len(types) <= 1:
            break
        p = _pick(model, types, device)
        if p is None or types[p] not in ops:
            return None
        v = ops[types[p]](vals[p - 1], vals[p + 1])
        types = types[:p - 1] + ["VAL"] + types[p + 2:]
        vals = vals[:p - 1] + [v] + vals[p + 2:]
        q = p - 1
        while q - 1 >= 0 and q + 1 < len(types) and types[q - 1] == "(" and types[q + 1] == ")":
            types = types[:q - 1] + ["VAL"] + types[q + 2:]
            vals = vals[:q - 1] + [vals[q]] + vals[q + 2:]
            q -= 1
    return vals[0] if (len(types) == 1 and types[0] == "VAL") else None


class Engine:
    """Holds one shared structure-only controller and a registry of domains."""

    def __init__(self, model, device="cpu"):
        self.model = model.to(device)
        self.device = device
        self.domains = {}

    def register(self, name, ops):
        """ops: {op_symbol -> callable(a, b)}; op symbols must be among + - * /."""
        self.domains[name] = ops
        return self

    def evaluate(self, types, vals, domain):
        return reduce_exact(types, vals, self.domains[domain], self.model, self.device)
