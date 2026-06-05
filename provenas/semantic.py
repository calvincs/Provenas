"""semantic — a neural embedding tool over the knowledge graph (TransE).

The neural half of the fabric finally grips the symbolic graph. Learns a vector per entity and
per relation such that  h + r ≈ t  for true triples (TransE; Bordes et al. 2013). From that one
learned geometry we get three reasoning powers the symbolic store can't give on its own:

  - similar(sym)        nearest entities in embedding space  -> "what is like X" (semantic match)
  - predict(h, r)       rank candidate objects for (h, r, ?) -> ADJACENCY / link prediction:
                        plausible relationships the graph doesn't explicitly store
  - plausible(h, r, t)  a confidence in [0,1] for a proposed edge -> verify a guess before asserting

Trained on aibox (tiny: a few hundred triples, CPU-fast). Deterministic given the seed.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class SemanticIndex:
    def __init__(self, kg, dim=24, seed=0):
        ents = sorted({s for t in kg.triples for s in (t[0], t[2])})
        rels = sorted({t[1] for t in kg.triples})
        self.ents, self.rels = ents, rels
        self.ei = {e: i for i, e in enumerate(ents)}
        self.ri = {r: i for i, r in enumerate(rels)}
        self.triples = [(self.ei[s], self.ri[r], self.ei[o]) for s, r, o in kg.triples]
        torch.manual_seed(seed)
        b = 6.0 / dim ** 0.5
        self.E = nn.Embedding(len(ents), dim)
        self.R = nn.Embedding(len(rels), dim)
        nn.init.uniform_(self.E.weight, -b, b)
        nn.init.uniform_(self.R.weight, -b, b)

    def _score(self, h, r, t):                                # higher = more plausible
        return -(self.E(h) + self.R(r) - self.E(t)).abs().sum(-1)

    def fit(self, epochs=500, lr=0.01, margin=1.0):
        tr = torch.tensor(self.triples)
        h, r, t = tr[:, 0], tr[:, 1], tr[:, 2]
        n = len(self.ents)
        opt = torch.optim.Adam(list(self.E.parameters()) + list(self.R.parameters()), lr=lr)
        g = torch.Generator().manual_seed(0)
        last = 0.0
        for _ in range(epochs):
            with torch.no_grad():                             # TransE: renormalize entities
                w = self.E.weight.data
                self.E.weight.data = w / w.norm(dim=1, keepdim=True).clamp_min(1e-8)
            flip = torch.rand(len(t), generator=g) < 0.5      # corrupt head or tail
            hn = torch.randint(0, n, (len(h),), generator=g)
            tn = torch.randint(0, n, (len(t),), generator=g)
            hh = torch.where(flip, hn, h)
            tt = torch.where(flip, t, tn)
            loss = torch.relu(margin - self._score(h, r, t) + self._score(hh, r, tt)).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            last = float(loss.detach())
        return last

    def similar(self, sym, k=5):
        i = self.ei[sym]
        with torch.no_grad():
            d = (self.E.weight - self.E.weight[i]).norm(dim=1)
        return [self.ents[j] for j in d.argsort().tolist() if j != i][:k]

    def predict(self, h_sym, r_sym, k=5, exclude=()):
        hi = torch.tensor(self.ei[h_sym])
        ri = torch.tensor(self.ri[r_sym])
        allt = torch.arange(len(self.ents))
        with torch.no_grad():
            s = self._score(hi.expand_as(allt), ri.expand_as(allt), allt)
        ranked = [self.ents[j] for j in s.argsort(descending=True).tolist()]
        ranked = [e for e in ranked if e not in exclude]
        return ranked[:k]

    def plausible(self, h_sym, r_sym, t_sym):
        ranked = self.predict(h_sym, r_sym, k=len(self.ents))
        rank = ranked.index(t_sym) if t_sym in ranked else len(ranked)
        return 1.0 - rank / max(1, len(self.ents) - 1)

    def vectors(self):
        return self.E.weight.detach().numpy(), self.ents
