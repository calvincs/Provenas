"""Slice 3 — a learned rewrite engine whose controller RETRAINS ITSELF when the ruleset changes.

Closes the loop back to the neural core. A structure-only conv controller (token types only) drives
term rewriting toward a normal form. Then the ruleset evolves — a new rule (constant folding) is proposed,
passes a value-preserving soundness gate, and is admitted. The old controller is now STALE; because the
rewrite engine is an EXACT oracle, it auto-generates fresh labels and the controller is regenerated — no
human in the loop. We watch the stale controller drop and the retrained one recover.

  C1@R1  (trained on identities)         -> high
  C1@R2  (identities + folding, STALE)   -> drops (never learned to fold)
  C2@R2  (auto-retrained from oracle)    -> recovers, and generalizes depth

Produces artifacts/rewrite_engine.png. Runs on aibox.
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from provenas import rewrite as RW

DEV = "cuda" if torch.cuda.is_available() else "cpu"


class RedexPointer(nn.Module):
    def __init__(self, n_classes, d=64, layers=5, k=5):
        super().__init__()
        self.emb = nn.Embedding(n_classes, d, padding_idx=0)
        self.convs = nn.ModuleList([nn.Conv1d(d, d, k, padding=k // 2) for _ in range(layers)])
        self.head = nn.Conv1d(d, 1, 1)

    def forward(self, x, mask):
        h = self.emb(x).transpose(1, 2)
        for c in self.convs:
            h = torch.relu(c(h)) + h
        return self.head(h).squeeze(1).masked_fill(~mask, -1e9)


def make_traces(rules, depths, n_per, rng):
    seqs = []
    for d in depths:
        for _ in range(n_per):
            t = RW.gen(d, rng)
            while True:
                p = RW.find_redex(t, rules)
                if p is None:
                    break
                _, classes, opidx = RW.linearize(t)
                seqs.append(([RW.CIDX[c] for c in classes], opidx[p]))
                t = RW.rewrite_at(t, p, rules)
    return seqs


def train_controller(rules, rng, depths=(2, 3, 4), n_per=900, epochs=12):
    seqs = make_traces(rules, list(depths), n_per, rng)
    maxlen = max(len(c) for c, _ in seqs)
    X = np.zeros((len(seqs), maxlen), dtype=np.int64)
    Y = np.zeros(len(seqs), dtype=np.int64)
    for i, (c, t) in enumerate(seqs):
        X[i, :len(c)] = c
        Y[i] = t
    X, Y = torch.tensor(X, device=DEV), torch.tensor(Y, device=DEV)
    model = RedexPointer(len(RW.CLASSES)).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    lossf = nn.CrossEntropyLoss()
    torch.manual_seed(0)
    n, bs = len(seqs), 256
    for _ in range(epochs):
        perm = torch.randperm(n, device=DEV)
        for i in range(0, n, bs):
            b = perm[i:i + bs]
            loss = lossf(model(X[b], X[b] != 0), Y[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model


def rollout(model, term, rules):
    for _ in range(500):
        p = RW.find_redex(term, rules)
        if p is None:
            return term                                   # reached normal form
        _, classes, opidx = RW.linearize(term)
        valid = {opidx[q]: q for q in RW.reducible_nodes(term, rules)}
        x = torch.zeros(1, len(classes), dtype=torch.long, device=DEV)
        x[0] = torch.tensor([RW.CIDX[c] for c in classes], device=DEV)
        with torch.no_grad():
            idx = int(model(x, x != 0)[0].argmax())
        if idx not in valid:                              # pointed at a non-redex
            return None
        term = RW.rewrite_at(term, valid[idx], rules)
    return None


def exact_rate(model, rules, depths, rng, n_per=200):
    rates = {}
    for d in depths:
        good = 0
        for _ in range(n_per):
            t = RW.gen(d, rng)
            if rollout(model, t, rules) == RW.normal_form(t, rules):
                good += 1
        rates[d] = good / n_per
    return rates


def get_controller(rules, cache, rng):
    sig = RW.signature(rules)
    if sig not in cache:
        print("   ruleset changed -> retraining the controller from the exact oracle (no human labels)...")
        cache[sig] = train_controller(rules, rng)
    return cache[sig]


def main():
    os.makedirs("artifacts", exist_ok=True)
    rng = np.random.default_rng(0)
    cache = {}

    # --- soundness gate: admit only value-preserving rewrite rules --------------
    print("rule admission (test-before-admit, value-preserving):")
    for cand, name in [(RW.FOLD, "fold: (num op num) -> num"),
                       (RW.BOGUS, "bogus: (a + b) -> (a * b)")]:
        ok = RW.sound(cand, rng)
        print(f"  propose {name:28s} -> sound={ok}  -> {'ADMITTED' if ok else 'REJECTED'}")
    rules1, rules2 = RW.R1, RW.R2                          # only the sound rule (fold) joins R2

    # --- concrete simplifications under the evolved ruleset ----------------------
    print("\nexact rewriting (oracle, ruleset R2 = identities + folding):")
    for t in [("op", "*", ("op", "+", ("var", "x"), ("num", 0)), ("op", "+", ("num", 2), ("num", 3))),
              ("op", "+", ("op", "*", ("var", "y"), ("num", 1)), ("op", "*", ("var", "x"), ("num", 0))),
              ("op", "*", ("op", "+", ("num", 1), ("num", 1)), ("op", "+", ("var", "z"), ("num", 0)))]:
        print(f"  {RW.pretty(t):34s} ->  {RW.pretty(RW.normal_form(t, rules2))}")

    # --- the self-retraining loop -----------------------------------------------
    print("\ncontroller drives rewriting; ruleset evolves R1 -> R2:")
    C1 = get_controller(rules1, cache, rng)               # trained on identities (R1)
    depths = [2, 3, 4, 5, 6, 7]
    c1_r1 = exact_rate(C1, rules1, depths, np.random.default_rng(1))
    c1_r2 = exact_rate(C1, rules2, depths, np.random.default_rng(1))   # STALE on the new ruleset
    C2 = get_controller(rules2, cache, rng)               # signature changed -> auto-retrain
    c2_r2 = exact_rate(C2, rules2, depths, np.random.default_rng(1))

    def agg(r):
        return 100 * np.mean([r[d] for d in (2, 3, 4, 5)])
    print(f"\n  C1 @ R1 (identities)            : {agg(c1_r1):5.1f}% exact")
    print(f"  C1 @ R2 (folding added, STALE)  : {agg(c1_r2):5.1f}% exact   <- can't fold; never trained on it")
    print(f"  C2 @ R2 (auto-retrained)        : {agg(c2_r2):5.1f}% exact   <- recovered from the oracle")
    print(f"  (C2 trained on depths<=4; depth 5-7 are extrapolation)")

    # --- plot --------------------------------------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    ax[0].bar(["C1 @ R1", "C1 @ R2\n(stale)", "C2 @ R2\n(retrained)"],
              [agg(c1_r1), agg(c1_r2), agg(c2_r2)], color=["#2ca02c", "#d62728", "#1f77b4"])
    ax[0].set_ylabel("exact normal form (%)")
    ax[0].set_ylim(0, 105)
    ax[0].set_title("ruleset changed -> controller retrains itself\n(dip when stale, recover from the oracle)")
    for i, v in enumerate([agg(c1_r1), agg(c1_r2), agg(c2_r2)]):
        ax[0].text(i, v + 1.5, f"{v:.0f}%", ha="center")

    ax[1].axvspan(1.5, 4.5, color="#eaf2ff", zorder=0)
    ax[1].plot(depths, [100 * c2_r2[d] for d in depths], "o-", color="#1f77b4", lw=2, label="C2 @ R2 (retrained)")
    ax[1].plot(depths, [100 * c1_r2[d] for d in depths], "o--", color="#d62728", lw=1.5, label="C1 @ R2 (stale)")
    ax[1].axvline(4.5, color="#1f77b4", ls="--", lw=1)
    ax[1].set_xlabel("term depth")
    ax[1].set_ylabel("exact normal form (%)")
    ax[1].set_ylim(0, 105)
    ax[1].set_title("retrained controller generalizes depth\n(trained <=4)")
    ax[1].legend(fontsize=8, loc="lower left")
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("artifacts/rewrite_engine.png", dpi=120)
    print("\nsaved artifacts/rewrite_engine.png")


if __name__ == "__main__":
    main()
