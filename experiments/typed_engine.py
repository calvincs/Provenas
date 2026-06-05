"""Phase 5 — the typed engine: comparisons + if/else, driven by a structure-only controller.

Two results on one fabric:
  A. EXACT mixed-type reduction (the capability) — numbers, comparisons that produce booleans,
     boolean logic, and if/else BRANCHING with short-circuit, evaluated exactly. Demonstrated by a
     few worked programs incl. a safe-division `if` whose dead branch (10/0) is never touched.
  B. A STRUCTURE-ONLY controller (a translation-equivariant conv pointer over token TYPES — it never
     sees a value) learns to pick the next redex and DRIVES the whole reduction. Trained on shallow
     expressions (depth <= 4), it generalizes to deeper ones (5-7), evaluating them exactly — the
     same length/depth-generalization trick as the arithmetic scratchpad, now over a typed grammar
     with branching.

Produces artifacts/typed_engine.png. Runs on aibox.
"""
from __future__ import annotations

import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from provenas import typed as T

DEV = "cuda" if torch.cuda.is_available() else "cpu"


# ----------------------------------------------------------------- A. capability demo
def demo():
    progs = {
        "mixed types  (a+b)*c compared, AND'd":
            ("bin", "&",
             ("bin", ">", ("bin", "*", ("bin", "+", ("lit", 3, "num"), ("lit", 4, "num")),
                           ("lit", 2, "num")), ("lit", 10, "num")),
             ("bin", "<", ("lit", 1, "num"), ("lit", 2, "num"))),
        "branch returns a number":
            ("if", ("bin", ">", ("lit", 3, "num"), ("lit", 5, "num")),
             ("bin", "*", ("lit", 2, "num"), ("lit", 2, "num")),
             ("bin", "+", ("lit", 100, "num"), ("lit", 1, "num"))),
        "short-circuit safe division (10/0 never runs)":
            ("if", ("bin", "==", ("bin", "-", ("lit", 5, "num"), ("lit", 5, "num")), ("lit", 0, "num")),
             ("lit", 0, "num"),
             ("bin", "/", ("lit", 10, "num"), ("bin", "-", ("lit", 5, "num"), ("lit", 5, "num")))),
    }
    print("A. exact mixed-type evaluation:")
    for label, node in progs.items():
        v, ty = T.TypedEngine().evaluate(node)
        print(f"  {label}\n     {T.render(node)}  =  {v}  ({ty})")


# ----------------------------------------------------------------- B. the controller
class RedexPointer(nn.Module):
    """Conv pointer over token-type ids -> a score per position; argmax = next redex. No positional
    encoding (translation-equivariant) -> generalizes to longer/deeper sequences."""
    def __init__(self, n_classes, d=64, layers=6, k=7):
        super().__init__()
        self.emb = nn.Embedding(n_classes, d, padding_idx=0)
        self.convs = nn.ModuleList([nn.Conv1d(d, d, k, padding=k // 2) for _ in range(layers)])
        self.head = nn.Conv1d(d, 1, 1)

    def forward(self, x, mask):
        h = self.emb(x).transpose(1, 2)
        for c in self.convs:
            h = torch.relu(c(h)) + h
        logit = self.head(h).squeeze(1)
        return logit.masked_fill(~mask, -1e9)


def make_traces(depths, n_per, rng):
    seqs = []
    for d in depths:
        for _ in range(n_per):
            want = "num" if rng.random() < 0.6 else "bool"
            n = T.gen(d, want, rng)
            while n[0] != "lit":
                _, classes, opidx = T.linearize(n)
                p = T.find_redex(n)
                seqs.append(([T.CIDX[c] for c in classes], opidx[p]))
                n = T.apply_redex(n, p)
    return seqs


def batchify(seqs, maxlen):
    X = np.zeros((len(seqs), maxlen), dtype=np.int64)
    Y = np.zeros(len(seqs), dtype=np.int64)
    for i, (cls, tgt) in enumerate(seqs):
        X[i, :len(cls)] = cls
        Y[i] = tgt
    return torch.tensor(X), torch.tensor(Y)


def rollout(model, n):
    for _ in range(2000):
        if n[0] == "lit":
            return n[1]
        _, classes, opidx = T.linearize(n)
        valid = {opidx[p] for p in T.valid_redexes(n)}
        x = torch.zeros(1, len(classes), dtype=torch.long, device=DEV)
        x[0] = torch.tensor([T.CIDX[c] for c in classes], device=DEV)
        with torch.no_grad():
            logit = model(x, x != 0)
        idx = int(logit[0].argmax())
        if idx not in valid:                       # pointed at a non-redex -> failed
            return None
        inv = {v: k for k, v in opidx.items()}
        n = T.apply_redex(n, inv[idx])
    return None


def main():
    os.makedirs("artifacts", exist_ok=True)
    demo()

    rng = np.random.default_rng(0)
    train = make_traces([0, 1, 2, 3, 4], 700, rng)
    maxlen = max(len(c) for c, _ in train) + 8
    X, Y = batchify(train, maxlen)
    X, Y = X.to(DEV), Y.to(DEV)
    print(f"\nB. structure-only controller: {len(train)} reduction steps "
          f"(depths<=4), maxlen {maxlen}, device {DEV}")

    model = RedexPointer(len(T.CLASSES)).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    lossf = nn.CrossEntropyLoss()
    torch.manual_seed(0)
    n, bs = len(train), 256
    for ep in range(12):
        perm = torch.randperm(n, device=DEV)
        tot = 0.0
        for i in range(0, n, bs):
            b = perm[i:i + bs]
            xb = X[b]
            logit = model(xb, xb != 0)
            loss = lossf(logit, Y[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss.detach()) * len(b)
        if ep % 3 == 0 or ep == 11:
            with torch.no_grad():
                acc = (model(X, X != 0).argmax(1) == Y).float().mean().item()
            print(f"  epoch {ep:2d}  loss {tot/n:.4f}  per-step pointer acc {acc*100:.1f}%")

    # depth generalization: end-to-end exact evaluation, in-dist (<=4) and extrapolation (5-7)
    print("\n  end-to-end: controller drives the FULL reduction, value must match the oracle")
    depths = list(range(1, 8))
    exact_rate, step_acc = [], []
    ev = np.random.default_rng(1)
    for d in depths:
        good = total = sright = ssteps = 0
        for _ in range(300):
            want = "num" if ev.random() < 0.6 else "bool"
            node = T.gen(d, want, ev)
            gold = T.eval_node(node)
            # per-step pointer accuracy along the oracle trace
            m = node
            while m[0] != "lit":
                _, classes, opidx = T.linearize(m)
                x = torch.zeros(1, len(classes), dtype=torch.long, device=DEV)
                x[0] = torch.tensor([T.CIDX[c] for c in classes], device=DEV)
                with torch.no_grad():
                    pred = int(model(x, x != 0)[0].argmax())
                sright += int(pred == opidx[T.find_redex(m)])
                ssteps += 1
                m = T.apply_redex(m, T.find_redex(m))
            got = rollout(model, node)
            good += int(got == gold)
            total += 1
        exact_rate.append(good / total)
        step_acc.append(sright / max(1, ssteps))
        tag = "train" if d <= 4 else "EXTRAP"
        print(f"    depth {d} ({tag:6s}): end-to-end exact {good/total*100:5.1f}%   "
              f"per-step pointer {sright/max(1,ssteps)*100:5.1f}%")

    # perf: exact structured reduction vs raw Python (honest cost of the controllable engine)
    pr = np.random.default_rng(2)
    nodes = [T.gen(4, "num" if pr.random() < 0.6 else "bool", pr) for _ in range(3000)]
    t0 = time.perf_counter()
    for nd in nodes:
        T.reduce_to_value(nd)
    t_red = time.perf_counter() - t0
    t0 = time.perf_counter()
    for nd in nodes:
        T.eval_node(nd)
    t_eval = time.perf_counter() - t0
    print(f"\n  perf (3000 exprs): local reduction {t_red*1e3:.0f} ms  vs  raw Python eval "
          f"{t_eval*1e3:.0f} ms  ({t_red/t_eval:.1f}x — the price of inspectable, controllable steps)")

    # ---- plot ----
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    for a in ax:
        a.axvspan(0.5, 4.5, color="#e8f0ff", zorder=0)
        a.axvline(4.5, color="#1f77b4", ls="--", lw=1)
        a.set_xlabel("expression depth")
        a.set_xticks(depths)
    ax[0].plot(depths, [r * 100 for r in exact_rate], "o-", color="#d62728", lw=2)
    ax[0].set_ylabel("end-to-end exact eval (%)")
    ax[0].set_ylim(0, 103)
    ax[0].set_title("controller drives typed reduction\n(trained ≤4, blue = train region)")
    ax[1].plot(depths, [s * 100 for s in step_acc], "o-", color="#2ca02c", lw=2)
    ax[1].set_ylabel("per-step redex-pointer acc (%)")
    ax[1].set_ylim(0, 103)
    ax[1].set_title("structure-only pointer accuracy")
    for a in ax:
        a.text(2.5, 6, "trained", color="#1f77b4", ha="center", fontsize=9)
        a.text(6.0, 6, "extrapolation", color="#444", ha="center", fontsize=9)
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("artifacts/typed_engine.png", dpi=120)
    print("\nsaved artifacts/typed_engine.png")


if __name__ == "__main__":
    main()
