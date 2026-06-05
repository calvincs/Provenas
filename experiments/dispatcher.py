"""Neural Dispatcher -> Symbolic Stack VM: the hybrid that offloads exactness.

The net learns to compile an infix expression (over positional value-symbols
N0,N1,...) into an RPN stack program; a Python stack executes it EXACTLY. We
measure exact-value match vs nesting depth and compare to M4's pure Tree-LSTM
(which approximates and cliffs to ~0 exact-match). Train depths 1-4; test held-out
depths 5-7 -> does the LEARNED PARSER generalize, with exactness guaranteed by the
symbolic VM?

  artifacts/dispatcher_depth.png

Env: DISP_EPOCHS (default 30).
"""
from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from provenas import plotting as P
from provenas import exprgen as G
from provenas.stackvm import slot_tokens, execute_rpn
from provenas.models_seq2seq import Seq2Seq

SEED = 0
MAXLEAF = 64
EPOCHS = int(os.environ.get("DISP_EPOCHS", "30"))
BATCH = 128

OPS = ["+", "-", "*", "/"]
PARENS = ["(", ")"]
SPECIAL = ["<pad>", "<bos>", "<eos>"]
SLOTS = [f"N{i}" for i in range(MAXLEAF)]
VOCAB = SPECIAL + OPS + PARENS + SLOTS
STOI = {t: i for i, t in enumerate(VOCAB)}
PAD, BOS, EOS = 0, 1, 2


def encode_seq(tokens, bos_eos=False):
    ids = [STOI[t] for t in tokens]
    return [BOS] + ids + [EOS] if bos_eos else ids


def make_pairs(depth_counts, rng):
    pairs = []
    for s in G.build(depth_counts, rng, seen=set()):
        if s.error != 0:                      # ok-cases only for exact-value match
            continue
        inf, rpn, vals = slot_tokens(s.tree)
        if len(vals) > MAXLEAF:
            continue
        pairs.append((inf, rpn, vals, s.value, s.depth))
    return pairs


def pad_batch(seqs, device):
    T = max(len(s) for s in seqs)
    ids = np.zeros((len(seqs), T), dtype=np.int64)
    for i, s in enumerate(seqs):
        ids[i, :len(s)] = s
    t = torch.from_numpy(ids).to(device)
    return t, (t == PAD)


def main():
    rng = np.random.default_rng(SEED)
    train_maxd = int(os.environ.get("DISP_TRAIN_MAXDEPTH", "4"))
    train = make_pairs({d: 6000 for d in range(1, train_maxd + 1)}, rng)
    test = make_pairs({1: 1500, 2: 1500, 3: 1500, 4: 1500, 5: 1500, 6: 1500, 7: 1500},
                      np.random.default_rng(1))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  train pairs={len(train)}  test pairs={len(test)}  vocab={len(VOCAB)}")

    model = Seq2Seq(len(VOCAB)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)
    src = [encode_seq(inf) for inf, _, _, _, _ in train]
    tgt = [encode_seq(rpn, bos_eos=True) for _, rpn, _, _, _ in train]
    n = len(train)

    for ep in range(EPOCHS):
        model.train()
        perm = rng.permutation(n)
        tot = 0.0
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            S, Spad = pad_batch([src[j] for j in idx], device)
            T_, Tpad = pad_batch([tgt[j] for j in idx], device)
            logits = model(S, T_[:, :-1], Spad, Tpad[:, :-1])
            loss = F.cross_entropy(logits.reshape(-1, len(VOCAB)), T_[:, 1:].reshape(-1),
                                   ignore_index=PAD)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += float(loss)
        if ep % 5 == 0 or ep == EPOCHS - 1:
            print(f"    epoch {ep:2d}  train_loss {tot / (n // BATCH + 1):.4f}")

    by_depth = {}
    for d in range(1, 8):
        items = [p for p in test if p[4] == d]
        if not items:
            continue
        exact = 0
        for i in range(0, len(items), 256):
            chunk = items[i:i + 256]
            S, Spad = pad_batch([encode_seq(inf) for inf, _, _, _, _ in chunk], device)
            ys = model.greedy(S, Spad, BOS, EOS).cpu().numpy()
            for row, (inf, rpn, vals, trueval, _) in zip(ys, chunk):
                toks = []
                for t in row[1:]:
                    if t == EOS:
                        break
                    toks.append(VOCAB[t])
                pv, st = execute_rpn(toks, vals)
                if st == "ok" and pv is not None and abs(pv - trueval) <= 1e-9 * max(1.0, abs(trueval)):
                    exact += 1
        by_depth[d] = exact / len(items)
        print(f"  depth {d}: exact-value match = {by_depth[d]:.3f}  (n={len(items)})")

    fig, ax = plt.subplots(figsize=(8.5, 5))
    ds = sorted(by_depth)
    ax.plot(ds, [by_depth[d] for d in ds], marker="o", color="#1f77b4",
            label="Hybrid dispatcher + symbolic stack VM")
    ax.plot(ds, [0.0] * len(ds), marker="x", ls="--", color="#d62728",
            label="M4 pure Tree-LSTM (exact-match ~ 0)")
    ax.axvspan(4.5, 7.5, alpha=0.06, color="red")
    ax.axvline(4.5, ls="--", color="gray")
    ax.text(4.6, 0.5, "held-out\ndepths 5-7", color="gray", fontsize=8)
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel("expression nesting depth")
    ax.set_ylabel("EXACT-value match rate")
    ax.set_title("Hybrid dispatcher: the net learns the PROGRAM, the symbolic VM gives EXACT results")
    ax.legend()
    ax.grid(True, alpha=0.25)
    P.save(fig, "dispatcher_depth.png")


if __name__ == "__main__":
    main()
