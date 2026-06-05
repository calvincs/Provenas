"""Direction 2: variables + get/set/delete memory (a register machine).

The net compiles a straight-line program (`V0 = 3 + 4; V1 = V0 * 2; ...`) into
LOAD/STORE/op instructions over a stack + keyed memory; Python executes them
EXACTLY. Trained on 2-6 statements, tested to 14 -> does symbol binding + the
growing memory generalize past the trained program length?

  artifacts/variables_depth.png
"""
from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from provenas import plotting as P
from provenas.models_seq2seq import Seq2Seq
from provenas import regvm as R

SEED = 0
MAXVAR, MAXCONST = 16, 48
EPOCHS = int(os.environ.get("VAR_EPOCHS", "30"))
BATCH = 128

OPS = ["+", "-", "*", "/"]
KW = ["=", ";", "PUSH", "LOAD", "STORE"]
SPECIAL = ["<pad>", "<bos>", "<eos>"]
VARS = [f"V{i}" for i in range(MAXVAR)]
CONSTS = [f"N{i}" for i in range(MAXCONST)]
VOCAB = SPECIAL + OPS + KW + VARS + CONSTS
STOI = {t: i for i, t in enumerate(VOCAB)}
PAD, BOS, EOS = 0, 1, 2


def enc(tokens, bos_eos=False):
    ids = [STOI[t] for t in tokens]
    return [BOS] + ids + [EOS] if bos_eos else ids


def make_pairs(ks, per_k, rng):
    pairs = []
    for k in ks:
        got = 0
        while got < per_k:
            stmts, consts = R.gen_program(k, rng)
            if len(consts) > MAXCONST:
                continue
            val, st = R.execute_true(stmts, consts)
            if st != "ok":                       # ok-cases for exact-value match
                continue
            pairs.append((R.source_tokens(stmts), R.instr_tokens(stmts), consts, val, k))
            got += 1
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
    kmax = int(os.environ.get("VAR_TRAIN_KMAX", "6"))
    train = make_pairs(range(2, kmax + 1), 6000, rng)
    test = make_pairs(range(2, 15), 800, np.random.default_rng(1))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  train={len(train)} test={len(test)} vocab={len(VOCAB)}")

    model = Seq2Seq(len(VOCAB)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)
    src = [enc(s) for s, _, _, _, _ in train]
    tgt = [enc(t, bos_eos=True) for _, t, _, _, _ in train]
    n = len(train)
    for ep in range(EPOCHS):
        model.train()
        perm = rng.permutation(n)
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

    by_k = {}
    for k in range(2, 15):
        items = [p for p in test if p[4] == k]
        if not items:
            continue
        exact = 0
        for i in range(0, len(items), 256):
            chunk = items[i:i + 256]
            S, Spad = pad_batch([enc(s) for s, _, _, _, _ in chunk], device)
            ys = model.greedy(S, Spad, BOS, EOS, max_steps=200).cpu().numpy()
            for row, (s, t, consts, val, kk) in zip(ys, chunk):
                toks = []
                for tid in row[1:]:
                    if tid == EOS:
                        break
                    toks.append(VOCAB[tid])
                pv, stt = R.execute_instructions(toks, consts, kk - 1)
                if stt == "ok" and pv is not None and abs(pv - val) <= 1e-9 * max(1.0, abs(val)):
                    exact += 1
        by_k[k] = exact / len(items)
        print(f"  {k:2d} statements: exact-value match = {by_k[k]:.3f}  (n={len(items)})")

    fig, ax = plt.subplots(figsize=(8.5, 5))
    ks = sorted(by_k)
    ax.plot(ks, [by_k[k] for k in ks], marker="o", color="#9467bd",
            label="Register machine (net compiles; dict memory + stack execute exactly)")
    ax.axvspan(6.5, ks[-1] + 0.5, alpha=0.06, color="red")
    ax.axvline(6.5, ls="--", color="gray")
    ax.text(6.6, 0.5, "more statements\nthan trained", color="gray", fontsize=8, va="center")
    ax.set_ylim(-0.03, 1.05)
    ax.set_xlabel("program length (number of statements / variables)")
    ax.set_ylabel("EXACT-value match rate")
    ax.set_title("Direction 2: variables + get/set/delete memory\n"
                 "net learns symbol binding; the keyed store makes it exact")
    ax.legend(loc="lower left")
    ax.grid(True, alpha=0.25)
    P.save(fig, "variables_depth.png")


if __name__ == "__main__":
    main()
