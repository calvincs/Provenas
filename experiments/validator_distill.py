"""InputValidator distillation — the classification sweet spot.

Distills detect_type (string -> {int,float,email,url,bool,other}) with a small
char-GRU. Shows distillation SUCCEEDS where the calculator failed (high accuracy),
then breaks it with a length-distribution shift (train <=30 chars, test 200) —
contrasting the calculator's magnitude cliff with a different failure axis.

  artifacts/validator.png
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from provenas import plotting as P
from provenas.chartok import batch_encode, VOCAB
from provenas.metrics import confusion
from provenas.models_char import CharGRU
from provenas.validator import TYPE_CLASSES, build

SEED = 0


@torch.no_grad()
def accuracy(model, strings, labels, device):
    model.eval()
    preds = []
    for i in range(0, len(strings), 1024):
        ids = batch_encode(strings[i:i + 1024], device, maxlen=256)
        preds.append(model(ids).argmax(1).cpu().numpy())
    pred = np.concatenate(preds)
    return float((pred == labels).mean()), pred


def main():
    rng = np.random.default_rng(SEED)
    tr_s, tr_y = build(40_000, rng, maxlen=30)
    te_s, te_y = build(8_000, np.random.default_rng(1), maxlen=30)
    long_s, long_y = build(8_000, np.random.default_rng(2), maxlen=200)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CharGRU(VOCAB, len(TYPE_CLASSES)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    y = torch.tensor(tr_y, device=device)
    n, B = len(tr_s), 256

    for ep in range(12):
        model.train()
        perm = rng.permutation(n)
        for i in range(0, n, B):
            idx = perm[i:i + B]
            ids = batch_encode([tr_s[j] for j in idx], device, maxlen=64)
            loss = F.cross_entropy(model(ids), y[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()

    acc_in, pred_in = accuracy(model, te_s, te_y, device)
    acc_long, _ = accuracy(model, long_s, long_y, device)
    print(f"  detect_type: in-dist acc={acc_in:.4f}   length-shift(200char) acc={acc_long:.4f}")

    M = confusion(te_y, pred_in, len(TYPE_CLASSES))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    P.confusion_matrix_plot(ax1, M, TYPE_CLASSES)
    ax1.set_title(f"detect_type confusion (in-dist acc={acc_in:.3f})")
    ax2.bar(["in-dist\n(<=30 char)", "length-shift\n(200 char)"], [acc_in, acc_long],
            color=["#2ca02c", "#d62728"])
    ax2.set_ylim(0, 1.02)
    ax2.set_ylabel("accuracy")
    ax2.set_title("Robust to length shift (train <=30 chars, test 200)")
    for i, v in enumerate([acc_in, acc_long]):
        ax2.text(i, v, f"{v:.3f}", ha="center", va="bottom")
    fig.suptitle("InputValidator distillation — the classification sweet spot (contrast the calculator)")
    P.save(fig, "validator.png")


if __name__ == "__main__":
    main()
