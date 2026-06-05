"""A small bidirectional char-GRU classifier for string -> category tasks."""
from __future__ import annotations

import torch.nn as nn


class CharGRU(nn.Module):
    def __init__(self, vocab, n_classes, emb=32, hid=64):
        super().__init__()
        self.emb = nn.Embedding(vocab, emb, padding_idx=0)
        self.gru = nn.GRU(emb, hid, batch_first=True, bidirectional=True)
        self.head = nn.Linear(2 * hid, n_classes)

    def forward(self, ids):
        out, _ = self.gru(self.emb(ids))
        mask = (ids != 0).unsqueeze(-1).float()         # mean-pool over non-pad
        pooled = (out * mask).sum(1) / mask.sum(1).clamp(min=1)
        return self.head(pooled)
