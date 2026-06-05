"""Digit/op/paren tokenizer for path C (the Transformer).

Char-level granularity (each digit, '.', sign, operator, paren, space is its own
token) — i.e. the project's reusable digit tokenizer, so the model could in
principle learn place value. A <cls> token is prepended for pooling.
"""
from __future__ import annotations

import numpy as np

SPECIAL = ["<pad>", "<cls>"]
CHARS = list("0123456789.+-*/() ")
VOCAB = SPECIAL + CHARS
STOI = {c: i for i, c in enumerate(VOCAB)}
VOCAB_SIZE = len(VOCAB)
PAD, CLS = 0, 1


def encode(string):
    return [CLS] + [STOI[c] for c in string]


def batch_encode(strings, device=None):
    """Return (ids LongTensor (B,T), key_padding_mask BoolTensor (B,T) True=pad)."""
    import torch
    seqs = [encode(s) for s in strings]
    T = max(len(s) for s in seqs)
    ids = np.full((len(seqs), T), PAD, dtype=np.int64)
    mask = np.ones((len(seqs), T), dtype=bool)
    for i, s in enumerate(seqs):
        ids[i, :len(s)] = s
        mask[i, :len(s)] = False
    ids = torch.from_numpy(ids)
    mask = torch.from_numpy(mask)
    if device is not None:
        ids, mask = ids.to(device), mask.to(device)
    return ids, mask
