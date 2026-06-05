"""General char-level tokenizer (printable ASCII + <unk>)."""
from __future__ import annotations

import numpy as np

PAD, UNK = 0, 1
_CHARS = [chr(c) for c in range(32, 127)]
STOI = {c: i + 2 for i, c in enumerate(_CHARS)}
VOCAB = len(_CHARS) + 2


def encode(s, maxlen):
    return [STOI.get(c, UNK) for c in s[:maxlen]] or [UNK]


def batch_encode(strings, device, maxlen=256):
    import torch
    seqs = [encode(s, maxlen) for s in strings]
    T = max(len(s) for s in seqs)
    ids = np.zeros((len(seqs), T), dtype=np.int64)
    for i, s in enumerate(seqs):
        ids[i, :len(s)] = s
    return torch.from_numpy(ids).to(device)
