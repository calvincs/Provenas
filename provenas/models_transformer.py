"""Path C: a tiny encoder-only Transformer over the expression STRING.

Must learn operator precedence and grammar from raw tokens. Pools a prepended
<cls> token into the SAME RegressionHead / ErrorHead used everywhere else, so the
only thing distinguishing it from path D is that it must parse.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .tokenize_expr import VOCAB_SIZE, PAD
from .models import RegressionHead, ErrorHead


class TinyTransformer(nn.Module):
    def __init__(self, d_model=64, nhead=4, nlayers=3, ff=256, dropout=0.1, max_len=1024):
        super().__init__()
        self.max_len = max_len
        self.tok = nn.Embedding(VOCAB_SIZE, d_model, padding_idx=PAD)
        self.pos = nn.Embedding(max_len, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward=ff, dropout=dropout,
            batch_first=True, norm_first=True, activation="gelu")
        self.enc = nn.TransformerEncoder(layer, nlayers)
        self.reg_head = RegressionHead(d_model)
        self.err_head = ErrorHead(d_model)

    def forward(self, ids, key_padding_mask):
        T = ids.shape[1]
        pos = torch.arange(T, device=ids.device).unsqueeze(0)
        x = self.tok(ids) + self.pos(pos)
        h = self.enc(x, src_key_padding_mask=key_padding_mask)
        cls = h[:, 0]                       # the <cls> position
        return self.reg_head(cls), self.err_head(cls)
