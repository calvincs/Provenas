"""A small Transformer encoder-decoder (seq2seq) for infix -> RPN compilation."""
from __future__ import annotations

import torch
import torch.nn as nn


class Seq2Seq(nn.Module):
    def __init__(self, vocab, d_model=128, nhead=4, layers=3, ff=256, dropout=0.1, max_len=512):
        super().__init__()
        self.src_emb = nn.Embedding(vocab, d_model, padding_idx=0)
        self.tgt_emb = nn.Embedding(vocab, d_model, padding_idx=0)
        self.pos = nn.Embedding(max_len, d_model)
        self.tr = nn.Transformer(d_model, nhead, layers, layers, ff, dropout,
                                 batch_first=True, norm_first=True, activation="gelu")
        self.gen = nn.Linear(d_model, vocab)

    def _embed(self, x, emb):
        pos = torch.arange(x.size(1), device=x.device).unsqueeze(0)
        return emb(x) + self.pos(pos)

    def forward(self, src, tgt, src_pad, tgt_pad):
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size(1), device=tgt.device)
        out = self.tr(self._embed(src, self.src_emb), self._embed(tgt, self.tgt_emb),
                      tgt_mask=tgt_mask, src_key_padding_mask=src_pad,
                      tgt_key_padding_mask=tgt_pad, memory_key_padding_mask=src_pad)
        return self.gen(out)

    @torch.no_grad()
    def greedy(self, src, src_pad, bos, eos, max_steps=400):
        self.eval()
        device = src.device
        mem = self.tr.encoder(self._embed(src, self.src_emb), src_key_padding_mask=src_pad)
        ys = torch.full((src.size(0), 1), bos, dtype=torch.long, device=device)
        done = torch.zeros(src.size(0), dtype=torch.bool, device=device)
        for _ in range(max_steps):
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(ys.size(1), device=device)
            out = self.tr.decoder(self._embed(ys, self.tgt_emb), mem, tgt_mask=tgt_mask,
                                  memory_key_padding_mask=src_pad)
            nxt = self.gen(out[:, -1]).argmax(-1)
            nxt = torch.where(done, torch.full_like(nxt, eos), nxt)
            ys = torch.cat([ys, nxt.unsqueeze(1)], dim=1)
            done = done | (nxt == eos)
            if bool(done.all()):
                break
        return ys
