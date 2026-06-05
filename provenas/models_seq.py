"""Sequence models for the stateful experiment."""
from __future__ import annotations

import torch.nn as nn


class BalanceGRU(nn.Module):
    """M1: carries the balance in an opaque recurrent hidden vector. Input per
    step is [is_deposit, is_withdraw, amount/scale]; output is balance/scale."""

    def __init__(self, hid=64):
        super().__init__()
        self.gru = nn.GRU(3, hid, batch_first=True)
        self.head = nn.Linear(hid, 1)

    def forward(self, x):
        out, _ = self.gru(x)
        return self.head(out)


class StepMLP(nn.Module):
    """M0: a pure step function (balance_in, op, amount) -> balance_out. The
    wrapper threads the balance between calls; the net never holds state."""

    def __init__(self, hid=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, hid), nn.ReLU(),
            nn.Linear(hid, hid), nn.ReLU(),
            nn.Linear(hid, 1))

    def forward(self, x):
        return self.net(x)
