"""Torch models.

RegressionHead and ErrorHead are factored out of CalcMLP so the M4 Transformer
and Tree-LSTM can reuse the exact same heads, output space, and loss — which is
what makes the cross-architecture comparison fair.
"""
from __future__ import annotations

import torch.nn as nn

from .calculator import ERR_CLASSES
from .encoders import IN_DIM

N_ERR = len(ERR_CLASSES)


class RegressionHead(nn.Module):
    """Pooled representation -> single scalar (in the path's target space)."""

    def __init__(self, in_dim):
        super().__init__()
        self.fc = nn.Linear(in_dim, 1)

    def forward(self, h):
        return self.fc(h)


class ErrorHead(nn.Module):
    """Pooled representation -> error-class logits {ok, ZeroDiv, Overflow}."""

    def __init__(self, in_dim, n_classes=N_ERR):
        super().__init__()
        self.fc = nn.Linear(in_dim, n_classes)

    def forward(self, h):
        return self.fc(h)


class CalcMLP(nn.Module):
    """Fat MLP trunk + multi-head (paths A and B; one class, two encodings)."""

    def __init__(self, in_dim=IN_DIM, width=128, trunk_out=64):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, width), nn.ReLU(),
            nn.Linear(width, width), nn.ReLU(),
            nn.Linear(width, trunk_out), nn.ReLU(),
        )
        self.reg_head = RegressionHead(trunk_out)
        self.err_head = ErrorHead(trunk_out)

    def forward(self, x):
        h = self.trunk(x)
        return self.reg_head(h), self.err_head(h)
