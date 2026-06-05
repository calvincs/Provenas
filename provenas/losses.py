"""The masked multi-head loss, shared by every architecture (MLP, Transformer,
Tree-LSTM).

Regression loss applies ONLY on ok rows and is averaged over the ok count (not
the batch size) — otherwise error-heavy batches silently shrink the regression
gradient. Error-row targets (NaN/garbage) are zeroed via torch.where *before*
squaring so they can never poison the graph.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .calculator import OK


def masked_multihead_loss(reg_pred, reg_target, err_logits, err_target,
                          w_reg=1.0, w_err=1.0, class_weight=None):
    """reg_pred/reg_target: (B,1); err_logits: (B,C); err_target: (B,) long.

    Returns (total_loss, reg_loss_detached, err_loss_detached).
    """
    mask = err_target == OK                              # (B,)
    n_ok = mask.sum().clamp(min=1).to(reg_pred.dtype)
    diff = reg_pred - reg_target                         # (B,1)
    diff = torch.where(mask.unsqueeze(1), diff, torch.zeros_like(diff))
    diff = torch.nan_to_num(diff, nan=0.0)               # belt-and-suspenders
    reg_loss = (diff * diff).sum() / n_ok

    err_loss = F.cross_entropy(err_logits, err_target, weight=class_weight)
    total = w_reg * reg_loss + w_err * err_loss
    return total, reg_loss.detach(), err_loss.detach()
