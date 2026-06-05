"""Evaluation metrics: relative error, per-stratum slicing, confusion. Numpy."""
from __future__ import annotations

import numpy as np


def relative_error(pred, true, eps=1e-12):
    pred = np.asarray(pred, dtype=np.float64).reshape(-1)
    true = np.asarray(true, dtype=np.float64).reshape(-1)
    return np.abs(pred - true) / (np.abs(true) + eps)


def per_stratum_summary(values, joint_stratum, n_strata):
    """stratum -> (median, p10, p90, n), over finite values only."""
    out = {}
    js = np.asarray(joint_stratum)
    v = np.asarray(values, dtype=np.float64)
    for s in range(n_strata):
        m = js == s
        if not m.any():
            continue
        vs = v[m]
        vs = vs[np.isfinite(vs)]
        if len(vs) == 0:
            continue
        out[s] = (float(np.median(vs)),
                  float(np.percentile(vs, 10)),
                  float(np.percentile(vs, 90)),
                  int(m.sum()))
    return out


def confusion(true_idx, pred_idx, n_classes):
    M = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(M, (np.asarray(true_idx, dtype=int), np.asarray(pred_idx, dtype=int)), 1)
    return M


def per_class_recall(M):
    M = np.asarray(M, dtype=np.float64)
    diag = np.diag(M)
    row = M.sum(axis=1)
    return np.divide(diag, row, out=np.zeros_like(diag), where=row > 0)
