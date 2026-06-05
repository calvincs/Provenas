"""Input/output encodings. Pure numpy (float32 out); torch-free.

Two parallel paths, same input width (9) so the only difference between path A
and path B is the encoding semantics and the target space:

  Path A (raw):  per operand [clip(x/1e5), sign*log10(|x|+1)/19]  + op one-hot
  Path B (log):  per operand [sign(x), log(|x|+1)/LOG_SCALE_E]    + op one-hot

CRITICAL: A and B regress in different target spaces, so their raw losses are
NOT comparable. Always decode predictions to original units before computing any
cross-path metric (relative error). decode_target_A / decode_target_B do that.
"""
from __future__ import annotations

import numpy as np

from .calculator import OPS

N_OPS = len(OPS)
IN_DIM = 2 + 2 + N_OPS  # 9

NORM_SCALE = 1e5          # top of the training magnitude range (path A scale)
CLIP = 6.0
LOG10_SCALE = 19.0        # ~ log10 of the max |x| we ever encode
LOG_SCALE_E = float(np.log(1e19))  # ~43.75; keeps natural-log features ~O(1)


def _op_onehot(op_idx):
    op_idx = np.asarray(op_idx, dtype=int)
    oh = np.zeros((len(op_idx), N_OPS), dtype=np.float32)
    oh[np.arange(len(op_idx)), op_idx] = 1.0
    return oh


# ---------- Path A: raw normalized scalars (+ magnitude hint) ----------
def _raw_feats(x):
    x = np.asarray(x, dtype=np.float64)
    norm = np.clip(x / NORM_SCALE, -CLIP, CLIP)
    hint = np.sign(x) * np.log10(np.abs(x) + 1.0) / LOG10_SCALE
    return np.stack([norm, hint], axis=1).astype(np.float32)


def encode_inputs_raw(a, b, op_idx):
    return np.concatenate([_raw_feats(a), _raw_feats(b), _op_onehot(op_idx)], axis=1)


def encode_target_A(y):
    return (np.asarray(y, dtype=np.float64) / NORM_SCALE).astype(np.float32)[:, None]


def decode_target_A(t):
    return np.asarray(t, dtype=np.float64).reshape(-1) * NORM_SCALE


# ---------- Path B: log-magnitude + sign ----------
def _log_feats(x):
    x = np.asarray(x, dtype=np.float64)
    sgn = np.sign(x)
    logm = np.log(np.abs(x) + 1.0) / LOG_SCALE_E
    return np.stack([sgn, logm], axis=1).astype(np.float32)


def encode_inputs_log(a, b, op_idx):
    return np.concatenate([_log_feats(a), _log_feats(b), _op_onehot(op_idx)], axis=1)


def encode_target_B(y):
    y = np.asarray(y, dtype=np.float64)
    t = np.sign(y) * np.log(np.abs(y) + 1.0) / LOG_SCALE_E
    return t.astype(np.float32)[:, None]


def decode_target_B(t):
    u = np.asarray(t, dtype=np.float64).reshape(-1) * LOG_SCALE_E
    mag = np.exp(np.clip(np.abs(u), 0.0, 700.0)) - 1.0  # clip avoids exp overflow on masked rows
    return np.sign(u) * mag


# Registry so the train loop / experiments stay encoder-agnostic.
PATHS = {
    "A": dict(encode_inputs=encode_inputs_raw,
              encode_target=encode_target_A,
              decode_target=decode_target_A),
    "B": dict(encode_inputs=encode_inputs_log,
              encode_target=encode_target_B,
              decode_target=decode_target_B),
}
