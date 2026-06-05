"""Dataset assembly: labeling, error injection, f32-exactness flags, npz I/O.

Design notes:
- M1-M3 train on ops {add, subtract, multiply, divide}. `power` is deliberately
  held out (reserved as the "untrained operator" for a later interpretability
  probe), so it never appears in the data even though it occupies an op one-hot
  slot.
- ZeroDivisionError cases force divide with b == 0.
- OverflowError cases multiply two huge operands so the float64 product is inf.
  (Genuine Python overflow needs |result| > ~1.8e308; this is honest, not a
  float32-threshold shortcut. The float32 precision cliff lives in the metrics,
  not the error label.)
"""
from __future__ import annotations

import os

import numpy as np

from . import strata as S
from .calculator import OP_INDEX, OK
from .generate import sample_operands, run_ops_vec

REG_OPS = ["add", "subtract", "multiply", "divide"]
M1_OPS = ["add", "subtract", "multiply"]


def f32_exact(x):
    """True where x is exactly representable in float32."""
    x = np.asarray(x, dtype=np.float64)
    with np.errstate(over="ignore", invalid="ignore"):
        return x.astype(np.float32).astype(np.float64) == x


def _finalize(a, b, op, rng):
    result, error = run_ops_vec(op, a, b)
    stratum_a = S.classify(a).astype(np.int8)
    stratum_b = S.classify(b).astype(np.int8)
    joint = np.maximum(stratum_a, stratum_b).astype(np.int8)
    ok = error == OK
    f_out = np.zeros(len(a), dtype=bool)
    if ok.any():
        f_out[ok] = f32_exact(result[ok])
    return {
        "a": a.astype(np.float64),
        "b": b.astype(np.float64),
        "op": op.astype(np.int8),
        "result": result,
        "error": error.astype(np.int8),
        "stratum_a": stratum_a,
        "stratum_b": stratum_b,
        "joint_stratum": joint,
        "f32_exact_in": (f32_exact(a) & f32_exact(b)),
        "f32_exact_out": f_out,
    }


def _ok_cases(n, ops, allowed_strata, rng, oversample=1.3):
    op_choices = np.array([OP_INDEX[o] for o in ops])
    out_a, out_b, out_op = [], [], []
    have = 0
    while have < n:
        m = int((n - have) * oversample) + 32
        a, _ = sample_operands(m, allowed_strata, rng)
        b, _ = sample_operands(m, allowed_strata, rng)
        op = rng.choice(op_choices, size=m).astype(np.int8)
        _, err = run_ops_vec(op, a, b)
        keep = err == OK
        out_a.append(a[keep])
        out_b.append(b[keep])
        out_op.append(op[keep])
        have += int(keep.sum())
    return (np.concatenate(out_a)[:n],
            np.concatenate(out_b)[:n],
            np.concatenate(out_op)[:n])


def _zero_div_cases(n, allowed_strata, rng):
    a, _ = sample_operands(n, allowed_strata, rng)
    b = np.zeros(n, dtype=np.float64)
    op = np.full(n, OP_INDEX["divide"], dtype=np.int8)
    return a, b, op


def _overflow_cases(n, rng, lo=1e160, hi=1e200):
    def huge(k):
        u = rng.uniform(np.log10(lo), np.log10(hi), size=k)
        sign = rng.choice(np.array([-1.0, 1.0]), size=k)
        return sign * np.power(10.0, u)

    a, b = huge(n), huge(n)
    op = np.full(n, OP_INDEX["multiply"], dtype=np.int8)
    return a, b, op


def build(n, rng, ops=REG_OPS, allowed_strata=S.TRAIN_STRATA,
          frac_zero_div=0.0, frac_overflow=0.0):
    """Assemble ~n labeled samples with the requested error mix. Returns a dict
    of parallel numpy arrays (see _finalize for keys)."""
    n_zero = int(round(n * frac_zero_div))
    n_over = int(round(n * frac_overflow))
    n_ok = n - n_zero - n_over

    pa, pb, po = _ok_cases(n_ok, ops, allowed_strata, rng)
    parts = [(pa, pb, po)]
    if n_zero:
        parts.append(_zero_div_cases(n_zero, allowed_strata, rng))
    if n_over:
        parts.append(_overflow_cases(n_over, rng))

    a = np.concatenate([p[0] for p in parts])
    b = np.concatenate([p[1] for p in parts])
    op = np.concatenate([p[2] for p in parts])
    perm = rng.permutation(len(a))
    return _finalize(a[perm], b[perm], op[perm], rng)


def eval_grid(ops, strata_idx, n_per_cell, rng):
    """Deterministic-style sweep over (op x stratum) for cliff plots. Operands
    are sampled per stratum; most cells are ok-cases (numeric results)."""
    A, B, OP = [], [], []
    for s in strata_idx:
        for o in ops:
            A.append(S.sample_magnitude(s, rng, size=n_per_cell))
            B.append(S.sample_magnitude(s, rng, size=n_per_cell))
            OP.append(np.full(n_per_cell, OP_INDEX[o], dtype=np.int8))
    return _finalize(np.concatenate(A), np.concatenate(B), np.concatenate(OP), rng)


def save(path, data):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.savez_compressed(path, **data)
    return path


def load(path):
    z = np.load(path)
    return {k: z[k] for k in z.files}
