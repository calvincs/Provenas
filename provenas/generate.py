"""Stratified input generation (pipeline stage 2). Pure numpy + the oracle."""
from __future__ import annotations

import numpy as np

from . import strata as S
from .calculator import OPS, run_op, ERR_INDEX


def run_ops_vec(op_idx, a, b):
    """Apply the oracle row-wise. Returns (result float64 [NaN on error], error int8).

    Loops in Python; fine for ~1e5-1e6 rows (sub-second)."""
    n = len(a)
    result = np.empty(n, dtype=np.float64)
    error = np.empty(n, dtype=np.int8)
    for i in range(n):
        r, e = run_op(OPS[int(op_idx[i])], a[i], b[i])
        error[i] = ERR_INDEX[e]
        result[i] = np.nan if r is None else r
    return result, error


def sample_operands(n, allowed_strata, rng):
    """Sample n signed operands, each from a uniformly chosen allowed stratum.
    Returns (values float64, stratum_idx int8)."""
    allowed = np.asarray(allowed_strata)
    strat = rng.choice(allowed, size=n).astype(np.int8)
    vals = np.empty(n, dtype=np.float64)
    for s in np.unique(strat):
        m = strat == s
        vals[m] = S.sample_magnitude(int(s), rng, size=int(m.sum()))
    return vals, strat
