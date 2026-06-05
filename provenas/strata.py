"""Magnitude strata for stratified sampling and error slicing.

Training range = strata 0-3 (|x| < 1e5). Strata 4-7 are eval-only. The bounds
are chosen so the two failure regimes land at *different* magnitudes and show up
as separate knees on a log-x error plot:
  - extrapolation cliff at ~1e5 (edge of the trained magnitude range)
  - float32 precision cliff at 2**24 ~= 1.67e7 (integers stop being exact)
"""
from __future__ import annotations

import numpy as np

# (name, lo, hi) on |x|, half-open [lo, hi).
STRATA = [
    ("sub_unit",        1e-3,  1e0),
    ("unit",            1e0,   1e1),
    ("small",           1e1,   1e3),
    ("mid",             1e3,   1e5),
    ("large",           1e5,   1.6e7),
    ("precision_cliff", 1.6e7, 1e10),
    ("extrapolation",   1e10,  1e15),
    ("extreme",         1e15,  1e19),
]
STRATA_NAMES = [s[0] for s in STRATA]
N_STRATA = len(STRATA)
TRAIN_STRATA = (0, 1, 2, 3)        # |x| < 1e5
EVAL_ONLY_STRATA = (4, 5, 6, 7)
TRAIN_MAX_ABS = 1e5
F32_PRECISION_LIMIT = 2 ** 24      # 16_777_216

_LOS = np.array([s[1] for s in STRATA], dtype=np.float64)


def sample_magnitude(stratum_idx, rng, size=None):
    """Log-uniform |x| within a stratum, with a random sign. Returns signed
    float(s). `rng` is a numpy Generator."""
    _, lo, hi = STRATA[stratum_idx]
    u = rng.uniform(np.log10(lo), np.log10(hi), size=size)
    mag = np.power(10.0, u)
    sign = rng.choice(np.array([-1.0, 1.0]), size=size)
    return sign * mag


def classify(x):
    """Return the stratum index for a scalar/array, keyed on |x| (clamped to
    [0, N_STRATA-1])."""
    ax = np.abs(np.asarray(x, dtype=np.float64))
    idx = np.searchsorted(_LOS, ax, side="right") - 1
    return np.clip(idx, 0, N_STRATA - 1)
