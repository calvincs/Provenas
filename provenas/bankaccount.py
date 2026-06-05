"""Stateful target: a running balance over a sequence of deposit/withdraw ops.

Output depends on internal state (the balance), not just the current argument — so
it breaks the purity assumption that lets a stateless net work. Used to test
whether a network can *hold state across calls* (M1 recurrent) and whether it
drifts vs threading state through inputs (M0).
"""
from __future__ import annotations

import numpy as np


def make_sequences(n, length, rng, amt_max=100.0):
    """Returns ops (n,length) in {+1 deposit, -1 withdraw}, amts (n,length), and
    the true running balances (n,length) starting from 0."""
    ops = rng.choice(np.array([1.0, -1.0]), size=(n, length))
    amts = rng.uniform(0.0, amt_max, size=(n, length))
    balances = np.cumsum(ops * amts, axis=1)
    return ops, amts, balances
