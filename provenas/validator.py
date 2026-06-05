"""InputValidator target: string -> category. The classification 'sweet spot'
(branch-rich Python logic, categorical output) — the optimistic counterweight to
the calculator's exact-computation difficulty.
"""
from __future__ import annotations

import re
import string as _string

import numpy as np

TYPE_CLASSES = ["int", "float", "email", "url", "bool", "other"]
TYPE_IDX = {c: i for i, c in enumerate(TYPE_CLASSES)}

_INT = re.compile(r"^[+-]?\d+$")
_FLOAT = re.compile(r"^[+-]?(\d+\.\d*|\.\d+)$")
_EMAIL = re.compile(r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$")
_URL = re.compile(r"^https?://[^\s]+$")


def detect_type(s):
    """Ground-truth oracle: classify a string into one of TYPE_CLASSES."""
    if _INT.match(s):
        return "int"
    if _FLOAT.match(s):
        return "float"
    if _EMAIL.match(s):
        return "email"
    if _URL.match(s):
        return "url"
    if s.lower() in ("true", "false"):
        return "bool"
    return "other"


def _word(rng, lo, hi, alphabet):
    k = int(rng.integers(lo, max(lo + 1, hi + 1)))
    return "".join(rng.choice(list(alphabet)) for _ in range(k))


def gen_string(cls, rng, maxlen=30):
    """Generate a string intended to be `cls`; length scales with maxlen for the
    email/url/other classes (so maxlen=200 exercises length extrapolation)."""
    if cls == "int":
        return ("-" if rng.random() < 0.3 else "") + str(int(rng.integers(0, 10 ** 6)))
    if cls == "float":
        return f"{rng.uniform(-1000, 1000):.{int(rng.integers(1, 4))}f}"
    if cls == "email":
        u = _word(rng, 3, max(4, maxlen // 3), _string.ascii_lowercase + _string.digits)
        d = _word(rng, 3, 7, _string.ascii_lowercase)
        return f"{u}@{d}.{rng.choice(['com', 'org', 'net', 'io'])}"
    if cls == "url":
        d = _word(rng, 3, 8, _string.ascii_lowercase)
        path = _word(rng, 0, max(1, maxlen - 15), _string.ascii_lowercase + "/-_")
        return f"http{'s' if rng.random() < 0.5 else ''}://{d}.com/{path}"
    if cls == "bool":
        return str(rng.choice(["true", "false", "True", "False"]))
    alpha = list(_string.ascii_letters + _string.digits + " .,!?-_")
    return "".join(rng.choice(alpha) for _ in range(int(rng.integers(1, maxlen + 1))))


def build(n, rng, maxlen=30):
    """Generate n (string, label) pairs; labels come from the detect_type oracle."""
    strings, labels = [], np.empty(n, dtype=np.int64)
    for i in range(n):
        cls = TYPE_CLASSES[int(rng.integers(len(TYPE_CLASSES)))]
        s = gen_string(cls, rng, maxlen)
        strings.append(s)
        labels[i] = TYPE_IDX[detect_type(s)]
    return strings, labels
