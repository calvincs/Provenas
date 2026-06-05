"""Compound-expression generator for M4 (parsing vs computing).

Builds random arithmetic expression trees over {+ - * /} and emits, for the SAME
expression: the parse TREE (for path D), the rendered STRING with MINIMAL,
correctness-preserving parentheses (for path C), the ground-truth value, and the
error label (ZeroDivisionError / Overflow / ok), computed through the same
`Calculator` semantics as M1-M3.

Minimal parens are load-bearing: fully-parenthesized strings make precedence
trivial (just bracket-matching) and would collapse the C-vs-D gap. We parenthesize
a child only when precedence requires it.

A self-check asserts Calculator.evaluate(string) matches the tree walk for every
sample, guaranteeing the string and tree denote the same expression.
"""
from __future__ import annotations

import math

import numpy as np

from .calculator import run_op, evaluate as _eval_string, ERR_INDEX

SYM = {"+": "add", "-": "subtract", "*": "multiply", "/": "divide"}
SYMS = list(SYM.keys())
PREC = {"+": 1, "-": 1, "*": 2, "/": 2}
LEAF_PREC = 99


class Node:
    __slots__ = ("kind", "value", "op", "left", "right", "depth")

    def __init__(self, kind, value=None, op=None, left=None, right=None):
        self.kind = kind
        self.value = value
        self.op = op
        self.left = left
        self.right = right
        self.depth = 0 if kind == "leaf" else 1 + max(left.depth, right.depth)


def _prec(node):
    return LEAF_PREC if node.kind == "leaf" else PREC[node.op]


def sample_operand(rng):
    r = rng.random()
    if r < 0.10:
        return 0.0                       # oversample zero (divisor -> ZeroDivisionError)
    if r < 0.70:
        return float(int(rng.integers(-20, 21)))
    return round(float(rng.uniform(-20, 20)), 1)


def gen(depth, rng):
    """Build an expression tree of exactly `depth` (one child depth-1, the other
    uniform in [0, depth-1]; random side) for controlled, varied-shape nesting."""
    if depth == 0:
        return Node("leaf", value=sample_operand(rng))
    op = SYMS[rng.integers(len(SYMS))]
    deep = gen(depth - 1, rng)
    shallow = gen(int(rng.integers(0, depth)), rng)
    if rng.random() < 0.5:
        left, right = deep, shallow
    else:
        left, right = shallow, deep
    return Node("op", op=op, left=left, right=right)


def _fmt(v):
    return str(int(v)) if v == int(v) else f"{v:.1f}"


def render(node):
    """Render with minimal, correctness-preserving parentheses."""
    if node.kind == "leaf":
        return _fmt(node.value)
    p = PREC[node.op]
    left = render(node.left)
    if _prec(node.left) < p:
        left = f"({left})"
    right = render(node.right)
    if _prec(node.right) <= p:          # left-associative: right child of equal prec needs parens
        right = f"({right})"
    return f"{left} {node.op} {right}"


def eval_tree(node):
    """(value|None, error_label) via Calculator semantics; first error wins (DFS, left-first)."""
    if node.kind == "leaf":
        return node.value, "ok"
    lv, le = eval_tree(node.left)
    if le != "ok":
        return None, le
    rv, re = eval_tree(node.right)
    if re != "ok":
        return None, re
    return run_op(SYM[node.op], lv, rv)


class Sample:
    __slots__ = ("tree", "string", "value", "error", "depth", "n_ops")

    def __init__(self, tree, string, value, error, depth, n_ops):
        self.tree = tree
        self.string = string
        self.value = value          # float; NaN on error
        self.error = error          # int error-class index
        self.depth = depth
        self.n_ops = n_ops


def _count_ops(node):
    if node.kind == "leaf":
        return 0
    return 1 + _count_ops(node.left) + _count_ops(node.right)


def make_sample(depth, rng):
    tree = gen(depth, rng)
    string = render(tree)
    value, error = eval_tree(tree)
    # self-check: Python's own parse+eval of the string must agree with the tree
    v2, e2 = _eval_string(string)
    if e2 != error or (error == "ok" and not _same(value, v2)):
        raise AssertionError(f"render/eval mismatch: {string!r} tree=({value},{error}) str=({v2},{e2})")
    return Sample(tree, string,
                  float("nan") if value is None else float(value),
                  ERR_INDEX[error], tree.depth, _count_ops(tree))


def _same(a, b):
    if a is None or b is None:
        return a is b
    return a == b or math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12)


def build(depth_counts, rng, dedup=True, seen=None):
    """depth_counts: dict {depth: n}. Returns a list[Sample], string-deduped
    against `seen` (a set, updated in place) when dedup is True."""
    seen = set() if seen is None else seen
    out = []
    for depth, n in depth_counts.items():
        got = 0
        guard = 0
        while got < n:
            guard += 1
            if guard > 50 * n + 1000:
                break
            s = make_sample(depth, rng)
            if dedup:
                if s.string in seen:
                    continue
                seen.add(s.string)
            out.append(s)
            got += 1
    return out
