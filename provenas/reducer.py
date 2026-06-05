"""Iterative-reduction 'scratchpad' for expression evaluation.

State = a token sequence of TYPES {VAL, +, -, *, /, (, )} with a parallel value
list. At each step a controller points to the next operator to reduce; the symbolic
VM computes that one op EXACTLY, splices in a new VAL, and strips redundant parens —
repeat until one VAL remains. Because the per-step decision is LOCAL (deepest parens,
then highest precedence, then leftmost), a model that learns it should generalize to
ANY depth — the scratchpad attack on the length cliff. The model sees only TYPES
(structure), never values: reduction ORDER is purely structural.
"""
from __future__ import annotations

from .calculator import run_op
from .exprgen import PREC

OPSET = {"+", "-", "*", "/"}
SYMNAME = {"+": "add", "-": "subtract", "*": "multiply", "/": "divide"}
TYPES = ["<pad>", "VAL", "+", "-", "*", "/", "(", ")"]
TIDX = {t: i for i, t in enumerate(TYPES)}


def tree_to_state(tree):
    """Initial (types, vals) for an expression, minimal parens (matches exprgen)."""
    types, vals = [], []

    def emit(n):
        if n.kind == "leaf":
            types.append("VAL")
            vals.append(n.value)             # value-agnostic (numbers, lists, bools, ...)
            return
        p = PREC[n.op]
        lpar = (n.left.kind == "op" and PREC[n.left.op] < p)
        rpar = (n.right.kind == "op" and PREC[n.right.op] <= p)
        if lpar:
            types.append("("); vals.append(None)
        emit(n.left)
        if lpar:
            types.append(")"); vals.append(None)
        types.append(n.op); vals.append(None)
        if rpar:
            types.append("("); vals.append(None)
        emit(n.right)
        if rpar:
            types.append(")"); vals.append(None)

    emit(tree)
    return types, vals


def reducible_positions(types):
    out = []
    for i, t in enumerate(types):
        if t in OPSET and i > 0 and i + 1 < len(types) and types[i - 1] == "VAL" and types[i + 1] == "VAL":
            out.append(i)
    return out


def _op_depth(types, i):
    return sum(t == "(" for t in types[:i]) - sum(t == ")" for t in types[:i])


def next_reduction(types):
    """Oracle policy: deepest parens, then highest precedence, then leftmost."""
    cand = reducible_positions(types)
    if not cand:
        return None
    return max(cand, key=lambda i: (_op_depth(types, i), PREC[types[i]], -i))


def _neighbor_op(types, j):
    return types[j] if 0 <= j < len(types) and types[j] in OPSET else None


def valid_reductions(types):
    """Positions safe to reduce NOW under a purely LOCAL rule: an op may reduce iff
    its immediate operator-neighbors (within paren scope) don't outrank it. Multiple
    can be valid at once (independent subexpressions) — any is correct. This removes
    the global 'find the deepest paren' step, so the decision is fixed-radius."""
    out = []
    for p in reducible_positions(types):
        lp = _neighbor_op(types, p - 2)
        rp = _neighbor_op(types, p + 2)
        if (lp is None or PREC[lp] < PREC[types[p]]) and (rp is None or PREC[rp] <= PREC[types[p]]):
            out.append(p)
    return out


def local_next(types):
    v = valid_reductions(types)
    return v[0] if v else None


def apply_reduction(types, vals, p):
    """Reduce the op at position p (exact). Returns (types, vals, status)."""
    v, e = run_op(SYMNAME[types[p]], vals[p - 1], vals[p + 1])
    if e != "ok":
        return None, None, e
    types = types[:p - 1] + ["VAL"] + types[p + 2:]
    vals = vals[:p - 1] + [v] + vals[p + 2:]
    q = p - 1                                   # strip redundant parens around the new VAL
    while q - 1 >= 0 and q + 1 < len(types) and types[q - 1] == "(" and types[q + 1] == ")":
        types = types[:q - 1] + ["VAL"] + types[q + 2:]
        vals = vals[:q - 1] + [vals[q]] + vals[q + 2:]
        q -= 1
    return types, vals, "ok"


def reduce_to_value(types, vals, policy, max_steps=200):
    """Run reductions under `policy(types)->pos` until one VAL remains.
    Returns (value|None, status, steps) where steps = [(types, pos), ...]."""
    types, vals = list(types), list(vals)
    steps = []
    for _ in range(max_steps):
        if len(types) <= 1:
            break
        p = policy(types)
        if p is None or types[p] not in OPSET:
            return None, "Stuck", steps
        steps.append((list(types), p))
        types, vals, e = apply_reduction(types, vals, p)
        if e != "ok":
            return None, e, steps
    if len(types) == 1 and types[0] == "VAL":
        return vals[0], "ok", steps
    return None, "BadState", steps
