"""Symbolic stack VM + tokenizers for the neural-dispatcher experiment.

The network never touches values. Each operand becomes a positional SYMBOL
(N0, N1, ... numbered left-to-right), and the exact values live in an external
list (the "memory"). The network learns to emit a stack PROGRAM (RPN over the
symbols); Python's stack executes it EXACTLY. This is the plan's path F (hybrid
dispatcher) + M2 (external store): offload exactness to symbolic tools, let the
net do control.
"""
from __future__ import annotations

from .calculator import run_op
from .exprgen import PREC, LEAF_PREC

SYM = {"+": "add", "-": "subtract", "*": "multiply", "/": "divide"}


def _prec(n):
    return LEAF_PREC if n.kind == "leaf" else PREC[n.op]


def slot_tokens(tree):
    """Return (infix_tokens, rpn_tokens, values).

    Leaves are numbered left-to-right as N0, N1, ...; `values[k]` is the exact
    value of slot Nk. The same left-to-right numbering is used for both the infix
    and the RPN token streams, so the net only has to learn OPERATOR placement —
    values are pure symbols it copies in order.
    """
    leaf_id, values = {}, []

    def assign(n):
        if n.kind == "leaf":
            leaf_id[id(n)] = len(values)
            values.append(n.value)
        else:
            assign(n.left)
            assign(n.right)

    assign(tree)

    def infix(n):
        if n.kind == "leaf":
            return [f"N{leaf_id[id(n)]}"]
        p = PREC[n.op]
        left = infix(n.left)
        if _prec(n.left) < p:
            left = ["("] + left + [")"]
        right = infix(n.right)
        if _prec(n.right) <= p:
            right = ["("] + right + [")"]
        return left + [n.op] + right

    def rpn(n):
        if n.kind == "leaf":
            return [f"N{leaf_id[id(n)]}"]
        return rpn(n.left) + rpn(n.right) + [n.op]

    return infix(tree), rpn(tree), values


def execute_rpn(rpn_tokens, values):
    """Execute a predicted RPN program on a real stack with exact values. Returns
    (value|None, status), status in {ok, ZeroDivisionError, OverflowError,
    BadProgram}. A structurally-wrong program -> BadProgram (the net's fault)."""
    stack = []
    for tok in rpn_tokens:
        if tok and tok[0] == "N":
            try:
                k = int(tok[1:])
            except ValueError:
                return None, "BadProgram"
            if k >= len(values):
                return None, "BadProgram"
            stack.append(values[k])
        elif tok in SYM:
            if len(stack) < 2:
                return None, "BadProgram"
            b = stack.pop()
            a = stack.pop()
            v, e = run_op(SYM[tok], a, b)
            if e != "ok":
                return None, e
            stack.append(v)
        else:
            return None, "BadProgram"
    return (stack[0], "ok") if len(stack) == 1 else (None, "BadProgram")
