"""The target `Calculator` and its ground-truth oracle.

`run_op` is the canonical oracle every downstream stage reads. It operates on
Python floats and canonicalizes inf/nan results into `OverflowError` so the
error label is op-independent: Python's native behavior is inconsistent
(`1/0` raises, but `1e200*1e200` silently returns `inf`), and we need one clean
3-class label set {ok, ZeroDivisionError, OverflowError} for the error head.
"""
from __future__ import annotations

import ast as _ast
import math

OPS = ["add", "subtract", "multiply", "divide", "power"]
OP_INDEX = {name: i for i, name in enumerate(OPS)}

ERR_CLASSES = ["ok", "ZeroDivisionError", "OverflowError"]
ERR_INDEX = {name: i for i, name in enumerate(ERR_CLASSES)}
OK, ZERO_DIV, OVERFLOW = 0, 1, 2

# Symbols used by the compound-expression generator (M4) and `evaluate`.
OP_SYMBOL = {"add": "+", "subtract": "-", "multiply": "*", "divide": "/", "power": "**"}


def _canonical(value):
    """Map a raw Python arithmetic result to (result|None, error_label)."""
    if isinstance(value, complex):  # e.g. (-2.0) ** 0.5 -> complex
        return None, "OverflowError"
    v = float(value)
    if math.isinf(v) or math.isnan(v):
        return None, "OverflowError"
    return v, "ok"


def run_op(op, a, b):
    """Run a single binary op on floats -> (result|None, error_label).

    error_label is one of {"ok", "ZeroDivisionError", "OverflowError"}.
    """
    a = float(a)
    b = float(b)
    try:
        if op == "add":
            raw = a + b
        elif op == "subtract":
            raw = a - b
        elif op == "multiply":
            raw = a * b
        elif op == "divide":
            if b == 0.0:
                return None, "ZeroDivisionError"
            raw = a / b
        elif op == "power":
            raw = a ** b  # 0.0 ** -1 raises ZeroDivisionError (caught below)
        else:
            raise ValueError(f"unknown op: {op!r}")
    except ZeroDivisionError:
        return None, "ZeroDivisionError"
    except OverflowError:
        return None, "OverflowError"
    return _canonical(raw)


class Calculator:
    """Ground-truth target object. Methods raise real exceptions; the
    behavior matches `run_op` exactly (the same canonicalization)."""

    def add(self, a, b):
        return self._dispatch("add", a, b)

    def subtract(self, a, b):
        return self._dispatch("subtract", a, b)

    def multiply(self, a, b):
        return self._dispatch("multiply", a, b)

    def divide(self, a, b):
        return self._dispatch("divide", a, b)

    def power(self, a, b):
        return self._dispatch("power", a, b)

    @staticmethod
    def _dispatch(op, a, b):
        result, error = run_op(op, a, b)
        if error == "ZeroDivisionError":
            raise ZeroDivisionError(f"{op}({a}, {b})")
        if error == "OverflowError":
            raise OverflowError(f"{op}({a}, {b})")
        return result

    def evaluate(self, expr_string):
        """Parse and evaluate an arithmetic expression string with the same
        canonicalized semantics as `run_op`. Returns (result|None, error_label).

        Error propagation is depth-first, left-before-right (first error wins) —
        identical to the M4 AST walk, so this serves as the generator self-check.
        Raises ValueError on unsupported syntax.
        """
        tree = _ast.parse(expr_string, mode="eval")
        return _eval_ast(tree.body)


_AST_OP = {
    _ast.Add: "add",
    _ast.Sub: "subtract",
    _ast.Mult: "multiply",
    _ast.Div: "divide",
    _ast.Pow: "power",
}


def _eval_ast(node):
    if isinstance(node, _ast.BinOp):
        op_name = _AST_OP.get(type(node.op))
        if op_name is None:
            raise ValueError(f"unsupported operator: {type(node.op).__name__}")
        left, lerr = _eval_ast(node.left)
        if lerr != "ok":
            return None, lerr
        right, rerr = _eval_ast(node.right)
        if rerr != "ok":
            return None, rerr
        return run_op(op_name, left, right)
    if isinstance(node, _ast.UnaryOp):
        if isinstance(node.op, _ast.USub):
            val, err = _eval_ast(node.operand)
            return (None, err) if err != "ok" else (-val, "ok")
        if isinstance(node.op, _ast.UAdd):
            return _eval_ast(node.operand)
        raise ValueError(f"unsupported unary op: {type(node.op).__name__}")
    if isinstance(node, _ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value), "ok"
    raise ValueError(f"unsupported node: {type(node).__name__}")


def evaluate(expr_string):
    """Module-level parse+evaluate with run_op semantics -> (value|None, error_label)."""
    tree = _ast.parse(expr_string, mode="eval")
    return _eval_ast(tree.body)
