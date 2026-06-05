"""A typed expression engine — comparisons, boolean logic, and if/else over mixed types.

The last computation-side Lego. Until now the engine was homogeneous (a tree of one type). Here
values carry a TYPE (num | bool) and operators carry a SIGNATURE, so one expression can cross types:
arithmetic (num,num→num), comparison (num,num→**bool**), boolean (bool,bool→bool), and a conditional
`if c then a else b` that BRANCHES (bool, T, T → T).

Two things are genuinely new versus the arithmetic reducer:
  - a TYPE TRANSITION: comparisons turn numbers into booleans inside one expression.
  - BRANCHING with SHORT-CIRCUIT: the `if` reduces its condition FIRST, collapses to the chosen
    branch, and never evaluates the dead branch (so `if b==0 then 0 else a/b` is safe).

Reduction stays LOCAL and exact (the project's thesis): `find_redex` picks the next site by a local,
lazy rule; `apply_redex` computes it with the exact tool table. `linearize` exposes the state as a
stream of token TYPES (never values) — the structure-only signal the controller learns from.

AST nodes (tuples):  ("lit", value, type) | ("bin", sym, left, right) | ("if", cond, then, else)
"""
from __future__ import annotations

import operator

SIG = {
    "+": ("num", "num", "num"), "-": ("num", "num", "num"),
    "*": ("num", "num", "num"), "/": ("num", "num", "num"),
    "<": ("num", "num", "bool"), ">": ("num", "num", "bool"),
    "<=": ("num", "num", "bool"), ">=": ("num", "num", "bool"),
    "==": ("num", "num", "bool"), "!=": ("num", "num", "bool"),
    "&": ("bool", "bool", "bool"), "|": ("bool", "bool", "bool"),
}
PREC = {"*": 5, "/": 5, "+": 4, "-": 4,
        "<": 3, ">": 3, "<=": 3, ">=": 3, "==": 2, "!=": 2, "&": 1, "|": 0}
COMPUTE = {
    "+": operator.add, "-": operator.sub, "*": operator.mul, "/": operator.truediv,
    "<": operator.lt, ">": operator.gt, "<=": operator.le, ">=": operator.ge,
    "==": operator.eq, "!=": operator.ne,
    "&": lambda a, b: a and b, "|": lambda a, b: a or b,
}
ARITH, CMP, BOOL = ["+", "-", "*"], ["<", ">", "<=", ">=", "==", "!="], ["&", "|"]

# token-type vocabulary for the structure-only controller (values never appear)
CLASSES = ["PAD", "NUM", "BOOL", "+", "-", "*", "/", "<", ">", "<=", ">=", "==", "!=",
           "&", "|", "IF", "THEN", "ELSE", "LP", "RP"]
CIDX = {c: i for i, c in enumerate(CLASSES)}


# ----------------------------------------------------------------- oracle (Python)
def eval_node(n):
    k = n[0]
    if k == "lit":
        return n[1]
    if k == "bin":
        return COMPUTE[n[1]](eval_node(n[2]), eval_node(n[3]))
    return eval_node(n[2]) if eval_node(n[1]) else eval_node(n[3])   # if: short-circuit


def type_of(n):
    if n[0] == "lit":
        return n[2]
    if n[0] == "bin":
        return SIG[n[1]][2]
    return type_of(n[2])                                              # if: branch type


# ----------------------------------------------------------------- local reduction
def find_redex(n, path=()):
    """Path to the next site to reduce: leftmost-innermost, conditions before branches (lazy)."""
    k = n[0]
    if k == "lit":
        return None
    if k == "bin":
        if n[2][0] == "lit" and n[3][0] == "lit":
            return path
        return find_redex(n[2], path + (0,)) or find_redex(n[3], path + (1,))
    # if: collapse once the condition is a value, else reduce only inside the condition
    if n[1][0] == "lit":
        return path
    return find_redex(n[1], path + ("c",))


def _reduce_here(n):
    if n[0] == "bin":
        return ("lit", COMPUTE[n[1]](n[2][1], n[3][1]), SIG[n[1]][2])
    return n[2] if n[1][1] else n[3]                                  # if: pick branch


def apply_redex(n, path):
    if path == ():
        return _reduce_here(n)
    h, rest = path[0], path[1:]
    if n[0] == "bin":
        if h == 0:
            return ("bin", n[1], apply_redex(n[2], rest), n[3])
        return ("bin", n[1], n[2], apply_redex(n[3], rest))
    if h == "c":
        return ("if", apply_redex(n[1], rest), n[2], n[3])
    if h == "t":
        return ("if", n[1], apply_redex(n[2], rest), n[3])
    return ("if", n[1], n[2], apply_redex(n[3], rest))


def reduce_to_value(n, max_steps=100000):
    steps = 0
    while n[0] != "lit":
        p = find_redex(n)
        if p is None:
            raise RuntimeError(("stuck", n))
        n = apply_redex(n, p)
        steps += 1
        if steps > max_steps:
            raise RuntimeError("too many steps")
    return n[1], n[2], steps


# ----------------------------------------------------------------- linearize / render
def linearize(node):
    """Flatten to (tokens, token-classes, {node_path: operator_token_index})."""
    toks, classes, opidx = [], [], {}

    def emit(t, c):
        toks.append(t)
        classes.append(c)

    def rec(n, path, parent_prec):
        k = n[0]
        if k == "lit":
            emit(repr(n[1]), "NUM") if n[2] == "num" else emit("T" if n[1] else "F", "BOOL")
            return
        if k == "bin":
            sym, p = n[1], PREC[n[1]]
            need = p < parent_prec
            if need:
                emit("(", "LP")
            rec(n[2], path + (0,), p)
            opidx[path] = len(toks)
            emit(sym, sym)
            rec(n[3], path + (1,), p + 1)
            if need:
                emit(")", "RP")
            return
        wrap = path != ()
        if wrap:
            emit("(", "LP")
        opidx[path] = len(toks)
        emit("if", "IF")
        for child, kw in ((n[1], None), (n[2], "THEN"), (n[3], "ELSE")):
            if kw:
                emit(kw.lower(), kw)
            emit("(", "LP")
            rec(child, path + ({None: "c", "THEN": "t", "ELSE": "e"}[kw],), 0)
            emit(")", "RP")
        if wrap:
            emit(")", "RP")

    rec(node, (), 0)
    return toks, classes, opidx


def render(node):
    return " ".join(str(t) for t in linearize(node)[0])


def redex_index(node):
    p = find_redex(node)
    if p is None:
        return None
    return linearize(node)[2][p]


def valid_redexes(n, path=()):
    """All structurally reducible sites (bin with both operands literal, or if with literal cond)."""
    k = n[0]
    if k == "lit":
        return []
    if k == "bin":
        here = [path] if n[2][0] == "lit" and n[3][0] == "lit" else []
        return here + valid_redexes(n[2], path + (0,)) + valid_redexes(n[3], path + (1,))
    here = [path] if n[1][0] == "lit" else []
    return (here + valid_redexes(n[1], path + ("c",))
            + valid_redexes(n[2], path + ("t",)) + valid_redexes(n[3], path + ("e",)))


# ----------------------------------------------------------------- generator
def _split(depth, rng, k=2):
    ds = [depth - 1] + [int(rng.integers(0, depth)) for _ in range(k - 1)]
    rng.shuffle(ds)
    return ds


def gen(depth, want, rng):
    if depth <= 0:
        if want == "num":
            return ("lit", int(rng.integers(-5, 6)), "num")
        return ("lit", bool(rng.integers(0, 2)), "bool")
    r = rng.random()
    if want == "num":
        if r < 0.72:
            dl, dr = _split(depth, rng)
            return ("bin", ARITH[int(rng.integers(len(ARITH)))], gen(dl, "num", rng), gen(dr, "num", rng))
        dc, dt, de = _split(depth, rng, 3)
        return ("if", gen(dc, "bool", rng), gen(dt, "num", rng), gen(de, "num", rng))
    if r < 0.5:
        dl, dr = _split(depth, rng)
        return ("bin", CMP[int(rng.integers(len(CMP)))], gen(dl, "num", rng), gen(dr, "num", rng))
    if r < 0.8:
        dl, dr = _split(depth, rng)
        return ("bin", BOOL[int(rng.integers(len(BOOL)))], gen(dl, "bool", rng), gen(dr, "bool", rng))
    dc, dt, de = _split(depth, rng, 3)
    return ("if", gen(dc, "bool", rng), gen(dt, "bool", rng), gen(de, "bool", rng))


class TypedEngine:
    """Exact evaluation of mixed-type expressions via local typed reduction.
    Computation lives in the swappable COMPUTE tool table; control is the local redex rule."""
    def __init__(self, compute=None):
        self.compute = compute or COMPUTE

    def evaluate(self, node):
        v, t, _ = reduce_to_value(node)
        return v, t
