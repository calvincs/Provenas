"""rewrite — a learned term-rewriting engine, and the hook that regenerates the controller.

Terms over {+, *}, numbers, and symbolic variables. A RULESET (algebraic identities + constant
folding) defines a normal form; the structure-only controller picks which redex to rewrite, the engine
applies the matching rule exactly. The point of Slice 3: the rewrite ORACLE is exact, so it can LABEL
unlimited training data — when the ruleset changes (a rule is admitted), the controller is regenerated
from the oracle automatically. The symbolic engine teaches the net; no human labels.

A rule is admitted only if it is value-PRESERVING (the rewrite analog of Slice 2's test-before-admit):
lhs and rhs must evaluate equal over random variable assignments.

Terms:     ("num", n) | ("var", name) | ("op", sym, l, r)
Patterns:  add ("?", name) matching any subterm
Rules:     ("pat", lhs, rhs)  |  ("fold",)
"""
from __future__ import annotations

FOLD_OP = {"+": lambda a, b: a + b, "*": lambda a, b: a * b}
PREC = {"+": 1, "*": 2}
VARS = ["x", "y", "z"]
# the controller sees token SYMBOLS, including the special constants 0/1 the rules discriminate on
# (it still cannot see the magnitude of an ordinary number — only that it is "some number")
CLASSES = ["PAD", "NUM0", "NUM1", "NUM", "VAR", "+", "*", "LP", "RP"]
CIDX = {c: i for i, c in enumerate(CLASSES)}


# ----------------------------------------------------------------- match / apply
def match(pat, term, b):
    if pat[0] == "?":
        name = pat[1]
        if name in b:
            return b if b[name] == term else None
        nb = dict(b)
        nb[name] = term
        return nb
    if pat[0] in ("num", "var"):
        return b if pat == term else None
    if pat[0] == "op" and term[0] == "op" and pat[1] == term[1]:
        b1 = match(pat[2], term[2], b)
        return match(pat[3], term[3], b1) if b1 is not None else None
    return None


def subst(pat, b):
    if pat[0] == "?":
        return b[pat[1]]
    if pat[0] == "op":
        return ("op", pat[1], subst(pat[2], b), subst(pat[3], b))
    return pat


def applies_at(term, rule):
    if rule[0] == "fold":
        return {} if term[0] == "op" and term[2][0] == "num" and term[3][0] == "num" else None
    return match(rule[1], term, {})                       # ("pat", lhs, rhs)


def apply_at(term, rule):
    if rule[0] == "fold":
        return ("num", FOLD_OP[term[1]](term[2][1], term[3][1]))
    return subst(rule[2], match(rule[1], term, {}))


def first_rule(term, rules):
    for r in rules:
        if applies_at(term, r) is not None:
            return r
    return None


# ----------------------------------------------------------------- reduction
def find_redex(term, rules, path=()):
    """Leftmost-innermost node where some rule applies (the oracle's canonical choice)."""
    if term[0] != "op":
        return None
    return (find_redex(term[2], rules, path + (0,))
            or find_redex(term[3], rules, path + (1,))
            or (path if first_rule(term, rules) is not None else None))


def reducible_nodes(term, rules, path=()):
    if term[0] != "op":
        return []
    here = [path] if first_rule(term, rules) is not None else []
    return here + reducible_nodes(term[2], rules, path + (0,)) + reducible_nodes(term[3], rules, path + (1,))


def rewrite_at(term, path, rules):
    if path == ():
        return apply_at(term, first_rule(term, rules))
    h, rest = path[0], path[1:]
    if h == 0:
        return ("op", term[1], rewrite_at(term[2], rest, rules), term[3])
    return ("op", term[1], term[2], rewrite_at(term[3], rest, rules))


def normal_form(term, rules, cap=10000):
    for _ in range(cap):
        p = find_redex(term, rules)
        if p is None:
            return term
        term = rewrite_at(term, p, rules)
    raise RuntimeError("no normal form (non-terminating ruleset?)")


# ----------------------------------------------------------------- eval / soundness
def evaluate(term, assign):
    if term[0] == "num":
        return term[1]
    if term[0] == "var":
        return assign[term[1]]
    return FOLD_OP[term[1]](evaluate(term[2], assign), evaluate(term[3], assign))


def _pvars(pat):
    if pat[0] == "?":
        return {pat[1]}
    if pat[0] == "op":
        return _pvars(pat[2]) | _pvars(pat[3])
    return set()


def sound(rule, rng, n=60):
    """A rewrite rule is admissible only if lhs == rhs in value over random instantiations."""
    if rule[0] == "fold":
        return True
    pvars = _pvars(rule[1]) | _pvars(rule[2])
    for _ in range(n):
        b = {pv: gen(int(rng.integers(0, 2)), rng) for pv in pvars}
        L, R = subst(rule[1], b), subst(rule[2], b)
        assign = {s: int(rng.integers(-4, 5)) for s in VARS}
        if evaluate(L, assign) != evaluate(R, assign):
            return False
    return True


# ----------------------------------------------------------------- linearize / gen
def linearize(term):
    toks, classes, opidx = [], [], {}

    def emit(t, c):
        toks.append(t)
        classes.append(c)

    def rec(n, path, parent_prec):
        if n[0] == "num":
            emit(str(n[1]), "NUM0" if n[1] == 0 else "NUM1" if n[1] == 1 else "NUM")
            return
        if n[0] == "var":
            emit(n[1], "VAR")
            return
        if n[0] == "?":                                       # pattern variable (display only)
            emit("?" + str(n[1]), "VAR")
            return
        p = PREC[n[1]]
        need = p < parent_prec
        if need:
            emit("(", "LP")
        rec(n[2], path + (0,), p)
        opidx[path] = len(toks)
        emit(n[1], n[1])
        rec(n[3], path + (1,), p + 1)
        if need:
            emit(")", "RP")

    rec(term, (), 0)
    return toks, classes, opidx


def pretty(term):
    return " ".join(linearize(term)[0])


def gen(depth, rng):
    if depth <= 0:
        if rng.random() < 0.5:
            return ("var", VARS[int(rng.integers(len(VARS)))])
        return ("num", int(rng.integers(0, 4)))           # 0..3 -> identity + foldable cases occur
    sym = "+" if rng.random() < 0.5 else "*"
    dl, dr = depth - 1, int(rng.integers(0, depth))
    if rng.random() < 0.5:
        dl, dr = dr, dl
    return ("op", sym, gen(dl, rng), gen(dr, rng))


def signature(rules):
    return tuple(rules)                                   # rules are nested tuples -> hashable


# ----------------------------------------------------------------- a starter ruleset
A, B = ("?", "a"), ("?", "b")
ADD0 = ("pat", ("op", "+", A, ("num", 0)), A)
ADD0b = ("pat", ("op", "+", ("num", 0), A), A)
MUL1 = ("pat", ("op", "*", A, ("num", 1)), A)
MUL1b = ("pat", ("op", "*", ("num", 1), A), A)
MUL0 = ("pat", ("op", "*", A, ("num", 0)), ("num", 0))
MUL0b = ("pat", ("op", "*", ("num", 0), A), ("num", 0))
FOLD = ("fold",)
BOGUS = ("pat", ("op", "+", A, B), ("op", "*", A, B))     # a+b -> a*b : NOT value-preserving

R1 = [ADD0, ADD0b, MUL1, MUL1b, MUL0, MUL0b]              # identities only
R2 = R1 + [FOLD]                                          # + constant folding

# named defaults, for seeding a knowledge base's rewrite table (Slice-3 ruleset)
DEFAULTS = [("add0", ADD0), ("add0b", ADD0b), ("mul1", MUL1), ("mul1b", MUL1b),
            ("mul0", MUL0), ("mul0b", MUL0b), ("fold", FOLD)]


def rule_str(rule):
    if rule[0] == "fold":
        return "(num op num) -> num   [constant folding]"
    return f"{pretty(rule[1])}  ->  {pretty(rule[2])}"


def parse(s):
    """Tiny recursive-descent parser for the rewrite term language: + * ( ) over numbers and variables."""
    import re
    toks = re.findall(r"\d+|[a-zA-Z_]\w*|[+*()]", s)
    pos = [0]

    def peek():
        return toks[pos[0]] if pos[0] < len(toks) else None

    def factor():
        t = toks[pos[0]]
        pos[0] += 1
        if t == "(":
            e = expr()
            if peek() == ")":
                pos[0] += 1
            return e
        return ("num", int(t)) if t.isdigit() else ("var", t)

    def term():
        n = factor()
        while peek() == "*":
            pos[0] += 1
            n = ("op", "*", n, factor())
        return n

    def expr():
        n = term()
        while peek() == "+":
            pos[0] += 1
            n = ("op", "+", n, term())
        return n

    return expr()
