"""Register/variable VM (stack + keyed memory) for the variables experiment.

Straight-line programs: statement i is `Vi = A <op> B`, where A,B are constants
(Nk) or prior variables (Vj, j<i). The net compiles the program to instructions
over a stack + a dict memory: PUSH Nk, LOAD Vj (get), op, STORE Vi (set). Python
executes them EXACTLY. Tests symbol binding + get/set across a growing memory.
"""
from __future__ import annotations

import numpy as np

from .calculator import run_op

SYMNAME = {"+": "add", "-": "subtract", "*": "multiply", "/": "divide"}
OPSYM = ["+", "-", "*", "/"]


def gen_program(k, rng, const_max=20):
    """k statements; returns (stmts, const_values). Each operand is ('const', idx)
    or ('var', j<i)."""
    consts, stmts = [], []

    def operand(i):
        if i > 0 and rng.random() < 0.6:
            return ("var", int(rng.integers(0, i)))
        consts.append(float(int(rng.integers(1, const_max + 1))))
        return ("const", len(consts) - 1)

    for i in range(k):
        a = operand(i)
        op = OPSYM[int(rng.integers(4))]
        b = operand(i)
        stmts.append({"var": i, "a": a, "op": op, "b": b})
    return stmts, consts


def execute_true(stmts, consts):
    """Reference execution with exact Python. Returns (value|None, status)."""
    mem = {}

    def val(o):
        return consts[o[1]] if o[0] == "const" else mem[o[1]]

    for s in stmts:
        v, e = run_op(SYMNAME[s["op"]], val(s["a"]), val(s["b"]))
        if e != "ok":
            return None, e
        mem[s["var"]] = v
    return mem[stmts[-1]["var"]], "ok"


def _opnd_src(o):
    return f"V{o[1]}" if o[0] == "var" else f"N{o[1]}"


def source_tokens(stmts):
    toks = []
    for s in stmts:
        toks += [f"V{s['var']}", "=", _opnd_src(s["a"]), s["op"], _opnd_src(s["b"]), ";"]
    return toks


def instr_tokens(stmts):
    toks = []
    for s in stmts:
        for o in (s["a"], s["b"]):
            toks += (["LOAD", f"V{o[1]}"] if o[0] == "var" else ["PUSH", f"N{o[1]}"])
        toks += [s["op"], "STORE", f"V{s['var']}"]
    return toks


def execute_instructions(toks, consts, result_var):
    """Execute predicted instruction tokens on stack + dict; return (value, status).
    A malformed program -> BadProgram (the net's fault)."""
    stack, mem, i = [], {}, 0
    while i < len(toks):
        t = toks[i]
        if t == "PUSH":
            i += 1
            if i >= len(toks) or not toks[i].startswith("N"):
                return None, "BadProgram"
            k = int(toks[i][1:])
            if k >= len(consts):
                return None, "BadProgram"
            stack.append(consts[k])
        elif t == "LOAD":
            i += 1
            if i >= len(toks) or not toks[i].startswith("V"):
                return None, "BadProgram"
            key = int(toks[i][1:])
            if key not in mem:
                return None, "BadProgram"
            stack.append(mem[key])
        elif t == "STORE":
            i += 1
            if i >= len(toks) or not toks[i].startswith("V") or not stack:
                return None, "BadProgram"
            mem[int(toks[i][1:])] = stack.pop()
        elif t in SYMNAME:
            if len(stack) < 2:
                return None, "BadProgram"
            b = stack.pop()
            a = stack.pop()
            v, e = run_op(SYMNAME[t], a, b)
            if e != "ok":
                return None, e
            stack.append(v)
        else:
            return None, "BadProgram"
        i += 1
    return (mem[result_var], "ok") if result_var in mem else (None, "BadProgram")
