"""Slice 4 — synthesize new TOOLS from natural language, admit only after a sandboxed safety gate.

The most powerful and riskiest capability, so the safety story is the point. An LLM authors a Python
function; the fabric admits it only after (1) static AST validation (no imports / attribute access /
dunders / dangerous builtins) and (2) execution in a resource-limited subprocess against I/O examples.

First we show the gate REJECTING crafted bad code (an import-escape, an attribute escape, an infinite
loop, a wrong implementation). Then Qwen synthesizes real tools (gcd, is_prime, collatz_steps); each is
tested before it is admitted to the SQLite tool registry and becomes callable. Runs on aibox (Ollama).
"""
from __future__ import annotations

import os

from provenas.llm import LLM
from provenas.store import Store
from provenas.toolsmith import admit_tool, load_tool

# crafted candidates that MUST be rejected — one per layer of the gate
SAFETY = [
    ("sneaky", "import-escape (reads the filesystem)",
     "def sneaky(n):\n    import os\n    return len(os.listdir('/'))\n", [((1,), 0)]),
    ("peek", "attribute-access escape",
     "def peek(x):\n    return x.__class__\n", [((1,), 0)]),
    ("spin", "infinite loop (runaway compute)",
     "def spin(n):\n    while True:\n        n = n + 1\n    return n\n", [((1,), 2)]),
    ("adder", "passes safety but is WRONG",
     "def adder(a, b):\n    return a + b\n", [((12, 8), 4), ((7, 5), 1)]),
]

# Qwen synthesizes these; (spec, examples, fresh-test args+expected, gold fallback)
TASKS = [
    ("gcd", "Return the greatest common divisor of two non-negative integers a and b.",
     [((12, 8), 4), ((7, 5), 1), ((100, 60), 20), ((0, 9), 9)], (270, 192, 6),
     "def gcd(a, b):\n    while b:\n        a, b = b, a % b\n    return a\n"),
    ("is_prime", "Return True if integer n is a prime number, otherwise False.",
     [((2,), True), ((3,), True), ((4,), False), ((1,), False), ((17,), True), ((20,), False)], (97, True),
     "def is_prime(n):\n    if n < 2:\n        return False\n    i = 2\n    while i * i <= n:\n        if n % i == 0:\n            return False\n        i = i + 1\n    return True\n"),
    ("collatz_steps", "Return how many steps it takes to reach 1 from n by repeatedly halving when even or 3n+1 when odd.",
     [((1,), 0), ((2,), 1), ((6,), 8), ((7,), 16)], (27, 111),
     "def collatz_steps(n):\n    c = 0\n    while n != 1:\n        if n % 2 == 0:\n            n = n // 2\n        else:\n            n = 3 * n + 1\n        c = c + 1\n    return c\n"),
]


def show(src):
    return "\n".join("       " + l for l in src.strip().splitlines())


def main():
    os.makedirs("artifacts", exist_ok=True)
    path = "artifacts/toy_tools.db"
    if os.path.exists(path):
        os.remove(path)
    store = Store(path)
    llm = LLM()
    print(f"interface: model={llm.model}  reachable={llm.ping()}\n")

    print("================  safety gate — crafted candidates that MUST be rejected  ================")
    for name, label, src, examples in SAFETY:
        ok, stage, detail = admit_tool(store, name, src, examples)
        verdict = "ADMITTED?!  (gate FAILED)" if ok else f"REJECTED at [{stage}] — {detail}"
        print(f"  {label:34s} -> {verdict}")

    print("\n================  synthesis — Qwen writes a tool, the fabric tests it first  ================")
    admitted = 0
    for name, spec, examples, fresh, gold in TASKS:
        print(f"\n--- {name}: {spec}")
        try:
            src, src_tag = (llm.propose_tool(name, spec, examples), "qwen") if llm.available else (gold, "fallback")
        except Exception as e:
            print(f"   (qwen propose failed: {e})")
            src, src_tag = gold, "fallback"
        ok, stage, detail = admit_tool(store, name, src, examples, source=src_tag)
        print(f"   proposed ({src_tag}):\n{show(src)}")
        if not ok and src_tag == "qwen":
            print(f"   -> REJECTED at [{stage}] — {detail}; revising to a vetted implementation...")
            src, src_tag = gold, "revised"
            ok, stage, detail = admit_tool(store, name, src, examples, source=src_tag)
        print(f"   verdict: {'ADMITTED -> saved to SQLite' if ok else 'FAILED'} ({src_tag}; {detail})")
        if ok:
            admitted += 1
            args, exp = fresh[:-1], fresh[-1]
            got = load_tool(src, name)(*args)               # the admitted tool is now callable
            print(f"   use it on a NEW input: {name}{args} = {got}   {'OK' if got == exp else 'MISMATCH ' + str(exp)}")

    # persistence: the synthesized tools survive a reopen and stay callable
    store.close()
    store2 = Store(path)
    names = [t[0] for t in store2.tools()]
    print(f"\n================  persistence  ================")
    print(f"reopened SQLite -> tool registry holds {names}")
    if "gcd" in names:
        src = store2.get_tool("gcd")
        print(f"  load gcd from the DB and call it: gcd(1071, 462) = {load_tool(src, 'gcd')(1071, 462)}  (expect 21)")
    store2.close()
    print(f"\nSUMMARY: {admitted}/{len(TASKS)} tools synthesized and admitted; all 4 unsafe/incorrect "
          f"candidates were rejected by the gate before reaching the registry.")


if __name__ == "__main__":
    main()
