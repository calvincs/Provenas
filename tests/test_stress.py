"""Stress / adversarial tests — try to break the engine and the safety gates. Pure standard library.

Covers: inference termination on cyclic/recursive rules, idempotence, malformed input, deep rewriting,
weird entity strings, and a battery of sandbox-escape attempts the toolsmith MUST reject.

    pytest tests/test_stress.py
"""
from provenas.kg import KnowledgeGraph
from provenas.infer import Rule, forward_chain, forward_chain_prov
from provenas.store import Store
from provenas import qa, rewrite as RW
from provenas.toolsmith import validate_ast, admit_tool


# --------------------------------------------------------------- inference can't run away (Datalog)
def test_symmetric_rule_terminates():
    kg = KnowledgeGraph()
    kg.assert_("a", "near", "b")
    forward_chain(kg, [Rule([("?x", "near", "?y")], ("?y", "near", "?x"), "sym")])   # would loop if naive
    assert ("b", "near", "a") in kg.triples and ("a", "near", "b") in kg.triples


def test_transitive_closure_on_a_cycle_terminates():
    kg = KnowledgeGraph()
    ring = ["n%d" % i for i in range(12)]
    for i in range(len(ring)):
        kg.assert_(ring[i], "to", ring[(i + 1) % len(ring)])
    forward_chain(kg, [Rule([("?x", "to", "?y"), ("?y", "to", "?z")], ("?x", "to", "?z"), "trans")])
    # full closure over a 12-cycle = every ordered pair (12*12), and it terminates
    assert sum(1 for s, r, o in kg.triples if r == "to") == len(ring) * len(ring)


def test_self_loop_and_duplicate_assert():
    kg = KnowledgeGraph()
    kg.assert_("x", "r", "x")
    kg.assert_("x", "r", "x")                       # idempotent
    assert len([t for t in kg.triples if t == ("x", "r", "x")]) == 1
    forward_chain(kg, [Rule([("?a", "r", "?b")], ("?a", "r", "?b"), "noop")])   # head==body, no growth
    assert ("x", "r", "x") in kg.triples


def test_unbound_head_variable_is_skipped():
    kg = KnowledgeGraph()
    kg.assert_("a", "p", "b")
    forward_chain(kg, [Rule([("?x", "p", "?y")], ("?x", "q", "?z"), "bad")])   # ?z unbound in head
    assert not any(r == "q" for _, r, _ in kg.triples)


def test_unicode_and_weird_entities():
    kg = KnowledgeGraph()
    for s, r, o in [("naïve", "is", "café"), ("x y", "has space", "v"), ("", "empty", "subj")]:
        kg.assert_(s, r, o)
    assert sorted(b["?o"] for b in kg.query(("naïve", "is", "?o"))) == ["café"]
    _, prov = forward_chain_prov(kg, [])
    assert ("naïve", "is", "café") in kg.triples


# --------------------------------------------------------------- qa robustness
def test_qa_empty_kb_and_malformed():
    s = Store(":memory:")
    assert qa.run_action(s, {"action": "query", "pattern": ["?x", "is_a", "thing"]})["answer"] == []
    assert qa.run_action(s, {"action": "check", "triple": ["a", "b", "c"]})["answer"] is False
    assert qa.run_action(s, {})["kind"] == "error"             # no 'action' key -> graceful


# --------------------------------------------------------------- rewrite stays exact & terminating
def test_rewrite_deep_nesting_terminates_and_preserves_value():
    expr = ("var", "x")
    for _ in range(60):                                        # ((((x*1)+0)*1)+0) ... 60 deep
        expr = ("op", "+", ("op", "*", expr, ("num", 1)), ("num", 0))
    nf = RW.normal_form(expr, RW.R2)
    assert nf == ("var", "x")                                  # collapses entirely
    assert RW.evaluate(expr, {"x": 5}) == RW.evaluate(nf, {"x": 5}) == 5


def test_rewrite_parser_roundtrips():
    for s in ["1", "x", "1 + 2 * 3", "(1 + 2) * 3", "x * (y + 0)"]:
        assert RW.pretty(RW.parse(s))                          # parses without error, renders


# --------------------------------------------------------------- the sandbox MUST reject every escape
ESCAPES = {
    "import":        "def f(n):\n    import os\n    return 1\n",
    "from-import":   "def f(n):\n    from os import system\n    return 1\n",
    "dunder-import": "def f(n):\n    return __import__('os')\n",
    "eval":          "def f(n):\n    return eval('1')\n",
    "exec":          "def f(n):\n    return exec('x=1')\n",
    "open":          "def f(n):\n    return open('/etc/passwd')\n",
    "getattr":       "def f(n):\n    return getattr(n, 'real')\n",
    "attr-class":    "def f(n):\n    return n.__class__\n",
    "attr-method":   "def f(n):\n    return [].append\n",
    "builtins":      "def f(n):\n    return __builtins__\n",
    "type":          "def f(n):\n    return type(n)\n",
    "globals":       "def f(n):\n    return globals()\n",
    "two-funcs":     "def f(n):\n    return 1\ndef g(n):\n    return 2\n",
    "wrong-name":    "def other(n):\n    return n\n",
}


def test_sandbox_rejects_all_static_escapes():
    for label, src in ESCAPES.items():
        ok, _ = validate_ast(src, "f")
        assert not ok, f"AST gate let through: {label}"


def test_sandbox_kills_runtime_abuse():
    # infinite loop -> killed by CPU rlimit / timeout
    ok, stage, _ = admit_tool(None, "spin", "def spin(n):\n    while True:\n        n += 1\n    return n\n", [((1,), 2)])
    assert not ok and stage == "sandbox"
    # recursion bomb -> RecursionError -> non-zero exit
    ok, stage, _ = admit_tool(None, "rec", "def rec(n):\n    return rec(n + 1)\n", [((1,), 0)])
    assert not ok and stage == "sandbox"
    # memory bomb -> RLIMIT_AS -> MemoryError -> non-zero exit
    ok, stage, _ = admit_tool(None, "hog", "def hog(n):\n    x = [0] * (10 ** 9)\n    return len(x)\n", [((1,), 1)])
    assert not ok and stage == "sandbox"


def test_sandbox_admits_a_legitimately_complex_tool():
    src = ("def is_prime(n):\n"
           "    if n < 2:\n        return False\n"
           "    i = 2\n"
           "    while i * i <= n:\n"
           "        if n % i == 0:\n            return False\n"
           "        i += 1\n"
           "    return True\n")
    ok, stage, _ = admit_tool(None, "is_prime", src, [((2,), True), ((15,), False), ((97,), True)])
    assert ok and stage == "admitted"
