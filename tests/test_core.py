"""Core engine tests — pure standard library, no torch/numpy/LLM. This is the suite a downloader runs.

    pytest tests/test_core.py
"""
from provenas.kg import KnowledgeGraph
from provenas.kgvm import KGMachine
from provenas.infer import (Rule, forward_chain, forward_chain_prov, explain,
                            paths_between, induce_path_rule)
from provenas.learn import validate_rule
from provenas.store import Store
from provenas import qa, domains
from provenas import rewrite as RW
from provenas import typed as T
from provenas.toolsmith import validate_ast, run_sandboxed, load_tool, admit_tool


# --------------------------------------------------------------- knowledge graph + inference
def _family():
    kg = KnowledgeGraph()
    for t in [("tom", "parent", "bob"), ("bob", "parent", "ann"), ("bob", "parent", "cy")]:
        kg.assert_(*t)
    return kg


def test_kg_assert_query_retract():
    kg = _family()
    assert sorted(b["?x"] for b in kg.query(("bob", "parent", "?x"))) == ["ann", "cy"]
    kg.retract("bob", "parent", "ann")
    assert sorted(b["?x"] for b in kg.query(("bob", "parent", "?x"))) == ["cy"]


def test_kgvm_combine():
    kg = KnowledgeGraph()
    for t in [("dog", "is_a", "mammal"), ("cat", "is_a", "mammal"), ("dog", "lives_in", "house")]:
        kg.assert_(*t)
    mammals = KGMachine(kg).subjects_with("is_a", "mammal").result()
    house = KGMachine(kg).subjects_with("lives_in", "house").result()
    assert KGMachine(kg).select(*mammals).intersect(house).result() == {"dog"}


def test_inference_transitive_and_guard():
    kg = _family()
    rules = [Rule([("?x", "parent", "?y")], ("?x", "ancestor", "?y"), "base"),
             Rule([("?x", "parent", "?y"), ("?y", "ancestor", "?z")], ("?x", "ancestor", "?z"), "step"),
             Rule([("?p", "parent", "?a"), ("?p", "parent", "?b"), ("?a", "!=", "?b")],
                  ("?a", "sibling", "?b"), "sib")]
    forward_chain(kg, rules)
    assert ("tom", "ancestor", "ann") in kg.triples           # transitive
    assert ("ann", "sibling", "cy") in kg.triples             # != guard
    assert ("ann", "sibling", "ann") not in kg.triples        # guard excludes self


def test_provenance_proof():
    kg = _family()
    rules = [Rule([("?x", "parent", "?y"), ("?y", "parent", "?z")], ("?x", "grandparent", "?z"), "gp")]
    _, prov = forward_chain_prov(kg, rules)
    proof = "\n".join(explain(("tom", "grandparent", "ann"), prov))
    assert "gp" in proof and "(tom parent bob)" in proof and "(bob parent ann)" in proof


def test_paths_and_induce():
    kg = _family()
    forward_chain(kg, [Rule([("?x", "parent", "?y")], ("?x", "ancestor", "?y"), "base"),
                       Rule([("?x", "parent", "?y"), ("?y", "ancestor", "?z")], ("?x", "ancestor", "?z"), "step")])
    assert paths_between(kg, "tom", "ann", max_len=3)
    assert ("parent", "parent") in induce_path_rule(kg, [("tom", "ann"), ("tom", "cy")])


def test_learn_gate():
    kg = KnowledgeGraph()
    kg.assert_("p", "parent", "a")
    kg.assert_("p", "parent", "b")
    naive = Rule([("?p", "parent", "?x"), ("?p", "parent", "?y")], ("?x", "sib", "?y"), "naive")
    guard = Rule([("?p", "parent", "?x"), ("?p", "parent", "?y"), ("?x", "!=", "?y")], ("?x", "sib", "?y"), "g")
    assert not validate_rule(kg, [naive], "sib", [("a", "b")], [("a", "a")])[0]
    assert validate_rule(kg, [guard], "sib", [("a", "b")], [("a", "a")])[0]


# --------------------------------------------------------------- store / qa / domains
def test_store_persistence(tmp_path):
    p = str(tmp_path / "kb.db")
    s = Store(p)
    s.assert_("a", "parent", "b")
    s.add_rule(Rule([("?x", "parent", "?y")], ("?x", "ancestor", "?y"), "anc"))
    s.add_tool("inc", "def inc(n):\n    return n + 1\n", [((1,), 2)])
    s.add_rewrite("add0", RW.ADD0)
    s.close()
    s2 = Store(p)                                              # reopen: everything durable
    assert ("a", "parent", "b") in s2.triples()
    assert len(s2.rules()) == 1 and s2.get_tool("inc")
    assert s2.rewrites() == [RW.ADD0]                         # json -> tuple round-trip preserved
    s2.close()


def test_qa_run_action_and_context():
    s = Store(":memory:")
    domains.load(s, "rbac")
    assert qa.run_action(s, {"action": "check", "triple": ["alice", "can", "prod_deploy"]})["answer"] is True
    assert set(qa.run_action(s, {"action": "query", "pattern": ["?x", "can", "view_wiki"]})["answer"]) \
        == {"alice", "bob", "dana"}
    assert qa.run_action(s, {"action": "bogus"})["kind"] == "error"    # malformed action -> graceful
    assert "has_role" in qa.context_from_store(s)


def test_all_domain_packs():
    for d in domains.DOMAINS:
        s = Store(":memory:")
        domains.load(s, d["name"])
        for q in d["questions"]:
            r = qa.run_action(s, dict(q["gold"]))
            got = set(r["answer"]) if isinstance(q["expect"], set) else r["answer"]
            assert got == q["expect"], (d["name"], q["q"], got, q["expect"])


# --------------------------------------------------------------- rewrite engine
def test_rewrite_parse_and_simplify():
    t = RW.parse("(x + 0) * (2 + 3)")
    assert RW.normal_form(t, RW.R1) == ("op", "*", ("var", "x"), ("op", "+", ("num", 2), ("num", 3)))
    assert RW.normal_form(t, RW.R2) == ("op", "*", ("var", "x"), ("num", 5))   # folding added
    assert RW.evaluate(t, {"x": 7}) == RW.evaluate(RW.normal_form(t, RW.R2), {"x": 7}) == 35


# --------------------------------------------------------------- typed engine (hand-built terms)
def test_typed_reducer():
    n = ("bin", "*", ("bin", "+", ("lit", 3, "num"), ("lit", 4, "num")), ("lit", 2, "num"))
    assert T.reduce_to_value(n)[:2] == (14, "num")
    cmp = ("bin", "<", ("lit", 3, "num"), ("lit", 4, "num"))   # num -> bool transition
    assert T.reduce_to_value(cmp)[:2] == (True, "bool")
    sc = ("if", ("bin", "==", ("lit", 1, "num"), ("lit", 0, "num")),
          ("bin", "/", ("lit", 1, "num"), ("lit", 0, "num")), ("lit", 7, "num"))
    assert T.reduce_to_value(sc)[0] == 7 == T.eval_node(sc)     # short-circuit: dead 1/0 never runs


# --------------------------------------------------------------- toolsmith safety gate
def test_toolsmith_admits_safe_tool():
    src = "def gcd(a, b):\n    while b:\n        a, b = b, a % b\n    return a\n"
    assert validate_ast(src, "gcd")[0]
    ok, out = run_sandboxed(src, "gcd", [[12, 8], [0, 9]])
    assert ok and out == [4, 9]
    assert load_tool(src, "gcd")(48, 36) == 12
    ok, stage, _ = admit_tool(None, "gcd", src, [((12, 8), 4), ((7, 5), 1)])
    assert ok and stage == "admitted"


def test_toolsmith_rejects_unsafe_and_wrong():
    assert not validate_ast("def f(n):\n import os\n return 1\n", "f")[0]
    assert not validate_ast("def f(x):\n return x.__class__\n", "f")[0]
    assert not validate_ast("def f(x):\n return open('/etc/passwd')\n", "f")[0]
    ok, stage, _ = admit_tool(None, "gcd", "def gcd(a, b):\n    return a + b\n", [((12, 8), 4)])
    assert not ok and stage == "tests"
