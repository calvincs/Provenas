"""Core engine tests — pure standard library, no torch/numpy/LLM. This is the suite a downloader runs.

    pytest tests/test_core.py
"""
import json

from provenas.kg import KnowledgeGraph
from provenas.kgvm import KGMachine
from provenas.infer import (Rule, forward_chain, forward_chain_prov, explain,
                            paths_between, induce_path_rule)
from provenas.learn import admit_rule, validate_rule
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


# --------------------------------------------------------------- materialized closure / Tier-3 ops
def test_closure_is_cached_and_invalidated():
    s = Store(":memory:")
    s.assert_("tom", "parent", "bob")
    s.add_rule(Rule([("?x", "parent", "?y")], ("?x", "ancestor", "?y"), "anc"))
    kg, prov = s.closure()
    assert ("tom", "ancestor", "bob") in kg.triples and ("tom", "ancestor", "bob") in prov
    assert s.closure()[0] is kg                                # same revision -> cached object
    s.assert_("bob", "parent", "ann")                          # write invalidates
    kg2, _ = s.closure()
    assert kg2 is not kg and ("bob", "ancestor", "ann") in kg2.triples
    s.retract("bob", "parent", "ann")
    assert ("bob", "ancestor", "ann") not in s.closure()[0].triples


def test_closure_warm_start_from_disk(tmp_path):
    p = str(tmp_path / "kb.db")
    s = Store(p)
    s.assert_("a", "parent", "b")
    s.add_rule(Rule([("?x", "parent", "?y")], ("?x", "ancestor", "?y"), "anc"))
    s.closure()                                                # materializes the derived table
    s.close()
    s2 = Store(p)                                              # fresh process: no in-memory cache
    kg, prov = s2.closure()
    assert ("a", "ancestor", "b") in kg.triples and prov[("a", "ancestor", "b")][0] == "anc"
    s2.close()


def test_rule_disable_enable():
    s = Store(":memory:")
    s.assert_("a", "parent", "b")
    s.add_rule(Rule([("?x", "parent", "?y")], ("?x", "ancestor", "?y"), "anc"))
    assert ("a", "ancestor", "b") in s.closure()[0].triples
    assert s.set_rule_active("anc", False)
    assert ("a", "ancestor", "b") not in s.closure()[0].triples
    assert len(s.rules()) == 0 and len(s.rules(all=True)) == 1  # kept on record
    s.set_rule_active("anc", True)
    assert ("a", "ancestor", "b") in s.closure()[0].triples
    assert not s.set_rule_active("nope", False)


def test_regression_cases_gate_rule_admission():
    s = Store(":memory:")
    s.assert_("p", "parent", "a")
    s.assert_("p", "parent", "b")
    # pin: nothing is currently anyone's "sib"
    case_action = {"action": "check", "triple": ["a", "sib", "a"]}
    s.add_case("no-self-sib", case_action, qa.run_action(s, case_action)["answer"])
    naive = Rule([("?p", "parent", "?x"), ("?p", "parent", "?y")], ("?x", "sib", "?y"), "naive")
    ok, rep = admit_rule(s, naive, "sib", [("a", "b")], [])    # examples pass, but the case flips
    assert not ok and rep["case_flips"] == ["no-self-sib"]
    guarded = Rule([("?p", "parent", "?x"), ("?p", "parent", "?y"), ("?x", "!=", "?y")],
                   ("?x", "sib", "?y"), "sib")
    ok, rep = admit_rule(s, guarded, "sib", [("a", "b")], [("a", "a")])
    assert ok and not rep["case_flips"] and len(s.rules()) == 1


def test_schema_strict_mode():
    import pytest
    s = Store(":memory:")
    s.declare("parent", "p is the parent of c")
    s.set_meta("strict", "1")
    s.assert_("a", "parent", "b")                              # declared: fine
    with pytest.raises(ValueError):
        s.assert_("a", "parnet", "b")                          # typo'd relation rejected
    r = qa.run_action(s, {"action": "assert", "triple": ["a", "parnet", "b"]})
    assert r["kind"] == "error" and "not declared" in r["trace"]
    s.set_meta("strict", "0")
    s.assert_("a", "adhoc", "b")                               # strict off: free-form again


def test_decision_log():
    s = Store(":memory:")
    s.assert_("a", "parent", "b")
    qa.run_action(s, {"action": "check", "triple": ["a", "parent", "b"]})
    qa.run_action(s, {"action": "query", "pattern": ["?x", "parent", "b"]})
    kinds = [k for _, k, _ in s.recent_log(10)]
    assert kinds.count("decide") == 2


def test_http_service():
    import threading
    import urllib.request
    import urllib.error
    from provenas.server import serve
    store, server = serve(":memory:", port=0, token="sekrit", with_llm=False)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = "http://127.0.0.1:%d" % server.server_address[1]
    def call(path, data=None, token="sekrit"):
        req = urllib.request.Request(base + path, data=json.dumps(data).encode() if data else None,
                                     headers={"Authorization": "Bearer " + token})
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read())
    try:
        assert call("/health")["ok"] is True
        assert call("/action", {"action": "assert", "triple": ["tom", "parent", "bob"]})["answer"] is True
        r = call("/action", {"action": "check", "triple": ["tom", "parent", "bob"]})
        assert r["answer"] is True and "(given fact)" in r["trace"]
        assert call("/facts")["facts"] == [["tom", "parent", "bob"]]
        try:
            call("/health", token="wrong")
            assert False, "should have been a 401"
        except urllib.error.HTTPError as e:
            assert e.code == 401
    finally:
        server.shutdown()
        store.close()


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


# --------------------------------------------------------------- indexed query / semi-naive engine
def test_query_indexes_match_brute_force():
    import random
    rnd = random.Random(7)
    kg = KnowledgeGraph()
    ents, rels = [f"e{i}" for i in range(12)], ["a", "b", "c"]
    for _ in range(60):
        kg.assert_(rnd.choice(ents), rnd.choice(rels), rnd.choice(ents))
    def brute(pat):
        out = []
        for t in kg.triples:
            b, ok = {}, True
            for p, v in zip(pat, t):
                if isinstance(p, str) and p.startswith("?"):
                    if b.get(p, v) != v:
                        ok = False
                        break
                    b[p] = v
                elif p != v:
                    ok = False
                    break
            if ok:
                out.append(b)
        return out
    for pat in [("e1", "a", "?x"), ("?x", "a", "e2"), ("e1", "?r", "e2"), ("?x", "b", "?y"),
                ("?x", "?r", "e3"), ("e4", "?r", "?o"), ("?x", "a", "?x"), ("?x", "?r", "?y"),
                ("e1", "a", "e2"), ("zz", "a", "?x"), ("?x", "zz", "?y")]:
        assert sorted(map(repr, kg.query(pat))) == sorted(map(repr, brute(pat))), pat


def test_semi_naive_matches_independent_closure():
    import random
    rnd = random.Random(11)
    kg, parent = KnowledgeGraph(), {}
    for i in range(1, 200):                                    # random tree rooted at n0
        parent[i] = rnd.randrange(i)
        kg.assert_(f"n{parent[i]}", "parent", f"n{i}")
    forward_chain(kg, [Rule([("?x", "parent", "?y")], ("?x", "ancestor", "?y"), "base"),
                       Rule([("?x", "parent", "?y"), ("?y", "ancestor", "?z")], ("?x", "ancestor", "?z"), "step")])
    for i in (5, 50, 150, 199):                                # ground truth by walking parent links
        anc, j = set(), i
        while j in parent:
            j = parent[j]
            anc.add(f"n{j}")
        assert {b["?x"] for b in kg.query(("?x", "ancestor", f"n{i}"))} == anc


def test_stratified_negation():
    kg = KnowledgeGraph()
    for t in [("alice", "has_role", "admin"), ("bob", "has_role", "admin"), ("bob", "suspended", "true")]:
        kg.assert_(*t)
    forward_chain(kg, [Rule([("?u", "has_role", "admin"), ("?u", "~suspended", "?any")],
                            ("?u", "can", "login"), "login")])
    assert ("alice", "can", "login") in kg.triples             # not suspended -> allowed
    assert ("bob", "can", "login") not in kg.triples           # deny-unless works


def test_negation_sees_derived_lower_stratum():
    kg = KnowledgeGraph()
    kg.assert_("a", "parent", "b")
    rules = [Rule([("?x", "parent", "?y")], ("?x", "ancestor", "?y"), "anc"),
             Rule([("?x", "parent", "?y"), ("?x", "~ancestor", "?y")], ("?x", "weird", "?y"), "w")]
    forward_chain(kg, rules)                                   # ancestor stratum completes first
    assert ("a", "weird", "b") not in kg.triples


def test_unstratified_ruleset_rejected():
    import pytest
    bad = [Rule([("?x", "~p", "?y")], ("?x", "q", "?y"), "r1"),
           Rule([("?x", "q", "?y")], ("?x", "p", "?y"), "r2")]
    with pytest.raises(ValueError):
        forward_chain(KnowledgeGraph(), bad)
    ok, rep = validate_rule(KnowledgeGraph(), bad, "q", [], [])     # the gate reports, not crashes
    assert not ok and "stratified" in rep.get("error", "")


def test_comparison_guards_are_numeric():
    kg = KnowledgeGraph()
    for person, age in [("ann", "9"), ("bob", "18"), ("cy", "40")]:
        kg.assert_(person, "age", age)
    forward_chain(kg, [Rule([("?p", "age", "?a"), ("?a", ">=", "18")], ("?p", "is", "adult"), "adult")])
    assert {s for s, _, o in kg.triples if o == "adult"} == {"bob", "cy"}   # "9" < "18" numerically


# --------------------------------------------------------------- regressions
def test_guard_position_does_not_matter():
    # a '!=' guard placed BEFORE the atoms that bind its variables must not vacuously pass
    kg = KnowledgeGraph()
    kg.assert_("p", "parent", "a")
    kg.assert_("p", "parent", "b")
    early = Rule([("?x", "!=", "?y"), ("?p", "parent", "?x"), ("?p", "parent", "?y")],
                 ("?x", "sib", "?y"), "early-guard")
    forward_chain(kg, [early])
    assert ("a", "sib", "b") in kg.triples
    assert ("a", "sib", "a") not in kg.triples


def test_guard_over_unbound_var_fails_closed():
    # a guard whose variable the body never binds cannot be checked -> derive nothing, not everything
    kg = KnowledgeGraph()
    kg.assert_("p", "parent", "a")
    dangling = Rule([("?p", "parent", "?x"), ("?x", "!=", "?nowhere")], ("?x", "weird", "?x"), "dangling")
    forward_chain(kg, [dangling])
    assert ("a", "weird", "a") not in kg.triples


def test_rewrite_parse_rejects_bad_input():
    import pytest
    for bad in ["", "x - 1", "x / 2", "(x + 1", "x 1", "x +", "* x", "x + )"]:
        with pytest.raises(ValueError):
            RW.parse(bad)
    assert RW.parse("2 * (x + 1)") == ("op", "*", ("num", 2), ("op", "+", ("var", "x"), ("num", 1)))


def test_llm_loads_tolerant_but_not_corrupting():
    from provenas.llm import _loads
    assert _loads('{"note": "is a?b ok"}') == {"note": "is a?b ok"}        # '?' in a string survives
    assert _loads('{"head": [?x, "parent", ?y],}') == {"head": ["?x", "parent", "?y"]}
    assert _loads('```json\n{"a": 1}\n```') == {"a": 1}


def test_qa_rejects_malformed_actions():
    s = Store(":memory:")
    s.assert_("a", "parent", "b")
    for bad in [{"action": "check"}, {"action": "query", "pattern": ["?x", "parent"]},
                {"action": "assert", "triple": ["a", None, "c"]}, {"action": "check", "triple": "a b c"}]:
        assert qa.run_action(s, bad)["kind"] == "error"
    assert ("a", None, "c") not in s.triples()


def test_solver_empty_goal():
    from provenas.solver import answer, ALL_TOOLS
    assert answer([], KnowledgeGraph(), [], None, ALL_TOOLS) == set()


# --------------------------------------------------------------- toolsmith safety gate (cont.)
def test_toolsmith_rejects_unsafe_and_wrong():
    assert not validate_ast("def f(n):\n import os\n return 1\n", "f")[0]
    assert not validate_ast("def f(x):\n return x.__class__\n", "f")[0]
    assert not validate_ast("def f(x):\n return open('/etc/passwd')\n", "f")[0]
    ok, stage, _ = admit_tool(None, "gcd", "def gcd(a, b):\n    return a + b\n", [((12, 8), 4)])
    assert not ok and stage == "tests"
