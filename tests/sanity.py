"""Fast sanity checks (no training). Run: python -m tests.sanity"""
from __future__ import annotations

import numpy as np
import torch

from provenas.calculator import run_op, Calculator
from provenas import encoders as E
from provenas import dataset
from provenas.losses import masked_multihead_loss


def check_oracle():
    assert run_op("divide", 1, 0) == (None, "ZeroDivisionError")
    assert run_op("multiply", 1e200, 1e200) == (None, "OverflowError")
    assert run_op("add", 1e308, 1e308) == (None, "OverflowError")
    assert run_op("add", 2, 3)[0] == 5.0
    c = Calculator()
    assert c.evaluate("(3 + 4) * 2 - 5 / (1 + 1)") == (11.5, "ok")
    assert c.evaluate("1 / 0")[1] == "ZeroDivisionError"
    try:
        c.divide(1, 0)
        assert False, "expected ZeroDivisionError"
    except ZeroDivisionError:
        pass
    print("oracle OK")


def check_encoders():
    y = np.array([0.0, 1.0, -1234.5, 1e8, -1e-3, 12345678.0])
    for path in ("A", "B"):
        enc = E.PATHS[path]
        back = enc["decode_target"](enc["encode_target"](y))
        assert np.allclose(back, y, rtol=1e-5, atol=1e-6), (path, back, y)
    X = E.encode_inputs_raw(np.array([1.0, 2.0]), np.array([3.0, 4.0]), np.array([0, 1]))
    assert X.shape == (2, E.IN_DIM), X.shape
    assert E.encode_inputs_log(np.array([1.0]), np.array([0.0]), np.array([3])).shape == (1, E.IN_DIM)
    print("encoders OK")


def check_dataset():
    rng = np.random.default_rng(0)
    d = dataset.build(5000, rng, ops=dataset.REG_OPS,
                      frac_zero_div=0.15, frac_overflow=0.15)
    err = d["error"]
    assert set(np.unique(err)).issubset({0, 1, 2})
    assert (err == 1).sum() > 0 and (err == 2).sum() > 0, "missing injected errors"
    # error rows have NaN result; ok rows are finite
    assert np.isnan(d["result"][err != 0]).all()
    assert np.isfinite(d["result"][err == 0]).all()
    print(f"dataset OK  (ok={int((err==0).sum())} zerodiv={int((err==1).sum())} overflow={int((err==2).sum())})")


def check_loss():
    torch.manual_seed(0)
    reg = torch.randn(8, 1, requires_grad=True)
    tgt = torch.randn(8, 1)
    tgt[4:] = float("nan")            # error rows carry NaN targets
    tgt = torch.nan_to_num(tgt, nan=0.0)
    err = torch.tensor([0, 0, 0, 0, 1, 1, 2, 2])
    logits = torch.randn(8, 3, requires_grad=True)
    loss, rl, el = masked_multihead_loss(reg, tgt, logits, err)
    loss.backward()
    assert torch.isfinite(loss), loss
    assert torch.allclose(reg.grad[4:], torch.zeros(4, 1), atol=1e-7), reg.grad
    assert not torch.allclose(reg.grad[:4], torch.zeros(4, 1)), "ok rows should have grad"
    print("loss OK")


def check_exprgen():
    from collections import Counter
    from provenas import exprgen as G
    rng = np.random.default_rng(0)
    samples = G.build({1: 1500, 2: 1500, 3: 1500, 4: 1500, 5: 800, 6: 800, 7: 800}, rng)
    err = Counter(int(s.error) for s in samples)
    depths = Counter(s.depth for s in samples)
    assert err.get(1, 0) > 0, "no ZeroDivisionError samples generated"
    for d in (1, 2, 3, 4, 5, 6, 7):
        assert depths[d] > 0, f"no depth-{d} samples"
    maxlen = max(len(s.string) for s in samples)
    print(f"exprgen OK  n={len(samples)} err={dict(err)} maxlen={maxlen}")
    for d in (1, 4, 7):
        ex = next(s for s in samples if s.depth == d)
        print(f"    d{d}: {ex.string!r} = {ex.value} (err={ex.error})")


def check_tree_lstm():
    from provenas import exprgen as G
    from provenas.models_tree import BinaryTreeLSTM
    rng = np.random.default_rng(1)
    trees = [G.make_sample(d, rng).tree for d in (0, 1, 2, 3, 4, 5)]
    m = BinaryTreeLSTM(h_dim=32).eval()
    with torch.no_grad():
        r1, e1 = m(trees, "cpu")                 # depth-batched
        r2, e2 = m.forward_recursive(trees, "cpu")
    assert torch.allclose(r1, r2, atol=1e-5), (r1, r2)
    assert torch.allclose(e1, e2, atol=1e-5)
    print("tree-lstm batched==recursive OK")


def check_stackvm():
    from provenas import exprgen as G
    from provenas.stackvm import slot_tokens, execute_rpn
    rng = np.random.default_rng(0)
    ok = bad = 0
    for s in G.build({1: 200, 2: 200, 3: 200, 4: 200, 5: 200}, rng, seen=set()):
        if s.error != 0:
            continue
        _, rpn, vals = slot_tokens(s.tree)
        pv, st = execute_rpn(rpn, vals)
        if st == "ok" and abs(pv - s.value) <= 1e-9 * max(1.0, abs(s.value)):
            ok += 1
        else:
            bad += 1
    assert bad == 0, f"stackvm mismatches: {bad}"
    print(f"stackvm OK (ground-truth RPN reproduces {ok} values exactly)")


def check_reducer():
    from provenas import exprgen as G
    from provenas.reducer import tree_to_state, reduce_to_value, next_reduction
    rng = np.random.default_rng(0)
    ok = bad = 0
    for s in G.build({1: 200, 2: 200, 3: 200, 4: 200, 5: 200}, rng, seen=set()):
        if s.error != 0:
            continue
        types, vals = tree_to_state(s.tree)
        v, st, _ = reduce_to_value(types, vals, next_reduction)
        if st == "ok" and abs(v - s.value) <= 1e-9 * max(1.0, abs(s.value)):
            ok += 1
        else:
            bad += 1
    assert bad == 0, f"reducer mismatches: {bad}"
    print(f"reducer OK (oracle reduction reproduces {ok} values exactly)")


def check_kg():
    from provenas.kg import KnowledgeGraph
    kg = KnowledgeGraph()
    for t in [("dog", "is_a", "mammal"), ("cat", "is_a", "mammal"),
              ("mammal", "is_a", "animal"), ("dog", "has", "fur"), ("cat", "has", "fur")]:
        kg.assert_(*t)
    assert sorted(b["?x"] for b in kg.query(("?x", "is_a", "mammal"))) == ["cat", "dog"]
    assert kg.ancestors("dog") == {"mammal", "animal"}          # transitive inference
    assert ("has", "fur") in kg.discover_properties({"dog", "cat"})["common_facts"]
    kg.retract("cat", "has", "fur")                              # disassociate
    assert ("has", "fur") not in kg.discover_properties({"dog", "cat"})["common_facts"]
    print("kg OK (assert/retract, query, inference, discovery all exact)")


def _animals_kg():
    from provenas.kg import KnowledgeGraph
    kg = KnowledgeGraph()
    for t in [("dog", "is_a", "mammal"), ("cat", "is_a", "mammal"), ("whale", "is_a", "mammal"),
              ("shark", "is_a", "fish"), ("mammal", "is_a", "vertebrate"), ("fish", "is_a", "vertebrate"),
              ("vertebrate", "has_part", "backbone"), ("dog", "lives_in", "land"),
              ("whale", "lives_in", "water"), ("shark", "lives_in", "water")]:
        kg.assert_(*t)
    return kg


def check_kgvm():
    from provenas.kgvm import KGMachine
    kg = _animals_kg()
    # all entities that are animals-of-type vertebrate (reverse closure)
    verts = KGMachine(kg).select("vertebrate").descendants("is_a").result()
    assert {"mammal", "fish", "dog", "cat", "whale", "shark"} == verts, verts
    # combination (the aha): vertebrates that live in water  ->  whale, shark
    water = KGMachine(kg).subjects_with("lives_in", "water").result()
    aquatic_verts = KGMachine(kg).select("vertebrate").descendants("is_a").intersect(water).result()
    assert aquatic_verts == {"whale", "shark"}, aquatic_verts
    print("kgvm OK (traverse, reverse-closure, intersect-combine exact)")


def check_infer():
    from provenas.infer import Rule, forward_chain, paths_between, induce_path_rule
    kg = _animals_kg()
    rules = [
        Rule([("?x", "is_a", "?y"), ("?y", "is_a", "?z")], ("?x", "is_a", "?z"), "is_a transitive"),
        Rule([("?x", "is_a", "?y"), ("?y", "has_part", "?z")], ("?x", "has_part", "?z"), "inherit part"),
    ]
    derived = forward_chain(kg, rules)
    assert ("whale", "is_a", "vertebrate") in derived              # transitivity
    assert ("whale", "has_part", "backbone") in derived           # inherited property
    paths = paths_between(kg, "whale", "vertebrate", max_len=2)
    assert any(p[-1][1] == "vertebrate" for p in paths)           # a connecting chain exists
    # discover the rule "grandtype = is_a then is_a" from positives:
    rule = induce_path_rule(kg, [("dog", "vertebrate"), ("shark", "vertebrate")])
    assert ("is_a", "is_a") in rule, rule
    print("infer OK (forward-chain derive, path-find, rule-induce exact)")


def check_typed():
    from provenas import typed as T
    rng = np.random.default_rng(0)
    ok = 0
    for _ in range(4000):
        want = "num" if rng.random() < 0.6 else "bool"
        depth = int(rng.integers(0, 7))
        n = T.gen(depth, want, rng)
        v, ty, _ = T.reduce_to_value(n)
        assert v == T.eval_node(n), (T.render(n), v, T.eval_node(n))   # reducer == oracle
        assert ty == T.type_of(n)                                      # type tracked
        ok += 1
    # short-circuit: the dead branch (1/0) is never evaluated
    sc = ("if", ("bin", "==", ("lit", 1, "num"), ("lit", 0, "num")),
          ("bin", "/", ("lit", 1, "num"), ("lit", 0, "num")), ("lit", 7, "num"))
    v, ty, _ = T.reduce_to_value(sc)
    assert v == 7 and ty == "num", (v, ty)
    # the structure-only stream carries token TYPES, never values
    _, classes, _ = T.linearize(("bin", "+", ("lit", 3, "num"), ("lit", 4, "num")))
    assert classes == ["NUM", "+", "NUM"], classes
    print(f"typed OK (reducer==oracle on {ok} mixed-type exprs; short-circuit if/else safe)")


def check_store():
    from provenas.store import Store
    from provenas.infer import Rule, forward_chain_prov, explain
    st = Store(":memory:")
    st.assert_("a", "parent", "b").assert_("b", "parent", "c")
    st.add_rule(Rule([("?x", "parent", "?y"), ("?y", "parent", "?z")], ("?x", "grandparent", "?z"), "gp"))
    assert st.counts() == (2, 1)
    kg = st.to_kg()                                          # rules round-trip SQLite(json) and fire
    derived, prov = forward_chain_prov(kg, st.rules())
    assert ("a", "grandparent", "c") in derived
    proof = "\n".join(explain(("a", "grandparent", "c"), prov))
    assert "gp" in proof and "(a parent b)" in proof        # provenance proof back to base facts
    print("store OK (sqlite persist + rule round-trip + provenance proof)")


def check_learn():
    from provenas.kg import KnowledgeGraph
    from provenas.infer import Rule
    from provenas.learn import validate_rule
    kg = KnowledgeGraph()
    for t in [("p", "parent", "a"), ("p", "parent", "b")]:
        kg.assert_(*t)
    naive = Rule([("?p", "parent", "?x"), ("?p", "parent", "?y")], ("?x", "sibling", "?y"), "naive")
    guard = Rule([("?p", "parent", "?x"), ("?p", "parent", "?y"), ("?x", "!=", "?y")],
                 ("?x", "sibling", "?y"), "guarded")
    pos, neg = [("a", "b")], [("a", "a"), ("b", "b")]            # self-sibling is a negative
    ok_n, _ = validate_rule(kg, [naive], "sibling", pos, neg)
    ok_g, _ = validate_rule(kg, [guard], "sibling", pos, neg)
    assert not ok_n, "naive rule should be REJECTED (over-generates self-siblings)"
    assert ok_g, "guarded rule should be ADMITTED"
    print("learn OK (!= guard works; gate rejects over-generating rule, admits guarded)")


def check_rewrite():
    from provenas import rewrite as RW
    t = ("op", "*", ("op", "+", ("var", "x"), ("num", 0)), ("op", "+", ("num", 2), ("num", 3)))
    nf1 = RW.normal_form(t, RW.R1)                          # identities only: (2+3) stays
    nf2 = RW.normal_form(t, RW.R2)                          # + folding: -> x * 5
    assert nf1 == ("op", "*", ("var", "x"), ("op", "+", ("num", 2), ("num", 3))), nf1
    assert nf2 == ("op", "*", ("var", "x"), ("num", 5)), nf2
    assign = {"x": 7, "y": 1, "z": 2}                       # rewriting is value-preserving
    assert RW.evaluate(t, assign) == RW.evaluate(nf2, assign) == 35
    assert RW.sound(RW.FOLD, np.random.default_rng(0))      # soundness gate
    assert not RW.sound(RW.BOGUS, np.random.default_rng(0))
    print("rewrite OK (normal-form exact, value-preserving, soundness gate rejects unsound rule)")


def check_toolsmith():
    from provenas.toolsmith import validate_ast, run_sandboxed, load_tool, admit_tool
    good = "def gcd(a, b):\n    while b:\n        a, b = b, a % b\n    return a\n"
    assert validate_ast(good, "gcd")[0]
    ok, out = run_sandboxed(good, "gcd", [[12, 8], [7, 5], [0, 9]])
    assert ok and out == [4, 1, 9], (ok, out)
    assert load_tool(good, "gcd")(48, 36) == 12                 # validated tool is callable
    assert not validate_ast("def f(n):\n import os\n return 1\n", "f")[0]          # import blocked
    assert not validate_ast("def f(x):\n return x.__class__\n", "f")[0]            # attribute blocked
    assert not validate_ast("def f(x):\n return open('/etc/passwd')\n", "f")[0]    # forbidden name
    okb, stage, _ = admit_tool(None, "gcd", "def gcd(a,b):\n return a+b\n",
                               [((12, 8), 4)])                  # wrong output
    assert not okb and stage == "tests", (okb, stage)
    print("toolsmith OK (AST blocks import/attr/forbidden; sandbox runs safe tool; tests reject wrong)")


def check_qa():
    from provenas.store import Store
    from provenas import qa, domains
    st = Store(":memory:")
    domains.load(st, "rbac")                                 # shared domain pack seeds facts + rules
    r = qa.run_action(st, {"action": "query", "pattern": ["?x", "can", "view_wiki"]})
    assert set(r["answer"]) == {"alice", "bob", "dana"}, r
    r = qa.run_action(st, {"action": "check", "triple": ["alice", "can", "prod_deploy"]})
    assert r["answer"] is True and "can-do" in r["trace"]    # proof references the rule
    assert qa.run_action(st, {"action": "check", "triple": ["dana", "can", "query_db"]})["answer"] is False
    qa.run_action(st, {"action": "assert", "triple": ["dana", "has_role", "analyst"]})
    assert qa.run_action(st, {"action": "check", "triple": ["dana", "can", "query_db"]})["answer"] is True
    ctx = qa.context_from_store(st)                          # domain-agnostic context for the CLI
    assert "has_role" in ctx and "alice" in ctx
    print("qa OK (run_action query/check/proof/assert; domain load + store-derived context)")


def check_domains():
    from provenas.store import Store
    from provenas import qa, domains
    for d in domains.DOMAINS:                                # every pack's gold actions must match expectations
        st = Store(":memory:")
        domains.load(st, d["name"])
        for q in d["questions"]:
            r = qa.run_action(st, dict(q["gold"]))
            got = set(r["answer"]) if isinstance(q["expect"], set) else r["answer"]
            assert got == q["expect"], (d["name"], q["q"], got, q["expect"])
    print(f"domains OK ({len(domains.DOMAINS)} packs; all gold actions match expected answers)")


def check_rewrite_store():
    from provenas.store import Store
    from provenas import rewrite as RW
    st = Store(":memory:")
    for name, rule in RW.DEFAULTS:
        st.add_rewrite(name, rule, source="default")
    rules = st.rewrites()                                    # rewrite rules round-trip SQLite (json -> tuples)
    assert len(rules) == len(RW.DEFAULTS)
    t = RW.parse("(x + 0) * (2 + 3)")                        # parser + exact simplify against stored rules
    assert RW.normal_form(t, rules) == ("op", "*", ("var", "x"), ("num", 5)), RW.normal_form(t, rules)
    print("rewrite-store OK (rewrite rules persist in SQLite; parse + simplify exact)")


if __name__ == "__main__":
    check_oracle()
    check_encoders()
    check_dataset()
    check_loss()
    check_exprgen()
    check_tree_lstm()
    check_stackvm()
    check_reducer()
    check_kg()
    check_kgvm()
    check_infer()
    check_typed()
    check_store()
    check_learn()
    check_rewrite()
    check_toolsmith()
    check_qa()
    check_domains()
    check_rewrite_store()
    print("ALL SANITY OK")
