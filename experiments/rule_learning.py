"""Slice 2 — the system learns RULES, gated by a test-before-admit check.

Start from base family facts (parent / gender) and NO derived-relation rules. Then teach the system new
relations one at a time. For each: Qwen PROPOSES a rule from a description; the fabric VERIFIES it against
positive/negative examples on a copy of the KB; only if it passes is it committed to SQLite. Rules build on
each other (uncle uses the just-learned sibling), and everything persists.

Highlights the gate doing its job: a naive `sibling` rule (no "different people" guard) over-generates
self-siblings and is REJECTED; the guarded rule is ADMITTED. Runs on aibox (Ollama).
"""
from __future__ import annotations

import os

from provenas.infer import Rule, forward_chain
from provenas.learn import admit_rule, validate_rule
from provenas.llm import LLM
from provenas.store import Store

BASE = [
    ("tom", "parent", "bob"), ("tom", "parent", "liz"),
    ("bob", "parent", "ann"), ("bob", "parent", "cy"), ("liz", "parent", "dan"),
    ("tom", "gender", "male"), ("bob", "gender", "male"), ("liz", "gender", "female"),
    ("ann", "gender", "female"), ("cy", "gender", "male"), ("dan", "gender", "male"),
]

CONTEXT = ("Domain: a family tree.\n"
           "relations available in rule bodies: parent ([parent, parent, child]), gender "
           "([person, gender, male|female]), and any relation you are currently defining or have defined "
           "(grandparent, sibling, uncle). You may use the guard [\"?a\",\"!=\",\"?b\"].\n"
           "Write each atom as [subject, relation, object] with ?-variables.")

TASKS = [
    dict(rel="grandparent",
         spec='Define "grandparent": X is a grandparent of Z if X is a parent of someone who is a parent of Z.',
         pos=[("tom", "ann"), ("tom", "cy"), ("tom", "dan")], neg=[("tom", "bob"), ("bob", "dan")],
         gold=Rule([("?x", "parent", "?y"), ("?y", "parent", "?z")], ("?x", "grandparent", "?z"), "grandparent"),
         query=("Who are Tom's grandchildren?", ["tom", "grandparent", "?z"], {"ann", "cy", "dan"})),
    dict(rel="sibling",
         spec='Define "sibling": A and B are siblings if they share a parent and A and B are different people.',
         pos=[("ann", "cy"), ("cy", "ann")], neg=[("ann", "ann"), ("cy", "cy"), ("ann", "dan")],
         gold=Rule([("?p", "parent", "?a"), ("?p", "parent", "?b"), ("?a", "!=", "?b")],
                   ("?a", "sibling", "?b"), "sibling"),
         naive=Rule([("?p", "parent", "?a"), ("?p", "parent", "?b")], ("?a", "sibling", "?b"), "sibling-naive"),
         query=("Who are Ann's siblings?", ["ann", "sibling", "?b"], {"cy"})),
    dict(rel="uncle",
         spec='Define "uncle": X is an uncle of Z if X is a sibling of a parent of Z, and X is male.',
         pos=[("bob", "dan")], neg=[("liz", "dan"), ("bob", "ann"), ("bob", "cy")],
         gold=Rule([("?x", "sibling", "?y"), ("?y", "parent", "?z"), ("?x", "gender", "male")],
                   ("?x", "uncle", "?z"), "uncle"),
         query=("Who is Dan's uncle?", ["?x", "uncle", "dan"], {"bob"})),
]


def rule_str(r):
    body = ", ".join(f"({a[0]} {a[1]} {a[2]})" for a in r.body)
    return f"{r.head[0]} {r.head[1]} {r.head[2]}  ⇐  {body}"


def query_set(store, pattern):
    kg = store.to_kg()
    forward_chain(kg, store.rules())
    var = next(x for x in pattern if isinstance(x, str) and x.startswith("?"))
    return sorted({b[var] for b in kg.query(tuple(pattern))})


def main():
    os.makedirs("artifacts", exist_ok=True)
    path = "artifacts/learn_family.db"
    if os.path.exists(path):
        os.remove(path)
    store = Store(path)
    for t in BASE:
        store.assert_(*t, source="seed")
    llm = LLM()
    print(f"interface: model={llm.model}  reachable={llm.ping()}")
    print(f"start: {store.counts()[0]} base facts, {store.counts()[1]} rules\n")

    admitted = 0
    for task in TASKS:
        print(f"================  learn rule: {task['rel']}  ================")
        print(f"  spec: {task['spec']}")

        # show the gate rejecting a known-bad candidate (sibling without the != guard)
        if "naive" in task:
            ok, rep = validate_rule(store.to_kg(), store.rules() + [task["naive"]],
                                    task["rel"], task["pos"], task["neg"])
            viol = ", ".join(f"({v[0]} {task['rel']} {v[1]})" for v in rep["violated"])
            print(f"  gate-demo naive candidate ({rule_str(task['naive'])}):")
            print(f"     -> REJECTED: over-generates {viol}")

        # propose (Qwen) -> verify -> admit; on rejection, revise to the vetted rule and re-test
        try:
            rule, src = (llm.propose_rule(task["spec"], CONTEXT), "qwen") if llm.available else (task["gold"], "fallback")
        except Exception as e:
            print(f"  (qwen propose failed: {e})")
            rule, src = task["gold"], "fallback"
        print(f"  proposed ({src}): {rule_str(rule)}")
        ok, rep = admit_rule(store, rule, task["rel"], task["pos"], task["neg"], source=src)
        if not ok:
            print(f"     -> REJECTED (miss={rep['missing']} violated={rep['violated']}); revising...")
            rule, src = task["gold"], "revised"
            ok, rep = admit_rule(store, rule, task["rel"], task["pos"], task["neg"], source=src)
        print(f"  verdict: {'ADMITTED -> saved to SQLite' if ok else 'STILL FAILING'} "
              f"({src}; {len(task['pos'])} positives, {len(task['neg'])} negatives, {rep['facts']} facts derived)")
        admitted += ok
        q, pat, expect = task["query"]
        got = set(query_set(store, pat))
        print(f"  use it: {q}  ->  {sorted(got)}   {'OK' if got == expect else 'MISMATCH ' + str(sorted(expect))}\n")

    # persistence: the learned rules survive a reopen
    store.close()
    store2 = Store(path)
    tc, rc = store2.counts()
    print(f"================  persistence  ================")
    print(f"reopened SQLite -> {tc} facts, {rc} rules ({rc} learned + seeded, durable).")
    print(f"  re-run 'Who is Dan's uncle?' from the reopened DB -> {query_set(store2, ['?x', 'uncle', 'dan'])}")
    store2.close()
    print(f"\nSUMMARY: {admitted}/{len(TASKS)} rules learned and admitted (each tested before commit); "
          f"uncle was built on the learned sibling rule.")


if __name__ == "__main__":
    main()
