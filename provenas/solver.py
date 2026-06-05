"""solver — the combination problem-solver: the "aha" of putting ideas together.

A goal is a conjunction of constraints. No single mechanism answers them all — the solver has a
toolbox and discovers, by trial and error, the *smallest combination* of tools that reaches the
answer:

  - (base) direct lookup / taxonomy traversal      stored facts only
  - infer      run the rule engine (is_a transitivity, property inheritance) over a KG copy first
  - combine    intersect the per-constraint sets (KGMachine relational algebra)
  - semantic   resolve a "like X" constraint with the neural embedding tool

Constraints:
  ("type", C)        entities that are C (is_a* C, by traversal)
  ("prop", R, O)     entities with relation (e, R, O) — INHERITED props need `infer`
  ("like", X)        entities semantically like X — needs `semantic`

`solve` returns the minimal tool-set, its size (the "combination depth"), and how many subsets it
had to try (the trial-and-error count a learned controller later short-circuits). `features` exposes
a goal to that controller.
"""
from __future__ import annotations

from itertools import combinations

from provenas.infer import forward_chain
from provenas.kgvm import KGMachine

ALL_TOOLS = ("infer", "combine", "semantic")


def _resolve(c, kg, semantic, use_sem):
    if c[0] == "type":
        return KGMachine(kg).select(c[1]).descendants("is_a").result()
    if c[0] == "prop":
        return set(kg.group_by_relation(c[1], c[2]))
    if c[0] == "like":
        if not use_sem:
            return None
        return set(semantic.similar(c[1], k=3)) | {c[1]}
    raise ValueError(c)


def answer(goal, base_kg, rules, semantic, tools):
    """Answer a goal using exactly the given tool-set; None if a needed tool is missing."""
    kg = base_kg.copy()
    if "infer" in tools:
        forward_chain(kg, rules)
    sets = []
    for c in goal:
        s = _resolve(c, kg, semantic, "semantic" in tools)
        if s is None:
            return None
        sets.append(s)
    if len(sets) > 1 and "combine" not in tools:
        return None
    res = set(sets[0])
    for s in sets[1:]:
        res &= s
    return res


def solve(goal, base_kg, rules, semantic):
    """Find the smallest tool combination that reaches the full-toolbox answer."""
    gold = answer(goal, base_kg, rules, semantic, ALL_TOOLS)
    attempts = 0
    for k in range(len(ALL_TOOLS) + 1):
        for subset in combinations(ALL_TOOLS, k):
            attempts += 1
            res = answer(goal, base_kg, rules, semantic, subset)
            if res is not None and res == gold:
                return dict(tools=subset, depth=k, attempts=attempts, answer=gold)
    return dict(tools=ALL_TOOLS, depth=len(ALL_TOOLS), attempts=attempts, answer=gold)


def features(goal, rel_vocab):
    """Goal -> fixed-width feature vector for the neural tool-controller."""
    rels = [c[1] for c in goal if c[0] == "prop"]
    base = [
        len(goal),
        sum(c[0] == "type" for c in goal),
        sum(c[0] == "prop" for c in goal),
        sum(c[0] == "like" for c in goal),
    ]
    return base + [float(r in rels) for r in rel_vocab]
