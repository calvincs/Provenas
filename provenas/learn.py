"""learn — Slice 2: admit a proposed RULE only after it passes a test against the knowledge base.

The governance spine applied to rules: a proposer (Qwen, or the symbolic `induce_path_rule`) PROPOSES a
rule; the fabric VERIFIES it before it is allowed to change anything. A rule is admitted only if, when
added to the current KB and forward-chained, it:
  - derives every POSITIVE example (it does its job),
  - derives NO NEGATIVE example (it doesn't over-generate / contradict known non-facts),
  - does not explode the fact set (a cheap termination/runaway guard).
Only then is it committed to SQLite with provenance. Rejected rules never touch the store.
"""
from __future__ import annotations

from provenas.infer import forward_chain


def validate_rule(base_kg, rules, head_rel, positives, negatives, fact_cap=2000, max_iter=50):
    """Test `rules` (which should include the candidate) against examples on a COPY of base_kg."""
    kg = base_kg.copy()
    forward_chain(kg, rules, max_iter=max_iter)
    facts = len(kg.triples)
    explosive = facts > fact_cap
    missing = [p for p in positives if (p[0], head_rel, p[1]) not in kg.triples]
    violated = [n for n in negatives if (n[0], head_rel, n[1]) in kg.triples]
    ok = not missing and not violated and not explosive
    return ok, dict(missing=missing, violated=violated, facts=facts, explosive=explosive)


def admit_rule(store, rule, head_rel, positives, negatives, source="qwen"):
    """Validate `rule` against the store's current KB+rules; commit to SQLite only if it passes."""
    kg = store.to_kg()
    candidate_rules = store.rules() + [rule]
    ok, report = validate_rule(kg, candidate_rules, head_rel, positives, negatives)
    detail = (f"{rule.name} rel={head_rel} facts={report['facts']} "
              f"miss={len(report['missing'])} viol={len(report['violated'])} explosive={report['explosive']}")
    if ok:
        store.add_rule(rule, source=source)
        store.log("admit_rule", detail)
    else:
        store.log("reject_rule", detail)
    return ok, report
