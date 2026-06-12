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

from provenas import qa
from provenas.infer import forward_chain, forward_chain_prov


def validate_rule(base_kg, rules, head_rel, positives, negatives, fact_cap=2000, max_iter=50):
    """Test `rules` (which should include the candidate) against examples on a COPY of base_kg."""
    kg = base_kg.copy()
    try:
        forward_chain(kg, rules, max_iter=max_iter)
    except ValueError as e:                                    # e.g. the candidate breaks stratification
        return False, dict(missing=list(positives), violated=[], facts=len(kg.triples),
                           explosive=False, error=str(e))
    facts = len(kg.triples)
    explosive = facts > fact_cap
    missing = [p for p in positives if (p[0], head_rel, p[1]) not in kg.triples]
    violated = [n for n in negatives if (n[0], head_rel, n[1]) in kg.triples]
    ok = not missing and not violated and not explosive
    return ok, dict(missing=missing, violated=violated, facts=facts, explosive=explosive)


def replay_cases(store, rules):
    """Re-answer the store's pinned regression cases under a candidate ruleset (on a KB copy);
    return the names of cases whose answer would CHANGE. The rule-admission analog of CI."""
    cases = store.cases()
    if not cases:
        return []
    kg = store.to_kg()
    try:
        _, prov = forward_chain_prov(kg, rules)
    except ValueError:                                         # candidate breaks the ruleset entirely
        return [name for name, _, _ in cases]
    return [name for name, action, expect in cases
            if qa.eval_action(kg, prov, action)["answer"] != expect]


def admit_rule(store, rule, head_rel, positives, negatives, source="qwen"):
    """Validate `rule` against the store's current KB+rules; commit to SQLite only if it passes
    BOTH the ± examples and the store's pinned regression cases (no previously pinned decision
    may flip)."""
    kg = store.to_kg()
    candidate_rules = store.rules() + [rule]
    ok, report = validate_rule(kg, candidate_rules, head_rel, positives, negatives)
    report["case_flips"] = replay_cases(store, candidate_rules) if ok else []
    if report["case_flips"]:
        ok = False
    detail = (f"{rule.name} rel={head_rel} facts={report['facts']} "
              f"miss={len(report['missing'])} viol={len(report['violated'])} explosive={report['explosive']}"
              + (f" case_flips={report['case_flips']}" if report["case_flips"] else ""))
    if ok:
        store.add_rule(rule, source=source)
        store.log("admit_rule", detail)
    else:
        store.log("reject_rule", detail)
    return ok, report
