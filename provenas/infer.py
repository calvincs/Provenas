"""infer — a small symbolic inference engine over the knowledge graph.

Three capabilities the reasoner needs:
  - forward_chain(kg, rules)   deductive closure: fire Horn-style rules to a fixpoint, deriving
                               new facts (e.g. is_a transitivity, property inheritance).
  - paths_between(kg, a, b)    relation-composition: what chain of relations connects a -> b
                               (the seed of "what do these two have in common / how are they linked").
  - induce_path_rule(kg, pos)  rule discovery (ILP-lite): find a 2-hop relation pattern that holds
                               for all positive (subject, object) pairs — a learned rule.

Rules use "?var" pattern variables, the same convention as KnowledgeGraph.query.

Body atoms come in three kinds:
  (s, rel, o)          a fact lookup (joined against the KG, index-accelerated)
  (a, GUARD, b)        a comparison guard: != == < <= > >=  (numeric when both sides are numbers)
  (s, ~rel, o)         stratified NEGATION: holds iff NO fact matches the pattern (NOT EXISTS)

Evaluation is stratified (rules whose heads feed a negation are completed first; a negation
through a recursive cycle raises ValueError) and semi-naive (each fixpoint round joins only
against the facts derived in the previous round).
"""
from __future__ import annotations

import operator
from collections import deque

GUARDS = {"!=", "==", "<", "<=", ">", ">="}
_CMP = {"<": operator.lt, "<=": operator.le, ">": operator.gt, ">=": operator.ge}


class Rule:
    def __init__(self, body, head, name=""):
        self.body = body        # list of (s, r, o) patterns with ?vars
        self.head = head        # (s, r, o) pattern with ?vars
        self.name = name

    def __repr__(self):
        return f"Rule({self.name or self.head})"


def _isvar(x):
    return isinstance(x, str) and x.startswith("?")


def _bind(pattern, triple, binding):
    b = dict(binding)
    for pat, val in zip(pattern, triple):
        if _isvar(pat):
            if pat in b and b[pat] != val:
                return None
            b[pat] = val
        elif pat != val:
            return None
    return b


def _matches(kg, pattern, binding):
    resolved = tuple(binding.get(x, x) if _isvar(x) else x for x in pattern)
    for triple in kg.candidates(resolved):          # index-narrowed; _bind does the final check
        b = _bind(pattern, triple, binding)
        if b is not None:
            yield b


def _matches_in(facts, pattern, binding):
    for triple in facts:                            # small delta set — a scan is fine
        b = _bind(pattern, triple, binding)
        if b is not None:
            yield b


def _is_lookup(atom):
    return atom[1] not in GUARDS and not atom[1].startswith("~")


def _guard_ok(b, atom):
    """A comparison guard over a variable the body never binds FAILS (it cannot be checked,
    so the binding is rejected rather than vacuously passed)."""
    l, r = b.get(atom[0], atom[0]), b.get(atom[2], atom[2])
    if _isvar(l) or _isvar(r):
        return False
    if atom[1] == "!=":
        return l != r
    if atom[1] == "==":
        return l == r
    try:
        return _CMP[atom[1]](float(l), float(r))    # numeric when both sides are numbers
    except (TypeError, ValueError):
        return _CMP[atom[1]](l, r)                  # else lexicographic (ISO dates order correctly)


def _neg_ok(kg, b, atom):
    """Negated atom (s, '~rel', o): holds iff NO fact matches the (partially bound) pattern.
    A variable still unbound here acts as a wildcard — NOT EXISTS semantics."""
    pat = (b.get(atom[0], atom[0]), atom[1][1:], b.get(atom[2], atom[2]))
    return not kg.query(pat)


def _solve_body(kg, body, init=None, delta=None):
    """Bindings satisfying all body atoms: positive lookups joined in body order against the
    indexed KG, then negations, then guards (both applied once the lookups have bound their
    variables, wherever they appear in the body).
    With `delta` (semi-naive evaluation) only bindings that use at least one delta fact in
    some positive atom are produced — callers join each round against last round's new facts."""
    pos = [a for a in body if _is_lookup(a)]
    bindings = []
    delta_rels = {r for _, r, _ in delta} if delta else set()
    for seed in (range(len(pos)) if delta is not None else [None]):
        if seed is not None and not _isvar(pos[seed][1]) and pos[seed][1] not in delta_rels:
            continue                                  # this atom's relation has no delta facts
        # the delta atom is joined FIRST so its (few) bindings restrict the indexed joins after it
        order = pos if seed is None else [pos[seed]] + pos[:seed] + pos[seed + 1:]
        bs = [dict(init) if init else {}]
        for i, atom in enumerate(order):
            if seed is not None and i == 0:
                bs = [b2 for b in bs for b2 in _matches_in(delta, atom, b)]
            else:
                bs = [b2 for b in bs for b2 in _matches(kg, atom, b)]
        bindings.extend(bs)
    for atom in body:
        if atom[1] in GUARDS:
            bindings = [b for b in bindings if _guard_ok(b, atom)]
        elif not _is_lookup(atom):
            bindings = [b for b in bindings if _neg_ok(kg, b, atom)]
    return bindings


def fire(kg, rule):
    """All ground head instantiations derivable from `rule` in one pass."""
    out = set()
    for b in _solve_body(kg, rule.body):
        h = tuple(b.get(x, x) for x in rule.head)
        if any(_isvar(x) for x in h):
            continue                                   # unbound head var → not ground
        out.add(h)
    return out


def stratify(rules):
    """Order rules into strata so every negated relation is fully derived in an earlier
    stratum. Raises ValueError when impossible (negation through a recursive cycle)."""
    strata, deps = {}, []
    for rule in rules:
        strata.setdefault(rule.head[1], 0)
        for a in rule.body:
            if a[1] in GUARDS:
                continue
            neg = a[1].startswith("~")
            br = a[1][1:] if neg else a[1]
            strata.setdefault(br, 0)
            deps.append((rule.head[1], br, neg))
    for _ in range(len(strata) + 1):
        changed = False
        for h, br, neg in deps:
            need = strata[br] + (1 if neg else 0)
            if strata[h] < need:
                strata[h], changed = need, True
        if not changed:
            groups = {}
            for rule in rules:
                groups.setdefault(strata[rule.head[1]], []).append(rule)
            return [groups[k] for k in sorted(groups)]
    raise ValueError("ruleset is not stratified: negation through a recursive cycle")


def _chain(kg, rules, max_iter, prov):
    """Semi-naive fixpoint for one stratum; asserts into kg, returns the newly derived set."""
    derived, delta = set(), None
    for _ in range(max_iter):
        new = {}
        for rule in rules:
            for b in _solve_body(kg, rule.body, delta=delta):
                head = tuple(b.get(x, x) for x in rule.head)
                if any(_isvar(x) for x in head) or head in kg.triples or head in new:
                    continue
                new[head] = None if prov is None else \
                    (rule.name, [tuple(b.get(x, x) for x in pat) for pat in rule.body if _is_lookup(pat)])
        if not new:
            break
        for h in new:
            kg.assert_(*h)
        if prov is not None:
            prov.update(new)
        derived |= set(new)
        delta = set(new)
    return derived


def forward_chain(kg, rules, max_iter=1000):
    """Assert all derivable facts into `kg` to a fixpoint (stratified, semi-naive); return the
    newly derived set. Raises ValueError if the ruleset is not stratified."""
    derived = set()
    for group in stratify(rules):
        derived |= _chain(kg, group, max_iter, None)
    return derived


def forward_chain_prov(kg, rules, max_iter=1000):
    """Like forward_chain, but record provenance: derived_fact -> (rule_name, [premise_facts])."""
    prov = {}
    for group in stratify(rules):
        _chain(kg, group, max_iter, prov)
    return set(prov.keys()), prov


def explain(target, prov, indent=0):
    """Render a proof tree for a (derived or base) ground fact, back to base facts."""
    pad = "  " * indent
    if target in prov:
        rule, premises = prov[target]
        lines = [f"{pad}{_fact(target)}   ⇐ rule[{rule}]"]
        for pr in premises:
            lines += explain(pr, prov, indent + 1)
        return lines
    return [f"{pad}{_fact(target)}   (given fact)"]


def _fact(t):
    return f"({t[0]} {t[1]} {t[2]})"


def paths_between(kg, a, b, max_len=3, max_paths=8):
    """Relation paths a -> ... -> b, each a list of (relation, next_node) steps."""
    out, q = [], deque([(a, [], {a})])
    while q and len(out) < max_paths:
        node, path, seen = q.popleft()
        if len(path) >= max_len:
            continue
        for s, r, o in kg.triples:
            if s != node or o in seen:
                continue
            step = path + [(r, o)]
            if o == b:
                out.append(step)
            else:
                q.append((o, step, seen | {o}))
    return out


def induce_path_rule(kg, positives):
    """Discover 2-hop (r1, r2) patterns holding for EVERY (subject, object) positive pair."""
    sets = []
    for s, o in positives:
        paths = set()
        for s1, r1, m in kg.triples:
            if s1 != s:
                continue
            for m2, r2, o2 in kg.triples:
                if m2 == m and o2 == o:
                    paths.add((r1, r2))
        sets.append(paths)
    return set.intersection(*sets) if sets else set()
