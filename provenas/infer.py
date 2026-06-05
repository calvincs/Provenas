"""infer — a small symbolic inference engine over the knowledge graph.

Three capabilities the reasoner needs:
  - forward_chain(kg, rules)   deductive closure: fire Horn-style rules to a fixpoint, deriving
                               new facts (e.g. is_a transitivity, property inheritance).
  - paths_between(kg, a, b)    relation-composition: what chain of relations connects a -> b
                               (the seed of "what do these two have in common / how are they linked").
  - induce_path_rule(kg, pos)  rule discovery (ILP-lite): find a 2-hop relation pattern that holds
                               for all positive (subject, object) pairs — a learned rule.

Rules use "?var" pattern variables, the same convention as KnowledgeGraph.query.
"""
from __future__ import annotations

from collections import deque


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
    for triple in kg.triples:
        b = _bind(pattern, triple, binding)
        if b is not None:
            yield b


def _solve_body(kg, body, init=None):
    """Bindings satisfying all body atoms. An atom with relation '!=' is an inequality GUARD
    (its two operands must differ) rather than a fact lookup."""
    bindings = [dict(init) if init else {}]
    for atom in body:
        if atom[1] == "!=":
            bindings = [b for b in bindings if b.get(atom[0], atom[0]) != b.get(atom[2], atom[2])]
        else:
            bindings = [b2 for b in bindings for b2 in _matches(kg, atom, b)]
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


def forward_chain(kg, rules, max_iter=50):
    """Assert all derivable facts into `kg` to a fixpoint; return the newly derived set."""
    derived = set()
    for _ in range(max_iter):
        new = set()
        for rule in rules:
            new |= fire(kg, rule)
        new -= kg.triples
        if not new:
            break
        for s, r, o in new:
            kg.assert_(s, r, o)
        derived |= new
    return derived


def forward_chain_prov(kg, rules, max_iter=50):
    """Like forward_chain, but record provenance: derived_fact -> (rule_name, [premise_facts])."""
    prov = {}
    for _ in range(max_iter):
        new = {}
        for rule in rules:
            for b in _solve_body(kg, rule.body):
                head = tuple(b.get(x, x) for x in rule.head)
                if any(_isvar(x) for x in head) or head in kg.triples or head in new:
                    continue
                premises = [tuple(b.get(x, x) for x in pat) for pat in rule.body if pat[1] != "!="]
                new[head] = (rule.name, premises)
        if not new:
            break
        for h, p in new.items():
            kg.assert_(*h)
            prov[h] = p
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
