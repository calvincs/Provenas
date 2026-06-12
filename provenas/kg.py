"""An associative knowledge graph — exact, inspectable relational memory.

The relational generalization of the register machine's keyed store: a tool the controller
can drive (assert / retract / query) and the storage core for reasoning. Backed by a triple
store (subject, relation, object) plus per-symbol property maps. Supports:

  - associate / disassociate     assert(s, r, o) / retract(s, r, o)
  - properties                   set_property / get_property
  - retrieval                    neighbors, relations, subjects
  - pattern query (mini-Datalog) query((s, r, o)) with "?var" wildcards -> bindings
  - inference                    transitive closure (e.g. is_a inheritance)
  - groups & property discovery  group_by_relation, common facts/ancestors/properties

This snaps into the engine the same way a tool table does: its ops are an instruction set a
controller can dispatch to. Everything is exact and inspectable (a real dict/set in memory).
"""
from __future__ import annotations

from collections import defaultdict


class KnowledgeGraph:
    def __init__(self):
        self.triples = set()                                   # {(s, r, o)}
        self.props = defaultdict(dict)                         # symbol -> {key: value}
        self._spo = defaultdict(lambda: defaultdict(set))      # s -> r -> {o}
        self._osr = defaultdict(lambda: defaultdict(set))      # o -> r -> {s}  (reverse index)
        self._rel = defaultdict(set)                           # r -> {(s, r, o)}

    # ---- associate / disassociate ----
    def assert_(self, s, r, o):
        if (s, r, o) not in self.triples:
            self.triples.add((s, r, o))
            self._spo[s][r].add(o)
            self._osr[o][r].add(s)
            self._rel[r].add((s, r, o))
        return self

    def retract(self, s, r, o):
        self.triples.discard((s, r, o))
        self._spo[s][r].discard(o)
        self._osr[o][r].discard(s)
        self._rel[r].discard((s, r, o))
        return self

    def set_property(self, sym, key, value):
        self.props[sym][key] = value
        return self

    def get_property(self, sym, key, default=None):
        return self.props[sym].get(key, default)

    # ---- retrieval ----
    def neighbors(self, s, r=None):
        if r is not None:
            return set(self._spo[s][r])
        return set().union(*self._spo[s].values()) if self._spo[s] else set()

    def relations(self, s):
        return {r for r, os in self._spo[s].items() if os}

    def subjects(self, r, o):
        return set(self._osr[o][r])                            # all s with (s, r, o)

    # ---- pattern query (variables are strings starting with '?') ----
    def candidates(self, pattern):
        """Triples that could match `pattern`, narrowed by the best available index
        (instead of a full scan). Variable slots are '?'-strings; constants are bound."""
        s, r, o = pattern
        sv = isinstance(s, str) and s.startswith("?")
        rv = isinstance(r, str) and r.startswith("?")
        ov = isinstance(o, str) and o.startswith("?")
        if not sv and not rv and not ov:
            return [(s, r, o)] if (s, r, o) in self.triples else []
        if not sv and not rv:
            return [(s, r, x) for x in self._spo.get(s, {}).get(r, ())] if s in self._spo else []
        if not rv and not ov:
            return [(x, r, o) for x in self._osr.get(o, {}).get(r, ())] if o in self._osr else []
        if not sv:
            return [(s, rr, x) for rr, xs in self._spo.get(s, {}).items() for x in xs
                    if ov or x == o]
        if not ov:
            return [(x, rr, o) for rr, xs in self._osr.get(o, {}).items() for x in xs]
        if not rv:
            return self._rel.get(r, ())
        return self.triples

    def query(self, pattern):
        s, r, o = pattern
        out = []
        for ts, tr, to in self.candidates(pattern):
            b, ok = {}, True
            for pat, val in ((s, ts), (r, tr), (o, to)):
                if isinstance(pat, str) and pat.startswith("?"):
                    if b.get(pat, val) != val:
                        ok = False
                        break
                    b[pat] = val
                elif pat != val:
                    ok = False
                    break
            if ok:
                out.append(b)
        return out

    # ---- inference: transitive closure of a relation (inheritance) ----
    def transitive(self, s, r):
        seen, stack = set(), list(self._spo[s][r])
        while stack:
            x = stack.pop()
            if x in seen:
                continue
            seen.add(x)
            stack.extend(self._spo[x][r])
        return seen

    def ancestors(self, s, r="is_a"):
        return self.transitive(s, r)

    # ---- groups & property discovery ----
    def group_by_relation(self, r, o):
        return set(self._osr[o][r])                            # all s with (s, r, o)

    def common_facts(self, symbols):
        symbols = list(symbols)
        if not symbols:
            return set()
        sets = [{(r, o) for r in self._spo[s] for o in self._spo[s][r]} for s in symbols]
        return set.intersection(*sets) if sets else set()

    def discover_properties(self, symbols):
        """What a group of symbols shares: direct facts, inherited ancestors, and properties."""
        symbols = list(symbols)
        if not symbols:
            return dict(common_facts=set(), common_ancestors=set(), common_props=set())
        anc = [self.ancestors(s) for s in symbols]
        prop = [set(self.props[s].items()) for s in symbols]
        return dict(
            common_facts=self.common_facts(symbols),
            common_ancestors=set.intersection(*anc) if anc else set(),
            common_props=set.intersection(*prop) if prop else set(),
        )

    def descendants(self, s, r="is_a"):
        """All x reachable up to s via r (reverse transitive) — the subtypes/instances of s."""
        seen, stack = set(), list(self._osr[s][r])
        while stack:
            x = stack.pop()
            if x in seen:
                continue
            seen.add(x)
            stack.extend(self._osr[x][r])
        return seen

    def copy(self):
        kg = KnowledgeGraph()
        for s, r, o in self.triples:
            kg.assert_(s, r, o)
        for sym, d in self.props.items():
            for k, v in d.items():
                kg.set_property(sym, k, v)
        return kg

    def __repr__(self):
        return f"KnowledgeGraph({len(self.triples)} triples, {len(self.props)} symbols w/ props)"
