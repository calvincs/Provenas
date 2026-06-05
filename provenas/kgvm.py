"""KGMachine — a controller-drivable instruction machine over the knowledge graph.

The KG's relational operations as a small composable instruction set with a working set of
symbols (`focus`). Reasoning becomes a *program*: select a starting point, traverse relations,
expand sub/super-types, filter by a property, then combine sets (intersect/union/exclude). This
is the substrate a controller drives — the same way the conv controller drives the arithmetic VM —
and the algebra the combination solver composes (each "tool" is a short KGMachine program; the
"aha" is intersecting two or three of them). Every step is recorded in `trace` for interpretability.
"""
from __future__ import annotations


class KGMachine:
    def __init__(self, kg):
        self.kg = kg
        self.focus = set()
        self.trace = []

    def _t(self, *step):
        self.trace.append(step)
        return self

    # ---- load ----
    def select(self, *syms):
        self.focus = set(syms)
        return self._t("select", *syms)

    def subjects_with(self, r, o):
        self.focus = set(self.kg.group_by_relation(r, o))     # all s with (s, r, o)
        return self._t("subjects_with", r, o)

    # ---- traverse ----
    def neighbors(self, r):
        self.focus = {o for s in self.focus for o in self.kg.neighbors(s, r)}
        return self._t("neighbors", r)

    def ancestors(self, r="is_a"):
        self.focus = {a for s in self.focus for a in self.kg.ancestors(s, r)}
        return self._t("ancestors", r)

    def descendants(self, r="is_a"):
        self.focus = {d for s in self.focus for d in self.kg.descendants(s, r)}
        return self._t("descendants", r)

    # ---- filter / combine ----
    def filter_has(self, r, o):
        self.focus = {s for s in self.focus if o in self.kg.neighbors(s, r)}
        return self._t("filter_has", r, o)

    def intersect(self, other):
        self.focus &= set(other)
        return self._t("intersect")

    def union(self, other):
        self.focus |= set(other)
        return self._t("union")

    def exclude(self, other):
        self.focus -= set(other)
        return self._t("exclude")

    def result(self):
        return set(self.focus)
