"""Phase 5 — associative knowledge graph demo (and self-check).

Builds a small taxonomy, then exercises every capability the framework now gains:
storage, associate/disassociate, pattern query, inheritance inference, groups, and
property discovery. Asserts correctness so it doubles as a test.
"""
from __future__ import annotations

from provenas.kg import KnowledgeGraph


def main():
    kg = KnowledgeGraph()
    # associate: a small taxonomy + facts
    for s, r, o in [
        ("dog", "is_a", "mammal"), ("cat", "is_a", "mammal"), ("whale", "is_a", "mammal"),
        ("mammal", "is_a", "animal"), ("fish", "is_a", "animal"), ("animal", "is_a", "living"),
        ("dog", "has", "fur"), ("cat", "has", "fur"), ("whale", "has", "blubber"),
        ("dog", "has", "legs"), ("cat", "has", "legs"), ("fish", "has", "gills"),
    ]:
        kg.assert_(s, r, o)
    for sym, n in [("dog", 4), ("cat", 4), ("whale", 0), ("fish", 0)]:
        kg.set_property(sym, "legs", n)
    print(f"  {kg}")

    # pattern query (mini-Datalog)
    mammals = sorted(b["?x"] for b in kg.query(("?x", "is_a", "mammal")))
    print(f"  query  ?x is_a mammal           -> {mammals}")
    assert mammals == ["cat", "dog", "whale"]

    # inference: inheritance via transitive closure
    dog_anc = kg.ancestors("dog")
    print(f"  infer  ancestors(dog)            -> {sorted(dog_anc)}")
    assert dog_anc == {"mammal", "animal", "living"}      # dog inherits is_a animal & living

    # groups
    grp = kg.group_by_relation("is_a", "mammal")
    print(f"  group  is_a mammal               -> {sorted(grp)}")
    assert grp == {"dog", "cat", "whale"}

    # property discovery: what do dog & cat share?
    disc = kg.discover_properties({"dog", "cat"})
    print(f"  discover(dog, cat): facts={sorted(disc['common_facts'])}")
    print(f"                      ancestors={sorted(disc['common_ancestors'])}  props={sorted(disc['common_props'])}")
    assert ("has", "fur") in disc["common_facts"] and ("has", "legs") in disc["common_facts"]
    assert disc["common_ancestors"] == {"mammal", "animal", "living"}
    assert ("legs", 4) in disc["common_props"]

    # disassociate: cats lose fur in this world -> shared 'fur' fact disappears
    kg.retract("cat", "has", "fur")
    disc2 = kg.discover_properties({"dog", "cat"})
    print(f"  retract cat has fur -> shared facts now {sorted(disc2['common_facts'])}")
    assert ("has", "fur") not in disc2["common_facts"]
    assert ("has", "legs") in disc2["common_facts"]        # still shared

    print("  KG demo OK — storage, associate/disassociate, query, inference, groups, discovery all exact.")


if __name__ == "__main__":
    main()
