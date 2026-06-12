"""Performance regression tests — the engine's published envelope, enforced.

Bounds are ~10x looser than measured (4x RTX 4090 box, CPU-side) so CI noise never flakes them,
but a return to pre-index/naive-evaluation behavior (100-1000x slower) fails loudly.
Measured reference: 50k base facts -> 287k closure in ~1.3s; cached store reads ~0ms.
"""
import time

from provenas.infer import Rule, forward_chain
from provenas.kg import KnowledgeGraph
from provenas.store import Store
from provenas import qa

RULES = [Rule([("?x", "parent", "?y")], ("?x", "ancestor", "?y"), "base"),
         Rule([("?x", "parent", "?y"), ("?y", "ancestor", "?z")], ("?x", "ancestor", "?z"), "step")]


def forest(n_facts, store=None):
    """A realistic org-chart shape: 5-child trees, depth ~5."""
    kg = store if store is not None else KnowledgeGraph()
    i = node = 0
    while i < n_facts:
        frontier, depth = [f"n{node}"], 0
        node += 1
        while frontier and depth < 5 and i < n_facts:
            nxt = []
            for p in frontier:
                for _ in range(5):
                    if i >= n_facts:
                        break
                    c = f"n{node}"
                    node += 1
                    kg.assert_(p, "parent", c)
                    nxt.append(c)
                    i += 1
            frontier, depth = nxt, depth + 1
    return kg


def test_chain_50k_facts_under_15s():
    kg = forest(50_000)
    t0 = time.perf_counter()
    forward_chain(kg, RULES)
    elapsed = time.perf_counter() - t0
    assert len(kg.triples) > 250_000                   # the closure actually got built
    assert elapsed < 15.0, f"forward_chain took {elapsed:.1f}s (naive-evaluation regression?)"


def test_indexed_query_under_50ms():
    kg = forest(20_000)
    forward_chain(kg, RULES)
    t0 = time.perf_counter()
    for _ in range(20):
        kg.query(("?x", "ancestor", "n7"))
        kg.query(("n0", "parent", "?x"))
    elapsed = (time.perf_counter() - t0) / 40
    assert elapsed < 0.05, f"indexed query took {1000 * elapsed:.1f}ms (index regression?)"


def test_store_reads_are_cached():
    s = Store(":memory:")
    forest(5_000, store=s)
    for rule in RULES:
        s.add_rule(rule)
    q = {"action": "query", "pattern": ["?x", "ancestor", "n7"]}
    first = qa.run_action(s, q)                        # pays for materialization once
    t0 = time.perf_counter()
    for _ in range(10):
        assert qa.run_action(s, q)["answer"] == first["answer"]
    per_read = (time.perf_counter() - t0) / 10
    assert per_read < 0.05, f"cached read took {1000 * per_read:.1f}ms (closure cache regression?)"
