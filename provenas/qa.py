"""qa — turn a structured action into an EXACT, proven answer against the store.

The single place the fabric answers questions: load the store's facts + rules into the KG, forward-chain
(with provenance), then run the action (query / check / assert) and return the answer plus a proof tree.
Used by the CLI and the demo harness so there is one canonical answer path.
"""
from __future__ import annotations

from provenas.infer import forward_chain_prov, explain, _fact

SCHEMA = ('Action schema (output exactly ONE):\n'
          '  query a set:  {"action":"query","pattern":["?x","<relation>","<value>"]}   (?x is the unknown; it may be any slot)\n'
          '  check a fact: {"action":"check","triple":["<subject>","<relation>","<object>"]}\n'
          '  add a fact:   {"action":"assert","triple":["<subject>","<relation>","<object>"]}\n'
          'Use only the listed relations and lowercase entity tokens; keep [subject, relation, object] '
          'in the relation\'s natural order (do not swap subject and object).')


def context_from_store(store, blurb=""):
    """Build the translate-context from whatever the KB currently knows (domain-agnostic).

    If a pack was loaded its curated vocab (stored as meta) leads — that's the cleanest guide to relation
    directions. For an ad-hoc KB (the user's own asserted facts), derive the context from the stored
    relations + a few sample facts so the LLM can see each relation's argument order."""
    blurb = blurb or store.get_meta("vocab", "")
    if blurb:
        return blurb + "\n" + SCHEMA
    rels = sorted(store.relations())
    ents = sorted(store.entities())
    facts = sorted(store.triples())[:10]
    head = f"Relations: {', '.join(rels) if rels else '(none yet)'}.\n"
    if facts:                                               # ground the relation directions by example
        head += ("Example facts (note the [subject, relation, object] order): "
                 + "; ".join(f"({s} {r} {o})" for s, r, o in facts) + ".\n")
    if ents:
        head += f"Known entities: {', '.join(ents[:60])}.\n"
    return head + SCHEMA


def run_action(store, action):
    """Execute a {query|check|assert} action exactly. Returns dict(kind, answer, trace)."""
    kg = store.to_kg()
    _, prov = forward_chain_prov(kg, store.rules())
    kind = action.get("action")
    if kind == "assert":
        store.assert_(*action["triple"], source="user")
        store.log("assert", action["triple"])
        return dict(kind="assert", answer=True, trace=f"stored {_fact(tuple(action['triple']))}")
    if kind == "check":
        t = tuple(action["triple"])
        ok = t in kg.triples
        trace = "\n".join(explain(t, prov)) if ok else f"{_fact(t)} is not derivable from the facts + rules."
        return dict(kind="check", answer=ok, trace=trace)
    if kind == "query":
        pat = tuple(action["pattern"])
        var = next((x for x in pat if isinstance(x, str) and x.startswith("?")), None)
        binds = kg.query(pat)
        ans = sorted({b[var] for b in binds}) if var else [bool(binds)]
        lines = []
        for v in ans[:8]:
            tgt = tuple(v if x == var else x for x in pat)
            lines += explain(tgt, prov)
        return dict(kind="query", answer=ans, trace="\n".join(lines))
    return dict(kind="error", answer=None, trace=f"unknown action {action!r}")


def show_answer(r):
    a = r["answer"]
    if r["kind"] == "check":
        return "yes" if a else "no"
    if r["kind"] == "query":
        return ", ".join(map(str, a)) if a else "(none)"
    if r["kind"] == "assert":
        return "added"
    return str(a)
