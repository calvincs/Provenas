"""Slice 1 toy — proving the full setup: Qwen (Ollama)  <->  fabric (KG + inference)  <->  SQLite.

One pipeline, run across THREE structurally different domains (kinship / access-control / diagnostics)
to stress the genericity: only the data changes. For each natural-language question:

    Qwen TRANSLATES it to a structured action  ->  the fabric EXECUTES it exactly (forward-chaining
    rules) and returns the answer WITH a proof tree  ->  Qwen NARRATES the exact result.

The LLM proposes; the fabric verifies and is the source of truth; SQLite persists everything. Closes
with a persistence check (reopen the DB) and a "learn a new fact" step that flips a decision and sticks.

Runs on aibox (Ollama local). If Qwen is unreachable, falls back to each question's gold action so the
fabric/DB tiers still demonstrate end-to-end; every line shows which path ran.
"""
from __future__ import annotations

import os

from provenas.domains import DOMAINS
from provenas.infer import forward_chain_prov, explain, _fact
from provenas.llm import LLM
from provenas.store import Store


SCHEMA = ('Action schema (output exactly ONE):\n'
          '  query a set:  {"action":"query","pattern":["?x","<relation>","<value>"]}   (?x is the unknown; it can be in any slot)\n'
          '  check a fact: {"action":"check","triple":["<subject>","<relation>","<object>"]}\n'
          '  add a fact:   {"action":"assert","triple":["<subject>","<relation>","<object>"]}\n'
          'Use only the listed relations and lowercase entity tokens. Map the question onto them.\n'
          'Relations are DIRECTIONAL: order each triple [subject, relation, object] by the relation\'s '
          'meaning; do not swap subject and object.')


def context_for(d):
    return f"Domain: {d['blurb']}.\n{d['vocab']}.\n{SCHEMA}"


# --------------------------------------------------------------------------- fabric execute
def execute(action, store):
    kg = store.to_kg()
    _, prov = forward_chain_prov(kg, store.rules())
    kind = action.get("action")
    if kind == "assert":
        store.assert_(*action["triple"], source="learned")
        store.log("assert", action["triple"])
        return True, f"stored {_fact(tuple(action['triple']))}"
    if kind == "check":
        t = tuple(action["triple"])
        if t in kg.triples:
            return True, "\n".join("   " + l for l in explain(t, prov))
        return False, f"   {_fact(t)} is not derivable from the facts + rules."
    if kind == "query":
        pat = tuple(action["pattern"])
        var = next((x for x in pat if isinstance(x, str) and x.startswith("?")), None)
        binds = kg.query(pat)
        ans = sorted({b[var] for b in binds}) if var else [not not binds]
        lines = []
        for v in ans[:3]:
            tgt = tuple(v if x == var else x for x in pat)
            lines += ["   " + l for l in explain(tgt, prov)]
        return set(ans), "\n".join(lines)
    return None, f"unknown action {action!r}"


def fmt_answer(a):
    if isinstance(a, set):
        return sorted(a)
    return a


def ok(answer, expect):
    if isinstance(expect, set):
        return isinstance(answer, set) and answer == expect
    return answer == expect


# --------------------------------------------------------------------------- run
def load(store, d):
    for t in d["triples"]:
        store.assert_(*t, source="seed")
    for r in d["rules"]:
        store.add_rule(r, source="seed")
    store.log("load", d["name"])


def get_action(llm, q, ctx):
    if llm.available:
        try:
            return llm.translate(q["q"], ctx), "qwen"
        except Exception as e:
            print(f"   (qwen translate failed: {e}; using fallback)")
    return dict(q["gold"]), "fallback"


def say(llm, q, answer, trace):
    if llm.available:
        try:
            return llm.narrate(q["q"], fmt_answer(answer), trace)
        except Exception:
            pass
    return ("Yes." if answer is True else "No." if answer is False else f"{fmt_answer(answer)}")


def main():
    import sys
    os.makedirs("artifacts", exist_ok=True)
    sel = sys.argv[1:] or ["family", "rbac", "diagnostics"]   # new packs (eligibility/config) opt-in
    domains = [d for d in DOMAINS if d["name"] in sel]
    llm = LLM()
    print(f"interface: model={llm.model}  reachable={llm.ping()}\n")

    passed = total = 0
    for d in domains:
        path = f"artifacts/toy_{d['name']}.db"
        if os.path.exists(path):
            os.remove(path)
        store = Store(path)
        load(store, d)
        tc, rc = store.counts()
        ctx = context_for(d)
        print(f"================  domain: {d['name']}  ({tc} facts, {rc} rules in SQLite)  ================")
        for q in d["questions"]:
            action, src = get_action(llm, q, ctx)
            answer, trace = execute(action, store)
            good = ok(answer, q["expect"])
            passed += good
            total += 1
            print(f"\nQ: {q['q']}")
            print(f"   action ({src}): {action}")
            print(f"   exact answer  : {fmt_answer(answer)}   {'OK' if good else 'MISMATCH exp ' + str(fmt_answer(q['expect']))}")
            if trace.strip():
                print("   why:\n" + trace)
            print(f"   narration     : {say(llm, q, answer, trace)}")
        store.close()

    # persistence + growth: reopen rbac, learn a fact via NL, watch a decision flip and persist
    if "rbac" not in sel:
        print(f"\nSUMMARY: {passed}/{total} questions answered exactly across {len(domains)} domain(s).")
        return
    print("\n================  persistence + learning (rbac)  ================")
    store = Store("artifacts/toy_rbac.db")
    print(f"reopened SQLite -> {store.counts()[0]} facts, {store.counts()[1]} rules still here (durable).")
    before, _ = execute({"action": "check", "triple": ["dana", "can", "query_db"]}, store)
    print(f"  before: can dana query_db?  {before}")
    grow_q = "Dana is now also an analyst."
    ctx = context_for(DOMAINS[1])
    action, src = get_action(llm, {"q": grow_q, "gold": {"action": "assert", "triple": ["dana", "has_role", "analyst"]}}, ctx)
    execute(action, store)
    print(f"  learned ({src}): '{grow_q}' -> {action}")
    after, trace = execute({"action": "check", "triple": ["dana", "can", "query_db"]}, store)
    print(f"  after:  can dana query_db?  {after}   (decision flipped, and persisted)")
    if trace.strip():
        print("  why now:\n" + trace)
    store.close()
    store2 = Store("artifacts/toy_rbac.db")
    recheck, _ = execute({"action": "check", "triple": ["dana", "can", "query_db"]}, store2)
    print(f"  reopened DB again -> can dana query_db?  {recheck}  (the learned fact stuck)")
    store2.close()

    print(f"\nSUMMARY: {passed}/{total} questions answered exactly across {len(DOMAINS)} domains "
          f"(same pipeline, only data swapped).")


if __name__ == "__main__":
    main()
