"""provenas CLI — talk to the neuro-symbolic fabric.

A small REPL over a persistent SQLite knowledge base. Type a question in plain English (the LLM
translates it into an exact fabric operation; the fabric computes the answer and a proof), or use a
`:command` to inspect/extend the system. The LLM only ever PROPOSES; the fabric verifies and is the
source of truth. Every change — facts, rules, tools — is test-before-admit.

Run:  python -m provenas [kb.db]        (default kb: provenas.db; needs Ollama for natural-language ask)
"""
from __future__ import annotations

import ast
import re
import shlex
import sys

from provenas import domains, qa, rewrite
from provenas.learn import admit_rule
from provenas.llm import LLM
from provenas.store import Store
from provenas.toolsmith import admit_tool, load_tool

HELP = """commands:
  <natural language>            ask the knowledge base (needs Ollama); prints answer + proof
  :load <name>                  seed a starter pack            (packs: %s)
  :assert <s> <r> <o>           add a fact
  :retract <s> <r> <o>          remove a fact
  :why <s> <r> <o>              show the proof for a fact
  :learn <rel> : <desc> :: <+pairs> | <-pairs>
                                teach a rule, tested before admit, e.g.
                                :learn sibling : share a parent, different people :: ann,cy cy,ann | ann,ann ann,dan
  :tool <name> : <desc> :: <examples>
                                synthesize a Python tool, sandbox-tested, e.g.
                                :tool gcd : gcd of a and b :: gcd(12,8)=4, gcd(7,5)=1
  :call <name> <args...> [as <s> <r>]
                                call a tool; with 'as' assert its result as a fact (so queries can use it)
  :simplify <expr>              exact algebraic simplify, e.g. :simplify (x + 0) * (2 + 3)  ->  x * 5
  :case <name>                  pin the LAST answer as a regression case (rule changes must preserve it)
  :cases                        list pinned cases
  :disable <rule>  :enable <rule>   deactivate / reactivate a rule (kept on record)
  :declare <rel> [doc]          declare a relation in the schema
  :strict on|off                strict mode: reject asserts whose relation is undeclared
  :facts [substr]  :rules  :tools  :rewrites  :schema  :log [n]  :kb    inspect the knowledge base
  :narrate on|off               LLM one-line narration of answers (default off)
  :help    :quit
""" % ", ".join(domains.names())


def _pairs(s):
    return [tuple(p.split(",")) for p in s.split() if "," in p]


def _split(rest, usage):
    """shlex.split with a usage hint instead of an opaque 'No closing quotation' error."""
    try:
        return shlex.split(rest)
    except ValueError as e:
        print(f"  bad quoting ({e}) — {usage}")
        return None


def _examples(s):
    out = []
    for m in re.finditer(r"\w+\s*\(([^)]*)\)\s*=\s*([^,]+)", s):
        out.append((ast.literal_eval("(" + m.group(1) + ",)"), ast.literal_eval(m.group(2).strip())))
    return out


def _show_proof(trace, pad="    "):
    return pad + trace.replace("\n", "\n" + pad) if trace.strip() else ""


def _rule_str(rule):
    return f"{rule.head[1]} ⇐ " + ", ".join("(%s %s %s)" % a for a in rule.body)


def _feedback(rep):
    msgs = []
    if rep.get("error"):
        msgs.append(f"it could not be evaluated: {rep['error']}")
    if rep["missing"]:
        msgs.append(f"it failed to derive required examples {rep['missing']} — check the "
                    "[subject, relation, object] direction (e.g. '(p parent c)' means p is the parent of c)")
    if rep["violated"]:
        msgs.append(f"it wrongly derived {rep['violated']} — add an inequality guard "
                    '["?a","!=","?b"] so a thing cannot relate to itself')
    return "\nThe previous rule was REJECTED because " + "; and ".join(msgs) + ". Give a corrected rule."


def cmd(line, store, llm, state):
    parts = line.split(None, 1)
    c = parts[0]
    rest = parts[1] if len(parts) > 1 else ""

    if c == ":load":
        d = domains.load(store, rest.strip())
        print(f"  loaded '{rest.strip()}'" if d else f"  no such pack (try: {', '.join(domains.names())})")

    elif c in (":assert", ":retract"):
        a = _split(rest, "usage: %s <s> <r> <o>" % c)
        if a is None:
            return
        if len(a) != 3:
            print("  usage: %s <s> <r> <o>" % c)
            return
        if c == ":assert":
            store.assert_(*a, source="user")
        else:
            store.retract(*a)
        print(f"  {'added' if c == ':assert' else 'removed'} ({a[0]} {a[1]} {a[2]})")

    elif c == ":why":
        a = _split(rest, "usage: :why <s> <r> <o>")
        if a is None:
            return
        if len(a) != 3:
            print("  usage: :why <s> <r> <o>")
            return
        action = {"action": "check", "triple": a}
        r = qa.run_action(store, action)
        state["last"] = (action, r)
        print(f"  {qa.show_answer(r)}")
        print(_show_proof(r["trace"]))

    elif c == ":learn":
        try:
            rel, body = rest.split(":", 1)
            desc, examples = body.split("::", 1)
            pos_s, neg_s = (examples.split("|", 1) + [""])[:2]
            rel = rel.strip()
            pos, neg = _pairs(pos_s), _pairs(neg_s)
        except ValueError:
            print("  usage: :learn <rel> : <desc> :: <+pairs> | <-pairs>")
            return
        if not llm.available:
            print("  llm offline — cannot propose a rule")
            return
        ctx = qa.context_from_store(store)
        try:
            rule = llm.propose_rule(f"Define '{rel}'. {desc.strip()}", ctx)
        except Exception as e:
            print(f"  propose failed: {e}")
            return
        print(f"  proposed: {_rule_str(rule)}")
        ok, rep = admit_rule(store, rule, rel, pos, neg, source="qwen")
        seen, tries = {_rule_str(rule)}, 0
        while not ok and tries < 2:                            # feed the gate's verdict back, revise, retest
            tries += 1
            try:
                rule = llm.propose_rule(f"Define '{rel}'. {desc.strip()}{_feedback(rep)}", ctx)
            except Exception as e:
                print(f"  (revision unparseable, retrying: {e})")
                continue                                   # a garbled reply costs a try, not the whole :learn
            print(f"  revised:  {_rule_str(rule)}")
            if _rule_str(rule) in seen:                        # LLM repeating itself — stop wasting calls
                print("  (no change from feedback — stopping)")
                break
            seen.add(_rule_str(rule))
            ok, rep = admit_rule(store, rule, rel, pos, neg, source="qwen-revised")
        print(f"  -> {'ADMITTED (saved)' if ok else 'REJECTED'}  "
              f"(missing={rep['missing']} violated={rep['violated']}"
              + (f" flips pinned cases {rep['case_flips']}" if rep.get("case_flips") else "")
              + (f" error={rep['error']}" if rep.get("error") else "") + ")")

    elif c == ":tool":
        try:
            name, body = rest.split(":", 1)
            desc, ex = body.split("::", 1)
            name = name.strip()
            examples = _examples(ex)
        except ValueError:
            print("  usage: :tool <name> : <desc> :: name(args)=result, ...")
            return
        if not examples:
            print("  need at least one example to test the tool before admitting it")
            return
        if llm.available:
            try:
                src = llm.propose_tool(name, desc.strip(), examples)
            except Exception as e:
                print(f"  propose failed: {e}")
                return
        else:
            print("  llm offline — cannot synthesize a tool")
            return
        ok, stage, detail = admit_tool(store, name, src, examples, source="qwen")
        print("  proposed:\n" + "\n".join("      " + l for l in src.strip().splitlines()))
        print(f"  -> {'ADMITTED (saved)' if ok else 'REJECTED at [' + stage + ']'}: {detail}")

    elif c == ":call":
        a = _split(rest, "usage: :call <name> <args...> [as <subject> <relation>]")
        if a is None:
            return
        if not a:
            print("  usage: :call <name> <args...> [as <subject> <relation>]")
            return
        dest = None
        if "as" in a:                                          # feed the result back into the KB as a fact
            i = a.index("as")
            a, dest = a[:i], a[i + 1:]
            if len(dest) != 2:
                print("  usage: ... as <subject> <relation>")
                return
        src = store.get_tool(a[0])
        if not src:
            print(f"  no tool '{a[0]}' (see :tools)")
            return
        try:
            args = [ast.literal_eval(x) for x in a[1:]]
        except (ValueError, SyntaxError):
            print("  arguments must be Python literals, e.g. :call gcd 48 36")
            return
        result = load_tool(src, a[0])(*args)
        print(f"  {a[0]}({', '.join(map(str, args))}) = {result}")
        if dest:
            store.assert_(dest[0], dest[1], str(result), source="computed")
            store.log("compute", f"{dest[0]} {dest[1]} {result}")
            print(f"  asserted ({dest[0]} {dest[1]} {result}) — now queryable")

    elif c == ":simplify":
        if not rest.strip():
            print("  usage: :simplify <expr>   e.g. :simplify (x + 0) * (2 + 3)")
            return
        rules = store.rewrites()
        if not rules:                                          # first use: seed the default ruleset into the KB
            for name, rule in rewrite.DEFAULTS:
                store.add_rewrite(name, rule, source="default")
            rules = store.rewrites()
            print("  (seeded default rewrite rules into this KB)")
        try:
            term = rewrite.parse(rest)
        except Exception as e:
            print(f"  parse error: {e}")
            return
        print(f"  {rewrite.pretty(term)}  ->  {rewrite.pretty(rewrite.normal_form(term, rules))}")

    elif c == ":case":
        name = rest.strip()
        if not name:
            print("  usage: :case <name>   (pins the last query/check answer as a regression case)")
            return
        last = state.get("last")
        if not last or last[1]["kind"] not in ("check", "query"):
            print("  ask a question (or :why a fact) first — :case pins the LAST answer")
            return
        store.add_case(name, last[0], last[1]["answer"])
        print(f"  pinned case '{name}': {last[0]} -> {last[1]['answer']}  (rule changes must preserve it)")

    elif c == ":cases":
        cs = store.cases()
        for name, action, expect in cs:
            print(f"  {name}: {action} -> {expect}")
        print(f"  [{len(cs)} pinned case(s)]  (every :learn must preserve these answers)")

    elif c in (":disable", ":enable"):
        name = rest.strip()
        if not name:
            print(f"  usage: {c} <rule-name>")
            return
        if store.set_rule_active(name, c == ":enable"):
            print(f"  rule '{name}' {'re-enabled' if c == ':enable' else 'disabled (kept on record)'}")
        else:
            print(f"  no rule named '{name}' (see :rules)")

    elif c == ":declare":
        a = rest.split(None, 1)
        if not a:
            print("  usage: :declare <relation> [doc]")
            return
        store.declare(a[0], a[1] if len(a) > 1 else "")
        print(f"  declared relation '{a[0]}'")

    elif c == ":strict":
        on = rest.strip() == "on"
        store.set_meta("strict", "1" if on else "0")
        print(f"  strict mode {'ON — asserts with undeclared relations are rejected' if on else 'off'}")

    elif c == ":schema":
        sch = store.schema()
        for rel, doc in sorted(sch.items()):
            print(f"  {rel}" + (f"  — {doc}" if doc else ""))
        strict = store.get_meta("strict", "0") == "1"
        print(f"  [{len(sch)} declared relation(s), strict mode {'ON' if strict else 'off'}]")

    elif c == ":rewrites":
        rs = store.rewrites()
        for rule in rs:
            print("  " + rewrite.rule_str(rule))
        print(f"  [{len(rs)} rewrite rule(s)]  (use :simplify <expr>)")

    elif c == ":facts":
        ts = [t for t in sorted(store.triples()) if rest.strip() in " ".join(t)]
        for s, r, o in ts[:60]:
            print(f"  ({s} {r} {o})")
        print(f"  [{len(ts)} fact(s)]")

    elif c == ":rules":
        active = {r.name for r in store.rules()}
        rs = store.rules(all=True)
        for rule in rs:
            flag = "" if rule.name in active else "  [DISABLED]"
            print(f"  {rule.name}: {rule.head[0]} {rule.head[1]} {rule.head[2]}  ⇐  "
                  + ", ".join("(%s %s %s)" % a for a in rule.body) + flag)
        print(f"  [{len(active)} active / {len(rs)} rule(s)]  (:disable/:enable <name>)")

    elif c == ":tools":
        for name, _, tests in store.tools():
            print(f"  {name}  ({len(tests)} tested example(s))")
        print(f"  [{len(store.tools())} tool(s)]")

    elif c == ":log":
        n = int(rest) if rest.strip().isdigit() else 12
        for _, kind, detail in store.recent_log(n):
            print(f"  {kind:14s} {detail}")

    elif c == ":kb":
        t, r = store.counts()
        print(f"  {t} facts, {r} rules, {len(store.tools())} tools")

    elif c == ":narrate":
        state["narrate"] = rest.strip() == "on"
        print(f"  narration {'on' if state['narrate'] else 'off'}")

    elif c == ":help":
        print(HELP)

    else:
        print(f"  unknown command {c} (try :help)")


def ask(line, store, llm, state):
    if not llm.available:
        print("  (LLM offline — natural-language ask needs Ollama; use :assert/:facts/:why)")
        return
    try:
        action = llm.translate(line, qa.context_from_store(store))
    except Exception as e:
        print(f"  could not translate: {e}")
        return
    r = qa.run_action(store, action)
    if r["kind"] in ("check", "query"):
        state["last"] = (action, r)                     # so :case can pin it
    print(f"  → {qa.show_answer(r)}")
    print(f"    [{action}]")
    proof = _show_proof(r["trace"])
    if proof:
        print(proof)
    if state["narrate"]:
        try:
            print("    " + llm.narrate(line, qa.show_answer(r), r["trace"]))
        except Exception:
            pass


USAGE = """usage: provenas [kb.db]

Opens (or creates) a persistent SQLite knowledge base and starts the REPL.
Default kb: provenas.db. Natural-language ask needs a model (see PROVENAS_LLM_*
environment variables); every :command works without one. Type :help inside.
"""


def main():
    kb = sys.argv[1] if len(sys.argv) > 1 else "provenas.db"
    if kb in ("-h", "--help") or kb.startswith("-"):
        print(USAGE, end="")
        return
    store = Store(kb)
    llm = LLM()
    ready = llm.ping()
    state = {"narrate": False}
    print(f"provenas — neuro-symbolic engine   (kb: {kb})")
    print(f"  llm: {llm.model} via {llm.backend} @ {llm.host}"
          f"  {'· ready' if ready else '· OFFLINE (:commands still work)'}   ·   :help")
    try:
        while True:
            try:
                line = input("provenas> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not line:
                continue
            if line in (":quit", ":exit", ":q"):
                break
            try:
                (cmd if line.startswith(":") else ask)(line, store, llm, state)
            except Exception as e:
                print(f"  error: {e}")
    finally:
        store.close()


if __name__ == "__main__":
    main()
