"""Use provenas as a library (no LLM needed) — build a KB, reason over it, get proofs.

    python examples/demo.py
"""
from provenas.store import Store
from provenas import domains, qa
from provenas.infer import Rule
from provenas.toolsmith import admit_tool, load_tool


def main():
    kb = Store(":memory:")                       # use Store("my.db") for a persistent, on-disk KB

    # 1. seed a starter pack (facts + rules), or assert your own facts
    domains.load(kb, "rbac")
    kb.assert_("erin", "has_role", "engineer")   # add a new user

    # 2. ask exact questions — every answer comes with a derivation
    r = qa.run_action(kb, {"action": "check", "triple": ["erin", "can", "prod_deploy"]})
    print("Can Erin deploy to prod?", "yes" if r["answer"] else "no")
    print(r["trace"])

    r = qa.run_action(kb, {"action": "query", "pattern": ["?u", "can", "view_wiki"]})
    print("\nWho can view the wiki?", sorted(r["answer"]))

    # 3. add your own rule (validate-before-admit happens via provenas.learn.admit_rule;
    #    here we just register a hand-written, trusted rule)
    kb.add_rule(Rule([("?u", "has_perm", "?p")], ("?u", "is_privileged", "yes"), "privileged"))
    r = qa.run_action(kb, {"action": "query", "pattern": ["?u", "is_privileged", "yes"]})
    print("Privileged users:", sorted(r["answer"]))

    # 4. admit a sandboxed tool, then call it
    ok, stage, detail = admit_tool(kb, "gcd",
                                   "def gcd(a, b):\n    while b:\n        a, b = b, a % b\n    return a\n",
                                   [((12, 8), 4), ((7, 5), 1)])
    print(f"\ntool gcd: {'admitted' if ok else 'rejected at ' + stage} ({detail})")
    if ok:
        print("gcd(48, 36) =", load_tool(kb.get_tool("gcd"), "gcd")(48, 36))

    kb.close()


if __name__ == "__main__":
    main()
