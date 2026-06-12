"""Real-life example 1 — an ACCESS AUDITOR: "who can touch prod, and why?"

The single most common audit question in any engineering org. Access is computed through
chains (person -> role -> inherited role -> permission -> resource) that nobody can see by
reading any one system. Here we encode a realistic org once, then answer audit questions
with PROOFS — the thing a SOC 2 auditor actually asks for.

Runs offline, no LLM, no third-party packages:   python examples/access_audit.py
"""
from provenas.infer import Rule
from provenas.store import Store
from provenas import qa


def banner(t):
    print(f"\n{'=' * 72}\n  {t}\n{'=' * 72}")


def ask(store, action, label):
    r = qa.run_action(store, action)
    print(f"\n[audit] {label}")
    print(f"        -> {qa.show_answer(r)}")
    if r["trace"].strip():
        print("        " + r["trace"].replace("\n", "\n        "))
    return r


store = Store(":memory:")                       # use a path ("org.db") to make it durable

# ---------------------------------------------------------------- the org, as facts
banner("1. Encode the org once (facts a script could export from GitHub/IAM/HR)")
FACTS = [
    # people -> roles
    ("alice", "has_role", "senior_eng"), ("bo", "has_role", "engineer"),
    ("carol", "has_role", "sre"), ("dev_dan", "has_role", "contractor"),
    ("erin", "has_role", "intern"),
    # role inheritance (a senior engineer IS an engineer, ...)
    ("senior_eng", "inherits", "engineer"), ("engineer", "inherits", "employee"),
    ("sre", "inherits", "engineer"), ("intern", "inherits", "employee"),
    ("contractor", "inherits", "employee"),
    # what roles grant
    ("employee", "grants", "read_wiki"), ("engineer", "grants", "push_code"),
    ("senior_eng", "grants", "approve_pr"), ("sre", "grants", "deploy_key"),
    ("contractor", "grants", "push_code"),
    # what resources require
    ("wiki", "requires", "read_wiki"), ("main_branch", "requires", "push_code"),
    ("prod_deploy", "requires", "deploy_key"), ("pr_merge", "requires", "approve_pr"),
    # the messy real-world bit: a suspension
    ("dev_dan", "suspended", "2026-06-02"),
]
for s, r, o in FACTS:
    store.assert_(s, r, o, source="org-export")

# ---------------------------------------------------------------- the policy, as rules
RULES = [
    # role inheritance is transitive
    Rule([("?r", "inherits", "?p"), ("?p", "inherits", "?g")], ("?r", "inherits", "?g"), "inherit-chain"),
    # you hold the permissions your role (or any inherited role) grants
    Rule([("?u", "has_role", "?r"), ("?r", "grants", "?p")], ("?u", "has_perm", "?p"), "perm-direct"),
    Rule([("?u", "has_role", "?r"), ("?r", "inherits", "?r2"), ("?r2", "grants", "?p")],
         ("?u", "has_perm", "?p"), "perm-inherited"),
    # raw capability: permission matches the resource's requirement
    Rule([("?u", "has_perm", "?p"), ("?a", "requires", "?p")], ("?u", "can", "?a"), "can"),
    # EFFECTIVE access: capable AND not suspended  (stratified negation: "unless")
    Rule([("?u", "can", "?a"), ("?u", "~suspended", "?when")], ("?u", "allowed", "?a"), "allowed"),
]
for rule in RULES:
    store.add_rule(rule, source="policy")
print(f"  {len(FACTS)} facts + {len(RULES)} rules stored. "
      "The closure (every implied permission) is materialized automatically.")

# ---------------------------------------------------------------- audit questions
banner("2. Ask the audit questions — every answer carries its derivation")
ask(store, {"action": "query", "pattern": ["?x", "allowed", "prod_deploy"]},
    "Who is ALLOWED to deploy to prod?")
ask(store, {"action": "check", "triple": ["dev_dan", "can", "main_branch"]},
    "Is the contractor CAPABLE of pushing to main?  (raw permission)")
ask(store, {"action": "check", "triple": ["dev_dan", "allowed", "main_branch"]},
    "...but is he ALLOWED?  (suspension wins — negation, not a special case)")
ask(store, {"action": "query", "pattern": ["erin", "allowed", "?x"]},
    "Everything the intern is allowed to touch")

# ---------------------------------------------------------------- what-if analysis
banner("3. What-if: how much access exists ONLY through role inheritance?")
store.set_rule_active("perm-inherited", False)
r = ask(store, {"action": "query", "pattern": ["?x", "allowed", "main_branch"]},
        "With inheritance DISABLED, who can still push to main?")
store.set_rule_active("perm-inherited", True)
r2 = ask(store, {"action": "query", "pattern": ["?x", "allowed", "main_branch"]},
         "Re-enabled: the full set again")
print(f"\n  -> {sorted(set(r2['answer']) - set(r['answer']))} hold push access only via inheritance.")

# ---------------------------------------------------------------- guardrails
banner("4. Guardrails: a typo can't silently corrupt the audit data")
store.set_meta("strict", "1")                   # relations were never declared loosely here:
for rel in ["has_role", "inherits", "grants", "requires", "has_perm", "can",
            "allowed", "suspended"]:
    store.declare(rel)
r = qa.run_action(store, {"action": "assert", "triple": ["frank", "hsa_role", "engineer"]})
print(f"  assert (frank hsa_role engineer)  ->  {r['kind']}: {r['trace']}")

banner("5. The decision log — every answer above is on record")
for _, kind, detail in reversed(store.recent_log(4)):
    print(f"  {kind:8s} {detail[:100]}")

store.close()
print("\nDone. Point this at a real org export and you have an instant access auditor.")
