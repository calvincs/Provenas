"""domains — reusable starter knowledge packs (facts + rules + sample questions).

Shared by the CLI (`:load <name>`) and the demo harness. Each pack is plain data: seed triples, a few
Horn rules, and example questions with expected answers. Add a pack here and it works everywhere.
"""
from __future__ import annotations

from provenas.infer import Rule as R

DOMAINS = [
    dict(
        name="family",
        blurb="a family tree (who is a parent of whom)",
        vocab=("relations are directional, written [subject, relation, object]:\n"
               "  (p parent c) — p is the parent of c\n"
               "  (a ancestor d) — a is the ancestor of d  (so 'D's ancestors' = query [\"?x\",\"ancestor\",\"d\"])\n"
               "  (g grandparent c) — g is the grandparent of c\n"
               "people: tom, bob, liz, ann, cy, dan, eve"),
        triples=[("tom", "parent", "bob"), ("tom", "parent", "liz"), ("bob", "parent", "ann"),
                 ("bob", "parent", "cy"), ("liz", "parent", "dan"), ("ann", "parent", "eve")],
        rules=[R([("?x", "parent", "?y")], ("?x", "ancestor", "?y"), "ancestor-base"),
               R([("?x", "parent", "?y"), ("?y", "ancestor", "?z")], ("?x", "ancestor", "?z"), "ancestor-step"),
               R([("?x", "parent", "?y"), ("?y", "parent", "?z")], ("?x", "grandparent", "?z"), "grandparent")],
        questions=[
            dict(q="Who are Tom's grandchildren?", expect={"ann", "cy", "dan"},
                 gold={"action": "query", "pattern": ["tom", "grandparent", "?x"]}),
            dict(q="Is Tom an ancestor of Eve?", expect=True,
                 gold={"action": "check", "triple": ["tom", "ancestor", "eve"]}),
            dict(q="Who are all of Cy's ancestors?", expect={"bob", "tom"},
                 gold={"action": "query", "pattern": ["?x", "ancestor", "cy"]}),
        ]),
    dict(
        name="rbac",
        blurb="access control: users hold roles, roles inherit roles and grant permissions, actions require permissions",
        vocab=("relations: has_role, inherits, grants, requires, has_perm, can.  "
               "users: alice, bob, dana.  roles: engineer, analyst, intern, employee.  "
               "permissions: deploy_perm, read_wiki, read_db.  actions: prod_deploy, view_wiki, query_db"),
        triples=[("alice", "has_role", "engineer"), ("bob", "has_role", "analyst"), ("dana", "has_role", "intern"),
                 ("engineer", "inherits", "employee"), ("analyst", "inherits", "employee"),
                 ("intern", "inherits", "employee"),
                 ("engineer", "grants", "deploy_perm"), ("employee", "grants", "read_wiki"),
                 ("analyst", "grants", "read_db"),
                 ("prod_deploy", "requires", "deploy_perm"), ("view_wiki", "requires", "read_wiki"),
                 ("query_db", "requires", "read_db")],
        rules=[R([("?u", "has_role", "?r"), ("?r", "inherits", "?r2")], ("?u", "has_role", "?r2"), "role-inherit"),
               R([("?u", "has_role", "?r"), ("?r", "grants", "?p")], ("?u", "has_perm", "?p"), "perm-from-role"),
               R([("?u", "has_perm", "?p"), ("?a", "requires", "?p")], ("?u", "can", "?a"), "can-do")],
        questions=[
            dict(q="Can Alice deploy to production?", expect=True,
                 gold={"action": "check", "triple": ["alice", "can", "prod_deploy"]}),
            dict(q="Can Dana query the database?", expect=False,
                 gold={"action": "check", "triple": ["dana", "can", "query_db"]}),
            dict(q="Who can view the wiki?", expect={"alice", "bob", "dana"},
                 gold={"action": "query", "pattern": ["?x", "can", "view_wiki"]}),
        ]),
    dict(
        name="diagnostics",
        blurb="troubleshooting: a symptom indicates a cause, a cause is fixed by an action",
        vocab=("relations are directional, written [subject, relation, object]:\n"
               "  [symptom, indicates, cause]   [cause, fixed_by, fix]   [fix, fixes, symptom]\n"
               "to find what fixes a symptom, query [?x, fixes, <symptom>].\n"
               "symptoms: no_boot, overheat.  causes: psu_fault, ram_fault, dust.  "
               "fixes: replace_psu, reseat_ram, clean_fans"),
        triples=[("no_boot", "indicates", "psu_fault"), ("no_boot", "indicates", "ram_fault"),
                 ("overheat", "indicates", "dust"), ("psu_fault", "fixed_by", "replace_psu"),
                 ("ram_fault", "fixed_by", "reseat_ram"), ("dust", "fixed_by", "clean_fans")],
        rules=[R([("?s", "indicates", "?c"), ("?c", "fixed_by", "?f")], ("?s", "resolved_by", "?f"), "resolve"),
               R([("?s", "resolved_by", "?f")], ("?f", "fixes", "?s"), "inverse-fix")],
        questions=[
            dict(q="How do I fix a machine that won't boot?", expect={"replace_psu", "reseat_ram"},
                 gold={"action": "query", "pattern": ["?f", "fixes", "no_boot"]}),
            dict(q="What resolves overheating?", expect={"clean_fans"},
                 gold={"action": "query", "pattern": ["?f", "fixes", "overheat"]}),
            dict(q="Is replacing the PSU a valid fix for no boot?", expect=True,
                 gold={"action": "check", "triple": ["no_boot", "resolved_by", "replace_psu"]}),
        ]),
    dict(
        name="eligibility",
        blurb="benefit eligibility: a person has attributes; a benefit needs an attribute",
        vocab=("relations: has_attr (person has_attr value), needs (benefit needs value), "
               "eligible_for (derived: person eligible_for benefit).\n"
               "people: alice, bob, carol, dave.  attributes: senior, adult, student, member.  "
               "benefits: senior_discount, student_discount, member_discount"),
        triples=[("alice", "has_attr", "senior"), ("alice", "has_attr", "member"),
                 ("bob", "has_attr", "student"), ("bob", "has_attr", "adult"),
                 ("carol", "has_attr", "senior"), ("dave", "has_attr", "adult"),
                 ("senior_discount", "needs", "senior"), ("student_discount", "needs", "student"),
                 ("member_discount", "needs", "member")],
        rules=[R([("?p", "has_attr", "?a"), ("?d", "needs", "?a")], ("?p", "eligible_for", "?d"), "eligible")],
        questions=[
            dict(q="What discounts is Alice eligible for?", expect={"senior_discount", "member_discount"},
                 gold={"action": "query", "pattern": ["alice", "eligible_for", "?d"]}),
            dict(q="Who is eligible for the student discount?", expect={"bob"},
                 gold={"action": "query", "pattern": ["?p", "eligible_for", "student_discount"]}),
            dict(q="Is Carol eligible for the member discount?", expect=False,
                 gold={"action": "check", "triple": ["carol", "eligible_for", "member_discount"]}),
        ]),
    dict(
        name="config",
        blurb="hardware compatibility: a part requires an interface, a board provides interfaces",
        vocab=("relations: requires (part requires interface), provides (board provides interface), "
               "compatible_with (derived: part compatible_with board).\n"
               "parts: gpu_x, ssd_y, ram_z.  boards: board_a, board_b.  "
               "interfaces: pcie4, sata, ddr5, pcie3"),
        triples=[("gpu_x", "requires", "pcie4"), ("ssd_y", "requires", "sata"), ("ram_z", "requires", "ddr5"),
                 ("board_a", "provides", "pcie4"), ("board_a", "provides", "sata"), ("board_a", "provides", "ddr5"),
                 ("board_b", "provides", "pcie3"), ("board_b", "provides", "sata")],
        rules=[R([("?part", "requires", "?i"), ("?board", "provides", "?i")],
                 ("?part", "compatible_with", "?board"), "compatible")],
        questions=[
            dict(q="What is gpu_x compatible with?", expect={"board_a"},
                 gold={"action": "query", "pattern": ["gpu_x", "compatible_with", "?b"]}),
            dict(q="Which parts are compatible with board_a?", expect={"gpu_x", "ssd_y", "ram_z"},
                 gold={"action": "query", "pattern": ["?part", "compatible_with", "board_a"]}),
            dict(q="Is ssd_y compatible with board_b?", expect=True,
                 gold={"action": "check", "triple": ["ssd_y", "compatible_with", "board_b"]}),
        ]),
]


def names():
    return [d["name"] for d in DOMAINS]


def get(name):
    return next((d for d in DOMAINS if d["name"] == name), None)


def load(store, name):
    """Seed a store with a pack's facts + rules. Returns the pack (or None)."""
    d = get(name)
    if d is None:
        return None
    for rel in sorted({t[1] for t in d["triples"]} | {r.head[1] for r in d["rules"]}):
        store.declare(rel)                              # schema first, so :strict stays satisfied
    for t in d["triples"]:
        store.assert_(*t, source="seed")
    for rule in d["rules"]:
        store.add_rule(rule, source="seed")
    store.set_meta("vocab", f"Domain: {d['blurb']}.\n{d['vocab']}")   # guides the LLM's translation
    store.log("load", name)
    return d
