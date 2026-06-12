"""Real-life example 2 — ELIGIBILITY SCREENING: "does this person qualify, and why?"

Benefits programs, clinic intake, legal aid, discounts — anywhere rules like "you qualify
if X and Y unless Z" live in a PDF and three people's heads. These decisions get APPEALED,
so "the system said no" is not an acceptable answer; "no, because income exceeds the
threshold" is. Comparison guards (>=, <) and negation (~disqualified) make the real rules
expressible, and pinned regression cases stop a rule change from silently flipping
last month's decisions.

Runs offline, no LLM, no third-party packages:   python examples/eligibility_screening.py
"""
from provenas.infer import Rule
from provenas.learn import admit_rule
from provenas.store import Store
from provenas import qa


def banner(t):
    print(f"\n{'=' * 72}\n  {t}\n{'=' * 72}")


def screen(store, person):
    r = qa.run_action(store, {"action": "query", "pattern": [person, "eligible_for", "?x"]})
    print(f"\n[screen] {person} qualifies for: {qa.show_answer(r)}")
    if r["trace"].strip():
        print("         " + r["trace"].replace("\n", "\n         "))
    return r


store = Store(":memory:")

# ---------------------------------------------------------------- applicants (intake data)
banner("1. Intake data: applicants as facts (values are plain strings; guards compare numerically)")
PEOPLE = [
    ("rosa",  "age", "71"), ("rosa",  "income", "21000"), ("rosa",  "household", "1"),
    ("marc",  "age", "34"), ("marc",  "income", "26500"), ("marc",  "household", "4"),
    ("jin",   "age", "67"), ("jin",   "income", "58000"), ("jin",   "household", "2"),
    ("tasha", "age", "29"), ("tasha", "income", "24000"), ("tasha", "household", "3"),
]
for t in PEOPLE:
    store.assert_(*t, source="intake")
store.assert_("tasha", "disqualified", "fraud-2025-441", source="case-review")
print(f"  {len(PEOPLE) + 1} facts stored for 4 applicants.")

# ---------------------------------------------------------------- the program rules
banner("2. The program rules — comparisons and 'unless' clauses, exactly as written in the policy")
RULES = [
    # Senior discount: age 65+
    Rule([("?p", "age", "?a"), ("?a", ">=", "65")],
         ("?p", "eligible_for", "senior_discount"), "senior-65plus"),
    # Income assistance: income under 30k, UNLESS disqualified by case review
    Rule([("?p", "income", "?i"), ("?i", "<", "30000"), ("?p", "~disqualified", "?why")],
         ("?p", "eligible_for", "income_assistance"), "assist-under-30k"),
    # Food support: household of 3+ AND income under 45k
    Rule([("?p", "household", "?h"), ("?h", ">=", "3"), ("?p", "income", "?i"), ("?i", "<", "45000")],
         ("?p", "eligible_for", "food_support"), "food-3plus-45k"),
]
for rule in RULES:
    store.add_rule(rule, source="policy-v1")
for rule in RULES:
    print(f"  {rule.name}: " + ", ".join("(%s %s %s)" % a for a in rule.body)
          + f"  =>  {rule.head[2]}")

# ---------------------------------------------------------------- screening, with reasons
banner("3. Screen everyone — each verdict carries the derivation an appeal would demand")
screen(store, "rosa")     # senior + assistance (71, 21k)
screen(store, "marc")     # assistance + food support (26.5k, household 4)
screen(store, "jin")      # senior only (67, but 58k income)
screen(store, "tasha")    # NOTHING despite 24k income — disqualified wins, with the case id

# ---------------------------------------------------------------- pin decisions; gate changes
banner("4. Pin this month's decisions — then watch the gate refuse a rule that flips one")
case = {"action": "check", "triple": ["tasha", "eligible_for", "income_assistance"]}
store.add_case("tasha-stays-denied", case, qa.run_action(store, case)["answer"])
print("  pinned: tasha's denial is now a regression case every future rule must preserve.")

# someone proposes a 'simpler' assistance rule that forgets the disqualification clause:
sloppy = Rule([("?p", "income", "?i"), ("?i", "<", "30000")],
              ("?p", "eligible_for", "income_assistance"), "assist-v2-sloppy")
ok, rep = admit_rule(store, sloppy, "eligible_for",
                     positives=[("rosa", "income_assistance"), ("marc", "income_assistance")],
                     negatives=[])
print(f"\n  proposed rule assist-v2-sloppy (drops the ~disqualified clause)")
print(f"  examples pass: {not rep['missing'] and not rep['violated']}   "
      f"pinned cases flipped: {rep['case_flips']}")
print(f"  -> {'ADMITTED' if ok else 'REJECTED — a rule that quietly re-approves a denied case never lands'}")

assert not ok and rep["case_flips"] == ["tasha-stays-denied"]
print(f"\n  rules in force are unchanged: {sorted(r.name for r in store.rules())}")

store.close()
print("\nDone. Policy in rules, decisions with reasons, changes gated by precedent.")
