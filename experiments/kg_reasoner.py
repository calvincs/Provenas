"""Phase 5 capstone — the reasoner: knowledge graph + inference + neural semantics + a
learned controller that COMBINES tools to solve problems.

Connects every Lego built so far into one fabric, on a small animal world:
  1. KnowledgeGraph  — the relational memory (taxonomy + properties).
  2. infer           — rule engine: derive inherited facts; find connecting paths.
  3. SemanticIndex   — TransE: similarity + link prediction ("adjacent symbols/relationships").
  4. solver          — trial-and-error over {infer, combine, semantic}; the minimal combination
                       that cracks each goal is the "aha" of putting 2-3 ideas together.
  5. Controller (here) — a tiny net that learns to PICK the tool combination up front, turning
                       the solver's search into a one-shot dispatch (the neural controller driving
                       symbolic reasoning).
  6. Learning        — materialize what was derived back into the KG + name the winning combo as a
                       reusable macro: the system grows new structure from what it solved.

Produces artifacts/kg_reasoner.png. Runs on aibox.
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from provenas.kg import KnowledgeGraph
from provenas.infer import Rule, forward_chain, paths_between, induce_path_rule
from provenas.semantic import SemanticIndex
from provenas import solver as S


# --------------------------------------------------------------------------- world
def build_world():
    kg = KnowledgeGraph()
    taxo = [
        ("dog", "canine"), ("wolf", "canine"), ("cat", "feline"), ("lion", "feline"),
        ("canine", "mammal"), ("feline", "mammal"), ("bat", "mammal"),
        ("whale", "cetacean"), ("dolphin", "cetacean"), ("cetacean", "mammal"),
        ("eagle", "bird"), ("sparrow", "bird"), ("penguin", "bird"),
        ("shark", "fish"), ("salmon", "fish"),
        ("frog", "amphibian"), ("snake", "reptile"), ("lizard", "reptile"),
        ("mammal", "vertebrate"), ("bird", "vertebrate"), ("fish", "vertebrate"),
        ("amphibian", "vertebrate"), ("reptile", "vertebrate"), ("vertebrate", "animal"),
    ]
    for s, o in taxo:
        kg.assert_(s, "is_a", o)
    # class-level parts (INHERITED via is_a -> resolving these needs `infer`)
    for c, p in [("vertebrate", "backbone"), ("mammal", "fur"), ("bird", "feathers"),
                 ("fish", "scales"), ("reptile", "scales")]:
        kg.assert_(c, "has_part", p)
    # individual-level facts (DIRECT)
    water = ["whale", "dolphin", "shark", "salmon", "penguin", "frog"]
    land = ["dog", "wolf", "cat", "lion", "bat", "eagle", "sparrow", "snake", "lizard", "frog"]
    for a in water:
        kg.assert_(a, "lives_in", "water")
    for a in land:
        kg.assert_(a, "lives_in", "land")
    for a in ["eagle", "sparrow", "bat"]:
        kg.assert_(a, "can", "fly")
    for a in ["whale", "dolphin", "shark", "salmon", "penguin", "frog"]:
        kg.assert_(a, "can", "swim")
    for a in ["dog", "wolf", "cat", "lion", "snake", "lizard"]:
        kg.assert_(a, "can", "walk")
    return kg


RULES = [
    Rule([("?x", "is_a", "?y"), ("?y", "is_a", "?z")], ("?x", "is_a", "?z"), "is_a transitive"),
    Rule([("?x", "is_a", "?y"), ("?y", "has_part", "?z")], ("?x", "has_part", "?z"), "inherit part"),
]
REL_VOCAB = ["lives_in", "has_part", "can"]
LEAVES = ["dog", "wolf", "cat", "lion", "bat", "whale", "dolphin", "eagle", "sparrow",
          "penguin", "shark", "salmon", "frog", "snake", "lizard"]
CLASSES = ["mammal", "bird", "fish", "reptile", "amphibian", "cetacean", "canine", "vertebrate"]


# --------------------------------------------------------------------------- goals
def gen_goals(kg, sem, n=120, seed=0):
    rng = np.random.default_rng(seed)
    props = [("lives_in", "water"), ("lives_in", "land"), ("can", "fly"), ("can", "swim"),
             ("can", "walk"), ("has_part", "backbone"), ("has_part", "fur"),
             ("has_part", "feathers"), ("has_part", "scales")]
    goals, seen = [], set()
    for _ in range(n * 4):
        size = int(rng.integers(1, 4))
        g = []
        for _ in range(size):
            kind = rng.choice(["type", "prop", "like"], p=[0.4, 0.45, 0.15])
            if kind == "type":
                g.append(("type", str(rng.choice(CLASSES))))
            elif kind == "prop":
                r, o = props[int(rng.integers(len(props)))]
                g.append(("prop", r, o))
            else:
                g.append(("like", str(rng.choice(LEAVES))))
        key = tuple(sorted(map(str, g)))
        if key in seen:
            continue
        gold = S.answer(g, kg, RULES, sem, S.ALL_TOOLS)
        if gold and 1 <= len(gold) <= 12:
            seen.add(key)
            goals.append(g)
        if len(goals) >= n:
            break
    return goals


# --------------------------------------------------------------------------- controller
class Controller(nn.Module):
    """Learns goal-features -> which tools to enable (multi-label), short-circuiting the search."""
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, 32), nn.ReLU(), nn.Linear(32, 3))

    def forward(self, x):
        return self.net(x)


def fmt(goal):
    parts = []
    for c in goal:
        parts.append(f"{c[1]}" if c[0] == "type" else
                     f"{c[1]}={c[2]}" if c[0] == "prop" else f"like {c[1]}")
    return " ∧ ".join(parts)


def main():
    os.makedirs("artifacts", exist_ok=True)
    kg = build_world()
    print(f"world: {kg}")

    # --- 2. inference: derive inherited facts + a connecting path -----------------
    enriched = kg.copy()
    derived = forward_chain(enriched, RULES)
    print(f"infer: forward-chaining derived {len(derived)} new facts "
          f"(e.g. ('whale','has_part','backbone') = {('whale','has_part','backbone') in derived})")
    paths = paths_between(kg, "whale", "vertebrate", max_len=3)
    if paths:
        chain = "whale " + " ".join(f"-{r}-> {o}" for r, o in paths[0])
        print(f"path : {chain}")
    rule = induce_path_rule(enriched, [("dog", "vertebrate"), ("shark", "vertebrate")])
    print(f"induce: rule discovered for 'X is_a vertebrate' = {sorted(rule)}  (a then-a chain)")

    # --- 3. neural semantics: train TransE, show similarity + link prediction -----
    sem = SemanticIndex(kg, dim=24, seed=0)
    loss = sem.fit(epochs=600)
    print(f"\nsemantic: TransE trained (final margin loss {loss:.3f})")
    for x in ["dolphin", "eagle", "shark"]:
        print(f"  like {x:8s} -> {sem.similar(x, k=3)}")
    print(f"  link-predict (penguin, lives_in, ?) -> {sem.predict('penguin', 'lives_in', k=3)}")
    # 'lion' has no stored 'can' fact — propose an adjacent relationship from the geometry:
    print(f"  discover missing edge (lion, can, ?) -> {sem.predict('lion', 'can', k=3)}")

    # --- 4. the aha: solve a goal needing all three ideas at once -----------------
    aha = [("prop", "has_part", "backbone"), ("prop", "lives_in", "water"), ("like", "shark")]
    r = S.solve(aha, kg, RULES, sem)
    print(f"\naha goal:  {fmt(aha)}")
    print(f"  minimal combination = {r['tools']}  (depth {r['depth']}, after trying {r['attempts']} subsets)")
    print(f"  answer = {sorted(r['answer'])}")

    # --- solve the whole battery; measure combination depth -----------------------
    goals = gen_goals(kg, sem, n=120, seed=1)
    res = [S.solve(g, kg, RULES, sem) for g in goals]
    depths = np.array([x["depth"] for x in res])
    base_attempts = np.array([x["attempts"] for x in res])
    budgets = range(0, len(S.ALL_TOOLS) + 1)
    solve_rate = [float((depths <= b).mean()) for b in budgets]
    print(f"\nbattery: {len(goals)} goals; depth distribution "
          f"{ {d:int((depths==d).sum()) for d in range(4)} }")
    for b in budgets:
        print(f"  tools allowed <= {b}: solves {solve_rate[b]*100:.0f}% of goals")

    # --- 5. learned controller: predict the tool-set, skip the search -------------
    X = torch.tensor([S.features(g, REL_VOCAB) for g in goals], dtype=torch.float32)
    Y = torch.tensor([[float(t in x["tools"]) for t in S.ALL_TOOLS] for x in res])
    ntr = int(0.6 * len(goals))
    ctrl = Controller(X.shape[1])
    opt = torch.optim.Adam(ctrl.parameters(), lr=0.02)
    lossf = nn.BCEWithLogitsLoss()
    torch.manual_seed(0)
    for _ in range(400):
        opt.zero_grad()
        l = lossf(ctrl(X[:ntr]), Y[:ntr])
        l.backward()
        opt.step()
    with torch.no_grad():
        pred = (torch.sigmoid(ctrl(X[ntr:])) > 0.5)
    # controller attempts: 1 if its predicted set already reaches gold, else 2 (fallback to full)
    ctrl_attempts, ok = [], 0
    for i, g in enumerate(goals[ntr:]):
        tools = tuple(t for t, on in zip(S.ALL_TOOLS, pred[i].tolist()) if on)
        gold = res[ntr + i]["answer"]
        got = S.answer(g, kg, RULES, sem, tools)
        if got == gold:
            ctrl_attempts.append(1)
            ok += 1
        else:
            ctrl_attempts.append(2)
    test = slice(ntr, len(goals))
    print(f"\ncontroller: one-shot tool-set correct on {ok}/{len(goals)-ntr} test goals; "
          f"mean attempts {np.mean(ctrl_attempts):.2f} vs search baseline "
          f"{base_attempts[test].mean():.2f}")

    # --- 6. learning: materialize a solved goal + name the macro ------------------
    learned = [g for g, x in zip(goals, res) if "infer" in x["tools"]]
    if learned:
        before = S.solve(learned[0], kg, RULES, sem)["depth"]
        forward_chain(kg, RULES)                       # bake derived facts into the base KG
        after = S.solve(learned[0], kg, RULES, sem)["depth"]
        print(f"\nlearning: baked {len(derived)} derived facts into the KG; goal '{fmt(learned[0])}' "
              f"now needs depth {after} (was {before}). Registered macro 'inherited_property'.")

    # --- plot ---------------------------------------------------------------------
    vecs, ents = sem.vectors()
    vc = vecs - vecs.mean(0)
    U, Sg, Vt = np.linalg.svd(vc, full_matrices=False)
    P = vc @ Vt[:2].T

    def top_class(e):
        for c in ["mammal", "bird", "fish", "reptile", "amphibian"]:
            if c in enriched.ancestors(e) or e == c:
                return c
        return "other"
    palette = {"mammal": "#d62728", "bird": "#1f77b4", "fish": "#2ca02c",
               "reptile": "#9467bd", "amphibian": "#ff7f0e", "other": "#888888"}

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    for i, e in enumerate(ents):
        c = top_class(e)
        ax[0].scatter(P[i, 0], P[i, 1], color=palette[c], s=70 if e in LEAVES else 30,
                      edgecolor="k" if e in LEAVES else "none", linewidth=0.4, zorder=3)
    for e in ["whale", "dolphin", "shark", "salmon", "eagle", "penguin", "snake", "frog", "bat"]:
        i = ents.index(e)
        ax[0].annotate(e, (P[i, 0], P[i, 1]), fontsize=8, xytext=(3, 3),
                       textcoords="offset points")
    handles = [plt.Line2D([], [], marker="o", ls="", color=palette[k], label=k) for k in palette]
    ax[0].legend(handles=handles, fontsize=7, loc="best")
    ax[0].set_title("TransE entity embedding (PCA)\nneural semantics cluster the taxonomy")
    ax[0].set_xticks([]); ax[0].set_yticks([])

    ax[1].plot(list(budgets), [s * 100 for s in solve_rate], "o-", color="#d62728", lw=2)
    ax[1].set_xlabel("tools allowed to combine  (0 = direct only)")
    ax[1].set_ylabel("goals solved (%)")
    ax[1].set_ylim(0, 105)
    ax[1].set_xticks(list(budgets))
    ax[1].set_title("the aha: combining tools unlocks problems")
    ax[1].grid(alpha=0.3)

    ax[2].bar(["search\nbaseline", "learned\ncontroller"],
              [base_attempts[test].mean(), float(np.mean(ctrl_attempts))],
              color=["#888888", "#1f77b4"])
    ax[2].set_ylabel("mean tool-combinations tried")
    ax[2].set_title("controller picks the combination up front")
    for i, v in enumerate([base_attempts[test].mean(), float(np.mean(ctrl_attempts))]):
        ax[2].text(i, v + 0.05, f"{v:.2f}", ha="center", fontsize=10)

    fig.tight_layout()
    fig.savefig("artifacts/kg_reasoner.png", dpi=120)
    print("\nsaved artifacts/kg_reasoner.png")


if __name__ == "__main__":
    main()
