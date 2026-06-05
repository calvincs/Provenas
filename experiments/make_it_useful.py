"""Item 3 — make it useful: ONE engine, MANY domains.

Trains ONE structure-only controller (on arithmetic reduction structure), drops it into
`provenas.engine.Engine`, then registers THREE domains — arithmetic, lists, boolean — as
simple op-tool tables and evaluates nested programs in each EXACTLY, generalizing depth, with
the SAME controller. Control is shared (learned, structural); computation is per-domain (exact
tools). This is the project's thesis as a reusable artifact + cross-domain pattern routing.

  artifacts/make_it_useful.png
"""
from __future__ import annotations

import operator

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from provenas import plotting as P
from provenas import exprgen as G
from provenas.exprgen import PREC, LEAF_PREC
from provenas.reducer import tree_to_state, local_next, reduce_to_value
from provenas.engine import ConvController, encode, Engine

SEED = 0


def interleave(a, b):
    out = []
    for i in range(max(len(a), len(b))):
        if i < len(a):
            out.append(a[i])
        if i < len(b):
            out.append(b[i])
    return out


DOMAINS = {
    "arithmetic": dict(ops=["+", "-", "*"],
                       opmap={"+": operator.add, "-": operator.sub, "*": operator.mul},
                       leaf=lambda r: int(r.integers(-9, 10)), show=str),
    "lists": dict(ops=["+", "*"],
                  opmap={"+": lambda a, b: a + b, "*": interleave},
                  leaf=lambda r: [int(r.integers(0, 9)) for _ in range(int(r.integers(1, 3)))], show=str),
    "boolean": dict(ops=["+", "*"],
                    opmap={"+": lambda a, b: a or b, "*": lambda a, b: a and b},
                    leaf=lambda r: bool(int(r.integers(0, 2))), show=lambda b: "T" if b else "F"),
    # fuzzy logic = the SAME engine, a different tool table: OR=max, AND=min (Zadeh), in [0,1]
    "fuzzy": dict(ops=["+", "*"],
                  opmap={"+": max, "*": min},
                  leaf=lambda r: round(float(r.uniform(0, 1)), 2), show=lambda x: f"{x:.2f}"),
}


def gen(depth, dom, rng):
    if depth == 0:
        return G.Node("leaf", value=dom["leaf"](rng))
    op = dom["ops"][int(rng.integers(len(dom["ops"])))]
    deep, shallow = gen(depth - 1, dom, rng), gen(int(rng.integers(0, depth)), dom, rng)
    l, r = (deep, shallow) if rng.random() < 0.5 else (shallow, deep)
    return G.Node("op", op=op, left=l, right=r)


def eval_tree(node, opmap):
    if node.kind == "leaf":
        return node.value
    return opmap[node.op](eval_tree(node.left, opmap), eval_tree(node.right, opmap))


def render(node, show):
    if node.kind == "leaf":
        return show(node.value)
    p = PREC[node.op]
    pr = lambda n: LEAF_PREC if n.kind == "leaf" else PREC[n.op]
    left = render(node.left, show)
    if pr(node.left) < p:
        left = f"({left})"
    right = render(node.right, show)
    if pr(node.right) <= p:
        right = f"({right})"
    return f"{left} {node.op} {right}"


def train_controller(device, rng, epochs=15):
    states, targets = [], []
    for s in G.build({1: 4000, 2: 4000, 3: 4000, 4: 4000}, rng, seen=set()):
        if s.error != 0:
            continue
        t, v = tree_to_state(s.tree)
        _, st, steps = reduce_to_value(t, v, local_next)
        if st != "ok":
            continue
        for stp, pos in steps:
            states.append(stp)
            targets.append(pos)
    targets = np.array(targets)
    model = ConvController().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)
    n = len(states)
    for _ in range(epochs):
        perm = rng.permutation(n)
        for i in range(0, n, 256):
            idx = perm[i:i + 256]
            ids, red = encode([states[j] for j in idx], device)
            loss = F.cross_entropy(model(ids).masked_fill(~red, -1e9),
                                   torch.tensor(targets[idx], device=device))
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = train_controller(device, np.random.default_rng(SEED))
    engine = Engine(model, device)
    for name, dom in DOMAINS.items():
        engine.register(name, dom["opmap"])

    print("\n=== one engine, three domains (depth-3 examples) ===")
    drng = np.random.default_rng(7)
    for name, dom in DOMAINS.items():
        tree = gen(3, dom, drng)
        types, vals = tree_to_state(tree)
        got = engine.evaluate(types, vals, name)
        true = eval_tree(tree, dom["opmap"])
        shown = dom["show"](got) if got is not None else "ERR"
        print(f"  [{name:10s}] {render(tree, dom['show'])}  =  {shown}   "
              f"({'ok' if got == true else 'MISMATCH'})")

    print("\n=== depth generalization (same controller, trained on depth<=4) ===")
    fig, ax = plt.subplots(figsize=(8.5, 5))
    for name, dom in DOMAINS.items():
        accs = []
        for d in range(1, 9):
            er = np.random.default_rng(100 + d)
            ok = 0
            for _ in range(300):
                tree = gen(d, dom, er)
                types, vals = tree_to_state(tree)
                ok += (engine.evaluate(types, vals, name) == eval_tree(tree, dom["opmap"]))
            accs.append(ok / 300)
        ax.plot(range(1, 9), accs, marker="o", label=name)
        print(f"  {name:10s}: {[round(a, 3) for a in accs]}")
    ax.axvspan(4.5, 8.5, alpha=0.06, color="red")
    ax.axvline(4.5, ls="--", color="gray")
    ax.text(4.6, 0.5, "trained 1-4\n-> extrapolation", color="gray", fontsize=8, va="center")
    ax.set_ylim(-0.03, 1.05)
    ax.set_xlabel("nesting depth")
    ax.set_ylabel("EXACT-match rate")
    ax.set_title("Item 3 — one structure-only controller, three domains, exact + depth-generalizing\n"
                 "(control is shared & learned; computation is per-domain exact tools)")
    ax.legend()
    ax.grid(True, alpha=0.25)
    P.save(fig, "make_it_useful.png")


if __name__ == "__main__":
    main()
