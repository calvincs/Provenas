"""Generate the shared M4 expression dataset ONCE (seeded) and serialize it.

Both path C and path D load this identical file, so they see byte-identical
expressions — a fairness requirement. Strings are deduped across all splits, so no
test string appears in train. Depths 1-4 are trained; depths 5-7 are held out for
the depth-scaling extrapolation probe.

  data/m4_expressions.pkl
"""
from __future__ import annotations

import os
import pickle

import numpy as np

from provenas import exprgen as G

SEED = 0
OUT = "data/m4_expressions.pkl"


def main():
    rng = np.random.default_rng(SEED)
    seen = set()
    train = G.build({1: 6000, 2: 6000, 3: 6000, 4: 6000}, rng, seen=seen)
    val = G.build({1: 1000, 2: 1000, 3: 1000, 4: 1000}, rng, seen=seen)
    test_indist = G.build({1: 2000, 2: 2000, 3: 2000, 4: 2000}, rng, seen=seen)
    test_depth = G.build({5: 2000, 6: 2000, 7: 2000}, rng, seen=seen)

    os.makedirs("data", exist_ok=True)
    with open(OUT, "wb") as f:
        pickle.dump({"train": train, "val": val,
                     "test_indist": test_indist, "test_depth": test_depth}, f)

    for name, split in [("train", train), ("val", val),
                        ("test_indist", test_indist), ("test_depth", test_depth)]:
        err = float(np.mean([s.error != 0 for s in split]))
        print(f"  {name:12s} n={len(split):6d}  error-rate={err:.3f}")
    print(f"  wrote {OUT}")


if __name__ == "__main__":
    main()
