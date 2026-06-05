"""Performance: distilled neural model vs the original Python function.

Measures inference throughput (calls/sec) for the calculator op:
  - Python `run_op` in a loop (the original static code)
  - Neural MLP, single-call (batch=1) on CPU      -> worst case for the net
  - Neural MLP, batched on CPU
  - Neural MLP, batched on GPU                     -> best case for the net

Speed is independent of the weights, so an untrained model is fine here. The
honest point: Python wins decisively per-call; the net only wins on large-batch
GPU throughput — and even then it trades exactness (see M1/M3) for speed.

  artifacts/perf_throughput.png
"""
from __future__ import annotations

import time

import numpy as np
import torch
import matplotlib.pyplot as plt

from provenas import plotting as P
from provenas.calculator import run_op, OPS
from provenas.encoders import PATHS
from provenas.models import CalcMLP

N = 200_000
REG_OPS = ["add", "subtract", "multiply"]


def _data(n):
    rng = np.random.default_rng(0)
    a = rng.uniform(-1e3, 1e3, n)
    b = rng.uniform(-1e3, 1e3, n)
    op = rng.integers(0, len(REG_OPS), n)
    return a, b, op


def bench_python(n):
    a, b, op = _data(n)
    t0 = time.perf_counter()
    for i in range(n):
        run_op(REG_OPS[op[i]], a[i], b[i])
    return n / (time.perf_counter() - t0)


def bench_neural(n, device, batch):
    a, b, op = _data(n)
    enc = PATHS["B"]
    model = CalcMLP().to(device).eval()
    X = torch.from_numpy(enc["encode_inputs"](a, b, op)).to(device)
    sync = (lambda: torch.cuda.synchronize()) if device == "cuda" else (lambda: None)
    with torch.no_grad():
        for i in range(0, min(5 * batch, n), batch):   # warmup
            model(X[i:i + batch])
        sync()
        t0 = time.perf_counter()
        for i in range(0, n, batch):
            model(X[i:i + batch])
        sync()
    return n / (time.perf_counter() - t0)


def main():
    results = {}
    results["Python\nrun_op (loop)"] = bench_python(N)
    results["Neural CPU\nbatch=1"] = bench_neural(20_000, "cpu", 1)   # fewer; it's slow
    results["Neural CPU\nbatch=4096"] = bench_neural(N, "cpu", 4096)
    if torch.cuda.is_available():
        results["Neural GPU\nbatch=65536"] = bench_neural(2_000_000, "cuda", 65536)

    for k, v in results.items():
        print(f"  {k.replace(chr(10), ' '):28s} {v:14,.0f} calls/sec")

    fig, ax = plt.subplots(figsize=(8.5, 5))
    names = list(results)
    vals = [results[k] for k in names]
    colors = ["#2ca02c", "#d62728", "#ff7f0e", "#1f77b4"][:len(names)]
    ax.bar(range(len(names)), vals, color=colors)
    ax.set_yscale("log")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, fontsize=9)
    ax.set_ylabel("throughput (calls/sec, log)")
    ax.set_title("Performance: Python function vs distilled neural net\n"
                 "(net only wins on large-batch GPU throughput — and trades exactness for it)")
    for i, v in enumerate(vals):
        ax.text(i, v, f" {v:,.0f}", ha="center", va="bottom", fontsize=8, rotation=0)
    ax.grid(True, axis="y", which="both", alpha=0.25)
    P.save(fig, "perf_throughput.png")


if __name__ == "__main__":
    main()
