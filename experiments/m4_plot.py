"""Aggregate the parallel M4 runs (artifacts/m4_runs/*.json) into the two
deliverable plots, with mean+/-std error bars across seeds.

  artifacts/m4_cvd_bars.png       headline fidelity comparison (in-distribution)
  artifacts/m4_depth_scaling.png  error vs nesting depth (incl. held-out depths 5-7)
"""
from __future__ import annotations

import glob
import json

import numpy as np
import matplotlib.pyplot as plt

from provenas import plotting as P

RUNS = "artifacts/m4_runs"
COLORS = ["#1f77b4", "#ff7f0e"]


def load():
    runs = {}
    for p in sorted(glob.glob(f"{RUNS}/*.json")):
        with open(p) as f:
            r = json.load(f)
        runs.setdefault(r["label"], []).append(r)
    return runs


def main():
    runs = load()
    labels = sorted(runs)
    n_seeds = min(len(v) for v in runs.values())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    x = np.arange(len(labels))
    for ax, key, title, ylabel, logy in [
            (ax1, "median_rel", "Numeric fidelity (lower = better)",
             "median relative error (ok rows)", True),
            (ax2, "err_acc", "Error detection (higher = better)",
             "error-head accuracy", False)]:
        means = [float(np.mean([r["fidelity"][key] for r in runs[l]])) for l in labels]
        stds = [float(np.std([r["fidelity"][key] for r in runs[l]])) for l in labels]
        ax.bar(x, means, yerr=stds, capsize=4, color=COLORS[:len(labels)])
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        if logy:
            ax.set_yscale("log")
            ax.grid(True, axis="y", which="both", alpha=0.25)
        else:
            ax.set_ylim(0, 1.02)
            ax.grid(True, axis="y", alpha=0.25)
        for xi, m in zip(x, means):
            ax.text(xi, m, f" {m:.3g}", ha="center", va="bottom", fontsize=8)
    fig.suptitle(f"M4 — parsing vs computing (in-distribution, mean±std over {n_seeds} seeds)")
    P.save(fig, "m4_cvd_bars.png")

    fig, ax = plt.subplots(figsize=(8.5, 5))
    for lab, color in zip(labels, COLORS):
        rs = runs[lab]
        depths = sorted({int(d) for r in rs for d in r["depth_curve"]})
        means, stds = [], []
        for d in depths:
            vals = [r["depth_curve"][str(d)] for r in rs if str(d) in r["depth_curve"]]
            means.append(np.mean(vals))
            stds.append(np.std(vals))
        means, stds = np.array(means), np.array(stds)
        ax.plot(depths, means, marker="o", label=lab, color=color)
        ax.fill_between(depths, np.maximum(means - stds, 1e-12), means + stds, alpha=0.18, color=color)
    ax.axvspan(4.5, 7.5, alpha=0.06, color="red")
    ax.axvline(4.5, ls="--", color="gray")
    ax.text(4.6, ax.get_ylim()[1] * 0.4, "held-out\ndepths 5-7", color="gray", fontsize=8, va="top")
    ax.set_yscale("log")
    ax.set_xlabel("expression nesting depth")
    ax.set_ylabel("median relative error (ok rows)")
    ax.set_title(f"M4 — depth scaling: trained 1-4, held-out 5-7 (mean±std over {n_seeds} seeds)")
    ax.legend()
    ax.grid(True, which="both", alpha=0.25)
    P.save(fig, "m4_depth_scaling.png")


if __name__ == "__main__":
    main()
