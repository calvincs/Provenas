"""Shared matplotlib helpers. Headless (Agg); writes PNGs to artifacts/."""
from __future__ import annotations

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ARTIFACTS = os.environ.get("PROVENAS_ARTIFACTS", "artifacts")


def save(fig, name):
    os.makedirs(ARTIFACTS, exist_ok=True)
    path = os.path.join(ARTIFACTS, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] wrote {path}")
    return path


def cliff_curve(ax, x, median, lo, hi, label=None, marker="o", **kw):
    line, = ax.plot(x, median, marker=marker, label=label, **kw)
    ax.fill_between(x, lo, hi, alpha=0.18, color=line.get_color())
    return line


def heatmap(ax, Z, extent, cmap="viridis", vmin=None, vmax=None):
    return ax.imshow(np.asarray(Z), origin="lower", aspect="auto",
                     extent=extent, cmap=cmap, vmin=vmin, vmax=vmax)


def confusion_matrix_plot(ax, M, labels, normalize=True):
    M = np.asarray(M, dtype=np.float64)
    if normalize:
        row = M.sum(axis=1, keepdims=True)
        Z = np.divide(M, row, out=np.zeros_like(M), where=row > 0)
    else:
        Z = M
    im = ax.imshow(Z, cmap="Blues", vmin=0, vmax=1 if normalize else None)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{Z[i, j]:.2f}", ha="center", va="center",
                    color="white" if Z[i, j] > 0.5 else "black", fontsize=8)
    return im
