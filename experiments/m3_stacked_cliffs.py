"""M3 deliverable: the stacked cliffs — raw (path A) vs log (path B) encoding.

Headline op is multiply, where log encoding's advantage is starkest: since
log(a*b) = log(a) + log(b), multiply becomes *addition* in log space, which a net
extrapolates. Both models train on operand strata 0-3 (|x| < 1e5), regression
only, then we evaluate across all 8 strata, decode to ORIGINAL units, and compare
relative error on a log-log axis. A float32-representation-floor reference marks
the best precision any float32 model could reach.

  artifacts/m3_stacked_cliffs.png
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from provenas import strata as S
from provenas import plotting as P
from provenas.dataset import build, eval_grid
from provenas.encoders import PATHS
from provenas.models import CalcMLP
from provenas.train import fit, predict
from provenas.metrics import relative_error, per_stratum_summary

SEED = 0
OP = "multiply"
FLOOR = 1e-9


def stratum_mid(s):
    _, lo, hi = S.STRATA[s]
    return 10 ** ((np.log10(lo) + np.log10(hi)) / 2)


def curve(summ):
    ss = sorted(summ)
    xs = [stratum_mid(s) for s in ss]
    med = [max(summ[s][0], FLOOR) for s in ss]
    lo = [max(summ[s][1], FLOOR) for s in ss]
    hi = [max(summ[s][2], FLOOR) for s in ss]
    return xs, med, lo, hi


def main():
    train = build(150_000, np.random.default_rng(SEED), ops=[OP], allowed_strata=S.TRAIN_STRATA)
    val = build(20_000, np.random.default_rng(SEED + 1), ops=[OP], allowed_strata=S.TRAIN_STRATA)

    grid = eval_grid([OP], list(range(S.N_STRATA)), 4000, np.random.default_rng(SEED + 99))
    ok = grid["error"] == 0

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for path, label in [("A", "path A (raw)"), ("B", "path B (log)")]:
        enc = PATHS[path]
        model = CalcMLP()
        model, _ = fit(model, train, val, enc["encode_inputs"], enc["encode_target"],
                       w_reg=1.0, w_err=0.0, epochs=60, seed=SEED, verbose=False)
        rhat, _ = predict(model, grid, enc["encode_inputs"], enc["decode_target"])
        rel = relative_error(rhat[ok], grid["result"][ok])
        summ = per_stratum_summary(rel, grid["joint_stratum"][ok], S.N_STRATA)
        xs, med, lo, hi = curve(summ)
        P.cliff_curve(ax, xs, med, lo, hi, label=label)
        in_dist = np.median([summ[s][0] for s in summ if s in S.TRAIN_STRATA])
        print(f"  {label:14s} in-dist median rel-err = {in_dist:.2e}")

    # float32 representation floor: best precision any float32 model could reach
    r = grid["result"][ok]
    with np.errstate(over="ignore", invalid="ignore"):
        f32rel = np.abs(r.astype(np.float32).astype(np.float64) - r) / (np.abs(r) + 1e-300)
    summ_f = per_stratum_summary(f32rel, grid["joint_stratum"][ok], S.N_STRATA)
    xs_f, med_f, _, _ = curve({s: (max(v[0], 1e-12), v[1], v[2], v[3]) for s, v in summ_f.items()})
    ax.plot(xs_f, med_f, "k:", lw=1.2, label="float32 precision floor (~6e-8)")

    ax.axvspan(S.TRAIN_MAX_ABS, 1e20, alpha=0.05, color="red")
    ax.axvline(S.TRAIN_MAX_ABS, ls="--", color="gray")
    ax.text(S.TRAIN_MAX_ABS, 0.5, " train edge 1e5  -> extrapolation", rotation=90,
            va="center", color="gray", fontsize=8, transform=ax.get_xaxis_transform())
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("operand magnitude |x|")
    ax.set_ylabel("relative error (median, p10-p90 band)")
    ax.set_title(f"M3 — {OP}: log encoding tames dynamic range, but neither extrapolates")
    ax.legend(loc="upper center", ncol=2, fontsize=8)
    ax.grid(True, which="both", alpha=0.25)
    P.save(fig, "m3_stacked_cliffs.png")


if __name__ == "__main__":
    main()
