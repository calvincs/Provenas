"""M2 deliverables: the error-classification head + masked multi-head loss.

Part 1 trains CalcMLP with BOTH heads on error-injected data (ok / ZeroDivision /
Overflow) using the log encoding (path B), exercising the full masked multi-head
loss, and reports the error-class confusion matrix.

Part 2 is the "learned error surface": P(ZeroDivisionError) across (a, b) for
op=divide, compared to Python's hard discontinuity at b=0. The boundary's
*sharpness depends on the encoding* — a sign-aware encoding (log/raw) gives the
net a near-exact zero-detector (razor boundary), so we deliberately use a SMOOTH
encoding of b (continuous through 0) with a visible gap between b=0 and the
nearest in-distribution divisor, which is what makes the soft band appear.

  artifacts/m2_confusion.png
  artifacts/m2_error_surface.png
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from provenas import strata as S
from provenas import plotting as P
from provenas.calculator import ERR_CLASSES, OP_INDEX, ZERO_DIV
from provenas.dataset import build, REG_OPS
from provenas.encoders import PATHS
from provenas.generate import run_ops_vec
from provenas.models import CalcMLP
from provenas.train import fit, predict, predict_proba
from provenas.metrics import confusion, per_class_recall

SEED = 0


def confusion_demo():
    train = build(200_000, np.random.default_rng(SEED), ops=REG_OPS,
                  allowed_strata=S.TRAIN_STRATA, frac_zero_div=0.15, frac_overflow=0.15)
    test = build(20_000, np.random.default_rng(SEED + 1), ops=REG_OPS,
                 allowed_strata=S.TRAIN_STRATA, frac_zero_div=0.15, frac_overflow=0.15)

    counts = np.bincount(train["error"], minlength=3)
    class_weight = (counts.sum() / (3 * np.maximum(counts, 1))).astype(np.float32)

    enc = PATHS["B"]
    model = CalcMLP()
    model, _ = fit(model, train, test, enc["encode_inputs"], enc["encode_target"],
                   w_reg=1.0, w_err=1.0, epochs=40, class_weight=class_weight,
                   seed=SEED, verbose=False)

    _, err_pred = predict(model, test, enc["encode_inputs"], enc["decode_target"])
    M = confusion(test["error"], err_pred, len(ERR_CLASSES))
    recall = per_class_recall(M)
    acc = float((err_pred == test["error"]).mean())
    print(f"  error-head accuracy = {acc:.4f}")
    for name, r in zip(ERR_CLASSES, recall):
        print(f"    recall[{name}] = {r:.4f}")

    fig, ax = plt.subplots(figsize=(5.5, 5))
    P.confusion_matrix_plot(ax, M, ERR_CLASSES, normalize=True)
    ax.set_title(f"M2 — error-class confusion (acc={acc:.3f})")
    P.save(fig, "m2_confusion.png")


def _divide_data(n, rng, frac_zero=0.4):
    """divide-only: ok with |b| in [1, 50]; ZeroDivisionError with b == 0.
    The gap (0, 1) has no training data, so a smooth classifier interpolates a
    visible soft band there."""
    a = rng.uniform(-50, 50, n)
    is_zero = rng.random(n) < frac_zero
    b = np.where(is_zero, 0.0,
                 rng.choice(np.array([-1.0, 1.0]), n) * rng.uniform(1.0, 50.0, n))
    op = np.full(n, OP_INDEX["divide"], dtype=np.int8)
    result, error = run_ops_vec(op, a, b)
    return {"a": a, "b": b, "op": op, "result": result, "error": error}


def _smooth_enc(a, b, _op):
    """Pure normalized scalars; b passes continuously through 0 (no sign tag)."""
    return np.stack([np.asarray(a, np.float64) / 50.0,
                     np.asarray(b, np.float64) / 50.0], axis=1).astype(np.float32)


def error_surface_demo():
    rng = np.random.default_rng(SEED + 5)
    train = _divide_data(120_000, rng)
    val = _divide_data(10_000, rng)

    def enc_t(y):
        return np.nan_to_num(np.asarray(y, np.float64) / 50.0, nan=0.0).astype(np.float32)[:, None]

    model = CalcMLP(in_dim=2)
    model, _ = fit(model, train, val, _smooth_enc, enc_t,
                   w_reg=0.0, w_err=1.0, epochs=40, seed=SEED, verbose=False)

    a_grid = np.linspace(-50, 50, 220)
    b_grid = np.linspace(-2, 2, 220)
    A, B = np.meshgrid(a_grid, b_grid, indexing="ij")
    X = _smooth_enc(A.reshape(-1), B.reshape(-1), None)
    Z = predict_proba(model, X)[:, ZERO_DIV].reshape(A.shape)

    # half-width of the learned band along b at a=0 (where P crosses 0.5)
    mid = Z[len(a_grid) // 2]
    over = b_grid[mid >= 0.5]
    band = float(over.max()) if over.size else 0.0
    print(f"  learned ZeroDiv band half-width ~ {band:.3f} (Python's is 0)")

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    im = P.heatmap(ax, Z, extent=[b_grid[0], b_grid[-1], a_grid[0], a_grid[-1]],
                   cmap="magma", vmin=0, vmax=1)
    ax.axvline(0.0, color="cyan", lw=1.5, ls="--")
    ax.text(0.05, a_grid[-1] * 0.86, "Python raises\nonly at b=0", color="cyan", fontsize=8, va="top")
    ax.set_xlabel("divisor b")
    ax.set_ylabel("dividend a")
    ax.set_title("M2 — learned P(ZeroDivisionError) for divide\nsoft band (smooth encoding) vs Python's hard discontinuity")
    fig.colorbar(im, ax=ax, label="P(ZeroDivisionError)")
    P.save(fig, "m2_error_surface.png")


def main():
    confusion_demo()
    error_surface_demo()


if __name__ == "__main__":
    main()
