"""M1 deliverable: the extrapolation cliff.

The cleanest "interpolate in-range, extrapolate off a cliff" demonstration:
train one small MLP per op on operands drawn from a box [-M_train, M_train], then
sweep the test operand magnitude outward and watch relative error. One model per
op (path A is literally the "single op" path) avoids a shared regression head
where multiply's ∝M^2 range swamps add/subtract.

Honest finding this exposes: the *linear* ops (add, subtract) extrapolate well —
a ReLU net learns the exact affine map and it keeps holding past the training
range — while the *nonlinear* op (multiply) interpolates in-range and then falls
off a cliff once operands leave [-M_train, M_train]. Plot -> artifacts/m1_cliff.png.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from provenas import plotting as P
from provenas.calculator import OP_INDEX
from provenas.generate import run_ops_vec
from provenas.models import CalcMLP
from provenas.train import fit, predict
from provenas.metrics import relative_error

SEED = 0
TRAIN_M = 50.0
OPS_M1 = ["add", "subtract", "multiply"]
TEST_MAGS = np.logspace(0, 8, 17)   # operand magnitude 1e0 .. 1e8
ERR_FLOOR = 1e-9


def _oracle(a, b, op):
    opa = np.full(len(a), OP_INDEX[op], dtype=np.int8)
    result, error = run_ops_vec(opa, a, b)
    return {"a": a, "b": b, "op": opa, "result": result, "error": error}


def make_box(n, M, op, rng):
    """Operands ~ U(-M, M) (linear box) for a single op."""
    return _oracle(rng.uniform(-M, M, n), rng.uniform(-M, M, n), op)


def make_at(M, n, op, rng):
    """Operands at magnitude ~M (|x| in [0.5M, M], random sign) for the sweep."""
    def s():
        return rng.choice(np.array([-1.0, 1.0]), n) * rng.uniform(0.5 * M, M, n)
    return _oracle(s(), s(), op)


def make_encoders(norm_scale, target_scale):
    """Pure normalized scalars, no clip, no op tag (single-op model). in_dim=2."""
    def enc_in(a, b, _op):
        a = np.asarray(a, np.float64) / norm_scale
        b = np.asarray(b, np.float64) / norm_scale
        return np.stack([a, b], axis=1).astype(np.float32)

    def enc_t(y):
        return (np.asarray(y, np.float64) / target_scale).astype(np.float32)[:, None]

    def dec_t(t):
        return np.asarray(t, np.float64).reshape(-1) * target_scale

    return dict(encode_inputs=enc_in, encode_target=enc_t, decode_target=dec_t)


def main():
    rng = np.random.default_rng(SEED)
    fig, ax = plt.subplots(figsize=(8.5, 5.5))

    for op in OPS_M1:
        tr = make_box(100_000, TRAIN_M, op, rng)
        va = make_box(10_000, TRAIN_M, op, rng)
        target_scale = float(np.percentile(np.abs(tr["result"]), 99)) or 1.0
        enc = make_encoders(TRAIN_M, target_scale)

        model = CalcMLP(in_dim=2)
        model, _ = fit(model, tr, va, enc["encode_inputs"], enc["encode_target"],
                       w_reg=1.0, w_err=0.0, epochs=120, lr=1e-3, seed=SEED, verbose=False)

        eval_rng = np.random.default_rng(SEED + 7)
        meds = []
        for M in TEST_MAGS:
            g = make_at(M, 4000, op, eval_rng)
            rhat, _ = predict(model, g, enc["encode_inputs"], enc["decode_target"])
            meds.append(float(np.median(relative_error(rhat, g["result"]))))
        meds = np.maximum(meds, ERR_FLOOR)
        ax.plot(TEST_MAGS, meds, marker="o", label=op)

        best = float(np.min(meds))
        plateau = float(meds[-1])
        print(f"  {op:9s} best in-range rel-err = {best:.2e}   extrapolation rel-err = {plateau:.2e}")

    ax.axvspan(TRAIN_M, TEST_MAGS[-1] * 1.4, alpha=0.05, color="red")
    ax.axvline(TRAIN_M, ls="--", color="gray")
    ax.text(TRAIN_M, 0.5, f" train edge |x|={TRAIN_M:g}", rotation=90, va="center",
            color="gray", fontsize=8, transform=ax.get_xaxis_transform())
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("test operand magnitude |x|")
    ax.set_ylabel("relative error (median over 4k samples)")
    ax.set_title("M1 — path A MLP (one per op): the extrapolation cliff")
    ax.legend()
    ax.grid(True, which="both", alpha=0.25)
    P.save(fig, "m1_cliff.png")


if __name__ == "__main__":
    main()
