"""Encoder-agnostic training loop + prediction helpers."""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .losses import masked_multihead_loss


def _default_device(device):
    return device or ("cuda" if torch.cuda.is_available() else "cpu")


def _make_tensors(data, encode_inputs, encode_target):
    X = torch.from_numpy(encode_inputs(data["a"], data["b"], data["op"]))
    y = torch.from_numpy(encode_target(data["result"]))
    y = torch.nan_to_num(y, nan=0.0)  # error rows are masked; their value is irrelevant
    err = torch.from_numpy(np.asarray(data["error"], dtype=np.int64))
    return X, y, err


def fit(model, train_data, val_data, encode_inputs, encode_target, *,
        w_reg=1.0, w_err=1.0, epochs=50, batch_size=1024, lr=1e-3,
        class_weight=None, device=None, patience=8, seed=0, verbose=True):
    device = _default_device(device)
    torch.manual_seed(seed)
    model = model.to(device)

    Xtr, ytr, etr = _make_tensors(train_data, encode_inputs, encode_target)
    Xva, yva, eva = _make_tensors(val_data, encode_inputs, encode_target)
    Xva, yva, eva = Xva.to(device), yva.to(device), eva.to(device)
    if class_weight is not None:
        class_weight = torch.as_tensor(class_weight, dtype=torch.float32, device=device)

    loader = DataLoader(TensorDataset(Xtr, ytr, etr), batch_size=batch_size,
                        shuffle=True, drop_last=False)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=4)

    history = []
    best, best_state, bad = float("inf"), None, 0
    for ep in range(epochs):
        model.train()
        for xb, yb, eb in loader:
            xb, yb, eb = xb.to(device), yb.to(device), eb.to(device)
            opt.zero_grad()
            reg, errl = model(xb)
            loss, _, _ = masked_multihead_loss(reg, yb, errl, eb, w_reg, w_err, class_weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        model.eval()
        with torch.no_grad():
            reg, errl = model(Xva)
            vloss, vrl, vel = masked_multihead_loss(reg, yva, errl, eva, w_reg, w_err, class_weight)
            vloss = float(vloss)
        sched.step(vloss)
        history.append({"epoch": ep, "val_loss": vloss,
                        "val_reg": float(vrl), "val_err": float(vel)})
        if verbose:
            print(f"  ep {ep:3d}  val_loss {vloss:.6f}  reg {float(vrl):.6f}  err {float(vel):.6f}")

        if vloss < best - 1e-6:
            best, bad = vloss, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                if verbose:
                    print(f"  early stop at epoch {ep}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


@torch.no_grad()
def predict(model, data, encode_inputs, decode_target, device=None, batch_size=16384):
    """Return (result_hat in original units, predicted error-class idx)."""
    device = _default_device(device)
    model = model.to(device).eval()
    X = torch.from_numpy(encode_inputs(data["a"], data["b"], data["op"]))
    regs, errs = [], []
    for i in range(0, len(X), batch_size):
        reg, errl = model(X[i:i + batch_size].to(device))
        regs.append(reg.cpu().numpy())
        errs.append(errl.argmax(1).cpu().numpy())
    result_hat = decode_target(np.concatenate(regs).reshape(-1))
    return result_hat, np.concatenate(errs)


@torch.no_grad()
def predict_proba(model, X_np, device=None, batch_size=16384):
    """Softmax error-class probabilities for already-encoded inputs X_np."""
    device = _default_device(device)
    model = model.to(device).eval()
    X = torch.from_numpy(np.asarray(X_np, dtype=np.float32))
    out = []
    for i in range(0, len(X), batch_size):
        _, errl = model(X[i:i + batch_size].to(device))
        out.append(torch.softmax(errl, dim=1).cpu().numpy())
    return np.concatenate(out)
