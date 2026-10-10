"""Small feed-forward fit of an explicit solution; MAPE / NRMSE in percent."""

from __future__ import annotations

from typing import Any

import numpy as np
import sympy as sp
import torch
import torch.nn as nn

from exam import config

x, C = sp.symbols("x C")


def fit_and_error(
    y_expr: sp.Expr,
    c_value: float,
    x_min: float,
    x_max: float,
    epochs: int | None = None,
    seed: int | None = None,
    hidden_size: int | None = None,
) -> tuple[float, float]:
    """Port of the Colab ``fit_and_error`` with identical formulas."""
    epochs = config.EPOCHS if epochs is None else epochs
    seed = config.SEED if seed is None else seed
    hidden_size = config.HIDDEN_SIZE if hidden_size is None else hidden_size

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    f = sp.lambdify(x, y_expr.subs(C, c_value), "numpy")
    xs_train = np.linspace(x_min, x_max, config.N_TRAIN)
    xs_test = rng.uniform(x_min, x_max, config.N_TEST)
    ys_train, ys_test = f(xs_train), f(xs_test)
    mid, half = (x_min + x_max) / 2, (x_max - x_min) / 2
    scale = np.max(np.abs(ys_train))
    if scale == 0 or not np.isfinite(scale):
        scale = 1.0
    to_t = lambda a: torch.tensor(a, dtype=torch.float32).reshape(-1, 1)
    Xtr = to_t((xs_train - mid) / half)
    positive = bool(np.all(ys_train > 0))
    if positive:
        log_tr = np.log(ys_train)
        mu = float(log_tr.mean())
        sd = float(log_tr.std()) or 1.0
        Ytr = to_t((log_tr - mu) / sd)
        decode = lambda out: np.exp(out * sd + mu)
    else:
        Ytr = to_t(ys_train / scale)
        decode = lambda out: out * scale
    Xte = to_t((xs_test - mid) / half)
    h = hidden_size
    net = nn.Sequential(
        nn.Linear(1, h),
        nn.Tanh(),
        nn.Linear(h, h),
        nn.Tanh(),
        nn.Linear(h, 1),
    )
    opt = torch.optim.Adam(net.parameters(), lr=config.LR)
    for _ in range(epochs):
        opt.zero_grad()
        loss = nn.functional.mse_loss(net(Xtr), Ytr)
        loss.backward()
        opt.step()
    with torch.no_grad():
        pred = decode(net(Xte).numpy().ravel())
    mape = 100 * np.mean(np.abs(pred - ys_test) / np.abs(ys_test))
    nrmse = 100 * np.sqrt(np.mean((pred - ys_test) ** 2)) / (ys_test.max() - ys_test.min())
    return float(mape), float(nrmse)


def fit_and_curves(
    y_expr: sp.Expr,
    c_value: float,
    x_min: float,
    x_max: float,
    epochs: int | None = None,
    seed: int | None = None,
    hidden_size: int | None = None,
) -> dict[str, Any]:
    """Same training as ``fit_and_error``, plus sorted curves for plotting."""
    epochs = config.EPOCHS if epochs is None else epochs
    seed = config.SEED if seed is None else seed
    hidden_size = config.HIDDEN_SIZE if hidden_size is None else hidden_size

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    f = sp.lambdify(x, y_expr.subs(C, c_value), "numpy")
    xs_train = np.linspace(x_min, x_max, config.N_TRAIN)
    xs_test = rng.uniform(x_min, x_max, config.N_TEST)
    ys_train, ys_test = np.asarray(f(xs_train), dtype=float), np.asarray(f(xs_test), dtype=float)
    mid, half = (x_min + x_max) / 2, (x_max - x_min) / 2
    scale = float(np.max(np.abs(ys_train)))
    if scale == 0 or not np.isfinite(scale):
        scale = 1.0
    to_t = lambda a: torch.tensor(a, dtype=torch.float32).reshape(-1, 1)
    Xtr = to_t((xs_train - mid) / half)
    positive = bool(np.all(ys_train > 0))
    if positive:
        log_tr = np.log(ys_train)
        mu = float(log_tr.mean())
        sd = float(log_tr.std()) or 1.0
        Ytr = to_t((log_tr - mu) / sd)
        decode = lambda out: np.exp(out * sd + mu)
    else:
        Ytr = to_t(ys_train / scale)
        decode = lambda out: out * scale
    Xte = to_t((xs_test - mid) / half)
    h = hidden_size
    net = nn.Sequential(
        nn.Linear(1, h),
        nn.Tanh(),
        nn.Linear(h, h),
        nn.Tanh(),
        nn.Linear(h, 1),
    )
    opt = torch.optim.Adam(net.parameters(), lr=config.LR)
    for _ in range(epochs):
        opt.zero_grad()
        loss = nn.functional.mse_loss(net(Xtr), Ytr)
        loss.backward()
        opt.step()
    with torch.no_grad():
        pred_test = decode(net(Xte).numpy().ravel())
        xs_plot = np.linspace(x_min, x_max, 200)
        Xplot = to_t((xs_plot - mid) / half)
        pred_plot = decode(net(Xplot).numpy().ravel())
        exact_plot = np.asarray(f(xs_plot), dtype=float)

    mape = 100 * np.mean(np.abs(pred_test - ys_test) / np.abs(ys_test))
    denom = ys_test.max() - ys_test.min()
    if denom == 0:
        nrmse = float("nan")
    else:
        nrmse = 100 * np.sqrt(np.mean((pred_test - ys_test) ** 2)) / denom

    order = np.argsort(xs_test)
    return {
        "mape": float(mape),
        "nrmse": float(nrmse),
        "x_test": xs_test[order],
        "y_exact_test": ys_test[order],
        "y_pred_test": pred_test[order],
        "x_plot": xs_plot,
        "y_exact_plot": exact_plot,
        "y_pred_plot": pred_plot,
    }
