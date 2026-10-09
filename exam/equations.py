"""Fifteen first-order ODEs y' = f(x, y) for the exam (blocks A, B, C)."""

from __future__ import annotations

import sympy as sp

x, y = sp.symbols("x y")

# SymPy right-hand sides keyed A1..A5, B1..B5, C1..C5.
RHS: dict[str, sp.Expr] = {
    "A1": 6 * x**2,
    "A2": 5 * x**4,
    "A3": 8 * x**3,
    "A4": 3 * x**2 + 2 * x,
    "A5": 10 * x**4 - 4 * x,
    "B1": 2 * x / y,
    "B2": x**2 / y,
    "B3": x / (3 * y),
    "B4": 4 * x**3 / y,
    "B5": x / y**2,
    "C1": 2 * x * y,
    "C2": x**2 * y,
    "C3": 3 * x**2 * y,
    "C4": 4 * x**3 * y,
    "C5": 6 * x * y,
}

# Printable ODE strings (same form the Transformer solver expects).
ODE_STRINGS: dict[str, str] = {
    "A1": "y' = 6*x**2",
    "A2": "y' = 5*x**4",
    "A3": "y' = 8*x**3",
    "A4": "y' = 3*x**2 + 2*x",
    "A5": "y' = 10*x**4 - 4*x",
    "B1": "y' = 2*x/y",
    "B2": "y' = x**2/y",
    "B3": "y' = x/(3*y)",
    "B4": "y' = 4*x**3/y",
    "B5": "y' = x/y**2",
    "C1": "y' = 2*x*y",
    "C2": "y' = x**2*y",
    "C3": "y' = 3*x**2*y",
    "C4": "y' = 4*x**3*y",
    "C5": "y' = 6*x*y",
}

EXERCISE_IDS: tuple[str, ...] = tuple(RHS.keys())
