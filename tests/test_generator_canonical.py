"""Tests for canonical scaling, canonicalize(), and ODE satisfaction."""

from __future__ import annotations

import random

import sympy as sp

from src.data.generator import (
    GenConfig,
    canonicalize,
    format_ode,
    format_solution,
    solution_satisfies_ode,
    split_outer_coeff,
    try_build_example,
    F_TEMPLATES,
    G_TEMPLATES,
    x,
    y,
    C,
)
from src.data.tokenizer import Tokenizer


def test_canonicalize_expands_logs() -> None:
    expr = sp.log(5 * x)
    out = canonicalize(expr)
    assert sp.simplify(out - (sp.log(5) + sp.log(x))) == 0


def test_split_outer_coeff_keeps_inner_linear() -> None:
    coeff, core = split_outer_coeff(3 * y - 4)
    assert coeff == 1
    assert sp.simplify(core - (3 * y - 4)) == 0


def test_canonical_scaling_examples() -> None:
    """Hand-checked targets from the project brief."""
    cases = [
        (8 * x, y, "log(y) = 4*x**2 + C"),
        (-16 / x, 1 / y, "y**2/2 = -16*log(x) + C"),
        (-10 * x**3, y**3, "-1/(2*y**2) = -5*x**4/2 + C"),
    ]
    for f_expr, g_expr, expected_sol in cases:
        g_coeff, g_core = split_outer_coeff(g_expr)
        G = sp.integrate(1 / g_core, y)
        F = sp.integrate(g_coeff * f_expr, x)
        sol = format_solution(G, F)
        ode = format_ode(f_expr, g_expr)
        assert sol == expected_sol, f"{ode}: got {sol!r}, expected {expected_sol!r}"
        assert solution_satisfies_ode(ode, sol)


def test_two_hundred_generated_satisfy_ode() -> None:
    cfg = GenConfig(coeff_min=-5, coeff_max=5, max_degree=3, seed=99)
    rng = random.Random(99)
    pairs: list[dict[str, str]] = []
    attempts = 0
    while len(pairs) < 200 and attempts < 8000:
        attempts += 1
        f_name = rng.choice(list(F_TEMPLATES))
        g_name = rng.choice(list(G_TEMPLATES))
        ex = try_build_example(f_name, g_name, rng, cfg)
        if ex is not None:
            pairs.append(ex)
    assert len(pairs) >= 200, f"Only got {len(pairs)} examples after {attempts} attempts"
    failures = []
    for ex in pairs:
        if not solution_satisfies_ode(ex["ode"], ex["solution"]):
            failures.append(f"{ex['ode']}  ||  {ex['solution']}")
    assert not failures, "ODE check failures:\n" + "\n".join(failures[:15])


def test_generated_canonical_round_trips() -> None:
    tok = Tokenizer(max_int=20)
    cfg = GenConfig(coeff_min=-4, coeff_max=4, max_degree=3, seed=11)
    rng = random.Random(11)
    pairs: list[dict[str, str]] = []
    attempts = 0
    while len(pairs) < 50 and attempts < 3000:
        attempts += 1
        ex = try_build_example(
            rng.choice(list(F_TEMPLATES)),
            rng.choice(list(G_TEMPLATES)),
            rng,
            cfg,
        )
        if ex is not None:
            pairs.append(ex)
    assert len(pairs) >= 50
    failures = []
    for ex in pairs:
        for field in ("ode", "solution"):
            if not tok.round_trip_ok(ex[field]):
                failures.append(f"{field}: {ex[field]} -> {tok.tokenize(ex[field])}")
    assert not failures, "Round-trip failures:\n" + "\n".join(failures[:20])
