"""CHECK 1: canonical ODE/solution must depend only on the product f*g."""

from __future__ import annotations

import random

import sympy as sp

from src.data.generator import (
    F_TEMPLATES,
    G_TEMPLATES,
    GenConfig,
    format_ode,
    format_solution,
    split_outer_coeff,
    strip_abs_in_logs,
    x,
    y,
)

# Rescaling factors for (f, g) -> (k*f, g/k).
K_VALUES = (sp.Integer(2), sp.Integer(-1), sp.Integer(3), sp.Rational(1, 2))


def _pipeline_from_fg(f_expr: sp.Expr, g_expr: sp.Expr) -> tuple[str, str] | None:
    """
    Same construction path as try_build_example (split -> integrate -> format/canonicalize).
    Returns (ode_str, sol_str) or None if the example would be discarded.
    """
    if f_expr == 0 or g_expr == 0:
        return None
    g_coeff, g_core = split_outer_coeff(g_expr)
    if g_core == 0:
        return None
    try:
        G = sp.integrate(1 / g_core, y)
        F = sp.integrate(g_coeff * f_expr, x)
    except Exception:
        return None
    if G is None or F is None:
        return None
    if isinstance(G, sp.Integral) or isinstance(F, sp.Integral):
        return None
    G = strip_abs_in_logs(G)
    F = strip_abs_in_logs(F)
    if (G.free_symbols | F.free_symbols) - {x, y}:
        return None
    return format_ode(f_expr, g_expr), format_solution(G, F)


def _sample_fg(rng: random.Random, cfg: GenConfig) -> tuple[sp.Expr, sp.Expr] | None:
    f_name = rng.choice(list(F_TEMPLATES))
    g_name = rng.choice(list(G_TEMPLATES))
    f_expr = F_TEMPLATES[f_name](x, rng, cfg)
    g_expr = G_TEMPLATES[g_name](y, rng, cfg)
    if f_expr is None or g_expr is None or f_expr == 0 or g_expr == 0:
        return None
    return f_expr, g_expr


def test_solution_depends_only_on_ode_product() -> None:
    """At least 200 (f,g) pairs: all (k*f, g/k) share identical canonical ODE+solution."""
    cfg = GenConfig(coeff_min=-5, coeff_max=5, max_degree=3, seed=123)
    rng = random.Random(123)
    checked = 0
    attempts = 0
    failures: list[str] = []

    while checked < 200 and attempts < 8000:
        attempts += 1
        sampled = _sample_fg(rng, cfg)
        if sampled is None:
            continue
        f0, g0 = sampled
        base = _pipeline_from_fg(f0, g0)
        if base is None:
            continue

        # Require every k-variant to succeed and match.
        variant_results: list[tuple[sp.Expr, str, str]] = []
        ok = True
        for k in K_VALUES:
            # Exact algebraic rescaling (also stress simplify-rewritten g).
            candidates = (k * f0, g0 / k), (sp.simplify(k * f0), sp.simplify(g0 / k))
            outs = []
            for f_k, g_k in candidates:
                if g_k == 0:
                    ok = False
                    break
                out = _pipeline_from_fg(f_k, g_k)
                if out is None:
                    ok = False
                    break
                outs.append((k, out[0], out[1]))
            if not ok:
                break
            variant_results.extend(outs)

        if not ok:
            continue

        ode0, sol0 = base
        matched = True
        for k, ode_k, sol_k in variant_results:
            if ode_k != ode0 or sol_k != sol0:
                c0, core0 = split_outer_coeff(g0)
                ck, corek = split_outer_coeff(g0 / k)
                failures.append(
                    f"k={k}: f={f0}, g={g0}\n"
                    f"  base ODE={ode0!r}\n"
                    f"  var  ODE={ode_k!r}\n"
                    f"  base SOL={sol0!r}\n"
                    f"  var  SOL={sol_k!r}\n"
                    f"  split(g)={(c0, core0)} split(g/k)={(ck, corek)}"
                )
                matched = False
                break
        if not matched:
            if len(failures) >= 10:
                break
            continue

        checked += 1

    assert not failures, (
        f"{len(failures)} uniqueness failures (showing up to 10):\n"
        + "\n\n".join(failures[:10])
    )
    assert checked >= 200, (
        f"Only checked {checked}/200 stable pairs after {attempts} attempts"
    )


def test_split_outer_coeff_primitive_linear() -> None:
    """6*y-8 and 3*y-4 (and sign flips) share the same primitive g_core."""
    cores = []
    for g in (6 * y - 8, 3 * y - 4, 4 - 3 * y, 8 - 6 * y, -(6 * y - 8)):
        _c, core = split_outer_coeff(g)
        cores.append(sp.expand(core))
    assert all(sp.simplify(c - cores[0]) == 0 for c in cores), cores
