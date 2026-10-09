"""Parse team answers and verify explicit solutions by differentiation."""

from __future__ import annotations

from typing import Optional

import numpy as np
import sympy as sp

from exam import config

x, y, C = sp.symbols("x y C")


def parse_answer(s: str) -> tuple[Optional[sp.Expr], str]:
    """Sympify an explicit y(x) string; rename the free constant to C if needed.

    Returns (expr_or_None, note). Notes include '' (ok), 'sin constante',
    or an error message (more than one free symbol / parse failure).
    """
    if s is None or not str(s).strip():
        return None, "respuesta vacía"
    try:
        # Do not pre-bind K/C aliases: any single free symbol (K, C, …) is
        # renamed to C below so renaming is testable and name-agnostic.
        expr = sp.sympify(str(s).strip(), locals={"x": x, "y": y, "C": C})
    except Exception as exc:  # noqa: BLE001
        return None, f"no se pudo parsear: {exc}"

    others = sorted(expr.free_symbols - {x}, key=lambda s: s.name)
    if len(others) == 0:
        return sp.sympify(expr), "sin constante"
    if len(others) == 1:
        sym = others[0]
        if sym != C:
            expr = expr.subs(sym, C)
        return sp.sympify(expr), ""
    names = ", ".join(s.name for s in others)
    return None, f"más de un símbolo libre además de x: {names}"


def verify_explicit(
    y_expr: sp.Expr,
    rhs: sp.Expr,
    *,
    x_range: tuple[float, float] | None = None,
) -> tuple[bool, str]:
    """Check y' == f(x, y) symbolically, with numeric fallback.

    Same core as ``simplify(diff(y,x) - rhs.subs(y, y_expr)) == 0``, plus
    20 random points when simplify does not reach 0.
    """
    try:
        residual = sp.simplify(sp.diff(y_expr, x) - rhs.subs(y, y_expr))
        if residual == 0:
            return True, ""
        # Numeric fallback
        xmin, xmax = x_range if x_range is not None else (0.5, 3.0)
        rng = np.random.default_rng(config.SEED)
        free = y_expr.free_symbols | residual.free_symbols
        has_c = C in free
        for _ in range(config.VERIFY_N_POINTS):
            xv = float(rng.uniform(xmin, xmax))
            subs = {x: xv}
            if has_c:
                # Prefer positive C so sqrt branches stay real on typical ranges.
                subs[C] = float(rng.uniform(0.5, 2.0))
            try:
                val = complex(residual.evalf(subs=subs))
            except Exception:
                return False, "fallback numérico falló"
            if abs(val) > config.VERIFY_TOL:
                return False, f"residuo simbólico={residual}; numérico={val}"
        return True, "ok por fallback numérico"
    except Exception as exc:  # noqa: BLE001
        return False, f"error en verificación: {exc}"


def corrupt_answer(s: str) -> str:
    """Negative-control corruption: append an extra term so the ODE fails."""
    return f"({s}) + x"


def x_range_for(exercise_id: str) -> tuple[float, float]:
    if exercise_id.startswith("C"):
        return config.X_RANGE_C
    if exercise_id.startswith("B"):
        return config.X_RANGE_B
    return config.X_RANGE_A
