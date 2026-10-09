"""Optional Transformer check via ``src.solve.solve`` (skippable if no model)."""

from __future__ import annotations

from typing import Any, Optional

import sympy as sp

from exam import config
from exam.verify import verify_explicit, x_range_for

x, y, C = sp.symbols("x y C")


def explicit(sol_str: str) -> list[sp.Expr]:
    """Solve an implicit solution string for y (ported helper)."""
    lhs, rhs = sol_str.split("=", 1)
    loc = {"x": x, "y": y, "C": C, "exp": sp.exp, "log": sp.log}
    eq = sp.Eq(sp.sympify(lhs, locals=loc), sp.sympify(rhs, locals=loc))
    return list(sp.solve(eq, y))


def _is_real_expr(expr: sp.Expr) -> bool:
    """Drop roots that still contain the imaginary unit."""
    try:
        if expr.has(sp.I):
            return False
        return True
    except Exception:
        return False


def check_transformer(
    exercise_id: str,
    ode_str: str,
    rhs: sp.Expr,
    model: Any,
    tokenizer: Any,
    device: Any = None,
    *,
    cache: Optional[dict[str, dict[str, Any]]] = None,
) -> str:
    """Return 'SÍ', 'NO', or '-' (skipped). Calls ``solve`` at most once per ODE."""
    if model is None or tokenizer is None:
        return "-"

    from src.solve import solve

    if cache is not None and ode_str in cache:
        result = cache[ode_str]
    else:
        result = solve(
            ode_str,
            model,
            tokenizer,
            beam_size=config.BEAM_SIZE,
            top_k=config.BEAM_SIZE,
            device=device,
        )
        if cache is not None:
            cache[ode_str] = result

    if not result.get("verified"):
        return "NO"
    sol = result.get("solution")
    if not sol:
        return "NO"

    try:
        roots = explicit(sol)
    except Exception:
        return "NO"

    x_range = x_range_for(exercise_id)
    for root in roots:
        if not _is_real_expr(root):
            continue
        # Rename any free constant other than x to C for verify_explicit.
        others = list(root.free_symbols - {x})
        y_expr = root
        if len(others) == 1 and others[0] != C:
            y_expr = root.subs(others[0], C)
        ok, _note = verify_explicit(y_expr, rhs, x_range=x_range)
        if ok:
            return "SÍ"
    return "NO"


def transformer_explicit_forms(
    ode_str: str,
    model: Any,
    tokenizer: Any,
    device: Any = None,
    *,
    cache: Optional[dict[str, dict[str, Any]]] = None,
) -> list[sp.Expr]:
    """Real explicit y-roots from a verified Transformer solution (for display)."""
    if model is None or tokenizer is None:
        return []
    from src.solve import solve

    if cache is not None and ode_str in cache:
        result = cache[ode_str]
    else:
        result = solve(
            ode_str,
            model,
            tokenizer,
            beam_size=config.BEAM_SIZE,
            top_k=config.BEAM_SIZE,
            device=device,
        )
        if cache is not None:
            cache[ode_str] = result
    if not result.get("verified") or not result.get("solution"):
        return []
    try:
        roots = explicit(result["solution"])
    except Exception:
        return []
    return [r for r in roots if _is_real_expr(r)]
