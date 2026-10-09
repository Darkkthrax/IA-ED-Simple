"""ODE check, round-trip, and uniqueness for new poly / inv_k templates."""

from __future__ import annotations

import random

import sympy as sp

from src.data.generator import (
    F_TEMPLATES,
    G_TEMPLATES,
    GenConfig,
    split_outer_coeff,
    solution_satisfies_ode,
    try_build_example,
    x,
    y,
)
from src.data.tokenizer import Tokenizer, expressions_equivalent
from tests.test_uniqueness import K_VALUES, _pipeline_from_fg


def _gen_for_template(
    *,
    f_name: str | None = None,
    g_name: str | None = None,
    n: int = 200,
    seed: int = 0,
) -> list[dict[str, str]]:
    cfg = GenConfig(coeff_min=-5, coeff_max=5, max_degree=4, seed=seed)
    rng = random.Random(seed)
    rows: list[dict[str, str]] = []
    attempts = 0
    f_choices = [f_name] if f_name else list(F_TEMPLATES)
    g_choices = [g_name] if g_name else list(G_TEMPLATES)
    while len(rows) < n and attempts < 20_000:
        attempts += 1
        ex = try_build_example(rng.choice(f_choices), rng.choice(g_choices), rng, cfg)
        if ex is None:
            continue
        if f_name is not None and ex["f_template"] != f_name:
            continue
        if g_name is not None and ex["g_template"] != g_name:
            continue
        rows.append(ex)
    assert len(rows) == n, (
        f"only got {len(rows)}/{n} for f={f_name} g={g_name} after {attempts} attempts"
    )
    return rows


def test_poly_template_registered() -> None:
    assert "poly" in F_TEMPLATES
    assert F_TEMPLATES["poly"].__name__ == "f_poly"


def test_inv_k_template_registered() -> None:
    assert "inv_k" in G_TEMPLATES
    assert G_TEMPLATES["inv_k"].__name__ == "g_inv_k"


def test_poly_200_satisfy_ode_and_round_trip() -> None:
    tok = Tokenizer(max_int=20)
    rows = _gen_for_template(f_name="poly", n=200, seed=101)
    failures: list[str] = []
    for ex in rows:
        if not solution_satisfies_ode(ex["ode"], ex["solution"]):
            failures.append(f"ode-check: {ex['ode']} || {ex['solution']}")
            continue
        for field in ("ode", "solution"):
            ids = tok.encode(ex[field], add_special=True)
            try:
                rebuilt = tok.ids_to_sympy(ids)
                original = tok.parse_any(ex[field])
            except Exception as exc:  # noqa: BLE001
                failures.append(f"round-trip {field}: {ex[field]} err={exc}")
                continue
            if not expressions_equivalent(original, rebuilt):
                failures.append(f"round-trip {field}: {ex[field]} -> {rebuilt}")
    assert not failures, "poly failures:\n" + "\n".join(failures[:20])


def test_inv_k_200_satisfy_ode_and_round_trip() -> None:
    tok = Tokenizer(max_int=20)
    rows = _gen_for_template(g_name="inv_k", n=200, seed=202)
    failures: list[str] = []
    for ex in rows:
        if not solution_satisfies_ode(ex["ode"], ex["solution"]):
            failures.append(f"ode-check: {ex['ode']} || {ex['solution']}")
            continue
        for field in ("ode", "solution"):
            ids = tok.encode(ex[field], add_special=True)
            try:
                rebuilt = tok.ids_to_sympy(ids)
                original = tok.parse_any(ex[field])
            except Exception as exc:  # noqa: BLE001
                failures.append(f"round-trip {field}: {ex[field]} err={exc}")
                continue
            if not expressions_equivalent(original, rebuilt):
                failures.append(f"round-trip {field}: {ex[field]} -> {rebuilt}")
    assert not failures, "inv_k failures:\n" + "\n".join(failures[:20])


def test_poly_and_inv_k_uniqueness_under_rescaling() -> None:
    """Canonical ODE/solution invariant under (k*f, g/k) for new templates."""
    cfg = GenConfig(coeff_min=-5, coeff_max=5, max_degree=4, seed=303)
    rng = random.Random(303)
    checked = 0
    attempts = 0
    failures: list[str] = []
    # Force at least one of the new templates in each sample.
    combos = (
        [("poly", g) for g in G_TEMPLATES]
        + [(f, "inv_k") for f in F_TEMPLATES]
    )
    while checked < 200 and attempts < 12_000:
        attempts += 1
        f_name, g_name = rng.choice(combos)
        f0 = F_TEMPLATES[f_name](x, rng, cfg)
        g0 = G_TEMPLATES[g_name](y, rng, cfg)
        if f0 is None or g0 is None or f0 == 0 or g0 == 0:
            continue
        base = _pipeline_from_fg(f0, g0)
        if base is None:
            continue
        ode0, sol0 = base
        ok = True
        for k in K_VALUES:
            out = _pipeline_from_fg(k * f0, g0 / k)
            if out is None or out[0] != ode0 or out[1] != sol0:
                c0, core0 = split_outer_coeff(g0)
                ck, corek = split_outer_coeff(g0 / k)
                failures.append(
                    f"f_name={f_name} g_name={g_name} k={k}: f={f0}, g={g0}\n"
                    f"  base ODE={ode0!r} SOL={sol0!r}\n"
                    f"  var  ODE={None if out is None else out[0]!r} "
                    f"SOL={None if out is None else out[1]!r}\n"
                    f"  split(g)={(c0, core0)} split(g/k)={(ck, corek)}"
                )
                ok = False
                break
        if not ok:
            if len(failures) >= 10:
                break
            continue
        checked += 1

    assert not failures, (
        f"{len(failures)} uniqueness failures:\n" + "\n\n".join(failures[:10])
    )
    assert checked >= 200, f"only checked {checked}/200 after {attempts} attempts"


def test_f_poly_shape_and_g_inv_k_shape() -> None:
    cfg = GenConfig(coeff_min=-5, coeff_max=5, max_degree=4, seed=1)
    rng = random.Random(1)
    for _ in range(50):
        f_expr = F_TEMPLATES["poly"](x, rng, cfg)
        assert isinstance(sp.expand(f_expr), sp.Add)
        poly = sp.Poly(sp.expand(f_expr), x)
        degrees = [i for i, c in enumerate(reversed(poly.all_coeffs())) if c != 0]
        assert len(degrees) == 2
        assert 1 <= min(degrees) < max(degrees) <= 4
        g_expr = G_TEMPLATES["inv_k"](y, rng, cfg)
        num, den = sp.fraction(sp.together(g_expr))
        assert num == 1
        assert den.free_symbols == {y}
        k = int(sp.Poly(sp.expand(den), y).nth(1))
        assert 2 <= k <= 5
