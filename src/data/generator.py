"""Synthetic separable ODE dataset generation via SymPy (generate backwards)."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

import sympy as sp
import yaml
from tqdm import tqdm

x, y = sp.symbols("x y", real=True, positive=True)
C = sp.Symbol("C")

# ---------------------------------------------------------------------------
# Expression builders (controlled grammar)
# ---------------------------------------------------------------------------

ExprBuilder = Callable[[sp.Symbol, random.Random, "GenConfig"], Optional[sp.Expr]]


@dataclass
class GenConfig:
    """Configuration for separable ODE generation."""

    n_samples: int = 100_000
    n_heldout: int = 5_000
    train_ratio: float = 0.90
    val_ratio: float = 0.05
    test_ratio: float = 0.05
    coeff_min: int = -5
    coeff_max: int = 5
    exclude_zero_coeff: bool = True
    max_degree: int = 3
    max_token_length: int = 64
    output_dir: str = "data/processed"
    family: str = "separable"
    seed: int = 42


def _rand_coeff(rng: random.Random, cfg: GenConfig, allow_zero: bool = False) -> int:
    """Sample a small integer coefficient."""
    lo, hi = cfg.coeff_min, cfg.coeff_max
    if hi < lo:
        raise ValueError("coeff_max must be >= coeff_min")
    for _ in range(50):
        c = rng.randint(lo, hi)
        if allow_zero or not cfg.exclude_zero_coeff or c != 0:
            return c
    return 1 if hi >= 1 else lo


def _rand_pos_coeff(rng: random.Random, cfg: GenConfig) -> int:
    """Sample a positive coefficient in [1, coeff_max] (fallback to 1..5)."""
    hi = max(1, abs(cfg.coeff_max))
    return rng.randint(1, hi)


def _rand_degree(rng: random.Random, cfg: GenConfig) -> int:
    return rng.randint(1, max(1, cfg.max_degree))


# --- f(x) templates ---


def f_const(var: sp.Symbol, rng: random.Random, cfg: GenConfig) -> sp.Expr:
    return sp.Integer(_rand_coeff(rng, cfg))


def f_linear(var: sp.Symbol, rng: random.Random, cfg: GenConfig) -> sp.Expr:
    a, b = _rand_coeff(rng, cfg), _rand_coeff(rng, cfg, allow_zero=True)
    return a * var + b


def f_power(var: sp.Symbol, rng: random.Random, cfg: GenConfig) -> sp.Expr:
    a = _rand_coeff(rng, cfg)
    n = _rand_degree(rng, cfg)
    return a * var**n


def f_inv(var: sp.Symbol, rng: random.Random, cfg: GenConfig) -> sp.Expr:
    return sp.Integer(_rand_coeff(rng, cfg)) / var


def f_exp(var: sp.Symbol, rng: random.Random, cfg: GenConfig) -> sp.Expr:
    a = _rand_coeff(rng, cfg)
    b = _rand_coeff(rng, cfg)
    return a * sp.exp(b * var)


def f_log(var: sp.Symbol, rng: random.Random, cfg: GenConfig) -> sp.Expr:
    a = _rand_coeff(rng, cfg)
    b = _rand_pos_coeff(rng, cfg)
    return a * sp.log(b * var)


def f_poly_inv(var: sp.Symbol, rng: random.Random, cfg: GenConfig) -> sp.Expr:
    a = _rand_coeff(rng, cfg)
    n = _rand_degree(rng, cfg)
    b = _rand_coeff(rng, cfg)
    return a * var**n + b / var


def f_rational_power(var: sp.Symbol, rng: random.Random, cfg: GenConfig) -> sp.Expr:
    a = _rand_coeff(rng, cfg)
    n = _rand_degree(rng, cfg)
    return a / var**n


def f_poly(var: sp.Symbol, rng: random.Random, cfg: GenConfig) -> sp.Expr:
    """Two-term polynomial: a*x**n + b*x**m with 1 <= m < n <= max_degree."""
    n_hi = max(2, cfg.max_degree)
    n = rng.randint(2, n_hi)
    m = rng.randint(1, n - 1)
    a = _rand_coeff(rng, cfg)
    b = _rand_coeff(rng, cfg)
    return a * var**n + b * var**m


F_TEMPLATES: dict[str, ExprBuilder] = {
    "const": f_const,
    "linear": f_linear,
    "power": f_power,
    "inv": f_inv,
    "exp": f_exp,
    "log": f_log,
    "poly_inv": f_poly_inv,
    "rational_power": f_rational_power,
    "poly": f_poly,
}


# --- g(y) templates (1/g must be integrable) ---


def g_const(var: sp.Symbol, rng: random.Random, cfg: GenConfig) -> sp.Expr:
    return sp.Integer(_rand_coeff(rng, cfg))


def g_linear(var: sp.Symbol, rng: random.Random, cfg: GenConfig) -> sp.Expr:
    a, b = _rand_coeff(rng, cfg), _rand_coeff(rng, cfg, allow_zero=True)
    return a * var + b


def g_power(var: sp.Symbol, rng: random.Random, cfg: GenConfig) -> sp.Expr:
    a = _rand_coeff(rng, cfg)
    n = _rand_degree(rng, cfg)
    return a * var**n


def g_inv(var: sp.Symbol, rng: random.Random, cfg: GenConfig) -> sp.Expr:
    return sp.Integer(_rand_coeff(rng, cfg)) / var


def g_exp(var: sp.Symbol, rng: random.Random, cfg: GenConfig) -> sp.Expr:
    a = _rand_coeff(rng, cfg)
    b = _rand_coeff(rng, cfg)
    return a * sp.exp(b * var)


def g_log(var: sp.Symbol, rng: random.Random, cfg: GenConfig) -> sp.Expr:
    # Use y*log(b*y) so 1/g = 1/(y log(...)) integrates to log(log(...)), not li.
    a = _rand_coeff(rng, cfg)
    b = _rand_pos_coeff(rng, cfg)
    return a * var * sp.log(b * var)


def g_rational_power(var: sp.Symbol, rng: random.Random, cfg: GenConfig) -> sp.Expr:
    a = _rand_coeff(rng, cfg)
    n = _rand_degree(rng, cfg)
    return a / var**n


def g_inv_k(var: sp.Symbol, rng: random.Random, cfg: GenConfig) -> sp.Expr:
    """Rational inverse: 1/(k*y) with integer k in [2, 5]."""
    k = rng.randint(2, 5)
    return 1 / (k * var)


G_TEMPLATES: dict[str, ExprBuilder] = {
    "const": g_const,
    "linear": g_linear,
    "power": g_power,
    "inv": g_inv,
    "exp": g_exp,
    "log": g_log,
    "rational_power": g_rational_power,
    "inv_k": g_inv_k,
}


# Combinations held out for generalization (never appear in train/val/test).
HELDOUT_COMBOS: set[tuple[str, str]] = {
    ("exp", "log"),
    ("log", "exp"),
    ("poly_inv", "exp"),
    ("rational_power", "log"),
    ("log", "rational_power"),
    ("exp", "rational_power"),
}


# Functions / atoms outside the v1 tokenizer vocabulary.
_ALLOWED_FUNCS = (sp.log, sp.exp, sp.Pow, sp.Add, sp.Mul)
UNSUPPORTED_ATOMS = (
    sp.Piecewise,
    sp.Abs,
    sp.Heaviside,
    sp.sign,
    sp.floor,
    sp.ceiling,
    sp.re,
    sp.im,
    sp.arg,
    sp.Integral,
    sp.Sum,
    sp.li,
    sp.Si,
    sp.Ci,
    sp.Ei,
    sp.erf,
    sp.erfc,
    sp.gamma,
    sp.polygamma,
    sp.besselj,
    sp.hyper,
)


# ---------------------------------------------------------------------------
# Cleaning / validation
# ---------------------------------------------------------------------------


@lru_cache(maxsize=8192)
def strip_abs_in_logs(expr: sp.Basic) -> sp.Expr:
    """Replace log(Abs(u)) with log(u); drop Abs elsewhere."""

    def _rewrite(e: sp.Expr) -> sp.Expr:
        if isinstance(e, sp.log) and len(e.args) == 1:
            arg = e.args[0]
            if isinstance(arg, sp.Abs):
                return sp.log(_rewrite(arg.args[0]))
            return sp.log(_rewrite(arg))
        if isinstance(e, sp.Abs):
            return _rewrite(e.args[0])
        if e.args:
            return e.func(*[_rewrite(a) for a in e.args])
        return e

    return _rewrite(sp.simplify(expr))


@lru_cache(maxsize=8192)
def canonicalize(expr: sp.Basic) -> sp.Expr:
    """
    Single canonical form for ODE RHS and solution sides.

    Expands products/sums and always expands logs (force=True) so
    log(5*x) -> log(5) + log(x) everywhere.
    """
    e = strip_abs_in_logs(expr)
    e = sp.expand_log(e, force=True)
    e = sp.expand(e)
    e = sp.expand_log(e, force=True)
    return e


@lru_cache(maxsize=8192)
def split_outer_coeff(expr: sp.Basic) -> tuple[sp.Expr, sp.Expr]:
    """
    Split a multiplicative outer numeric prefactor from expr.

    g_core is normalized so that (f, g) and (k*f, g/k) share the same core:
    - expand_log(force=True) so log(u**v) / log(a*y) forms match;
    - factor() pulls shared rational content out of sums (6*y-8 -> 2*(3*y-4),
      and -5*y*log(4*y) -> -5*y*(log(y)+log(4)));
    - polynomial-in-y cores are primitive over QQ with positive leading coefficient;
    - otherwise any overall minus sign is absorbed into coeff.
    Inner relative coefficients (e.g. 3 and -4 in 3*y-4) are preserved.
    """
    expr = sp.factor(sp.expand(sp.expand_log(sp.sympify(expr), force=True)))
    coeff, core = expr.as_coeff_Mul(rational=True)
    coeff = sp.sympify(coeff)
    core = sp.sympify(core)

    # Primitive content + sign for polynomials in y.
    if core != 0 and core.free_symbols <= {y} and core.is_polynomial(y):
        poly = sp.Poly(sp.expand(core), y, domain="QQ")
        content, prim = poly.primitive()
        coeff *= sp.sympify(content)
        core = prim.as_expr()
        if prim.LC() < 0:
            coeff *= -1
            core = (-prim).as_expr()
    elif core != 0 and core.could_extract_minus_sign():
        coeff *= -1
        core = sp.sympify(-core)

    return coeff, core


@lru_cache(maxsize=8192)
def _integrate_cached(expr: sp.Basic, var: sp.Basic) -> sp.Expr:
    """Memoized SymPy integrate for repeated identical integrands."""
    return sp.integrate(expr, var)


def contains_unsupported(expr: sp.Expr) -> bool:
    """Return True if expression contains nodes we refuse to keep."""
    if expr.has(*UNSUPPORTED_ATOMS):
        return True
    if expr.has(sp.I):
        return True
    # Reject floating-point leaves (prefer exact rationals/integers).
    for atom in expr.atoms(sp.Float):
        if atom != 0:
            return True
    # Reject any Function not in the allowed set (catches li, etc.).
    for fn in expr.atoms(sp.Function):
        if not isinstance(fn, _ALLOWED_FUNCS):
            return True
    return False


def estimate_token_length(expr: sp.Expr) -> int:
    """Cheap structural size proxy used before the tokenizer is available."""
    count = 0
    for node in sp.preorder_traversal(expr):
        count += 1
    return count


def format_ode(f_expr: sp.Expr, g_expr: sp.Expr) -> str:
    """Pretty ODE string: y' = f(x)*g(y), canonicalized."""
    # Avoid simplify() here — it can rewrite f(x)*g(y) into entangled forms
    # like log((a*x)**(b*y)), which obscures separability.
    if g_expr == 1:
        rhs: sp.Expr = f_expr
    elif f_expr == 1:
        rhs = g_expr
    else:
        rhs = sp.Mul(f_expr, g_expr, evaluate=True)
    rhs = canonicalize(rhs)
    return f"y' = {sp.sstr(rhs)}"


def format_solution(G: sp.Expr, F: sp.Expr) -> str:
    """Implicit solution G(y) = F(x) + C (canonicalized, no Abs in logs)."""
    left = canonicalize(G)
    right = canonicalize(F)
    return f"{sp.sstr(left)} = {sp.sstr(right)} + C"


@lru_cache(maxsize=8192)
def solution_satisfies_ode(ode_str: str, sol_str: str) -> bool:
    """
    Verify an implicit solution by differentiation.

    From G(y) = F(x) + C, predict y' = F'(x) / G'(y) and compare to the ODE RHS.
    """
    local = {"x": x, "y": y, "C": C, "Derivative": sp.Derivative, "exp": sp.exp, "log": sp.log}
    ode_s = ode_str.strip().replace("y'", "Derivative(y, x)")
    if "=" not in ode_s or "=" not in sol_str:
        return False
    ode_lhs, ode_rhs = ode_s.split("=", 1)
    sol_lhs, sol_rhs = sol_str.split("=", 1)
    try:
        rhs = sp.sympify(ode_rhs.strip(), locals=local)
        G = sp.sympify(sol_lhs.strip(), locals=local)
        F_plus_C = sp.sympify(sol_rhs.strip(), locals=local)
        F = sp.simplify(F_plus_C - C)
        dG = sp.diff(G, y)
        dF = sp.diff(F, x)
        if dG == 0:
            return False
        predicted = sp.simplify(dF / dG)
        diff = sp.simplify(canonicalize(predicted) - canonicalize(rhs))
        if diff == 0:
            return True
        # Numeric fallback at a few positive points.
        for xv, yv in ((0.7, 1.3), (1.1, 0.8), (1.7, 2.0)):
            try:
                val = complex(diff.evalf(subs={x: xv, y: yv}))
                if abs(val) > 1e-6:
                    return False
            except Exception:
                return False
        return True
    except Exception:
        return False


def _bump(stats: Optional[dict[str, int]], reason: str) -> None:
    if stats is not None:
        stats[reason] = stats.get(reason, 0) + 1


def try_build_example(
    f_name: str,
    g_name: str,
    rng: random.Random,
    cfg: GenConfig,
    stats: Optional[dict[str, int]] = None,
) -> Optional[dict[str, str]]:
    """
    Build one separable example backwards with canonical scaling.

    G is integral of 1/g_core (g without outer numeric prefactor); all outer
    numeric prefactors of g are moved onto the x-side integral.

    Returns dict with ode, solution, family, f_template, g_template — or None.
    """
    f_builder = F_TEMPLATES[f_name]
    g_builder = G_TEMPLATES[g_name]
    f_expr = f_builder(x, rng, cfg)
    g_expr = g_builder(y, rng, cfg)

    if f_expr is None or g_expr is None:
        _bump(stats, "null_expr")
        return None
    if f_expr == 0 or g_expr == 0:
        _bump(stats, "zero_factor")
        return None

    g_coeff, g_core = split_outer_coeff(g_expr)
    if g_core == 0:
        _bump(stats, "zero_g_core")
        return None

    try:
        inv_g = 1 / g_core
        G = _integrate_cached(sp.sympify(inv_g), y)
        # Move outer numeric prefactor of g onto the x side: G = g_coeff * F_f + C
        F = _integrate_cached(sp.sympify(g_coeff * f_expr), x)
    except Exception:
        _bump(stats, "integrate_error")
        return None

    if G is None or F is None:
        _bump(stats, "integrate_none")
        return None
    if isinstance(G, sp.Integral) or isinstance(F, sp.Integral):
        _bump(stats, "unevaluated_integral")
        return None

    G = strip_abs_in_logs(G)
    F = strip_abs_in_logs(F)

    if contains_unsupported(G) or contains_unsupported(F):
        _bump(stats, "unsupported_solution")
        return None
    if contains_unsupported(f_expr) or contains_unsupported(g_expr):
        _bump(stats, "unsupported_ode")
        return None

    # Reject free symbols other than x, y (integration constants from SymPy, etc.).
    allowed = {x, y}
    if (G.free_symbols | F.free_symbols) - allowed:
        _bump(stats, "extra_symbols")
        return None

    ode_str = format_ode(f_expr, g_expr)
    sol_str = format_solution(G, F)

    # Structural length guard (token length checked later by caller if needed).
    G_c, F_c = canonicalize(G), canonicalize(F)
    if estimate_token_length(G_c) + estimate_token_length(F_c) > cfg.max_token_length * 2:
        _bump(stats, "too_long_solution")
        return None
    if estimate_token_length(canonicalize(f_expr * g_expr)) > cfg.max_token_length:
        _bump(stats, "too_long_ode")
        return None

    if not solution_satisfies_ode(ode_str, sol_str):
        _bump(stats, "failed_ode_check")
        return None

    _bump(stats, "kept")
    return {
        "ode": ode_str,
        "solution": sol_str,
        "family": cfg.family,
        "f_template": f_name,
        "g_template": g_name,
    }


# ---------------------------------------------------------------------------
# Dataset assembly
# ---------------------------------------------------------------------------


def _combo_pool(heldout: bool) -> list[tuple[str, str]]:
    combos: list[tuple[str, str]] = []
    for f_name in F_TEMPLATES:
        for g_name in G_TEMPLATES:
            is_held = (f_name, g_name) in HELDOUT_COMBOS
            if heldout == is_held:
                combos.append((f_name, g_name))
    return combos


def generate_examples(
    n: int,
    cfg: GenConfig,
    rng: random.Random,
    heldout: bool = False,
    max_attempts_factor: int = 40,
    stats: Optional[dict[str, int]] = None,
) -> list[dict[str, str]]:
    """Generate up to n unique examples from seen or held-out template combos."""
    combos = _combo_pool(heldout=heldout)
    if not combos:
        raise RuntimeError("No template combinations available for generation")

    if stats is None:
        stats = {}

    seen: set[tuple[str, str]] = set()
    examples: list[dict[str, str]] = []
    attempts = 0
    max_attempts = max(n * max_attempts_factor, n)

    pbar = tqdm(total=n, desc="heldout" if heldout else "main", unit="ex")
    while len(examples) < n and attempts < max_attempts:
        attempts += 1
        f_name, g_name = rng.choice(combos)
        ex = try_build_example(f_name, g_name, rng, cfg, stats=stats)
        if ex is None:
            continue
        key = (ex["ode"], ex["solution"])
        if key in seen:
            _bump(stats, "duplicate")
            continue
        seen.add(key)
        examples.append(ex)
        pbar.update(1)
    pbar.close()

    if len(examples) < n:
        print(
            f"Warning: only generated {len(examples)}/{n} "
            f"({'heldout' if heldout else 'main'}) examples after {attempts} attempts"
        )
    return examples


def collect_generation_stats(
    examples: list[dict[str, str]],
    discard_stats: dict[str, int],
) -> dict[str, Any]:
    """Aggregate keep/discard counts, template histogram, and token-length stats."""
    from src.data.tokenizer import Tokenizer

    tok = Tokenizer(max_int=20)
    combo_counts: dict[str, int] = {}
    ode_lens: list[int] = []
    sol_lens: list[int] = []
    for ex in examples:
        key = f"{ex['f_template']}|{ex['g_template']}"
        combo_counts[key] = combo_counts.get(key, 0) + 1
        ode_lens.append(len(tok.tokenize(ex["ode"])))
        sol_lens.append(len(tok.tokenize(ex["solution"])))

    kept = len(examples)
    discarded = sum(v for k, v in discard_stats.items() if k != "kept")

    def _len_stats(vals: list[int]) -> dict[str, float]:
        if not vals:
            return {"min": 0, "mean": 0.0, "max": 0}
        return {
            "min": int(min(vals)),
            "mean": float(sum(vals) / len(vals)),
            "max": int(max(vals)),
        }

    return {
        "kept": kept,
        "discarded": discarded,
        "discard_reasons": {
            k: v for k, v in sorted(discard_stats.items()) if k != "kept"
        },
        "combo_counts": dict(sorted(combo_counts.items())),
        "ode_token_length": _len_stats(ode_lens),
        "solution_token_length": _len_stats(sol_lens),
    }


def print_generation_stats(stats: dict[str, Any]) -> None:
    """Pretty-print dataset generation statistics."""
    print("=== Dataset stats ===")
    print(f"kept:      {stats['kept']}")
    print(f"discarded: {stats['discarded']}")
    print("discard reasons:")
    for reason, count in stats["discard_reasons"].items():
        print(f"  {reason}: {count}")
    print("counts per (f, g) template:")
    for combo, count in stats["combo_counts"].items():
        f_name, g_name = combo.split("|", 1)
        print(f"  ({f_name}, {g_name}): {count}")
    ode_l = stats["ode_token_length"]
    sol_l = stats["solution_token_length"]
    print(
        f"ODE token length:      min={ode_l['min']}  mean={ode_l['mean']:.2f}  max={ode_l['max']}"
    )
    print(
        f"Solution token length: min={sol_l['min']}  mean={sol_l['mean']:.2f}  max={sol_l['max']}"
    )


def split_dataset(
    examples: list[dict[str, str]],
    cfg: GenConfig,
    rng: random.Random,
) -> dict[str, list[dict[str, str]]]:
    """Deduplicated train/val/test split with fixed ratios."""
    total_ratio = cfg.train_ratio + cfg.val_ratio + cfg.test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError(f"Split ratios must sum to 1.0, got {total_ratio}")

    shuffled = list(examples)
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_train = int(n * cfg.train_ratio)
    n_val = int(n * cfg.val_ratio)
    train = shuffled[:n_train]
    val = shuffled[n_train : n_train + n_val]
    test = shuffled[n_train + n_val :]
    return {"train": train, "val": val, "test": test}


def _public_fields(ex: dict[str, str]) -> dict[str, str]:
    out = {"ode": ex["ode"], "solution": ex["solution"], "family": ex["family"]}
    if "f_template" in ex:
        out["f_template"] = ex["f_template"]
    if "g_template" in ex:
        out["g_template"] = ex["g_template"]
    return out


def write_jsonl(path: Path, rows: Iterable[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(_public_fields(row), ensure_ascii=False) + "\n")


def generate_dataset(cfg: GenConfig) -> dict[str, Path]:
    """Generate full dataset and write JSONL splits. Returns output paths."""
    rng = random.Random(cfg.seed)
    main = generate_examples(cfg.n_samples, cfg, rng, heldout=False)
    heldout = generate_examples(cfg.n_heldout, cfg, rng, heldout=True)

    split_rng = random.Random(cfg.seed + 1)
    splits = split_dataset(main, cfg, split_rng)

    out = Path(cfg.output_dir)
    paths = {
        "train": out / "train.jsonl",
        "val": out / "val.jsonl",
        "test": out / "test.jsonl",
        "test_unseen": out / "test_unseen.jsonl",
    }
    write_jsonl(paths["train"], splits["train"])
    write_jsonl(paths["val"], splits["val"])
    write_jsonl(paths["test"], splits["test"])
    write_jsonl(paths["test_unseen"], heldout)

    try:
        import sympy as sp
        import torch

        sympy_version = sp.__version__
        torch_version = torch.__version__
    except Exception:
        sympy_version = None
        torch_version = None

    meta = {
        "seed": cfg.seed,
        "n_main": len(main),
        "n_train": len(splits["train"]),
        "n_val": len(splits["val"]),
        "n_test": len(splits["test"]),
        "n_test_unseen": len(heldout),
        "heldout_combos": sorted(list(HELDOUT_COMBOS)),
        "f_templates": sorted(F_TEMPLATES),
        "g_templates": sorted(G_TEMPLATES),
        "family": cfg.family,
        "sympy_version": sympy_version,
        "torch_version": torch_version,
    }
    out.mkdir(parents=True, exist_ok=True)
    with (out / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return paths


def cfg_from_mapping(data: dict[str, Any], seed: Optional[int] = None) -> GenConfig:
    """Build GenConfig from a YAML-loaded mapping (supports nested `data:`)."""
    block = data.get("data", data)
    return GenConfig(
        n_samples=int(block.get("n_samples", 100_000)),
        n_heldout=int(block.get("n_heldout", 5_000)),
        train_ratio=float(block.get("train_ratio", 0.90)),
        val_ratio=float(block.get("val_ratio", 0.05)),
        test_ratio=float(block.get("test_ratio", 0.05)),
        coeff_min=int(block.get("coeff_min", -5)),
        coeff_max=int(block.get("coeff_max", 5)),
        exclude_zero_coeff=bool(block.get("exclude_zero_coeff", True)),
        max_degree=int(block.get("max_degree", 3)),
        max_token_length=int(block.get("max_token_length", 64)),
        output_dir=str(block.get("output_dir", "data/processed")),
        family=str(block.get("family", "separable")),
        seed=int(seed if seed is not None else data.get("seed", 42)),
    )


def load_config(path: str | Path) -> GenConfig:
    with Path(path).open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return cfg_from_mapping(raw or {})


def sample_pairs(
    n: int = 10,
    seed: int = 0,
    cfg: Optional[GenConfig] = None,
) -> list[dict[str, str]]:
    """Convenience helper: generate n in-distribution examples."""
    cfg = cfg or GenConfig(n_samples=n, seed=seed)
    rng = random.Random(seed)
    return generate_examples(n, cfg, rng, heldout=False, max_attempts_factor=80)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate separable ODE dataset")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to YAML config",
    )
    parser.add_argument("--n-samples", type=int, default=None)
    parser.add_argument("--n-heldout", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.n_samples is not None:
        cfg.n_samples = args.n_samples
    if args.n_heldout is not None:
        cfg.n_heldout = args.n_heldout
    if args.output_dir is not None:
        cfg.output_dir = args.output_dir
    if args.seed is not None:
        cfg.seed = args.seed

    paths = generate_dataset(cfg)
    print("Wrote:")
    for name, path in paths.items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
