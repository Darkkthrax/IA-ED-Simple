"""Unit tests for prefix tokenizer round-trips."""

from __future__ import annotations

import random

import pytest
import sympy as sp

from src.data.generator import GenConfig, sample_pairs, try_build_example, F_TEMPLATES, G_TEMPLATES
from src.data.tokenizer import Tokenizer, expressions_equivalent

x, y = sp.symbols("x y")
C = sp.Symbol("C")


HANDCRAFTED = [
    "y' = x",
    "y' = x*y",
    "y' = exp(x)",
    "y' = 1/x",
    "y' = x*exp(y)",
    "y' = (2*x + 1)/y",
    "log(y) = x + C",
    "log(y) = x**2/2 + C",
    "y = exp(x) + C",
    "1/y = -x + C",
    "y**2/2 = x**3/3 + C",
    "exp(-y) = x + C",
    "log(2*y) = 3*x + C",
    "y' = -3/x**2",
    "y' = (x**2 + 1/x)*y",
]


def _random_expr(rng: random.Random) -> sp.Expr:
    """Build a random expression from the project grammar atoms."""
    kind = rng.choice(
        [
            "poly_x",
            "poly_y",
            "inv",
            "exp",
            "log",
            "pow",
            "mixed",
            "eq_sol",
            "eq_ode",
        ]
    )
    a = rng.randint(1, 5)
    b = rng.randint(1, 5)
    n = rng.randint(1, 3)
    if kind == "poly_x":
        return a * x**n + b
    if kind == "poly_y":
        return a * y**n + rng.randint(-3, 3)
    if kind == "inv":
        return sp.Integer(a) / x
    if kind == "exp":
        return a * sp.exp(b * x)
    if kind == "log":
        return a * sp.log(b * y)
    if kind == "pow":
        return a * y**n
    if kind == "mixed":
        return a * x**n + b / x
    if kind == "eq_sol":
        return sp.Eq(sp.log(y), a * x**n / n + C)
    # eq_ode
    return sp.Eq(sp.Derivative(y, x), a * x * sp.exp(b * y))


@pytest.fixture
def tokenizer() -> Tokenizer:
    return Tokenizer(max_int=20)


def test_handcrafted_round_trips(tokenizer: Tokenizer) -> None:
    for s in HANDCRAFTED:
        assert tokenizer.round_trip_ok(s), f"Round-trip failed for {s!r}"


def test_fifty_random_expression_round_trips(tokenizer: Tokenizer) -> None:
    rng = random.Random(123)
    failures: list[str] = []
    for i in range(50):
        expr = _random_expr(rng)
        if not tokenizer.round_trip_ok(expr):
            tokens = tokenizer.tokenize(expr)
            try:
                rebuilt = tokenizer.tokens_to_sympy(tokens)
            except Exception as exc:  # noqa: BLE001
                failures.append(f"{i}: parse error {expr} tokens={tokens} err={exc}")
                continue
            failures.append(f"{i}: {expr} -> {tokens} -> {rebuilt}")
    assert not failures, "Round-trip failures:\n" + "\n".join(failures)


def test_fifty_generated_ode_solution_round_trips(tokenizer: Tokenizer) -> None:
    cfg = GenConfig(coeff_min=-4, coeff_max=4, max_degree=3, seed=7)
    rng = random.Random(7)
    pairs = []
    attempts = 0
    while len(pairs) < 50 and attempts < 2000:
        attempts += 1
        f_name = rng.choice(list(F_TEMPLATES))
        g_name = rng.choice(list(G_TEMPLATES))
        ex = try_build_example(f_name, g_name, rng, cfg)
        if ex is not None:
            pairs.append(ex)

    assert len(pairs) >= 50, f"Only got {len(pairs)} examples"
    failures = []
    for ex in pairs:
        for field in ("ode", "solution"):
            if not tokenizer.round_trip_ok(ex[field]):
                tokens = tokenizer.tokenize(ex[field])
                failures.append(f"{field}: {ex[field]} -> {tokens}")
    assert not failures, "Generated round-trip failures:\n" + "\n".join(failures[:20])


def test_encode_decode_ids_preserves_tokens(tokenizer: Tokenizer) -> None:
    text = "y' = x*exp(y)"
    ids = tokenizer.encode(text, add_special=True)
    assert ids[0] == tokenizer.vocab.sos_id
    assert ids[-1] == tokenizer.vocab.eos_id
    tokens = tokenizer.decode_ids(ids, strip_special=True)
    assert tokens == tokenizer.tokenize(text)


def test_sample_pairs_smoke() -> None:
    pairs = sample_pairs(n=5, seed=1)
    assert len(pairs) == 5
    for p in pairs:
        assert "ode" in p and "solution" in p and p["family"] == "separable"


def test_equivalence_helper() -> None:
    a = sp.Eq(sp.log(y), x + C)
    b = sp.Eq(sp.log(y), C + x)
    assert expressions_equivalent(a, b)
