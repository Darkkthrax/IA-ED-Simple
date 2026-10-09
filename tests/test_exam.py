"""Tests for the exam deliverable pipeline."""

from __future__ import annotations

import sympy as sp

from exam.answers import ANSWERS
from exam.answers_demo import ANSWERS_DEMO
from exam.equations import EXERCISE_IDS, RHS
from exam.report import run
from exam.verify import corrupt_answer, parse_answer, verify_explicit, x_range_for


def test_demo_answers_verify_and_corrupt_fails() -> None:
    for eid in EXERCISE_IDS:
        ans = ANSWERS_DEMO[eid]
        y_expr, note = parse_answer(ans)
        assert y_expr is not None, (eid, note)
        ok, vnote = verify_explicit(y_expr, RHS[eid], x_range=x_range_for(eid))
        assert ok, (eid, vnote, ans)

        bad = corrupt_answer(ans)
        y_bad, note_bad = parse_answer(bad)
        assert y_bad is not None, (eid, note_bad)
        ok_bad, _ = verify_explicit(y_bad, RHS[eid], x_range=x_range_for(eid))
        assert not ok_bad, f"{eid} corrupted answer unexpectedly verified: {bad}"


def test_constant_renaming_k_to_c() -> None:
    expr, note = parse_answer("2*x**3 + K")
    assert note == ""
    assert expr is not None
    assert expr.free_symbols <= {sp.Symbol("x"), sp.Symbol("C")}
    assert sp.Symbol("K") not in expr.free_symbols
    assert sp.simplify(expr - (2 * sp.Symbol("x") ** 3 + sp.Symbol("C"))) == 0


def test_pipeline_all_none() -> None:
    assert all(v is None for v in ANSWERS.values())
    df = run(use_demo=False, model=None, tokenizer=None, output_dir="exam/output_test_none")
    assert len(df) == 15
    assert list(df.columns) == [
        "id", "ODE", "respuesta", "SÍ/NO", "MAPE %", "NRMSE %", "Transformer",
    ]
    assert (df["SÍ/NO"] == "PENDIENTE").all()
    assert (df["Transformer"] == "-").all()


def test_dataframe_columns_with_demo_smoke(tmp_path) -> None:
    # Smoke only: skip heavy NN by temporarily lowering epochs via config.
    import exam.config as cfg

    old = cfg.EPOCHS
    cfg.EPOCHS = 5
    try:
        df = run(
            use_demo=True,
            model=None,
            tokenizer=None,
            output_dir=tmp_path,
        )
    finally:
        cfg.EPOCHS = old
    assert len(df) == 15
    assert list(df.columns) == [
        "id", "ODE", "respuesta", "SÍ/NO", "MAPE %", "NRMSE %", "Transformer",
    ]
    assert (df["SÍ/NO"] == "SÍ").all()
    assert (df["Transformer"] == "-").all()
