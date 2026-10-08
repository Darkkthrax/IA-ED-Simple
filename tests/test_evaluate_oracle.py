"""Oracle and negative-control checks for the symbolic evaluation path."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from src.data.generator import GenConfig, try_build_example, write_jsonl
from src.evaluate import corrupt_solution_candidate, evaluate


ROOT = Path(__file__).resolve().parents[1]


def _write_val_split(tmp_path: Path, n: int = 20) -> Path:
    cfg = GenConfig(coeff_min=-5, coeff_max=5, max_degree=3, seed=7)
    import random

    rng = random.Random(0)
    rows: list[dict] = []
    seen: set[tuple[str, str]] = set()
    attempts = 0
    while len(rows) < n and attempts < 4000:
        attempts += 1
        ex = try_build_example("linear", "const", rng, cfg)
        if ex is None:
            continue
        key = (ex["ode"], ex["solution"])
        if key in seen:
            continue
        seen.add(key)
        rows.append(ex)
    assert len(rows) == n
    path = tmp_path / "val.jsonl"
    write_jsonl(path, rows)
    meta = {
        "sympy_version": __import__("sympy").__version__,
        "torch_version": __import__("torch").__version__,
    }
    (tmp_path / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return path


def test_oracle_top1_and_corrupt_rejected(tmp_path) -> None:
    _write_val_split(tmp_path, n=20)
    raw = yaml.safe_load((ROOT / "configs" / "default.yaml").read_text(encoding="utf-8"))
    raw["data"]["output_dir"] = str(tmp_path)

    args = argparse.Namespace(
        checkpoint=None,
        split="val",
        max_examples=20,
        data_dir=str(tmp_path),
        device="cpu",
        oracle=True,
        oracle_corrupt=False,
    )
    summary = evaluate(raw, args)
    assert summary["n"] == 20
    assert summary["top1_symbolic_acc"] == 1.0
    assert summary["oracle"] is True

    args_neg = argparse.Namespace(
        checkpoint=None,
        split="val",
        max_examples=20,
        data_dir=str(tmp_path),
        device="cpu",
        oracle=True,
        oracle_corrupt=True,
    )
    summary_neg = evaluate(raw, args_neg)
    assert summary_neg["n"] == 20
    assert summary_neg["top1_symbolic_acc"] == 0.0
    assert summary_neg["oracle_corrupt"] is True


def test_corrupt_solution_candidate_changes_string() -> None:
    s = "y = x**2/2 + C"
    bad = corrupt_solution_candidate(s)
    assert bad != s
