"""Dataset collate and encode/decode equivalence on generated records."""

from __future__ import annotations

import json
import random

import torch

from src.data.dataset import ODEDataset, make_collate_fn
from src.data.generator import (
    F_TEMPLATES,
    G_TEMPLATES,
    GenConfig,
    try_build_example,
    write_jsonl,
)
from src.data.tokenizer import Tokenizer, expressions_equivalent


def test_collate_pads_batch_first() -> None:
    pad_id = 0
    batch = [([1, 2, 3], [4, 5, 6, 7]), ([8, 9], [4, 5, 6])]
    out = make_collate_fn(pad_id)(batch)
    assert out.src.shape == (2, 3)
    assert out.tgt_in.shape == (2, 3)
    assert out.tgt_out.shape == (2, 3)
    assert out.src_key_padding_mask.shape == out.src.shape
    assert out.tgt_key_padding_mask.shape == out.tgt_in.shape
    assert out.src_key_padding_mask.dtype == torch.bool
    assert bool(out.src_key_padding_mask[1, 2])
    assert not bool(out.src_key_padding_mask[0, 0])
    assert int(out.tgt_out[1, -1]) == pad_id
    assert not bool(out.tgt_key_padding_mask[1].any())


def test_encode_decode_equivalent_on_100_records(tmp_path) -> None:
    tokenizer = Tokenizer(max_int=20)
    cfg = GenConfig(coeff_min=-5, coeff_max=5, max_degree=3, seed=42)
    rng = random.Random(0)
    rows: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    attempts = 0
    while len(rows) < 100 and attempts < 4000:
        attempts += 1
        f_name = rng.choice(list(F_TEMPLATES))
        g_name = rng.choice(list(G_TEMPLATES))
        ex = try_build_example(f_name, g_name, rng, cfg)
        if ex is None:
            continue
        key = (ex["ode"], ex["solution"])
        if key in seen:
            continue
        seen.add(key)
        rows.append(ex)
    assert len(rows) == 100, f"only generated {len(rows)} records in {attempts} attempts"

    path = tmp_path / "train.jsonl"
    write_jsonl(path, rows)
    dataset = ODEDataset(path, tokenizer, max_seq_len=128)
    assert dataset.skipped + len(dataset) == 100

    failures: list[str] = []
    for ex in rows:
        for field in ("ode", "solution"):
            text = ex[field]
            ids = tokenizer.encode(text, add_special=True)
            tokens = tokenizer.decode_ids(ids)
            try:
                rebuilt = tokenizer.ids_to_sympy(ids)
                from_tokens = tokenizer.tokens_to_sympy(tokens)
                original = tokenizer.parse_any(text)
            except Exception as exc:  # noqa: BLE001
                failures.append(f"{field}: {text} err={exc}")
                continue
            if not expressions_equivalent(original, rebuilt) or not expressions_equivalent(
                original, from_tokens
            ):
                failures.append(f"{field}: {text} -> {tokens} -> {rebuilt}")
    assert not failures, "equivalence failures:\n" + "\n".join(failures[:20])


def test_ode_dataset_ignores_extra_fields_and_loads_legacy(tmp_path) -> None:
    tokenizer = Tokenizer(max_int=20)
    legacy = tmp_path / "legacy.jsonl"
    modern = tmp_path / "modern.jsonl"
    rows_legacy = [
        {"ode": "y' = x", "solution": "y = x**2/2 + C", "family": "separable"},
        {"ode": "y' = 2*x", "solution": "y = x**2 + C", "family": "separable"},
    ]
    rows_modern = [
        {
            "ode": "y' = x",
            "solution": "y = x**2/2 + C",
            "family": "separable",
            "f_template": "linear",
            "g_template": "const",
            "extra": "ignored",
        },
        {
            "ode": "y' = 2*x",
            "solution": "y = x**2 + C",
            "family": "separable",
            "f_template": "linear",
            "g_template": "const",
        },
    ]
    legacy.write_text(
        "\n".join(json.dumps(r) for r in rows_legacy) + "\n", encoding="utf-8"
    )
    modern.write_text(
        "\n".join(json.dumps(r) for r in rows_modern) + "\n", encoding="utf-8"
    )
    ds_legacy = ODEDataset(legacy, tokenizer, max_seq_len=128, use_cache=False)
    ds_modern = ODEDataset(modern, tokenizer, max_seq_len=128, use_cache=False)
    assert len(ds_legacy) == 2
    assert len(ds_modern) == 2
    assert ds_legacy[0][0] == ds_modern[0][0]
    assert ds_legacy[0][1] == ds_modern[0][1]


def test_tokenization_cache_reused_and_invalidated(tmp_path) -> None:
    tokenizer = Tokenizer(max_int=20)
    path = tmp_path / "train.jsonl"
    rows = [
        {"ode": "y' = x", "solution": "y = x**2/2 + C", "family": "separable"},
        {"ode": "y' = 2*x", "solution": "y = x**2 + C", "family": "separable"},
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    cold = ODEDataset(path, tokenizer, max_seq_len=128, use_cache=True)
    assert cold.cache_hit is False
    assert cold.cache_path is not None and cold.cache_path.exists()

    warm = ODEDataset(path, tokenizer, max_seq_len=128, use_cache=True)
    assert warm.cache_hit is True
    assert len(warm) == len(cold)
    assert warm.samples == cold.samples

    # Changing max_seq_len must invalidate the cache key.
    other = ODEDataset(path, tokenizer, max_seq_len=64, use_cache=True)
    assert other.cache_hit is False

    # Changing file content must invalidate.
    path.write_text(
        path.read_text(encoding="utf-8")
        + json.dumps({"ode": "y' = 3*x", "solution": "y = 3*x**2/2 + C", "family": "separable"})
        + "\n",
        encoding="utf-8",
    )
    changed = ODEDataset(path, tokenizer, max_seq_len=128, use_cache=True)
    assert changed.cache_hit is False
    assert len(changed) == 3
