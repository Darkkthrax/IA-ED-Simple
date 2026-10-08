"""Resume, dedupe, and finalize behavior of scripts/generate_dataset.py."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import yaml

from src.data.generator import HELDOUT_COMBOS
from src.data.tokenizer import build_tokenizer_from_config

ROOT = Path(__file__).resolve().parents[1]


def _load_script():
    path = ROOT / "scripts" / "generate_dataset.py"
    spec = importlib.util.spec_from_file_location("generate_dataset_script", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _cfg(module, tmp_path: Path):
    cfg, raw = module.load_raw_config(ROOT / "configs" / "default.yaml")
    cfg.output_dir = str(tmp_path)
    tok = build_tokenizer_from_config(raw)
    return cfg, tok


def test_pool_quotas_match_config_ratio() -> None:
    module = _load_script()
    cfg, _raw = module.load_raw_config(ROOT / "configs" / "default.yaml")
    main, held = module.pool_quotas(30000, cfg)
    assert main + held == 30000
    assert held == round(30000 * cfg.n_heldout / (cfg.n_samples + cfg.n_heldout))
    full_main, full_held = module.pool_quotas(cfg.n_samples + cfg.n_heldout, cfg)
    assert full_main == cfg.n_samples
    assert full_held == cfg.n_heldout


def test_int_above_max_boundary() -> None:
    module = _load_script()
    raw = yaml.safe_load((ROOT / "configs" / "default.yaml").read_text(encoding="utf-8"))
    tok = build_tokenizer_from_config(raw)
    assert tok.max_int == 20
    small = module.annotate_record(
        {
            "ode": "y' = 20*x",
            "solution": "y = 20*x + C",
            "family": "separable",
            "f_template": "linear",
            "g_template": "const",
        },
        tok,
    )
    big = module.annotate_record(
        {
            "ode": "y' = 21*x",
            "solution": "y = x + C",
            "family": "separable",
            "f_template": "linear",
            "g_template": "const",
        },
        tok,
    )
    assert small["has_int_above_max"] is False
    assert big["has_int_above_max"] is True
    assert small["ode_token_len"] == len(tok.tokenize("y' = 20*x"))


def test_resume_dedup_and_finalize(tmp_path) -> None:
    module = _load_script()
    cfg, tok = _cfg(module, tmp_path)
    first = module.generate_resumable(cfg, tok, target=16, max_seconds=60)
    assert first["total_unique"] == 16
    assert first["stop_reason"] == "target_reached"
    keys = []
    for line in (tmp_path / "all.jsonl").read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        keys.append((row["ode"], row["solution"]))
        assert set(row) >= {"ode", "solution", "family", "f_template", "g_template"}
    assert len(keys) == len(set(keys)) == 16

    again = module.generate_resumable(cfg, tok, target=16, max_seconds=30)
    assert again["attempts"] == 0
    assert again["total_unique"] == 16

    grown = module.generate_resumable(cfg, tok, target=28, max_seconds=60)
    assert grown["total_unique"] == 28
    assert grown["new_kept"] == 12
    keys2 = []
    for line in (tmp_path / "all.jsonl").read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        keys2.append((row["ode"], row["solution"]))
    assert len(keys2) == len(set(keys2)) == 28
    assert keys2[:16] == keys

    paths = module.finalize_splits(cfg)
    paths_again = module.finalize_splits(cfg)
    for name in ("train", "val", "test", "test_unseen"):
        assert paths[name].read_bytes() == paths_again[name].read_bytes()

    def _load(path: Path) -> list[dict]:
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]

    train, val, test, unseen = (
        _load(paths["train"]),
        _load(paths["val"]),
        _load(paths["test"]),
        _load(paths["test_unseen"]),
    )
    for row in train + val + test + unseen:
        assert set(row) >= {"ode", "solution", "family", "f_template", "g_template"}
        assert row["f_template"]
        assert row["g_template"]

    by_key = {}
    for line in (tmp_path / "all.jsonl").read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        by_key[(row["ode"], row["solution"])] = row

    split_keys = []
    for row in train + val + test:
        key = (row["ode"], row["solution"])
        split_keys.append(key)
        assert (by_key[key]["f_template"], by_key[key]["g_template"]) not in HELDOUT_COMBOS
    unseen_keys = []
    for row in unseen:
        key = (row["ode"], row["solution"])
        unseen_keys.append(key)
        assert (by_key[key]["f_template"], by_key[key]["g_template"]) in HELDOUT_COMBOS
    assert len(set(split_keys)) == len(split_keys)
    assert not (set(split_keys) & set(unseen_keys))
    assert set(split_keys) | set(unseen_keys) == set(by_key)

    n_main = len(split_keys)
    assert len(train) == int(n_main * cfg.train_ratio)
    assert len(val) == int(n_main * cfg.val_ratio)
    assert len(test) == n_main - len(train) - len(val)
    meta = json.loads((tmp_path / "meta.json").read_text(encoding="utf-8"))
    assert meta["seed"] == cfg.seed
    assert meta["n_train"] == len(train)
    assert meta["n_test_unseen"] == len(unseen)
