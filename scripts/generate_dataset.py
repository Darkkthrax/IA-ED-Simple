"""Resumable separable-ODE generation into one JSONL, with a final split step.

The one-shot writer in ``src.data.generator.generate_dataset`` produces:

* ``{output_dir}/train.jsonl``
* ``{output_dir}/val.jsonl``
* ``{output_dir}/test.jsonl``
* ``{output_dir}/test_unseen.jsonl``  (every held-out combo; not a slice of main)
* ``{output_dir}/meta.json``

Main examples are shuffled with ``random.Random(seed + 1)`` and cut by
``split_dataset`` (``train_ratio`` / ``val_ratio``; the remainder is test).
That shuffle covers the whole main pool, so the split is not updated
incrementally. This script appends to ``{output_dir}/all.jsonl`` and writes
the four split files only when ``--finalize`` is passed, using the same
``split_dataset`` and ``write_jsonl`` functions.

``--target`` is the total unique-record stop count (default 30000). It is
divided into a main quota and a held-out quota with the config ratio
``n_samples : n_heldout``. Defaults: 28571 main and 1429 held-out.
Passing ``--target`` equal to ``n_samples + n_heldout`` recovers 100000 + 5000.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import threading
import time
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import yaml  # noqa: E402

from src.data.generator import (  # noqa: E402
    F_TEMPLATES,
    G_TEMPLATES,
    HELDOUT_COMBOS,
    GenConfig,
    _combo_pool,
    cfg_from_mapping,
    split_dataset,
    try_build_example,
    write_jsonl,
)
from src.data.tokenizer import Tokenizer, build_tokenizer_from_config  # noqa: E402

RAW_FILENAME = "all.jsonl"


def _arm_hard_exit(seconds: float) -> None:
    """Kill the process if a single SymPy call outlives the soft deadline."""

    def _kill() -> None:
        print(f"HARD TIMEOUT after {seconds:.0f}s", flush=True)
        os._exit(124)

    timer = threading.Timer(seconds, _kill)
    timer.daemon = True
    timer.start()


def pool_quotas(target: int, cfg: GenConfig) -> tuple[int, int]:
    """Split a total unique target into ``(main, heldout)`` quotas."""
    if target < 0:
        raise ValueError(f"target must be >= 0, got {target}")
    denom = cfg.n_samples + cfg.n_heldout
    if denom <= 0:
        raise ValueError("n_samples + n_heldout must be positive")
    held = int(round(target * cfg.n_heldout / denom))
    held = max(0, min(target, held))
    return target - held, held


def raw_path(cfg: GenConfig) -> Path:
    return Path(cfg.output_dir) / RAW_FILENAME


def _dump_line(record: dict[str, Any]) -> str:
    return json.dumps(record, ensure_ascii=False) + "\n"


def _rewrite(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(_dump_line(record))


def load_raw(path: Path) -> list[dict[str, Any]]:
    """Load accumulation records, dropping a truncated final line and dupes."""
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    records: list[dict[str, Any]] = []
    dropped_tail = False
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            if i == len(lines) - 1:
                dropped_tail = True
                continue
            raise
    seen: set[tuple[str, str]] = set()
    unique: list[dict[str, Any]] = []
    for record in records:
        key = (record["ode"], record["solution"])
        if key in seen:
            continue
        seen.add(key)
        unique.append(record)
    if dropped_tail or len(unique) != len(records):
        print(
            f"Rewrote {path} (truncated_tail={dropped_tail}, "
            f"dropped_dupes={len(records) - len(unique)})",
            flush=True,
        )
        _rewrite(path, unique)
    return unique


def _has_int_above(expr: Any, max_int: int) -> bool:
    import sympy as sp

    for rat in expr.atoms(sp.Rational):
        if abs(int(rat.p)) > max_int or abs(int(rat.q)) > max_int:
            return True
    return False


def annotate_record(ex: dict[str, str], tokenizer: Tokenizer) -> dict[str, Any]:
    """Public identity plus template tags and cheap exit-stat fields."""
    ode_expr = tokenizer.parse_any(ex["ode"])
    sol_expr = tokenizer.parse_any(ex["solution"])
    return {
        "ode": ex["ode"],
        "solution": ex["solution"],
        "family": ex["family"],
        "f_template": ex["f_template"],
        "g_template": ex["g_template"],
        "ode_token_len": len(tokenizer.sympy_to_tokens(ode_expr)),
        "solution_token_len": len(tokenizer.sympy_to_tokens(sol_expr)),
        "has_int_above_max": _has_int_above(ode_expr, tokenizer.max_int)
        or _has_int_above(sol_expr, tokenizer.max_int),
    }


def _is_heldout(record: dict[str, Any]) -> bool:
    return (record["f_template"], record["g_template"]) in HELDOUT_COMBOS


def _ensure_stat_fields(record: dict[str, Any], tokenizer: Tokenizer) -> None:
    if "ode_token_len" in record and "has_int_above_max" in record:
        return
    annotated = annotate_record(record, tokenizer)
    record.update(annotated)


def generate_resumable(
    cfg: GenConfig,
    tokenizer: Tokenizer,
    target: int,
    max_seconds: float,
) -> dict[str, Any]:
    """Append unseen examples until ``target`` uniques or ``max_seconds``."""
    main_quota, held_quota = pool_quotas(target, cfg)
    path = raw_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    records = load_raw(path)
    seen: set[tuple[str, str]] = {(r["ode"], r["solution"]) for r in records}
    main_kept = sum(1 for r in records if not _is_heldout(r))
    held_kept = len(records) - main_kept

    # Resume from a different stream than the previous prefix of the file.
    # Python 3.14 accepts str seeds and rejects tuple seeds.
    rng = random.Random(f"{cfg.seed}-{len(records)}")
    main_combos = _combo_pool(heldout=False)
    held_combos = _combo_pool(heldout=True)
    held_attempt_p = cfg.n_heldout / (cfg.n_samples + cfg.n_heldout)

    attempts = 0
    duplicates = 0
    new_kept = 0
    stop_reason = "target_reached"
    t0 = time.perf_counter()
    deadline = t0 + max_seconds

    print(
        f"Appending to {path}  target={target}  "
        f"main_quota={main_quota}  heldout_quota={held_quota}  "
        f"already={len(records)} (main={main_kept}, heldout={held_kept})  "
        f"max_seconds={max_seconds}",
        flush=True,
    )

    fh = path.open("a", encoding="utf-8")
    last_print = t0
    try:
        while True:
            now = time.perf_counter()
            if now >= deadline:
                stop_reason = "max_seconds"
                break
            if main_kept >= main_quota and held_kept >= held_quota:
                stop_reason = "target_reached"
                break
            need_main = main_kept < main_quota
            need_held = held_kept < held_quota
            if need_main and need_held:
                use_held = rng.random() < held_attempt_p
            else:
                use_held = need_held
            combos = held_combos if use_held else main_combos
            f_name, g_name = rng.choice(combos)
            attempts += 1
            ex = try_build_example(f_name, g_name, rng, cfg)
            if ex is None:
                pass
            else:
                key = (ex["ode"], ex["solution"])
                if key in seen:
                    duplicates += 1
                else:
                    record = annotate_record(ex, tokenizer)
                    fh.write(_dump_line(record))
                    fh.flush()
                    records.append(record)
                    seen.add(key)
                    new_kept += 1
                    if _is_heldout(record):
                        held_kept += 1
                    else:
                        main_kept += 1
            if now - last_print >= 5.0:
                print(
                    f"  progress unique={len(records)} attempts={attempts} "
                    f"dups={duplicates} elapsed={now - t0:.1f}s",
                    flush=True,
                )
                last_print = now
    except KeyboardInterrupt:
        stop_reason = "interrupted"
        print("Interrupted; flushed records are on disk.", flush=True)
    finally:
        fh.flush()
        fh.close()

    elapsed = max(time.perf_counter() - t0, 0.0)
    stats = _stats_from_records(
        records,
        attempts=attempts,
        duplicates=duplicates,
        new_kept=new_kept,
        elapsed=elapsed,
        stop_reason=stop_reason,
        tokenizer=tokenizer,
        main_quota=main_quota,
        held_quota=held_quota,
    )
    return stats


def _stats_from_records(
    records: list[dict[str, Any]],
    *,
    attempts: int,
    duplicates: int,
    new_kept: int,
    elapsed: float,
    stop_reason: str,
    tokenizer: Tokenizer,
    main_quota: int,
    held_quota: int,
) -> dict[str, Any]:
    for record in records:
        _ensure_stat_fields(record, tokenizer)
    combo_counts = {(f, g): 0 for f in sorted(F_TEMPLATES) for g in sorted(G_TEMPLATES)}
    big = 0
    ode_max = 0
    sol_max = 0
    main_kept = 0
    held_kept = 0
    for record in records:
        key = (record["f_template"], record["g_template"])
        combo_counts[key] = combo_counts.get(key, 0) + 1
        if record["has_int_above_max"]:
            big += 1
        ode_max = max(ode_max, int(record["ode_token_len"]))
        sol_max = max(sol_max, int(record["solution_token_len"]))
        if key in HELDOUT_COMBOS:
            held_kept += 1
        else:
            main_kept += 1
    return {
        "total_unique": len(records),
        "attempts": attempts,
        "duplicates": duplicates,
        "new_kept": new_kept,
        "elapsed": elapsed,
        "stop_reason": stop_reason,
        "combo_counts": combo_counts,
        "n_int_above": big,
        "ode_token_max": ode_max,
        "solution_token_max": sol_max,
        "max_int": tokenizer.max_int,
        "main_quota": main_quota,
        "held_quota": held_quota,
        "main_kept": main_kept,
        "held_kept": held_kept,
    }


def print_stats(stats: dict[str, Any]) -> None:
    total = stats["total_unique"]
    attempts = stats["attempts"]
    duplicates = stats["duplicates"]
    successful = duplicates + stats["new_kept"]
    dup_rate = (duplicates / successful) if successful else 0.0
    elapsed = stats["elapsed"]
    aps = (attempts / elapsed) if elapsed > 0 else 0.0
    frac = (stats["n_int_above"] / total) if total else 0.0

    print("=== Resumable generation ===", flush=True)
    print(f"stop_reason: {stats['stop_reason']}", flush=True)
    print(f"total_unique_kept: {total}", flush=True)
    print(f"attempts_this_run: {attempts}", flush=True)
    print(
        f"duplicate_rate: {dup_rate:.6f}  "
        f"(duplicates/successful_builds this run; "
        f"dups={duplicates}, successful={successful})",
        flush=True,
    )
    print(f"attempts_per_sec: {aps:.4f}  (elapsed={elapsed:.2f}s)", flush=True)
    print(
        f"main_kept: {stats['main_kept']}/{stats['main_quota']}  "
        f"heldout_kept: {stats['held_kept']}/{stats['held_quota']}",
        flush=True,
    )
    print(
        f"fraction_int_above_max: {frac:.6f}  "
        f"({stats['n_int_above']}/{total})  max_int={stats['max_int']}",
        flush=True,
    )
    print(f"ode_token_length_max: {stats['ode_token_max']}", flush=True)
    print(f"solution_token_length_max: {stats['solution_token_max']}", flush=True)
    print("unique per (f, g):", flush=True)
    for (f_name, g_name), count in stats["combo_counts"].items():
        tag = " heldout" if (f_name, g_name) in HELDOUT_COMBOS else ""
        print(f"  ({f_name}, {g_name}): {count}{tag}", flush=True)


def finalize_splits(cfg: GenConfig) -> dict[str, Path]:
    """Read ``all.jsonl`` and write the same split files as ``generate_dataset``."""
    path = raw_path(cfg)
    if not path.exists():
        raise SystemExit(f"No accumulation file at {path}. Generate before --finalize.")
    records = load_raw(path)
    for record in records:
        if "f_template" not in record or "g_template" not in record:
            raise SystemExit(
                f"{path} is missing f_template/g_template; cannot separate held-out combos."
            )
    main = [r for r in records if not _is_heldout(r)]
    held = [r for r in records if _is_heldout(r)]

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
    write_jsonl(paths["test_unseen"], held)

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
        "n_test_unseen": len(held),
        "heldout_combos": sorted(list(HELDOUT_COMBOS)),
        "f_templates": sorted(F_TEMPLATES),
        "g_templates": sorted(G_TEMPLATES),
        "family": cfg.family,
        "sympy_version": sympy_version,
        "torch_version": torch_version,
    }
    out.mkdir(parents=True, exist_ok=True)
    with (out / "meta.json").open("w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    paths["meta"] = out / "meta.json"
    return paths


def load_raw_config(path: str | Path) -> tuple[GenConfig, dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    return cfg_from_mapping(raw), raw


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Resumable separable ODE dataset generation")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--max-seconds", type=float, default=600.0)
    parser.add_argument("--target", type=int, default=30000)
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override data.output_dir (all.jsonl location and finalize split writes).",
    )
    parser.add_argument(
        "--finalize",
        action="store_true",
        help="Read all.jsonl and write train/val/test/test_unseen. Does not generate.",
    )
    args = parser.parse_args(argv)

    cfg, raw = load_raw_config(args.config)
    if args.output_dir:
        cfg.output_dir = args.output_dir
    if args.finalize:
        paths = finalize_splits(cfg)
        print("Wrote:", flush=True)
        for name, path in paths.items():
            print(f"  {name}: {path}", flush=True)
        return

    tokenizer = build_tokenizer_from_config(raw)
    # Backstop if one SymPy call hangs past the loop's max-seconds check.
    _arm_hard_exit(float(args.max_seconds) + 20.0)
    stats = generate_resumable(cfg, tokenizer, args.target, float(args.max_seconds))
    print_stats(stats)


if __name__ == "__main__":
    main()
