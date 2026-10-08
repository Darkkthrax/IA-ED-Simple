"""STEP 1d: per-(f,g) duplicate rate over 100 attempts. Supports chunked runs."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.generator import (  # noqa: E402
    F_TEMPLATES,
    G_TEMPLATES,
    HELDOUT_COMBOS,
    load_config,
    try_build_example,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=0, help="Combo index to start at")
    parser.add_argument("--max-seconds", type=float, default=120.0)
    parser.add_argument("--attempts-per-combo", type=int, default=100)
    parser.add_argument(
        "--out",
        type=str,
        default="data/processed/audit_dup_rates.jsonl",
    )
    args = parser.parse_args()

    cfg = load_config(ROOT / "configs" / "default.yaml")
    combos = [
        (f, g)
        for f in sorted(F_TEMPLATES)
        for g in sorted(G_TEMPLATES)
        if (f, g) not in HELDOUT_COMBOS
    ]
    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    done = 0
    with out_path.open("a" if args.start > 0 else "w", encoding="utf-8") as fout:
        for idx in range(args.start, len(combos)):
            if time.perf_counter() - t0 >= args.max_seconds:
                print(f"TIME_CAP next_start={idx} done_this_run={done}", flush=True)
                return
            f_name, g_name = combos[idx]
            # Independent RNG per combo for reproducibility of this audit metric.
            rng = random.Random(42 + idx * 17)
            seen: set[tuple[str, str]] = set()
            success = 0
            dups = 0
            fails = 0
            for _ in range(args.attempts_per_combo):
                if time.perf_counter() - t0 >= args.max_seconds:
                    print(f"TIME_CAP mid-combo next_start={idx} done_this_run={done}", flush=True)
                    return
                ex = try_build_example(f_name, g_name, rng, cfg)
                if ex is None:
                    fails += 1
                    continue
                success += 1
                key = (ex["ode"], ex["solution"])
                if key in seen:
                    dups += 1
                else:
                    seen.add(key)
            unique = len(seen)
            dup_rate = (dups / success) if success else 0.0
            row = {
                "f": f_name,
                "g": g_name,
                "attempts": args.attempts_per_combo,
                "success": success,
                "fails": fails,
                "unique": unique,
                "duplicate_builds": dups,
                "duplicate_rate": dup_rate,
            }
            fout.write(json.dumps(row) + "\n")
            fout.flush()
            print(
                f"[{idx+1}/{len(combos)}] ({f_name},{g_name}) "
                f"success={success} unique={unique} dup_rate={dup_rate:.3f}",
                flush=True,
            )
            done += 1

    print(f"COMPLETE all {len(combos)} combos", flush=True)


if __name__ == "__main__":
    main()
