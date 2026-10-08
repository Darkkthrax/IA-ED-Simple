"""STEP 1a-c: integer/UNK/length audit on 2000 examples (seed 42). Hard 120s cap."""

from __future__ import annotations

import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import sympy as sp
import yaml

from src.data.generator import GenConfig, generate_examples, load_config
from src.data.tokenizer import Tokenizer


def _ints_in_text(tok: Tokenizer, text: str) -> list[int]:
    expr = tok.parse_any(text)
    out: list[int] = []
    for atom in expr.atoms(sp.Integer):
        out.append(abs(int(atom)))
    # Rationals contribute numerator/denominator integers
    for atom in expr.atoms(sp.Rational):
        if not isinstance(atom, sp.Integer):
            out.append(abs(int(atom.p)))
            out.append(abs(int(atom.q)))
    return out


def main() -> None:
    t0 = time.perf_counter()
    max_seconds = 120.0
    cfg = load_config(ROOT / "configs" / "default.yaml")
    cfg.n_samples = 2000
    cfg.seed = 42
    with (ROOT / "configs" / "default.yaml").open(encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    max_seq_len = int(raw["model"]["max_seq_len"])
    max_int = int(raw["tokenizer"]["max_int"])

    tok = Tokenizer(max_int=max_int)
    rng = random.Random(42)
    # Manual loop with wall-clock cap (generate_examples has no time limit).
    from src.data.generator import F_TEMPLATES, G_TEMPLATES, HELDOUT_COMBOS, try_build_example

    combos = [
        (f, g)
        for f in F_TEMPLATES
        for g in G_TEMPLATES
        if (f, g) not in HELDOUT_COMBOS
    ]
    examples = []
    seen: set[tuple[str, str]] = set()
    while len(examples) < 2000 and (time.perf_counter() - t0) < 110.0:
        f_name, g_name = rng.choice(combos)
        ex = try_build_example(f_name, g_name, rng, cfg)
        if ex is None:
            continue
        key = (ex["ode"], ex["solution"])
        if key in seen:
            continue
        seen.add(key)
        examples.append(ex)
    elapsed_gen = time.perf_counter() - t0
    print(f"generated={len(examples)} elapsed_gen={elapsed_gen:.1f}s", flush=True)
    if len(examples) < 2000:
        print("WARNING: fewer than 2000 examples (hit time cap)", flush=True)

    max_abs = 0
    n_over = 0
    unk_total = 0
    n_with_unk = 0
    ode_lens: list[int] = []
    sol_lens: list[int] = []
    over_examples: list[str] = []

    for ex in examples:
        if time.perf_counter() - t0 > max_seconds:
            print("AUDIT_TRUNCATED_BY_TIME", flush=True)
            break
        ints = _ints_in_text(tok, ex["ode"]) + _ints_in_text(tok, ex["solution"])
        local_max = max(ints) if ints else 0
        max_abs = max(max_abs, local_max)
        if local_max > max_int:
            n_over += 1
            if len(over_examples) < 5:
                over_examples.append(
                    f"max={local_max} ODE={ex['ode'][:80]} SOL={ex['solution'][:80]}"
                )

        ids_ode = tok.encode(ex["ode"], add_special=True)
        ids_sol = tok.encode(ex["solution"], add_special=True)
        ode_lens.append(len(ids_ode))
        sol_lens.append(len(ids_sol))
        u = sum(1 for i in ids_ode + ids_sol if i == tok.vocab.unk_id)
        unk_total += u
        if u:
            n_with_unk += 1

    # Demonstrate behavior on a large integer
    demo_n = max(max_abs, 25)
    demo_tokens = tok.tokenize(sp.Integer(demo_n))
    demo_ids = [tok.vocab.encode_token(t) for t in demo_tokens]
    print("=== STEP 1a integers ===", flush=True)
    print(f"max_abs_integer={max_abs}", flush=True)
    print(f"examples_with_int_gt_{max_int}={n_over} / {len(examples)}", flush=True)
    for s in over_examples:
        print(f"  sample: {s}", flush=True)

    print("=== STEP 1b tokenizer large-int / UNK ===", flush=True)
    print(
        f"behavior: _int_tokens digit-split when n>max_int "
        f"(demo n={demo_n} -> tokens={demo_tokens} ids={demo_ids}; "
        f"UNK id={tok.vocab.unk_id})",
        flush=True,
    )
    print(f"UNK_id_count_in_encoded_dataset={unk_total}", flush=True)
    print(f"examples_with_any_UNK={n_with_unk}", flush=True)

    print("=== STEP 1c lengths ===", flush=True)
    print(
        f"ODE len incl SOS/EOS: min={min(ode_lens)} mean={sum(ode_lens)/len(ode_lens):.2f} "
        f"max={max(ode_lens)}",
        flush=True,
    )
    print(
        f"SOL len incl SOS/EOS: min={min(sol_lens)} mean={sum(sol_lens)/len(sol_lens):.2f} "
        f"max={max(sol_lens)}",
        flush=True,
    )
    print(f"model.max_seq_len={max_seq_len}", flush=True)
    need = max(max(ode_lens), max(sol_lens))
    print(
        f"max_seq_len_sufficient={need <= max_seq_len} (need>={need})",
        flush=True,
    )
    print(f"total_elapsed={time.perf_counter()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
