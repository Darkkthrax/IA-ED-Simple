"""Evaluate a checkpoint with beam search and symbolic ODE checking."""

from __future__ import annotations

import argparse
import json
import re
import threading
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

import sympy as sp
import torch
import yaml

from src.data.generator import HELDOUT_COMBOS, solution_satisfies_ode
from src.data.tokenizer import Tokenizer, Vocab, build_tokenizer_from_config
from src.model.transformer import beam_search, build_model_from_config


_INT_RE = re.compile(r"\b\d+\b")


def _print_library_versions(meta_path: Optional[Path] = None) -> None:
    print(f"sympy={sp.__version__} torch={torch.__version__}", flush=True)
    if meta_path is None or not meta_path.exists():
        return
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return
    recorded = meta.get("sympy_version")
    if recorded is not None and str(recorded) != str(sp.__version__):
        print(
            f"WARNING: sympy version mismatch: runtime={sp.__version__} "
            f"meta.json={recorded}",
            flush=True,
        )


def _tokenizer_from_checkpoint(ckpt: dict[str, Any], cfg: dict[str, Any]) -> Tokenizer:
    state = ckpt.get("tokenizer")
    if state and "id_to_token" in state:
        id_to_token = list(state["id_to_token"])
        vocab = Vocab(
            token_to_id={t: i for i, t in enumerate(id_to_token)},
            id_to_token=id_to_token,
        )
        return Tokenizer(max_int=int(state["max_int"]), vocab=vocab)
    if state:
        return build_tokenizer_from_config({"tokenizer": state})
    return build_tokenizer_from_config(cfg)


def _load_split(path: Path, max_examples: Optional[int]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if max_examples is not None and len(rows) >= max_examples:
                break
    return rows


def _has_int_above_max(ode: str, solution: str, max_int: int, tokenizer: Tokenizer) -> bool:
    """True if ODE or solution contains an integer magnitude greater than max_int."""
    for text in (ode, solution):
        try:
            expr = tokenizer.parse_any(text)
            for rat in expr.atoms(sp.Rational):
                if abs(int(rat.p)) > max_int or abs(int(rat.q)) > max_int:
                    return True
        except Exception:
            for m in _INT_RE.findall(text):
                if abs(int(m)) > max_int:
                    return True
    return False


def _ids_to_solution_str(tokenizer: Tokenizer, ids: list[int]) -> Optional[str]:
    try:
        expr = tokenizer.ids_to_sympy(ids)
        if isinstance(expr, sp.Equality):
            return f"{sp.sstr(expr.lhs)} = {sp.sstr(expr.rhs)}"
        return sp.sstr(expr)
    except Exception:
        return None


def _ref_body_ids(tokenizer: Tokenizer, ref: str) -> list[int]:
    ref_ids = tokenizer.encode(ref, add_special=True)
    ref_body = ref_ids[1:-1] if ref_ids and ref_ids[0] == tokenizer.vocab.sos_id else ref_ids
    if ref_body and ref_body[-1] == tokenizer.vocab.eos_id:
        ref_body = ref_body[:-1]
    return list(ref_body)


def corrupt_solution_candidate(sol: str) -> str:
    """Deterministic corruption of a reference solution for negative control."""
    # Prefer changing an integer digit so the parse still succeeds but ODE check fails.
    for i, ch in enumerate(sol):
        if ch.isdigit() and ch != "0":
            repl = "9" if ch != "9" else "8"
            return sol[:i] + repl + sol[i + 1 :]
    for a, b in (("+", "-"), ("*", "+"), ("-", "+"), ("/", "*")):
        if a in sol:
            return sol.replace(a, b, 1)
    return sol + " + 1"


def _check_solution(
    ode: str,
    sol: str,
    timeout_s: float,
) -> bool:
    """Run ``solution_satisfies_ode`` with a soft timeout (daemon thread)."""
    box: list[bool] = [False]

    def _run() -> None:
        try:
            box[0] = bool(solution_satisfies_ode(ode, sol))
        except Exception:
            box[0] = False

    worker = threading.Thread(target=_run, daemon=True)
    worker.start()
    worker.join(timeout=timeout_s)
    if worker.is_alive():
        return False
    return box[0]


def evaluate(cfg: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    data_dir = Path(args.data_dir or cfg["data"]["output_dir"])
    _print_library_versions(data_dir / "meta.json")

    split_path = data_dir / f"{args.split}.jsonl"
    if not split_path.exists():
        raise SystemExit(f"Split not found: {split_path}")

    oracle = bool(getattr(args, "oracle", False))
    oracle_corrupt = bool(getattr(args, "oracle_corrupt", False))
    if oracle_corrupt:
        oracle = True

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )

    model = None
    model_cfg = cfg
    if oracle:
        if args.checkpoint:
            ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
            model_cfg = ckpt.get("config", cfg)
            tokenizer = _tokenizer_from_checkpoint(ckpt, model_cfg)
        else:
            tokenizer = build_tokenizer_from_config(cfg)
    else:
        if not args.checkpoint:
            raise SystemExit("--checkpoint is required unless --oracle is set")
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        model_cfg = ckpt.get("config", cfg)
        tokenizer = _tokenizer_from_checkpoint(ckpt, model_cfg)
        model = build_model_from_config(model_cfg, tokenizer)
        model.load_state_dict(ckpt["model"])
        model.to(device)
        model.eval()

    ev = cfg["evaluate"]
    beam_size = int(ev["beam_size"])
    top_k = int(ev["top_k"])
    max_decode_len = int(ev["max_decode_len"])
    timeout_s = float(ev["numeric_timeout_s"])
    max_seq_len = int(model_cfg.get("model", model_cfg).get("max_seq_len", 128))
    if model is not None:
        max_decode_len = min(max_decode_len, model.max_seq_len, max_seq_len)
    else:
        max_decode_len = min(max_decode_len, max_seq_len)

    rows = _load_split(split_path, args.max_examples)
    has_templates = all("f_template" in r and "g_template" in r for r in rows) if rows else False

    n = len(rows)
    top1_sym = 0
    topk_sym = 0
    token_exact = 0
    valid_but_diff = 0
    invalid_parse = 0
    verify_seconds = 0.0
    verify_calls = 0
    by_combo: dict[tuple[str, str], dict[str, int]] = defaultdict(
        lambda: {"n": 0, "top1": 0, "topk": 0}
    )
    by_heldout = {
        False: {"n": 0, "top1": 0, "topk": 0},
        True: {"n": 0, "top1": 0, "topk": 0},
    }
    by_big_int = {
        True: {"n": 0, "top1": 0, "topk": 0},
        False: {"n": 0, "top1": 0, "topk": 0},
    }
    failures: list[dict[str, str]] = []

    for idx, row in enumerate(rows):
        ode = row["ode"]
        ref = row["solution"]
        src_ids = tokenizer.encode(ode, add_special=True)
        if len(src_ids) > max_seq_len:
            big = _has_int_above_max(ode, ref, tokenizer.max_int, tokenizer)
            by_big_int[big]["n"] += 1
            if has_templates:
                key = (row["f_template"], row["g_template"])
                by_combo[key]["n"] += 1
                held = key in HELDOUT_COMBOS
                by_heldout[held]["n"] += 1
            if len(failures) < 10:
                failures.append(
                    {"ode": ode, "reference": ref, "top1": "<skipped: src too long>"}
                )
            continue

        if oracle:
            cand_text = corrupt_solution_candidate(ref) if oracle_corrupt else ref
            beams = [_ref_body_ids(tokenizer, cand_text)]
        else:
            assert model is not None
            src = torch.tensor([src_ids], dtype=torch.long, device=device)
            beams = beam_search(
                model,
                src,
                beam_size=beam_size,
                max_len=max_decode_len,
                num_return=top_k,
            )[0]

        ref_body = _ref_body_ids(tokenizer, ref)

        hit_top1 = False
        hit_topk = False
        top1_str = ""
        any_valid_diff = False
        for rank, cand_ids in enumerate(beams):
            if cand_ids == ref_body and rank == 0:
                token_exact += 1
            sol_str = _ids_to_solution_str(tokenizer, cand_ids)
            if sol_str is None:
                invalid_parse += 1
                if rank == 0:
                    top1_str = "<invalid parse>"
                continue
            if rank == 0:
                top1_str = sol_str
            t0 = time.perf_counter()
            ok = _check_solution(ode, sol_str, timeout_s=timeout_s)
            verify_seconds += time.perf_counter() - t0
            verify_calls += 1
            if ok:
                if rank == 0:
                    hit_top1 = True
                hit_topk = True
                if sol_str.strip() != ref.strip() and cand_ids != ref_body:
                    any_valid_diff = True
        if hit_top1:
            top1_sym += 1
        if hit_topk:
            topk_sym += 1
        if any_valid_diff:
            valid_but_diff += 1
        if not hit_top1 and len(failures) < 10:
            failures.append({"ode": ode, "reference": ref, "top1": top1_str or "<empty>"})

        big = _has_int_above_max(ode, ref, tokenizer.max_int, tokenizer)
        by_big_int[big]["n"] += 1
        by_big_int[big]["top1"] += int(hit_top1)
        by_big_int[big]["topk"] += int(hit_topk)
        if has_templates:
            key = (row["f_template"], row["g_template"])
            by_combo[key]["n"] += 1
            by_combo[key]["top1"] += int(hit_top1)
            by_combo[key]["topk"] += int(hit_topk)
            held = key in HELDOUT_COMBOS
            by_heldout[held]["n"] += 1
            by_heldout[held]["top1"] += int(hit_top1)
            by_heldout[held]["topk"] += int(hit_topk)

        if (idx + 1) % 10 == 0 or idx + 1 == n:
            print(f"evaluated {idx + 1}/{n}", flush=True)

    avg_verify_s = verify_seconds / max(1, verify_calls)
    summary = {
        "n": n,
        "top1_symbolic_acc": top1_sym / max(1, n),
        "topk_symbolic_acc": topk_sym / max(1, n),
        "token_exact_match": token_exact / max(1, n),
        "valid_but_diff_form": valid_but_diff,
        "invalid_parse_candidates": invalid_parse,
        "top_k": top_k,
        "beam_size": beam_size,
        "has_templates": has_templates,
        "oracle": oracle,
        "oracle_corrupt": oracle_corrupt,
        "avg_verify_seconds": avg_verify_s,
        "verify_calls": verify_calls,
    }

    mode = "oracle-corrupt" if oracle_corrupt else ("oracle" if oracle else "model")
    print("=== Evaluation ===", flush=True)
    print(
        f"mode={mode} split={args.split} n={n} checkpoint={args.checkpoint}",
        flush=True,
    )
    print(
        f"top1_symbolic_acc: {summary['top1_symbolic_acc']:.4f} ({top1_sym}/{n})",
        flush=True,
    )
    print(
        f"top{top_k}_symbolic_acc: {summary['topk_symbolic_acc']:.4f} ({topk_sym}/{n})",
        flush=True,
    )
    print(
        f"token_exact_match: {summary['token_exact_match']:.4f} ({token_exact}/{n})",
        flush=True,
    )
    print(
        f"valid_but_diff_form: {valid_but_diff} "
        "(symbolically valid candidate differs from reference string/tokens)",
        flush=True,
    )
    print(f"invalid_parse_candidates: {invalid_parse}", flush=True)
    print(
        f"avg_verify_seconds: {avg_verify_s:.4f} "
        f"(total={verify_seconds:.2f}s over {verify_calls} checks)",
        flush=True,
    )

    if has_templates:
        print("per (f_template, g_template):", flush=True)
        for key in sorted(by_combo):
            s = by_combo[key]
            tag = " heldout" if key in HELDOUT_COMBOS else ""
            print(
                f"  {key}: n={s['n']} top1={s['top1']/max(1,s['n']):.3f} "
                f"topk={s['topk']/max(1,s['n']):.3f}{tag}",
                flush=True,
            )
        print("held-out vs in-distribution:", flush=True)
        for held, label in ((False, "in_distribution"), (True, "held_out")):
            s = by_heldout[held]
            print(
                f"  {label}: n={s['n']} top1={s['top1']/max(1,s['n']):.4f} "
                f"topk={s['topk']/max(1,s['n']):.4f}",
                flush=True,
            )
        summary["by_combo"] = {
            f"{f}|{g}": dict(v) for (f, g), v in sorted(by_combo.items())
        }
        summary["in_distribution"] = dict(by_heldout[False])
        summary["held_out"] = dict(by_heldout[True])
    else:
        print(
            "per (f_template, g_template): UNAVAILABLE - split JSONL lacks "
            "f_template/g_template fields.",
            flush=True,
        )

    for flag, label in ((False, "no_int_above_max"), (True, "has_int_above_max")):
        s = by_big_int[flag]
        print(
            f"{label}: n={s['n']} top1={s['top1']/max(1,s['n']):.4f} "
            f"topk={s['topk']/max(1,s['n']):.4f}",
            flush=True,
        )

    print("failure examples (up to 10):", flush=True)
    for i, ex in enumerate(failures):
        print(f"  [{i}] ODE: {ex['ode']}", flush=True)
        print(f"      ref: {ex['reference']}", flush=True)
        print(f"      top1: {ex['top1']}", flush=True)

    return summary


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate ODE Transformer checkpoint")
    p.add_argument("--config", type=str, default="configs/default.yaml")
    p.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Required unless --oracle (tokenizer then comes from config).",
    )
    p.add_argument("--split", type=str, choices=("val", "test", "test_unseen"), required=True)
    p.add_argument("--max-examples", type=int, default=None)
    p.add_argument("--data-dir", type=str, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument(
        "--oracle",
        action="store_true",
        help="Use reference solution tokens (encode->decode->parse) instead of the model.",
    )
    p.add_argument(
        "--oracle-corrupt",
        action="store_true",
        help="Oracle negative control: corrupt each reference candidate before verify.",
    )
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)
    with open(args.config, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    evaluate(cfg, args)


if __name__ == "__main__":
    main()
