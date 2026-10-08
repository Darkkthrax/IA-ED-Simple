"""CLI and library entry point to solve an ODE string with a trained checkpoint."""

from __future__ import annotations

import argparse
import json
import threading
from pathlib import Path
from typing import Any, Optional

import sympy as sp
import torch
import yaml

from src.data.generator import solution_satisfies_ode
from src.data.tokenizer import Tokenizer, Vocab, build_tokenizer_from_config
from src.model.transformer import beam_search, build_model_from_config


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


def _ids_to_solution_str(tokenizer: Tokenizer, ids: list[int]) -> Optional[str]:
    try:
        expr = tokenizer.ids_to_sympy(ids)
        if isinstance(expr, sp.Equality):
            return f"{sp.sstr(expr.lhs)} = {sp.sstr(expr.rhs)}"
        return sp.sstr(expr)
    except Exception:
        return None


def _check_solution(ode: str, sol: str, timeout_s: float) -> bool:
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


def load_model_and_tokenizer(
    checkpoint: str | Path,
    config: Optional[dict[str, Any]] = None,
    device: Optional[torch.device] = None,
) -> tuple[torch.nn.Module, Tokenizer, dict[str, Any], torch.device]:
    """Rebuild model + tokenizer from a training checkpoint."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    cfg = ckpt.get("config") or config
    if cfg is None:
        raise ValueError("Checkpoint has no config and no config override was provided")
    tokenizer = _tokenizer_from_checkpoint(ckpt, cfg)
    model = build_model_from_config(cfg, tokenizer)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()
    return model, tokenizer, cfg, device


def solve(
    ode_str: str,
    model: torch.nn.Module,
    tokenizer: Tokenizer,
    *,
    beam_size: int = 5,
    top_k: int = 5,
    max_decode_len: int = 128,
    timeout_s: float = 2.0,
    device: Optional[torch.device] = None,
) -> dict[str, Any]:
    """Encode ``ode_str``, beam-search candidates, and symbolically verify them.

    Returns a dict with keys: ``ode``, ``verified``, ``solution``, ``candidates``.
    Never raises for invalid ODE strings; reports ``error`` instead.
    """
    if device is None:
        device = next(model.parameters()).device
    try:
        tokenizer.parse_ode_string(ode_str)
    except Exception as exc:  # noqa: BLE001
        return {
            "ode": ode_str,
            "verified": False,
            "solution": None,
            "candidates": [],
            "error": f"invalid ODE: {exc}",
        }

    try:
        src_ids = tokenizer.encode(ode_str, add_special=True)
    except Exception as exc:  # noqa: BLE001
        return {
            "ode": ode_str,
            "verified": False,
            "solution": None,
            "candidates": [],
            "error": f"encode failed: {exc}",
        }

    if len(src_ids) > model.max_seq_len:
        return {
            "ode": ode_str,
            "verified": False,
            "solution": None,
            "candidates": [],
            "error": f"source length {len(src_ids)} exceeds max_seq_len {model.max_seq_len}",
        }

    max_len = min(max_decode_len, model.max_seq_len)
    src = torch.tensor([src_ids], dtype=torch.long, device=device)
    beams = beam_search(
        model,
        src,
        beam_size=beam_size,
        max_len=max_len,
        num_return=top_k,
    )[0]

    candidates: list[dict[str, Any]] = []
    verified_sol: Optional[str] = None
    for cand_ids in beams:
        sol_str = _ids_to_solution_str(tokenizer, cand_ids)
        parsed = sol_str is not None
        ok = False
        if parsed and sol_str is not None:
            ok = _check_solution(ode_str, sol_str, timeout_s=timeout_s)
            if ok and verified_sol is None:
                verified_sol = sol_str
        candidates.append(
            {
                "tokens": cand_ids,
                "text": sol_str,
                "parsed": parsed,
                "verified": ok,
            }
        )

    return {
        "ode": ode_str,
        "verified": verified_sol is not None,
        "solution": verified_sol,
        "candidates": candidates,
        "error": None,
    }


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Solve an ODE with a trained checkpoint")
    p.add_argument("ode", type=str, help='ODE string, e.g. "y\' = x*exp(y)"')
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--config", type=str, default="configs/default.yaml")
    p.add_argument("--beam-size", type=int, default=None)
    p.add_argument("--top-k", type=int, default=None)
    p.add_argument("--show-all", action="store_true")
    p.add_argument("--device", type=str, default=None)
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)
    print(f"sympy={sp.__version__} torch={torch.__version__}", flush=True)
    with open(args.config, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    meta_path = Path(cfg.get("data", {}).get("output_dir", "data/processed")) / "meta.json"
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            recorded = meta.get("sympy_version")
            if recorded is not None and str(recorded) != str(sp.__version__):
                print(
                    f"WARNING: sympy version mismatch: runtime={sp.__version__} "
                    f"meta.json={recorded}",
                    flush=True,
                )
        except Exception:
            pass

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    try:
        model, tokenizer, model_cfg, device = load_model_and_tokenizer(
            args.checkpoint, config=cfg, device=device
        )
    except Exception as exc:  # noqa: BLE001
        print(f"error: failed to load checkpoint: {exc}", flush=True)
        return

    ev = model_cfg.get("evaluate", cfg.get("evaluate", {}))
    sol_cfg = model_cfg.get("solve", cfg.get("solve", {}))
    beam_size = int(args.beam_size or sol_cfg.get("beam_size", ev.get("beam_size", 5)))
    top_k = int(args.top_k or ev.get("top_k", beam_size))
    max_decode_len = int(sol_cfg.get("max_decode_len", ev.get("max_decode_len", 128)))
    timeout_s = float(ev.get("numeric_timeout_s", 2.0))

    result = solve(
        args.ode,
        model,
        tokenizer,
        beam_size=beam_size,
        top_k=top_k,
        max_decode_len=max_decode_len,
        timeout_s=timeout_s,
        device=device,
    )
    if result.get("error"):
        print(f"error: {result['error']}", flush=True)
        return

    if result["verified"]:
        print(f"VERIFIED: {result['solution']}", flush=True)
    else:
        print("No candidate verified against the ODE.", flush=True)

    if args.show_all or not result["verified"]:
        print("candidates:", flush=True)
        for i, cand in enumerate(result["candidates"]):
            text = cand["text"] if cand["parsed"] else "<invalid parse>"
            flag = "verified" if cand["verified"] else ("parsed" if cand["parsed"] else "unparsed")
            print(f"  [{i}] ({flag}) {text}", flush=True)


if __name__ == "__main__":
    main()
