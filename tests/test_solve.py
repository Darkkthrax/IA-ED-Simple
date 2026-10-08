"""End-to-end CLI/library smoke for src.solve with a tiny random model."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import torch
import yaml

from src.data.tokenizer import build_tokenizer_from_config
from src.model.transformer import build_model_from_config
from src.solve import solve


ROOT = Path(__file__).resolve().parents[1]


def _tiny_checkpoint(tmp_path: Path) -> Path:
    raw = yaml.safe_load((ROOT / "configs" / "default.yaml").read_text(encoding="utf-8"))
    raw["model"] = {
        "d_model": 32,
        "nhead": 4,
        "num_encoder_layers": 1,
        "num_decoder_layers": 1,
        "dim_feedforward": 64,
        "dropout": 0.0,
        "max_seq_len": 64,
    }
    tokenizer = build_tokenizer_from_config(raw)
    model = build_model_from_config(raw, tokenizer)
    ckpt_path = tmp_path / "tiny.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "config": raw,
            "tokenizer": {
                "max_int": tokenizer.max_int,
                "special_tokens": ["PAD", "SOS", "EOS", "UNK"],
                "id_to_token": list(tokenizer.vocab.id_to_token),
            },
            "sympy_version": __import__("sympy").__version__,
            "torch_version": torch.__version__,
        },
        ckpt_path,
    )
    return ckpt_path


def test_solve_function_does_not_crash(tmp_path) -> None:
    ckpt = _tiny_checkpoint(tmp_path)
    payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    tokenizer = build_tokenizer_from_config(payload["config"])
    # Rebuild vocab from checkpoint for fidelity.
    from src.solve import load_model_and_tokenizer

    model, tokenizer, _cfg, device = load_model_and_tokenizer(ckpt, device=torch.device("cpu"))
    result = solve("y' = x*exp(y)", model, tokenizer, beam_size=2, top_k=2, device=device)
    assert "verified" in result
    assert "candidates" in result
    assert result.get("error") is None or isinstance(result["error"], str)

    bad = solve("not an ode", model, tokenizer, beam_size=2, top_k=2, device=device)
    assert bad["verified"] is False
    assert bad["error"]


def test_solve_cli_end_to_end(tmp_path) -> None:
    ckpt = _tiny_checkpoint(tmp_path)
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.solve",
            "y' = x",
            "--checkpoint",
            str(ckpt),
            "--beam-size",
            "2",
            "--top-k",
            "2",
            "--show-all",
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr + proc.stdout
    assert "candidates:" in proc.stdout or "VERIFIED:" in proc.stdout or "No candidate" in proc.stdout
