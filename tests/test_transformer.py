"""Shape, mask, and parameter-count checks for the ODE Transformer."""

from __future__ import annotations

from pathlib import Path

import torch
import yaml

from src.data.tokenizer import Tokenizer, build_tokenizer_from_config
from src.model.transformer import (
    ODETransformer,
    _strip_sos_eos,
    beam_search,
    build_model_from_config,
    count_parameters,
    decoded_token_ids,
    greedy_decode,
)

ROOT = Path(__file__).resolve().parents[1]


def _small_model() -> ODETransformer:
    tok = Tokenizer(max_int=20)
    return ODETransformer(
        vocab_size=len(tok.vocab),
        pad_id=tok.vocab.pad_id,
        sos_id=tok.vocab.sos_id,
        eos_id=tok.vocab.eos_id,
        d_model=32,
        nhead=4,
        num_encoder_layers=1,
        num_decoder_layers=1,
        dim_feedforward=64,
        dropout=0.0,
        max_seq_len=32,
    )


def test_default_config_parameter_count(capsys) -> None:
    raw = yaml.safe_load((ROOT / "configs" / "default.yaml").read_text(encoding="utf-8"))
    tok = build_tokenizer_from_config(raw)
    assert tok.max_int == 20
    assert len(tok.vocab) == 37
    model = build_model_from_config(raw, tok)
    assert model.d_model == 256
    assert model.transformer.encoder.num_layers == 4
    assert model.transformer.decoder.num_layers == 4
    n = count_parameters(model)
    print(f"DEFAULT_PARAM_COUNT {n} vocab {len(tok.vocab)}")
    captured = capsys.readouterr()
    assert "DEFAULT_PARAM_COUNT" in captured.out
    assert n == sum(p.numel() for p in model.parameters())
    assert n > 1_000_000


def test_forward_shape_and_backward() -> None:
    torch.manual_seed(0)
    model = _small_model()
    model.train()
    src = torch.randint(1, model.vocab_size, (4, 6))
    tgt = torch.randint(1, model.vocab_size, (4, 5))
    src_pad = torch.zeros(4, 6, dtype=torch.bool)
    tgt_pad = torch.zeros(4, 5, dtype=torch.bool)
    logits = model(src, tgt, src_pad, tgt_pad)
    assert logits.shape == (4, 5, model.vocab_size)
    loss = logits.reshape(-1, model.vocab_size).sum()
    loss.backward()
    assert model.embed.weight.grad is not None
    assert float(model.embed.weight.grad.abs().sum()) > 0


def test_causal_mask_hides_future_tokens() -> None:
    torch.manual_seed(1)
    model = _small_model()
    model.eval()
    src = torch.randint(1, model.vocab_size, (2, 5))
    tgt = torch.randint(1, model.vocab_size, (2, 7))
    src_pad = torch.zeros(2, 5, dtype=torch.bool)
    tgt_pad = torch.zeros(2, 7, dtype=torch.bool)
    logits_a = model(src, tgt, src_pad, tgt_pad)
    changed = tgt.clone()
    changed[:, -1] = (changed[:, -1] + 3) % model.vocab_size
    changed[:, -1] = torch.where(changed[:, -1] == 0, torch.ones_like(changed[:, -1]), changed[:, -1])
    logits_b = model(src, changed, src_pad, tgt_pad)
    assert torch.allclose(logits_a[:, :-1, :], logits_b[:, :-1, :], atol=1e-5)
    assert not torch.allclose(logits_a[:, -1, :], logits_b[:, -1, :], atol=1e-5)


def test_src_padding_mask_ignores_masked_positions() -> None:
    torch.manual_seed(2)
    model = _small_model()
    model.eval()
    src = torch.randint(1, model.vocab_size, (2, 6))
    tgt = torch.randint(1, model.vocab_size, (2, 4))
    src_pad = torch.zeros(2, 6, dtype=torch.bool)
    src_pad[:, -1] = True
    tgt_pad = torch.zeros(2, 4, dtype=torch.bool)
    other = src.clone()
    other[:, -1] = (other[:, -1] + 5) % model.vocab_size
    other[:, -1] = torch.where(other[:, -1] == 0, torch.ones_like(other[:, -1]), other[:, -1])
    logits_a = model(src, tgt, src_pad, tgt_pad)
    logits_b = model(other, tgt, src_pad, tgt_pad)
    assert torch.allclose(logits_a, logits_b, atol=1e-5)


def test_greedy_decode_starts_with_sos() -> None:
    torch.manual_seed(3)
    model = _small_model()
    src = torch.randint(1, model.vocab_size, (3, 4))
    out = greedy_decode(model, src, max_len=6)
    assert out.shape[0] == 3
    assert out.shape[1] <= 6
    assert int(out[0, 0]) == model.sos_id
    single = greedy_decode(model, src[0], max_len=4)
    assert single.dim() == 1
    assert int(single[0]) == model.sos_id


def test_beam_size_one_matches_greedy() -> None:
    torch.manual_seed(4)
    model = _small_model()
    src = torch.randint(1, model.vocab_size, (20, 5))
    max_len = 12
    greedy = greedy_decode(model, src, max_len=max_len)
    beams = beam_search(model, src, beam_size=1, max_len=max_len, num_return=1)
    for i in range(20):
        g_ids = decoded_token_ids(greedy[i], model.eos_id, model.pad_id)
        g_body = _strip_sos_eos(g_ids, model.sos_id, model.eos_id)
        assert g_body == beams[i][0]
