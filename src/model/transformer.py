"""Encoder-decoder Transformer built on ``torch.nn.Transformer``."""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from src.data.tokenizer import Tokenizer


class SinusoidalPositionalEncoding(nn.Module):
    """Fixed sinusoids added to token embeddings. Shape is batch-first."""

    def __init__(self, d_model: int, max_len: int) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div)
        pe[:, 1::2] = torch.cos(position * div)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class ODETransformer(nn.Module):
    """Shared embedding, sinusoidal positions, tied output projection.

    Padding masks follow PyTorch: True marks tokens that must be ignored.
    The causal target mask is a boolean upper triangle (True = blocked).
    """

    def __init__(
        self,
        vocab_size: int,
        pad_id: int,
        sos_id: int,
        eos_id: int,
        d_model: int,
        nhead: int,
        num_encoder_layers: int,
        num_decoder_layers: int,
        dim_feedforward: int,
        dropout: float,
        max_seq_len: int,
    ) -> None:
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by nhead ({nhead})")
        self.vocab_size = vocab_size
        self.pad_id = pad_id
        self.sos_id = sos_id
        self.eos_id = eos_id
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.scale = math.sqrt(d_model)

        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        nn.init.normal_(self.embed.weight, mean=0.0, std=d_model**-0.5)
        with torch.no_grad():
            self.embed.weight[pad_id].zero_()
        self.pos = SinusoidalPositionalEncoding(d_model, max_seq_len)
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )

    def _embed(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.pos(self.embed(token_ids) * self.scale)

    def forward(
        self,
        src: torch.Tensor,
        tgt_in: torch.Tensor,
        src_key_padding_mask: torch.Tensor,
        tgt_key_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Return logits ``[batch, tgt_len, vocab]``."""
        if src.size(1) > self.max_seq_len or tgt_in.size(1) > self.max_seq_len:
            raise ValueError(
                f"Sequence length exceeds max_seq_len={self.max_seq_len} "
                f"(src={src.size(1)}, tgt={tgt_in.size(1)})"
            )
        tgt_mask = torch.triu(
            torch.ones(tgt_in.size(1), tgt_in.size(1), device=tgt_in.device, dtype=torch.bool),
            diagonal=1,
        )
        hidden = self.transformer(
            self._embed(src),
            self._embed(tgt_in),
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask,
        )
        return F.linear(hidden, self.embed.weight)


def build_model_from_config(
    cfg: dict[str, Any],
    tokenizer: Tokenizer,
    **overrides: Any,
) -> ODETransformer:
    """Build a model from the ``model:`` block. ``overrides`` replace those keys."""
    block = dict(cfg.get("model", cfg))
    block.update(overrides)
    return ODETransformer(
        vocab_size=len(tokenizer.vocab),
        pad_id=tokenizer.vocab.pad_id,
        sos_id=tokenizer.vocab.sos_id,
        eos_id=tokenizer.vocab.eos_id,
        d_model=int(block["d_model"]),
        nhead=int(block["nhead"]),
        num_encoder_layers=int(block["num_encoder_layers"]),
        num_decoder_layers=int(block["num_decoder_layers"]),
        dim_feedforward=int(block["dim_feedforward"]),
        dropout=float(block["dropout"]),
        max_seq_len=int(block["max_seq_len"]),
    )


def count_parameters(model: nn.Module) -> int:
    """Number of parameters, counting a tied weight once."""
    seen: set[int] = set()
    total = 0
    for param in model.parameters():
        ident = id(param)
        if ident in seen:
            continue
        seen.add(ident)
        total += param.numel()
    return total


def _ids_until_stop(row: list[int], eos_id: int, pad_id: int) -> list[int]:
    out: list[int] = []
    for token_id in row:
        if token_id == pad_id:
            break
        out.append(token_id)
        if token_id == eos_id:
            break
    return out


@torch.no_grad()
def greedy_decode(model: ODETransformer, src: torch.Tensor, max_len: int) -> torch.Tensor:
    """Greedy decode.

    ``src`` is ``[batch, src_len]`` or ``[src_len]``. Returns token ids starting
    with SOS, including the first EOS, shaped ``[batch, dec_len]`` (squeezed
    back to 1-D when ``src`` was 1-D). ``max_len`` counts SOS.
    """
    if max_len < 1:
        raise ValueError("max_len must be >= 1")
    if max_len > model.max_seq_len:
        raise ValueError(f"max_len {max_len} exceeds model.max_seq_len {model.max_seq_len}")

    was_training = model.training
    model.eval()
    single = src.dim() == 1
    if single:
        src = src.unsqueeze(0)
    if src.dim() != 2:
        raise ValueError(f"src must be rank 1 or 2, got {tuple(src.shape)}")

    device = src.device
    batch = src.size(0)
    src_pad = src.eq(model.pad_id)
    ys = torch.full((batch, 1), model.sos_id, dtype=torch.long, device=device)
    finished = torch.zeros(batch, dtype=torch.bool, device=device)
    try:
        for _ in range(max_len - 1):
            logits = model(src, ys, src_pad, ys.eq(model.pad_id))
            next_id = logits[:, -1, :].argmax(dim=-1)
            next_id = torch.where(finished, torch.full_like(next_id, model.pad_id), next_id)
            ys = torch.cat([ys, next_id.unsqueeze(1)], dim=1)
            finished = finished | next_id.eq(model.eos_id)
            if bool(finished.all()):
                break
    finally:
        model.train(was_training)

    if single:
        return ys.squeeze(0)
    return ys


def decoded_token_ids(row: torch.Tensor, eos_id: int, pad_id: int) -> list[int]:
    """Trim a decode row at PAD or the first EOS (EOS kept)."""
    values = row.detach().cpu().tolist()
    return _ids_until_stop([int(v) for v in values], eos_id, pad_id)


def _strip_sos_eos(ids: list[int], sos_id: int, eos_id: int) -> list[int]:
    out = list(ids)
    if out and out[0] == sos_id:
        out = out[1:]
    if out and out[-1] == eos_id:
        out = out[:-1]
    elif eos_id in out:
        out = out[: out.index(eos_id)]
    return out


@torch.no_grad()
def beam_search(
    model: ODETransformer,
    src: torch.Tensor,
    beam_size: int,
    max_len: int,
    num_return: int,
) -> list[list[list[int]]]:
    """Beam search with length-normalized log-probability scores.

    ``src`` is ``[batch, src_len]`` or ``[src_len]``. For each example returns up
    to ``num_return`` candidate token-id lists (no SOS/EOS), best first.
    ``max_len`` counts SOS, matching ``greedy_decode``.
    """
    if beam_size < 1:
        raise ValueError("beam_size must be >= 1")
    if num_return < 1:
        raise ValueError("num_return must be >= 1")
    if max_len < 1:
        raise ValueError("max_len must be >= 1")
    if max_len > model.max_seq_len:
        raise ValueError(f"max_len {max_len} exceeds model.max_seq_len {model.max_seq_len}")

    was_training = model.training
    model.eval()
    single = src.dim() == 1
    if single:
        src = src.unsqueeze(0)
    if src.dim() != 2:
        raise ValueError(f"src must be rank 1 or 2, got {tuple(src.shape)}")

    results: list[list[list[int]]] = []
    try:
        for i in range(src.size(0)):
            results.append(
                _beam_search_one(
                    model,
                    src[i : i + 1],
                    beam_size=beam_size,
                    max_len=max_len,
                    num_return=num_return,
                )
            )
    finally:
        model.train(was_training)
    return results


def _beam_search_one(
    model: ODETransformer,
    src: torch.Tensor,
    beam_size: int,
    max_len: int,
    num_return: int,
) -> list[list[int]]:
    """Beam search for a single padded source row ``[1, src_len]``."""
    device = src.device
    src_pad = src.eq(model.pad_id)
    # Each live hyp: (token_ids including SOS, cumulative log-prob)
    live: list[tuple[list[int], float]] = [([model.sos_id], 0.0)]
    finished: list[tuple[list[int], float]] = []

    for _ in range(max_len - 1):
        if not live:
            break
        ys = torch.tensor([hyp for hyp, _ in live], dtype=torch.long, device=device)
        src_rep = src.expand(ys.size(0), -1)
        src_pad_rep = src_pad.expand(ys.size(0), -1)
        logits = model(src_rep, ys, src_pad_rep, ys.eq(model.pad_id))
        log_probs = torch.log_softmax(logits[:, -1, :], dim=-1)
        topk_log, topk_idx = log_probs.topk(beam_size, dim=-1)

        candidates: list[tuple[list[int], float]] = []
        for b, (hyp, score) in enumerate(live):
            for k in range(beam_size):
                tok = int(topk_idx[b, k])
                new_score = score + float(topk_log[b, k])
                candidates.append((hyp + [tok], new_score))

        candidates.sort(key=lambda item: item[1] / max(1, len(item[0]) - 1), reverse=True)
        live = []
        for hyp, score in candidates:
            if hyp[-1] == model.eos_id:
                finished.append((hyp, score))
            elif len(live) < beam_size:
                live.append((hyp, score))
            if len(live) >= beam_size and len(finished) >= beam_size:
                break
        if len(finished) > beam_size * 4:
            finished.sort(
                key=lambda item: item[1] / max(1, len(item[0]) - 1),
                reverse=True,
            )
            finished = finished[: beam_size * 4]
        if len(finished) >= beam_size and not live:
            break

    pool = finished + live
    pool.sort(key=lambda item: item[1] / max(1, len(item[0]) - 1), reverse=True)
    out: list[list[int]] = []
    for hyp, _ in pool[:num_return]:
        out.append(_strip_sos_eos(hyp, model.sos_id, model.eos_id))
    return out
