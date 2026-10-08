"""PyTorch dataset for JSONL ODE splits produced by the generator."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import Dataset

from src.data.tokenizer import Tokenizer


@dataclass
class Seq2SeqBatch:
    """Batch-first tensors. Padding masks use True for positions to ignore."""

    src: torch.Tensor
    tgt_in: torch.Tensor
    tgt_out: torch.Tensor
    src_key_padding_mask: torch.Tensor
    tgt_key_padding_mask: torch.Tensor


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _cache_key(path: Path, max_int: int, max_seq_len: int) -> str:
    return f"{_file_sha256(path)}:max_int={max_int}:max_seq_len={max_seq_len}"


def _cache_path(path: Path, key: str) -> Path:
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
    return path.parent / f".{path.name}.tokcache.{digest}.pt"


class ODEDataset(Dataset):
    """Read a split JSONL and encode ``ode`` / ``solution`` with ``tokenizer``.

    Records whose source or target id length (including SOS/EOS) exceeds
    ``max_seq_len`` are skipped and counted on ``skipped``.

    Extra JSON fields (e.g. ``f_template``, ``g_template``) are ignored.
    Encoded id pairs are cached on disk under a key derived from the file
    content hash, ``tokenizer.max_int``, and ``max_seq_len``.
    """

    def __init__(
        self,
        path: str | Path,
        tokenizer: Tokenizer,
        max_seq_len: int,
        *,
        use_cache: bool = True,
    ) -> None:
        self.path = Path(path)
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.samples: list[tuple[list[int], list[int]]] = []
        self.skipped = 0
        self.cache_hit = False
        self.cache_path: Path | None = None

        key = _cache_key(self.path, tokenizer.max_int, max_seq_len)
        cache_file = _cache_path(self.path, key) if use_cache else None
        self.cache_path = cache_file

        if cache_file is not None and cache_file.exists():
            try:
                payload = torch.load(cache_file, map_location="cpu", weights_only=False)
                if payload.get("key") == key:
                    self.samples = [(list(s), list(t)) for s, t in payload["samples"]]
                    self.skipped = int(payload.get("skipped", 0))
                    self.cache_hit = True
                    return
            except Exception:
                pass

        with self.path.open("r", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                row = json.loads(line)
                src = tokenizer.encode(row["ode"], add_special=True)
                tgt = tokenizer.encode(row["solution"], add_special=True)
                if len(src) > max_seq_len or len(tgt) > max_seq_len:
                    self.skipped += 1
                    continue
                self.samples.append((src, tgt))

        if cache_file is not None:
            try:
                cache_file.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {"key": key, "samples": self.samples, "skipped": self.skipped},
                    cache_file,
                )
            except Exception:
                pass

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[list[int], list[int]]:
        return self.samples[idx]


def make_collate_fn(pad_id: int):
    """Pad with ``pad_id`` to the batch max length (batch dim first)."""

    def collate_fn(batch: list[tuple[list[int], list[int]]]) -> Seq2SeqBatch:
        srcs, tgts = zip(*batch)
        src = _pad(srcs, pad_id)
        tgt = _pad(tgts, pad_id)
        tgt_in = tgt[:, :-1]
        tgt_out = tgt[:, 1:]
        return Seq2SeqBatch(
            src=src,
            tgt_in=tgt_in,
            tgt_out=tgt_out,
            src_key_padding_mask=src.eq(pad_id),
            tgt_key_padding_mask=tgt_in.eq(pad_id),
        )

    return collate_fn


def _pad(seqs: tuple[list[int], ...] | list[list[int]], pad_id: int) -> torch.Tensor:
    max_len = max(len(seq) for seq in seqs)
    out = torch.full((len(seqs), max_len), pad_id, dtype=torch.long)
    for i, seq in enumerate(seqs):
        if seq:
            out[i, : len(seq)] = torch.tensor(seq, dtype=torch.long)
    return out
