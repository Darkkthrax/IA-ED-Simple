"""CPU overfit smoke test on 64 generated examples. Not the project trainer.

Hard wall clock: 120 seconds for the whole process (generation + train + greedy
eval). Training stops early so greedy_decode still runs inside that budget.
"""

from __future__ import annotations

import json
import os
import random
import sys
import tempfile
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
import yaml  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from src.data.dataset import ODEDataset, make_collate_fn  # noqa: E402
from src.data.generator import _combo_pool, cfg_from_mapping, try_build_example  # noqa: E402
from src.data.tokenizer import build_tokenizer_from_config  # noqa: E402
from src.model.transformer import (  # noqa: E402
    build_model_from_config,
    decoded_token_ids,
    greedy_decode,
)

WALL_S = 120.0
GEN_CAP_S = 40.0
EVAL_RESERVE_S = 25.0


def _arm_deadline(seconds: float) -> None:
    def _kill() -> None:
        print(f"HARD TIMEOUT after {seconds:.0f}s", flush=True)
        os._exit(124)

    timer = threading.Timer(seconds, _kill)
    timer.daemon = True
    timer.start()


def _five_points(losses: list[float]) -> list[tuple[int, float]]:
    n = len(losses)
    if n == 0:
        return []
    if n == 1:
        return [(0, losses[0])]
    idxs = []
    for i in range(5):
        idxs.append(round(i * (n - 1) / 4))
    # De-duplicate if fewer than 5 distinct steps, preserving order.
    seen: set[int] = set()
    points = []
    for idx in idxs:
        if idx in seen:
            continue
        seen.add(idx)
        points.append((idx, losses[idx]))
    return points


def main() -> None:
    _arm_deadline(WALL_S)
    t0 = time.perf_counter()
    hard_end = t0 + WALL_S
    train_end = hard_end - EVAL_RESERVE_S

    raw = yaml.safe_load((ROOT / "configs" / "default.yaml").read_text(encoding="utf-8"))
    cfg = cfg_from_mapping(raw)
    tok = build_tokenizer_from_config(raw)
    rng = random.Random(42)
    combos = _combo_pool(heldout=False)
    examples: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    attempts = 0
    gen_deadline = min(t0 + GEN_CAP_S, train_end)
    while len(examples) < 64 and time.perf_counter() < gen_deadline and attempts < 64 * 40:
        attempts += 1
        f_name, g_name = rng.choice(combos)
        ex = try_build_example(f_name, g_name, rng, cfg)
        if ex is None:
            continue
        key = (ex["ode"], ex["solution"])
        if key in seen:
            continue
        seen.add(key)
        examples.append(ex)

    gen_elapsed = time.perf_counter() - t0
    print(
        f"generated {len(examples)} examples in {attempts} attempts "
        f"({gen_elapsed:.1f}s) seed=42 heldout=False",
        flush=True,
    )
    if len(examples) < 64:
        print(
            f"Stopped: only {len(examples)}/64 examples before the generation cap. "
            "Not training.",
            flush=True,
        )
        return

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "train.jsonl"
        with path.open("w", encoding="utf-8") as fh:
            for ex in examples:
                fh.write(
                    json.dumps(
                        {"ode": ex["ode"], "solution": ex["solution"], "family": ex["family"]},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        dataset = ODEDataset(path, tok, max_seq_len=int(raw["model"]["max_seq_len"]))
    print(
        f"dataset kept={len(dataset)} skipped={dataset.skipped} "
        f"max_src={max(len(s) for s, _ in dataset.samples)} "
        f"max_tgt={max(len(t) for _, t in dataset.samples)}",
        flush=True,
    )
    if len(dataset) == 0:
        print("Stopped: every example exceeded max_seq_len.", flush=True)
        return

    torch.manual_seed(42)
    model = build_model_from_config(
        raw,
        tok,
        d_model=128,
        num_encoder_layers=2,
        num_decoder_layers=2,
        dropout=0.0,
    )
    model.to("cpu")
    model.train()
    lr = float(raw["train"]["lr"])
    weight_decay = float(raw["train"]["weight_decay"])
    grad_clip = float(raw["train"]["grad_clip"])
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=tok.vocab.pad_id)
    loader_gen = torch.Generator()
    loader_gen.manual_seed(42)
    loader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=True,
        collate_fn=make_collate_fn(tok.vocab.pad_id),
        num_workers=0,
        generator=loader_gen,
    )
    print(
        f"train setup: d_model=128 layers=2/2 nhead={raw['model']['nhead']} "
        f"ff={raw['model']['dim_feedforward']} dropout=0 "
        f"AdamW lr={lr} weight_decay={weight_decay} grad_clip={grad_clip} "
        f"batch=16 cpu",
        flush=True,
    )

    losses: list[float] = []
    train_t0 = time.perf_counter()
    while time.perf_counter() < train_end:
        for batch in loader:
            if time.perf_counter() >= train_end:
                break
            opt.zero_grad(set_to_none=True)
            logits = model(
                batch.src,
                batch.tgt_in,
                batch.src_key_padding_mask,
                batch.tgt_key_padding_mask,
            )
            loss = criterion(
                logits.reshape(-1, logits.size(-1)),
                batch.tgt_out.reshape(-1),
            )
            if not torch.isfinite(loss):
                print(f"non-finite loss at step {len(losses)}: {loss.item()}", flush=True)
                losses.append(float("nan"))
                break
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            losses.append(float(loss.item()))
        else:
            continue
        break

    train_elapsed = time.perf_counter() - train_t0
    points = _five_points(losses)
    print(f"train_steps: {len(losses)}  train_seconds: {train_elapsed:.1f}", flush=True)
    print("loss_at_5_points:", flush=True)
    for idx, value in points:
        print(f"  step {idx}: {value:.6f}", flush=True)

    if len(losses) >= 2 and not (losses[-1] < losses[0] - 0.05):
        print(
            "Loss did not clearly decrease between the first and last step. "
            "Hyperparameters were not changed.",
            flush=True,
        )

    model.eval()
    max_tgt = max(len(tgt) for _, tgt in dataset.samples)
    srcs = [src for src, _ in dataset.samples]
    max_src = max(len(src) for src in srcs)
    src_batch = torch.full((len(srcs), max_src), tok.vocab.pad_id, dtype=torch.long)
    for i, src in enumerate(srcs):
        src_batch[i, : len(src)] = torch.tensor(src, dtype=torch.long)
    preds = greedy_decode(model, src_batch, max_tgt)
    n_match = 0
    shown = 0
    for i, (_src, tgt) in enumerate(dataset.samples):
        pred_ids = decoded_token_ids(preds[i], tok.vocab.eos_id, tok.vocab.pad_id)
        if pred_ids == list(tgt):
            n_match += 1
        elif shown < 2:
            shown += 1
            print(
                f"mismatch[{i}] pred={tok.decode_ids(pred_ids)} tgt={tok.decode_ids(tgt)}",
                flush=True,
            )
    rate = n_match / len(dataset)
    print(
        f"exact_token_match: {n_match}/{len(dataset)} = {rate:.4f}",
        flush=True,
    )
    print(f"wall_seconds: {time.perf_counter() - t0:.1f}", flush=True)


if __name__ == "__main__":
    main()
