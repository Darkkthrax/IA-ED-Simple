"""Train the ODE encoder-decoder Transformer from a YAML config."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import threading
import time
from pathlib import Path
from typing import Any, Optional

import sympy as sp
import torch
import yaml
from torch.utils.data import DataLoader

from src.data.dataset import ODEDataset, make_collate_fn
from src.data.tokenizer import Tokenizer, Vocab, build_tokenizer_from_config
from src.model.transformer import (
    build_model_from_config,
    decoded_token_ids,
    greedy_decode,
)


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


def _set_seeds(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_device(explicit: Optional[str]) -> torch.device:
    if explicit:
        return torch.device(explicit)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _truthy_resume(value: Any) -> bool:
    if value is True:
        return True
    if isinstance(value, str) and value.strip().lower() in {"1", "true", "yes", "on"}:
        return True
    return False


def _lr_at_step(step: int, base_lr: float, warmup: int, name: str, total_steps: int) -> float:
    """Return the learning rate for a 1-based update step."""
    step = max(1, step)
    sched = name.strip().lower()
    if sched in {"inverse_sqrt", "inv_sqrt"}:
        if warmup <= 0:
            return base_lr * (step**-0.5)
        return base_lr * min(step / warmup, (warmup**0.5) / (step**0.5))
    if sched == "cosine":
        if warmup > 0 and step <= warmup:
            return base_lr * step / warmup
        denom = max(1, total_steps - warmup)
        progress = min(1.0, max(0.0, (step - warmup) / denom))
        return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))
    raise ValueError(
        f"Unknown train.scheduler={name!r}; implemented: cosine, inverse_sqrt (alias inv_sqrt)"
    )


def _tokenizer_state(tokenizer: Tokenizer, cfg: dict[str, Any]) -> dict[str, Any]:
    special = list(cfg.get("tokenizer", {}).get("special_tokens", ["PAD", "SOS", "EOS", "UNK"]))
    return {
        "max_int": tokenizer.max_int,
        "special_tokens": special,
        "id_to_token": list(tokenizer.vocab.id_to_token),
    }


def _tokenizer_from_state(state: dict[str, Any]) -> Tokenizer:
    if "id_to_token" in state:
        id_to_token = list(state["id_to_token"])
        token_to_id = {t: i for i, t in enumerate(id_to_token)}
        vocab = Vocab(token_to_id=token_to_id, id_to_token=id_to_token)
        return Tokenizer(max_int=int(state["max_int"]), vocab=vocab)
    return build_tokenizer_from_config(
        {"tokenizer": {"max_int": state["max_int"], "special_tokens": state["special_tokens"]}}
    )


def _save_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    step: int,
    config: dict[str, Any],
    tokenizer: Tokenizer,
    best_val_loss: float,
    scheduler_name: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "step": step,
            "config": config,
            "tokenizer": _tokenizer_state(tokenizer, config),
            "best_val_loss": best_val_loss,
            "scheduler_name": scheduler_name,
            "sympy_version": sp.__version__,
            "torch_version": torch.__version__,
        },
        path,
    )


def _arm_hard_exit(seconds: float) -> None:
    def _kill() -> None:
        print(f"HARD TIMEOUT after {seconds:.0f}s", flush=True)
        os._exit(124)

    timer = threading.Timer(seconds, _kill)
    timer.daemon = True
    timer.start()


@torch.no_grad()
def _eval_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    pad_id: int,
    eos_id: int,
    max_decode_examples: int = 200,
    deadline: Optional[float] = None,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    correct_tokens = 0
    n_batches = 0

    for batch in loader:
        if deadline is not None and time.perf_counter() >= deadline:
            break
        src = batch.src.to(device)
        tgt_in = batch.tgt_in.to(device)
        tgt_out = batch.tgt_out.to(device)
        src_pad = batch.src_key_padding_mask.to(device)
        tgt_pad = batch.tgt_key_padding_mask.to(device)
        logits = model(src, tgt_in, src_pad, tgt_pad)
        loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))
        total_loss += float(loss.item())
        n_batches += 1
        pred = logits.argmax(dim=-1)
        mask = tgt_out.ne(pad_id)
        correct_tokens += int((pred.eq(tgt_out) & mask).sum().item())
        total_tokens += int(mask.sum().item())

    # Exact-match greedy on the first max_decode_examples (one row at a time so
    # --max-seconds can interrupt without hanging on a huge batch decode).
    n_match = 0
    n_seen = 0
    for batch in loader:
        if n_seen >= max_decode_examples:
            break
        if deadline is not None and time.perf_counter() >= deadline:
            break
        tgt_out = batch.tgt_out
        for i in range(batch.src.size(0)):
            if n_seen >= max_decode_examples:
                break
            if deadline is not None and time.perf_counter() >= deadline:
                break
            src_i = batch.src[i].to(device)
            # tgt_out lacks SOS; +1 restores SOS budget used by greedy_decode.
            ref = decoded_token_ids(tgt_out[i], eos_id, pad_id)
            max_len = min(len(ref) + 1, model.max_seq_len)
            pred_row = greedy_decode(model, src_i, max_len=max_len)
            pred_full = decoded_token_ids(pred_row, eos_id, pad_id)
            pred_body = pred_full[1:] if pred_full and pred_full[0] == model.sos_id else pred_full
            if pred_body == ref:
                n_match += 1
            n_seen += 1

    return {
        "val_loss": total_loss / max(1, n_batches),
        "token_acc": correct_tokens / max(1, total_tokens),
        "exact_match": n_match / max(1, n_seen),
        "exact_match_n": float(n_seen),
    }


def train(cfg: dict[str, Any], args: argparse.Namespace) -> None:
    seed = int(cfg.get("seed", 42))
    _set_seeds(seed)

    data_dir = Path(args.data_dir or cfg["data"]["output_dir"])
    _print_library_versions(data_dir / "meta.json")
    train_cfg = dict(cfg["train"])
    if args.output_dir:
        train_cfg["checkpoint_dir"] = args.output_dir
    ckpt_dir = Path(train_cfg["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    device = _resolve_device(args.device)
    tokenizer = build_tokenizer_from_config(cfg)
    max_seq_len = int(cfg["model"]["max_seq_len"])

    train_ds = ODEDataset(data_dir / "train.jsonl", tokenizer, max_seq_len)
    val_ds = ODEDataset(data_dir / "val.jsonl", tokenizer, max_seq_len)
    print(
        f"train_ds n={len(train_ds)} skipped={train_ds.skipped} "
        f"cache_hit={train_ds.cache_hit}",
        flush=True,
    )
    print(
        f"val_ds n={len(val_ds)} skipped={val_ds.skipped} "
        f"cache_hit={val_ds.cache_hit}",
        flush=True,
    )
    if len(train_ds) == 0:
        raise SystemExit(f"No usable train examples in {data_dir / 'train.jsonl'}")

    num_workers = int(train_cfg["num_workers"])
    if device.type == "cpu":
        num_workers = 0
    collate = make_collate_fn(tokenizer.vocab.pad_id)
    train_loader = DataLoader(
        train_ds,
        batch_size=int(train_cfg["batch_size"]),
        shuffle=True,
        collate_fn=collate,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(train_cfg["batch_size"]),
        shuffle=False,
        collate_fn=collate,
        num_workers=num_workers,
    )

    model = build_model_from_config(cfg, tokenizer)
    model.to(device)

    base_lr = float(train_cfg["lr"])
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=base_lr,
        weight_decay=float(train_cfg["weight_decay"]),
    )
    criterion = torch.nn.CrossEntropyLoss(
        ignore_index=tokenizer.vocab.pad_id,
        label_smoothing=float(train_cfg["label_smoothing"]),
    )
    scheduler_name = str(train_cfg["scheduler"])
    grad_clip = float(train_cfg["grad_clip"])
    log_every = int(train_cfg["log_every"])
    epochs = int(train_cfg["epochs"])
    warmup = int(train_cfg["warmup_steps"])
    steps_per_epoch = max(1, len(train_loader))
    total_steps = epochs * steps_per_epoch
    if args.max_steps is not None:
        total_steps = min(total_steps, int(args.max_steps))

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    start_epoch = 0
    step = 0
    best_val_loss = float("inf")
    if args.resume is not None:
        resume_flag = _truthy_resume(args.resume)
    else:
        resume_flag = _truthy_resume(train_cfg.get("resume"))
    last_path = ckpt_dir / "last.pt"
    if resume_flag and last_path.exists():
        ckpt = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = int(ckpt.get("epoch", 0))
        step = int(ckpt.get("step", 0))
        best_val_loss = float(ckpt.get("best_val_loss", best_val_loss))
        if "tokenizer" in ckpt:
            tokenizer = _tokenizer_from_state(ckpt["tokenizer"])
        print(
            f"resumed from {last_path} epoch={start_epoch} step={step} "
            f"best_val_loss={best_val_loss:.6f}",
            flush=True,
        )

    t0 = time.perf_counter()
    hard_deadline = t0 + float(args.max_seconds) if args.max_seconds is not None else None
    if args.max_seconds is not None:
        # Soft train budget is max_seconds; allow a short validation tail, then hard-kill.
        _arm_hard_exit(float(args.max_seconds) + 45.0)
    metrics_path = ckpt_dir / "metrics.jsonl"
    stop_training = False

    for epoch in range(start_epoch, epochs):
        if stop_training:
            break
        model.train()
        epoch_loss_sum = 0.0
        epoch_steps = 0
        for batch in train_loader:
            if hard_deadline is not None and time.perf_counter() >= hard_deadline:
                print("max-seconds reached; stopping", flush=True)
                stop_training = True
                break
            if args.max_steps is not None and step >= int(args.max_steps):
                print("max-steps reached; stopping", flush=True)
                stop_training = True
                break

            step += 1
            lr = _lr_at_step(step, base_lr, warmup, scheduler_name, total_steps)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            src = batch.src.to(device)
            tgt_in = batch.tgt_in.to(device)
            tgt_out = batch.tgt_out.to(device)
            src_pad = batch.src_key_padding_mask.to(device)
            tgt_pad = batch.tgt_key_padding_mask.to(device)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits = model(src, tgt_in, src_pad, tgt_pad)
                loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            epoch_loss_sum += float(loss.item())
            epoch_steps += 1

            if step % log_every == 0 or step == 1:
                print(
                    f"step {step} epoch {epoch + 1}/{epochs} "
                    f"train_loss={float(loss.item()):.6f} lr={lr:.8f}",
                    flush=True,
                )

        if step == 0:
            continue

        # Persist weights before validation so --max-seconds cannot lose a run.
        finished_epoch = epoch + 1
        _save_checkpoint(
            last_path,
            model=model,
            optimizer=optimizer,
            epoch=finished_epoch,
            step=step,
            config=cfg,
            tokenizer=tokenizer,
            best_val_loss=best_val_loss,
            scheduler_name=scheduler_name,
        )

        # End-of-epoch validation (also after early stop mid-epoch if we made progress).
        eval_deadline = None
        decode_n = 200
        if hard_deadline is not None:
            # Bounded validation tail after the train budget.
            remaining = hard_deadline + 25.0 - time.perf_counter()
            if remaining < 1.0:
                print("skipping validation: no time left after max-seconds", flush=True)
                if stop_training:
                    break
                continue
            eval_deadline = time.perf_counter() + remaining
            if remaining < 10.0:
                decode_n = 0
        val_stats = _eval_epoch(
            model,
            val_loader,
            criterion,
            device,
            pad_id=tokenizer.vocab.pad_id,
            eos_id=tokenizer.vocab.eos_id,
            max_decode_examples=decode_n,
            deadline=eval_deadline,
        )
        print(
            f"epoch {epoch + 1}/{epochs} val_loss={val_stats['val_loss']:.6f} "
            f"token_acc={val_stats['token_acc']:.4f} "
            f"exact_match={val_stats['exact_match']:.4f} "
            f"(n={int(val_stats['exact_match_n'])})",
            flush=True,
        )
        row = {
            "epoch": epoch + 1,
            "step": step,
            "train_loss": epoch_loss_sum / max(1, epoch_steps),
            "val_loss": val_stats["val_loss"],
            "token_acc": val_stats["token_acc"],
            "exact_match": val_stats["exact_match"],
            "lr": optimizer.param_groups[0]["lr"],
        }
        with metrics_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row) + "\n")

        _save_checkpoint(
            last_path,
            model=model,
            optimizer=optimizer,
            epoch=finished_epoch,
            step=step,
            config=cfg,
            tokenizer=tokenizer,
            best_val_loss=min(best_val_loss, val_stats["val_loss"]),
            scheduler_name=scheduler_name,
        )
        if val_stats["val_loss"] < best_val_loss:
            best_val_loss = val_stats["val_loss"]
            _save_checkpoint(
                ckpt_dir / "best.pt",
                model=model,
                optimizer=optimizer,
                epoch=finished_epoch,
                step=step,
                config=cfg,
                tokenizer=tokenizer,
                best_val_loss=best_val_loss,
                scheduler_name=scheduler_name,
            )
            print(f"saved best.pt val_loss={best_val_loss:.6f}", flush=True)

        if stop_training:
            break

    print(f"training done step={step} elapsed={time.perf_counter() - t0:.1f}s", flush=True)


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train ODE Transformer")
    p.add_argument("--config", type=str, default="configs/default.yaml")
    p.add_argument("--data-dir", type=str, default=None)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--max-seconds", type=float, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Override train.resume (true/false). Default: use config.",
    )
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)
    with open(args.config, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    # Optional reduced-model overrides via env-free CLI are not added; smoke
    # tests pass a temporary YAML when needed.
    train(cfg, args)


if __name__ == "__main__":
    main()
