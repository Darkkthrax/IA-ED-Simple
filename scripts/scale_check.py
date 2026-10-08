"""CHECK 2: short probe of unique-example yield vs attempts (hard time/attempt caps)."""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

# Allow `python scripts/scale_check.py` from repo root.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.generator import (  # noqa: E402
    F_TEMPLATES,
    G_TEMPLATES,
    HELDOUT_COMBOS,
    load_config,
    try_build_example,
)


def _p(msg: str = "") -> None:
    print(msg, flush=True)


def _estimate_plateau(curve: list[tuple[int, int]]) -> str:
    """
    Rough saturation estimate from (attempts, unique) samples.

    Fits unique ≈ U_max * (1 - exp(-a / tau)) by matching the last two points
    when possible; clearly labeled as an ESTIMATE only.
    """
    if len(curve) < 3:
        return "ESTIMATE: insufficient samples to fit a plateau (need >= 3 points)."

    # Use late-window growth rate: if unique gain per attempt is tiny, near plateau.
    a1, u1 = curve[-2]
    a2, u2 = curve[-1]
    da = max(a2 - a1, 1)
    du = u2 - u1
    rate = du / da

    # Extrapolate assuming exponential approach: remaining ≈ rate * tau, tau ~ a2/3 heuristic
    # Better: from two late points and an early point, solve for U_max.
    a0, u0 = curve[0]
    # Avoid degenerate cases.
    if u2 <= u0 + 1:
        return f"ESTIMATE: unique count nearly flat (~{u2}); plateau ~= {u2}."

    # If still growing linearly-ish, plateau is well above current unique.
    if rate > 0.5:
        return (
            f"ESTIMATE: still growing fast (d_unique/d_attempts ~= {rate:.3f}); "
            f"plateau >> {u2} (not visible in this short run)."
        )

    # Fit U_max from assuming u(a) = U*(1 - exp(-a/tau)).
    # Use a0,u0 and a2,u2: u/U = 1-exp(-a/tau) => (1-u0/U)/(1-u2/U) = exp(-(a0-a2)/tau)
    # Search U in [u2+1, 50*u2] for best match to mid point.
    best_U = None
    best_err = float("inf")
    mid = curve[len(curve) // 2]
    for U in range(u2 + 1, max(u2 + 2, int(u2 * 20) + 50)):
        # Skip invalid
        if u0 >= U or u2 >= U or mid[1] >= U:
            continue
        # tau from end point: u2/U = 1 - exp(-a2/tau) => tau = -a2 / log(1 - u2/U)
        frac2 = u2 / U
        if frac2 <= 0 or frac2 >= 1:
            continue
        import math

        tau = -a2 / math.log(1.0 - frac2)
        if tau <= 0:
            continue
        pred_mid = U * (1.0 - math.exp(-mid[0] / tau))
        pred0 = U * (1.0 - math.exp(-a0 / tau))
        err = (pred_mid - mid[1]) ** 2 + (pred0 - u0) ** 2
        if err < best_err:
            best_err = err
            best_U = U

    if best_U is None:
        # Fallback: project with current rate decaying.
        proj = int(u2 + du / max(rate, 1e-6) * 0.1) if rate > 0 else u2
        return (
            f"ESTIMATE: fit unstable; naive projection ~= {max(proj, u2)} "
            f"(current unique={u2}, late rate={rate:.4f})."
        )

    return (
        f"ESTIMATE (exponential fit unique ~= Uinf * (1 - exp(-a/tau))): "
        f"plateau Uinf ~= {best_U}  "
        f"(observed unique={u2} at {a2} attempts; late rate={rate:.4f} unique/attempt). "
        f"Treat as rough only."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Scale / duplicate-saturation check")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Config path (ranges are not modified)",
    )
    parser.add_argument(
        "--target",
        type=int,
        default=100_000,
        help="Unique examples to aim for (usually not reached under short limits)",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=20_000,
        help="Hard cap on try_build_example calls (default: 20000)",
    )
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=120.0,
        help="Wall-clock limit in seconds (default: 120)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    target = args.target
    max_attempts = args.max_attempts
    max_seconds = float(args.max_seconds)

    rng = random.Random(cfg.seed)
    seen: set[tuple[str, str]] = set()
    seen_ode: set[str] = set()
    combo_unique_odes: dict[tuple[str, str], set[str]] = {}
    discard_stats: dict[str, int] = {}
    curve: list[tuple[int, int]] = []

    combos = [
        (f, g)
        for f in F_TEMPLATES
        for g in G_TEMPLATES
        if (f, g) not in HELDOUT_COMBOS
    ]

    attempts = 0
    stop_reason = "target_reached"
    t0 = time.perf_counter()

    _p(
        f"Config seed={cfg.seed} coeff=[{cfg.coeff_min},{cfg.coeff_max}] "
        f"max_degree={cfg.max_degree} max_token_length={cfg.max_token_length}"
    )
    _p(
        f"Target unique={target}  max_attempts={max_attempts}  "
        f"max_seconds={max_seconds}"
    )
    _p(f"In-distribution template combos: {len(combos)}")

    while len(seen) < target:
        if attempts >= max_attempts:
            stop_reason = "max_attempts"
            break
        if (time.perf_counter() - t0) >= max_seconds:
            stop_reason = "max_seconds"
            break

        attempts += 1
        f_name, g_name = rng.choice(combos)
        ex = try_build_example(f_name, g_name, rng, cfg, stats=discard_stats)
        if ex is not None:
            key = (ex["ode"], ex["solution"])
            if key in seen:
                discard_stats["duplicate"] = discard_stats.get("duplicate", 0) + 1
                if discard_stats.get("kept", 0) > 0:
                    discard_stats["kept"] -= 1
            else:
                seen.add(key)
                seen_ode.add(ex["ode"])
                combo_unique_odes.setdefault((f_name, g_name), set()).add(ex["ode"])

        if attempts % 200 == 0:
            curve.append((attempts, len(seen)))
            _p(
                f"  progress: attempts={attempts} unique={len(seen)} "
                f"dups={discard_stats.get('duplicate', 0)} "
                f"elapsed={time.perf_counter() - t0:.1f}s"
            )

    elapsed = max(time.perf_counter() - t0, 1e-9)
    if not curve or curve[-1][0] != attempts:
        curve.append((attempts, len(seen)))

    kept = len(seen)
    dups = discard_stats.get("duplicate", 0)
    successful = kept + dups
    dup_rate = (dups / successful) if successful else 0.0
    aps = attempts / elapsed

    ranked = sorted(
        ((fg, len(odes)) for fg, odes in combo_unique_odes.items()),
        key=lambda t: t[1],
    )

    _p()
    _p("=== Scale check results ===")
    _p(f"stop_reason:        {stop_reason}")
    _p(f"attempts:           {attempts}")
    _p(f"unique kept:        {kept}")
    _p(f"unique ODE strings: {len(seen_ode)}")
    _p(f"duplicate rate:     {dup_rate:.4f}  (dups={dups}, successful_builds={successful})")
    _p(f"attempts/second:    {aps:.2f}  (elapsed={elapsed:.2f}s)")
    _p("discarded by reason:")
    for reason, count in sorted(discard_stats.items()):
        if reason == "kept":
            continue
        _p(f"  {reason}: {count}")

    _p()
    _p("unique counts per (f, g) template combo:")
    for (f_name, g_name), n in ranked:
        _p(f"  ({f_name}, {g_name}): {n}")

    _p()
    _p(_estimate_plateau(curve))
    if kept >= target:
        _p(f"SUCCESS: reached {kept} unique examples.")
    else:
        _p(
            f"Stopped before target ({target}); unique={kept}. "
            f"Lowest unique-ODE combos:"
        )
        for (f_name, g_name), n in ranked[:10]:
            _p(f"  ({f_name}, {g_name}): {n}")


if __name__ == "__main__":
    main()
