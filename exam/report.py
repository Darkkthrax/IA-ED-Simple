"""Build the 15-row exam table, charts, and optional equation display."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import pandas as pd
import sympy as sp

from exam import config
from exam.answers import ANSWERS
from exam.answers_demo import ANSWERS_DEMO
from exam.equations import EXERCISE_IDS, ODE_STRINGS, RHS
from exam.nn_fit import fit_and_curves
from exam.transformer_check import check_transformer, transformer_explicit_forms
from exam.verify import parse_answer, verify_explicit, x_range_for

x, y = sp.symbols("x y")
y_prime = sp.Function("y")(x).diff(x)


def _load_answers(use_demo: bool) -> dict[str, Optional[str]]:
    src = ANSWERS_DEMO if use_demo else ANSWERS
    return {eid: src.get(eid) for eid in EXERCISE_IDS}


def run(
    use_demo: bool = False,
    model: Any = None,
    tokenizer: Any = None,
    device: Any = None,
    output_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Run verification + NN fit (+ optional Transformer) for all 15 exercises."""
    out = Path(output_dir) if output_dir is not None else Path(config.OUTPUT_DIR)
    out.mkdir(parents=True, exist_ok=True)

    answers = _load_answers(use_demo)
    solve_cache: dict[str, dict[str, Any]] = {}
    curves: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []

    for eid in EXERCISE_IDS:
        ode = ODE_STRINGS[eid]
        rhs = RHS[eid]
        ans = answers.get(eid)
        x_range = x_range_for(eid)

        verifica = "PENDIENTE"
        mape: Any = ""
        nrmse: Any = ""
        respuesta_show = "PENDIENTE" if ans is None else str(ans)

        if ans is not None:
            y_expr, note = parse_answer(ans)
            if y_expr is None:
                verifica = "NO"
                respuesta_show = f"{ans} [{note}]"
            else:
                ok, vnote = verify_explicit(y_expr, rhs, x_range=x_range)
                verifica = "SÍ" if ok else "NO"
                if note:
                    respuesta_show = f"{ans} ({note})"
                if ok:
                    try:
                        curve = fit_and_curves(
                            y_expr,
                            config.C_VALUE,
                            x_range[0],
                            x_range[1],
                        )
                        curves[eid] = curve
                        mape = round(curve["mape"], 4)
                        nrmse = round(curve["nrmse"], 4)
                    except Exception as exc:  # noqa: BLE001
                        mape = ""
                        nrmse = ""
                        verifica = "SÍ"
                        respuesta_show = f"{ans} [NN error: {exc}]"

        transformer = check_transformer(
            eid,
            ode,
            rhs,
            model,
            tokenizer,
            device,
            cache=solve_cache,
        )

        rows.append(
            {
                "id": eid,
                "ODE": ode,
                "respuesta": respuesta_show,
                "SÍ/NO": verifica,
                "MAPE %": mape,
                "NRMSE %": nrmse,
                "Transformer": transformer,
            }
        )

    df = pd.DataFrame(rows, columns=[
        "id", "ODE", "respuesta", "SÍ/NO", "MAPE %", "NRMSE %", "Transformer",
    ])
    csv_path = out / "exam_results.csv"
    df.to_csv(csv_path, index=False)

    _save_bar_chart(df, out / "exam_nn_error.png")
    _save_curve_grid(curves, out / "exam_curves.png")
    return df


def _save_bar_chart(df: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    vals = []
    for v in df["MAPE %"]:
        try:
            vals.append(float(v))
        except (TypeError, ValueError):
            vals.append(0.0)
    ax.bar(df["id"], vals, color="#2a6fbb")
    ax.set_ylabel("MAPE %")
    ax.set_title("Error de la red neuronal por ejercicio")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _save_curve_grid(curves: dict[str, dict[str, Any]], path: Path) -> None:
    fig, axes = plt.subplots(5, 3, figsize=(12, 14))
    axes_flat = list(axes.ravel())
    for i, eid in enumerate(EXERCISE_IDS):
        ax = axes_flat[i]
        ax.set_title(eid)
        data = curves.get(eid)
        if data is None:
            ax.text(0.5, 0.5, "PENDIENTE", ha="center", va="center", transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
            continue
        ax.plot(data["x_plot"], data["y_exact_plot"], label="exacta", lw=2)
        ax.plot(data["x_plot"], data["y_pred_plot"], "--", label="red", lw=1.5)
        ax.legend(fontsize=7)
    fig.suptitle("Solución exacta vs red neuronal", y=1.01)
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def show_equations(
    model: Any = None,
    tokenizer: Any = None,
    device: Any = None,
) -> None:
    """Notebook display: ODE and Transformer explicit forms in LaTeX."""
    sp.init_printing()
    try:
        from IPython.display import display
    except ImportError:
        display = print  # type: ignore[assignment]

    cache: dict[str, dict[str, Any]] = {}
    for eid in EXERCISE_IDS:
        rhs = RHS[eid]
        display(eid, sp.Eq(y_prime, rhs))
        roots = transformer_explicit_forms(
            ODE_STRINGS[eid], model, tokenizer, device, cache=cache
        )
        if not roots:
            display(f"{eid}: Transformer sin forma explícita real")
        else:
            for i, root in enumerate(roots):
                display(f"{eid} raíz {i}:", sp.Eq(y, root))
        print("-------------------------------------------------------\n")
