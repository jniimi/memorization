"""Cross-model overlay of cloze accuracy vs citation count.

Loads multiple run parquets (one per model), plots binned mean + 95% CI
and the per-model logistic fit on the same axes for each maskable
field. Used to compare e.g. Gemini 3 Flash Preview vs GPT-5.4-mini on
the same paper sample.

Usage:
    uv run scripts/compare_models.py \\
        data/runs/<run1>.parquet data/runs/<run2>.parquet \\
        --out data/analysis/cross_model.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from analyze import _fit_logistic, binned_accuracy
from cloze import MASKABLE_FIELDS


# Distinct, color-blind-friendly palette
_COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd",
           "#8c564b", "#e377c2", "#17becf"]


def _label_for(df: pd.DataFrame, fallback: str) -> str:
    """Use the model column if present, else fall back to file stem."""
    if "model" in df.columns and len(df) > 0:
        m = df["model"].iloc[0]
        if isinstance(m, str) and m:
            return m
    return fallback


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("run_parquets", nargs="+", type=Path,
                   help="one parquet per model")
    p.add_argument("--out", type=Path, required=True,
                   help="output PNG path")
    p.add_argument("--bin-width", type=float, default=0.3)
    p.add_argument("--title", type=str, default=None)
    p.add_argument("--labels", type=str, default=None,
                   help="comma-separated labels, one per run parquet "
                        "(overrides the model-column auto-label)")
    args = p.parse_args()

    explicit_labels: list[str] | None = None
    if args.labels:
        explicit_labels = [s.strip() for s in args.labels.split(",")]
        if len(explicit_labels) != len(args.run_parquets):
            sys.exit(
                f"--labels has {len(explicit_labels)} entries but "
                f"{len(args.run_parquets)} parquets given"
            )

    runs: dict[str, pd.DataFrame] = {}
    for i, path in enumerate(args.run_parquets):
        df = pd.read_parquet(path)
        df = df[df["error"].isna()].copy()
        df = df[df["cited_by_count"] > 0]
        if df.empty:
            print(f"skip empty {path.name}")
            continue
        if explicit_labels is not None:
            label = explicit_labels[i]
        else:
            label = _label_for(df, path.stem)
            if label in runs:
                label = f"{label} ({path.stem})"
        runs[label] = df

    if not runs:
        sys.exit("no non-empty runs")

    # Pre-compute binned and logistic fits per (model, field)
    binned_per_model: dict[str, pd.DataFrame] = {
        m: binned_accuracy(d, bin_width=args.bin_width)
        for m, d in runs.items()
    }
    logit_per_model: dict[str, dict[str, dict | None]] = {}
    for m, d in runs.items():
        d = d.copy()
        d["log_cit"] = np.log10(d["cited_by_count"].astype(float))
        per_field: dict[str, dict | None] = {}
        for f in MASKABLE_FIELDS:
            sub = d[d["field"] == f]
            x = sub["log_cit"].to_numpy()
            y = sub["score"].astype(float).to_numpy()
            per_field[f] = _fit_logistic(x, y)
        logit_per_model[m] = per_field

    # Determine common log-cit range for the curve grid
    log_min = min(
        np.log10(d["cited_by_count"].astype(float)).min()
        for d in runs.values()
    )
    log_max = max(
        np.log10(d["cited_by_count"].astype(float)).max()
        for d in runs.values()
    )
    grid = np.linspace(log_min, log_max, 240)

    n_fields = len(MASKABLE_FIELDS)
    n_cols = 2
    n_rows = (n_fields + 1) // n_cols
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(12, 3.1 * n_rows), sharex=True,
    )
    axes = axes.flatten()

    for ax, field in zip(axes, MASKABLE_FIELDS):
        for i, (model, df) in enumerate(runs.items()):
            color = _COLORS[i % len(_COLORS)]
            # binned dots + 95% CI
            sub = binned_per_model[model]
            sub = sub[sub["field"] == field]
            if len(sub) > 0:
                x = sub["log_mid"].to_numpy()
                y = sub["accuracy"].to_numpy()
                yerr = np.stack([
                    y - sub["ci_lo"].to_numpy(),
                    sub["ci_hi"].to_numpy() - y,
                ])
                ax.errorbar(
                    x, y, yerr=yerr, fmt="o", color=color, ecolor=color,
                    alpha=0.55, capsize=2, markersize=4,
                    label=f"{model}",
                )
            # logistic curve
            fit = logit_per_model[model][field]
            if fit and not np.isnan(fit.get("b1", np.nan)):
                p_curve = 1.0 / (1.0 + np.exp(
                    -(fit["b0"] + fit["b1"] * grid)
                ))
                ax.plot(grid, p_curve, color=color, lw=1.6, alpha=0.9)
                # Inflection mark
                if abs(fit["b1"]) > 1e-9:
                    inflection_log = -fit["b0"] / fit["b1"]
                    if log_min <= inflection_log <= log_max:
                        ax.axvline(
                            inflection_log, color=color,
                            linestyle="--", linewidth=0.8, alpha=0.4,
                        )

        ax.set_title(field, fontsize=10)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(alpha=0.2)
        ax.set_ylabel("P(score=1)")
        if ax in axes[-n_cols:]:
            ax.set_xlabel("log10(citation count)")

    # Single combined legend top-right of first panel
    axes[0].legend(fontsize=8, loc="lower right")

    for ax in axes[n_fields:]:
        ax.axis("off")

    suptitle = (
        args.title or
        f"Cross-model cloze accuracy vs citation "
        f"({len(runs)} models, log10 bin={args.bin_width})"
    )
    fig.suptitle(suptitle, fontsize=12, y=1.00)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
