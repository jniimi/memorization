"""Forest-plot of inflection points x condition x domain (CS-AI vs Oncology).

Reads the two bootstrap JSONs and draws a per-field 4-panel figure where
each row is one of the seven conditions and each row has two markers:
CS-AI (blue circle) and Oncology (red square), each with its 95%
paper-cluster bootstrap CI. The x-axis is log10(citation count). Use
this as the cross-domain replication figure in the main paper.

Run::

    uv run scripts/cross_domain_figure.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent

CONDS = ["GEM", "GEM-T", "mini", "mini-T", "full", "DS", "DS-T"]
LABELS = [
    "Gemini 3 Flash",
    "Gemini 3 Flash +reasoning",
    "GPT-5.4-mini",
    "GPT-5.4-mini +reasoning",
    "GPT-5.4 (full)",
    "DeepSeek V4-Flash",
    "DeepSeek V4-Flash +reasoning",
]
FIELDS = ["first_author", "title", "co_authors", "pages"]
DOMAINS = [
    ("CS-AI", "data/analysis/bootstrap_inflections.json", "#1f77b4", "o"),
    ("Oncology", "data/analysis/medicine_bootstrap.json", "#d62728", "s"),
]


def _load(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)["marginal"]


def main() -> None:
    payloads = [(name, _load(ROOT / p), c, m) for name, p, c, m in DOMAINS]

    fig, axes = plt.subplots(1, 4, figsize=(14, 4.6), sharey=True)
    y = np.arange(len(CONDS))[::-1]  # plot top-to-bottom in the listed order
    offsets = [+0.18, -0.18]

    for ax, field in zip(axes, FIELDS):
        for (name, data, color, marker), off in zip(payloads, offsets):
            xs, ys, errs_lo, errs_hi = [], [], [], []
            arrow_x, arrow_y = [], []
            for i, c in enumerate(CONDS):
                d = data[f"{c}|{field}"]
                pt, lo, hi = d["point"], d["ci_lo"], d["ci_hi"]
                if not np.isfinite(pt) or pt <= 0:
                    continue
                cap_hi = min(hi, 1e6) if np.isfinite(hi) else 1e6
                cap_lo = max(lo, 1.0) if np.isfinite(lo) else 1.0
                xs.append(pt)
                ys.append(y[i] + off)
                errs_lo.append(pt - cap_lo)
                errs_hi.append(cap_hi - pt)
                if np.isfinite(hi) and hi > 1e5:
                    arrow_x.append(cap_hi)
                    arrow_y.append(y[i] + off)
            if xs:
                ax.errorbar(
                    xs, ys,
                    xerr=[errs_lo, errs_hi],
                    fmt=marker, color=color, ecolor=color,
                    capsize=2, markersize=5, alpha=0.85,
                    label=name if ax is axes[0] else None,
                )
                # mark the extrapolation-capped CIs with a faint ">" arrow
                if arrow_x:
                    ax.scatter(arrow_x, arrow_y, marker=">",
                               color=color, alpha=0.7, s=24)

        ax.set_xscale("log")
        ax.set_xlim(10, 2e5)
        ax.set_xlabel("inflection (citations)")
        ax.set_title(field, fontsize=11)
        ax.grid(axis="x", alpha=0.3, which="both")
        ax.set_yticks(y)
        ax.tick_params(axis="y", length=0)

    # Set y-tick labels only on the leftmost panel; force visibility because
    # sharey=True otherwise suppresses them via the shared-axis label hiding.
    axes[0].set_yticklabels(LABELS)
    axes[0].tick_params(axis="y", labelleft=True)
    axes[0].set_ylim(-0.6, len(CONDS) - 0.4)

    axes[0].legend(loc="lower right", fontsize=8, frameon=True)
    fig.suptitle(
        "Inflection points (citation count at $P=0.5$), CS-AI vs. Oncology "
        "with 95% paper-cluster bootstrap CIs",
        fontsize=11, y=1.00,
    )
    fig.tight_layout()
    out = ROOT / "scirep" / "cross_domain_inflections.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
