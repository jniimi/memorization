"""Analysis: chance baseline + tier × field accuracy + log-linear slope.

Loads a run parquet, computes:
- raw accuracy per (tier, field) cell
- chance baseline (mode-frequency of ground truth in that tier+field
  on the master parquet)
- above-chance accuracy = raw - chance
- log-linear slope of score on log10(citation+1) per field (within full run)

Usage:
    uv run scripts/analyze.py path/to/run.parquet
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import chi2_contingency, norm

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cloze import MASKABLE_FIELDS, _format_field_value  # noqa: F401

ROOT = Path(__file__).resolve().parent.parent

_CANON_TIER_ORDER: dict[str, int] = {"low": 0, "mid": 1, "high": 2}


def _tier_sort_key(tier: str) -> tuple[int, int]:
    """Sort canonical low/mid/high before the log-spaced sN_lo-hi
    bands, and order each scheme by its numeric position."""
    if tier in _CANON_TIER_ORDER:
        return (0, _CANON_TIER_ORDER[tier])
    m = re.search(r"_(\d+)", tier)  # 's3_101-316' -> 101
    return (1, int(m.group(1)) if m else 1 << 30)


def _infer_tier_order(df_or_values) -> tuple[str, ...]:
    """Return the tiers present in `df_or_values` in display order.

    Accepts either a DataFrame (looks at ``pilot_tier`` if present
    else ``tier``) or an iterable of tier names. Replaces the older
    hard-coded ``("low", "mid", "high")`` so the log-spaced sN_lo-hi
    scheme used in the manuscript's runs also works without code
    changes.
    """
    if isinstance(df_or_values, pd.DataFrame):
        col = "pilot_tier" if "pilot_tier" in df_or_values.columns else "tier"
        values = df_or_values[col].dropna().unique().tolist()
    else:
        values = list(df_or_values)
    return tuple(sorted(set(values), key=_tier_sort_key))


def _format_for_field(record: dict, field_name: str) -> str:
    return _format_field_value(record, field_name)


def chance_baselines(master: pd.DataFrame) -> dict[tuple[str, str], float]:
    """For each (tier, field), the modal-value frequency on the master.

    This is the accuracy a 'blind guesser' would achieve if it always
    output the most common value in that tier / field. Actual chance for
    a model that randomly samples from the prior is lower, so this is a
    conservative ceiling on what could be obtained without memorization.
    """
    out: dict[tuple[str, str], float] = {}
    for tier in _infer_tier_order(master):
        sub = master[master["pilot_tier"] == tier]
        for f in MASKABLE_FIELDS:
            vals = [
                _format_for_field(r.to_dict(), f)
                for _, r in sub.iterrows()
            ]
            vals = [v for v in vals if v]  # drop empty (variable-mask aware)
            if not vals:
                out[(tier, f)] = float("nan")
                continue
            counts = pd.Series(vals).value_counts(normalize=True)
            out[(tier, f)] = float(counts.iloc[0])
    return out


def compute_tier_field_table(
    df: pd.DataFrame, baselines: dict[tuple[str, str], float]
) -> pd.DataFrame:
    df = df[df["error"].isna()].copy()
    tiers = _infer_tier_order(df)
    rows = []
    for f in MASKABLE_FIELDS:
        for tier in tiers:
            cell = df[(df["field"] == f) & (df["tier"] == tier)]
            if len(cell) == 0:
                rows.append(dict(field=f, tier=tier, n=0,
                                 raw_acc=float("nan"),
                                 chance=baselines.get((tier, f), float("nan")),
                                 adj_acc=float("nan")))
                continue
            raw = cell["score"].mean()
            chance = baselines.get((tier, f), float("nan"))
            adj = raw - chance if not np.isnan(chance) else float("nan")
            rows.append(dict(
                field=f, tier=tier, n=len(cell),
                raw_acc=raw, chance=chance, adj_acc=adj,
            ))
    return pd.DataFrame(rows)


def loglinear_slope(df: pd.DataFrame) -> pd.DataFrame:
    """LPM slope of score on log10(citation+1) (kept for backward compat)."""
    df = df[df["error"].isna()].copy()
    df["log_cit"] = np.log10(df["cited_by_count"].astype(float) + 1.0)
    out_rows = []
    for f in MASKABLE_FIELDS:
        sub = df[df["field"] == f]
        if len(sub) < 10 or sub["score"].nunique() < 2:
            out_rows.append(dict(
                field=f, n=len(sub),
                slope=float("nan"), intercept=float("nan"),
                pearson_r=float("nan"),
            ))
            continue
        x = sub["log_cit"].to_numpy()
        y = sub["score"].astype(float).to_numpy()
        slope, intercept = np.polyfit(x, y, 1)
        if x.std() > 0 and y.std() > 0:
            r = np.corrcoef(x, y)[0, 1]
        else:
            r = float("nan")
        out_rows.append(dict(
            field=f, n=len(sub), slope=slope, intercept=intercept,
            pearson_r=r,
        ))
    return pd.DataFrame(out_rows)


# ---------------------------------------------------------------------------
# Proper logistic regression + ICAART-style threshold estimation
# ---------------------------------------------------------------------------

def _fit_logistic(x: np.ndarray, y: np.ndarray) -> dict | None:
    """Fit P(y=1) = sigmoid(b0 + b1 * x) by maximum likelihood.

    Returns coefficients with standard errors derived from the inverse
    Hessian (BFGS), z-statistics, and two-sided p-values. Returns None
    if optimization fails or the design is degenerate.
    """
    if len(x) < 10 or len(np.unique(y)) < 2:
        return None

    def nll(theta):
        b0, b1 = theta
        z = b0 + b1 * x
        # numerically-stable log(1 + exp(z))
        return -np.sum(y * z - np.logaddexp(0.0, z))

    def grad(theta):
        b0, b1 = theta
        z = b0 + b1 * x
        p = 1.0 / (1.0 + np.exp(-z))
        return np.array([
            -np.sum(y - p),
            -np.sum((y - p) * x),
        ])

    res = minimize(nll, [0.0, 0.5], jac=grad, method="BFGS")
    if not res.success:
        return None
    b0, b1 = res.x
    try:
        cov = np.asarray(res.hess_inv)
        se = np.sqrt(np.maximum(np.diag(cov), 0.0))
    except Exception:
        return None
    if np.any(np.isnan(se)) or np.any(se == 0):
        return None
    z_stats = res.x / se
    p_vals = 2.0 * (1.0 - norm.cdf(np.abs(z_stats)))
    inflection_log_cit = -b0 / b1 if abs(b1) > 1e-9 else float("nan")
    inflection_cit = (
        10.0 ** inflection_log_cit - 1.0
        if not np.isnan(inflection_log_cit) else float("nan")
    )
    return {
        "b0": float(b0),
        "b1": float(b1),
        "se_b0": float(se[0]),
        "se_b1": float(se[1]),
        "z_b0": float(z_stats[0]),
        "z_b1": float(z_stats[1]),
        "p_b0": float(p_vals[0]),
        "p_b1": float(p_vals[1]),
        "inflection_log_cit": float(inflection_log_cit),
        "inflection_cit": float(inflection_cit),
    }


def logistic_per_field(df: pd.DataFrame) -> pd.DataFrame:
    """Per-field logistic regression of score on log10(cit+1).

    Reports the slope b1 (with z, p), intercept b0, inflection point
    where P(score=1) = 0.5 (= -b0/b1 in log10 space), and the
    inflection in raw citation count (= 10^x - 1).
    """
    df = df[df["error"].isna()].copy()
    df["log_cit"] = np.log10(df["cited_by_count"].astype(float) + 1.0)
    out_rows = []
    for f in MASKABLE_FIELDS:
        sub = df[df["field"] == f]
        n = len(sub)
        x = sub["log_cit"].to_numpy()
        y = sub["score"].astype(float).to_numpy()
        fit = _fit_logistic(x, y)
        if fit is None:
            out_rows.append({"field": f, "n": n, **{
                k: float("nan") for k in
                ("b0", "b1", "se_b1", "z_b1", "p_b1",
                 "inflection_cit", "inflection_log_cit")
            }})
            continue
        out_rows.append({
            "field": f,
            "n": n,
            "b0": fit["b0"],
            "b1": fit["b1"],
            "se_b1": fit["se_b1"],
            "z_b1": fit["z_b1"],
            "p_b1": fit["p_b1"],
            "inflection_log_cit": fit["inflection_log_cit"],
            "inflection_cit": fit["inflection_cit"],
        })
    return pd.DataFrame(out_rows)


# ---------------------------------------------------------------------------
# Tier comparisons
# ---------------------------------------------------------------------------

def _proportion_z_test(k1: int, n1: int, k2: int, n2: int) -> tuple[float, float]:
    if n1 == 0 or n2 == 0:
        return float("nan"), float("nan")
    p1 = k1 / n1
    p2 = k2 / n2
    p_pool = (k1 + k2) / (n1 + n2)
    if p_pool == 0 or p_pool == 1:
        return 0.0, 1.0
    se = np.sqrt(p_pool * (1.0 - p_pool) * (1.0 / n1 + 1.0 / n2))
    if se == 0:
        return 0.0, 1.0
    z = (p1 - p2) / se
    p = 2.0 * (1.0 - norm.cdf(abs(z)))
    return float(z), float(p)


# ---------------------------------------------------------------------------
# Continuous (log-spaced bin) analysis
# ---------------------------------------------------------------------------

def binned_accuracy(
    df: pd.DataFrame,
    bin_width: float = 0.3,
    min_cit: float | None = None,
    max_cit: float | None = None,
) -> pd.DataFrame:
    """Per-field accuracy in log10(citation) bins of width `bin_width`
    (default 0.3 = ~2x per bin).

    Returns a tidy DataFrame with one row per (field, bin), including
    bin midpoint in log space, midpoint in raw citations, n, accuracy,
    and Wald 95% confidence intervals on the proportion.
    """
    df = df[df["error"].isna()].copy()
    df = df[df["cited_by_count"] > 0]
    df["log_cit"] = np.log10(df["cited_by_count"].astype(float))
    lo = float(df["log_cit"].min()) if min_cit is None else float(np.log10(min_cit))
    hi = float(df["log_cit"].max()) if max_cit is None else float(np.log10(max_cit))
    edges = np.arange(np.floor(lo * 10) / 10, hi + bin_width, bin_width)
    rows = []
    for f in MASKABLE_FIELDS:
        sub = df[df["field"] == f].copy()
        sub["bin"] = pd.cut(
            sub["log_cit"], bins=edges, include_lowest=True
        )
        grouped = sub.groupby("bin", observed=True)
        for bin_int, g in grouped:
            if len(g) == 0:
                continue
            n = len(g)
            p = float(g["score"].mean())
            # Wald 95% CI (clipped to [0, 1])
            se = np.sqrt(max(p * (1 - p), 0.0) / max(n, 1))
            ci_lo = max(0.0, p - 1.96 * se)
            ci_hi = min(1.0, p + 1.96 * se)
            rows.append({
                "field": f,
                "log_lo": float(bin_int.left),
                "log_hi": float(bin_int.right),
                "log_mid": float((bin_int.left + bin_int.right) / 2),
                "cit_mid": float(10 ** ((bin_int.left + bin_int.right) / 2)),
                "cit_lo": float(10 ** bin_int.left),
                "cit_hi": float(10 ** bin_int.right),
                "n": n,
                "accuracy": p,
                "ci_lo": ci_lo,
                "ci_hi": ci_hi,
            })
    return pd.DataFrame(rows)


def plot_continuous(
    df: pd.DataFrame,
    logit: pd.DataFrame,
    out_path: Path,
    bin_width: float = 0.3,
) -> None:
    """One subplot per field. Binned mean accuracy with 95% CI error bars
    overlaid by the per-field fitted logistic curve."""
    binned = binned_accuracy(df, bin_width=bin_width)
    n_fields = len(MASKABLE_FIELDS)
    n_cols = 2
    n_rows = (n_fields + 1) // n_cols
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(11, 3.0 * n_rows), sharex=True
    )
    axes = axes.flatten()

    df_clean = df[df["error"].isna()].copy()
    df_clean = df_clean[df_clean["cited_by_count"] > 0]
    df_clean["log_cit"] = np.log10(df_clean["cited_by_count"].astype(float))
    log_min = df_clean["log_cit"].min()
    log_max = df_clean["log_cit"].max()
    grid = np.linspace(log_min, log_max, 200)

    for ax, field in zip(axes, MASKABLE_FIELDS):
        sub = binned[binned["field"] == field]
        # Binned mean with 95% CI
        if len(sub) > 0:
            x = sub["log_mid"].to_numpy()
            y = sub["accuracy"].to_numpy()
            yerr = np.stack(
                [y - sub["ci_lo"].to_numpy(), sub["ci_hi"].to_numpy() - y]
            )
            ax.errorbar(
                x, y, yerr=yerr, fmt="o", color="C0", ecolor="C0",
                alpha=0.7, capsize=2, markersize=4, label="binned mean ±95% CI",
            )
        # Fitted logistic curve
        row = logit[logit["field"] == field]
        if len(row) and not pd.isna(row["b1"].iloc[0]):
            b0 = float(row["b0"].iloc[0])
            b1 = float(row["b1"].iloc[0])
            p_curve = 1.0 / (1.0 + np.exp(-(b0 + b1 * grid)))
            ax.plot(grid, p_curve, "-", color="C3", linewidth=1.5,
                    label="logistic fit")
            # Inflection
            if abs(b1) > 1e-9:
                inflection_log = -b0 / b1
                if log_min <= inflection_log <= log_max:
                    ax.axvline(
                        inflection_log, color="grey",
                        linestyle="--", linewidth=0.8, alpha=0.6,
                    )
                    ax.text(
                        inflection_log + 0.05, 0.05,
                        f"P=0.5 @ cit≈{10 ** inflection_log:.0f}",
                        fontsize=8, color="grey",
                    )
        ax.set_title(field, fontsize=10)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(alpha=0.2)
        ax.set_ylabel("P(score=1)")
        if ax in axes[-n_cols:]:
            ax.set_xlabel("log10(citation count)")
        if ax is axes[0]:
            ax.legend(fontsize=8, loc="lower right")

    # Hide any unused subplots
    for ax in axes[n_fields:]:
        ax.axis("off")
    fig.suptitle(
        "Per-field cloze accuracy vs citation count "
        f"(binned width={bin_width} on log10)",
        fontsize=12, y=1.00,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def tier_tests(df: pd.DataFrame) -> pd.DataFrame:
    """For each field: chi-squared over (K tiers × correct/wrong) +
    pairwise z-tests with Bonferroni correction.

    Pair set is every consecutive tier transition (low→mid, mid→high,
    …) plus the (first, last) endpoint pair when there are ≥3 tiers;
    Bonferroni alpha is 0.05 / (pairs × fields). Works for both the
    canonical low/mid/high scheme and the log-spaced sN_lo-hi scheme.
    """
    df = df[df["error"].isna()].copy()
    tiers = _infer_tier_order(df)
    # Pairwise z-tests on every consecutive tier transition plus the
    # endpoints (so 3 tiers -> 3 pairs, 6 tiers -> 6 pairs, etc.).
    pairs: list[tuple[str, str]] = []
    for i in range(len(tiers) - 1):
        pairs.append((tiers[i], tiers[i + 1]))
    if len(tiers) >= 3:
        pairs.append((tiers[0], tiers[-1]))
    n_pairs = len(pairs) * len(MASKABLE_FIELDS)
    bonf_alpha = 0.05 / n_pairs if n_pairs > 0 else float("nan")
    rows = []
    for f in MASKABLE_FIELDS:
        sub = df[df["field"] == f]
        per_tier: dict[str, tuple[int, int]] = {}
        for t in tiers:
            cell = sub[sub["tier"] == t]
            per_tier[t] = (int(cell["score"].sum()), int(len(cell)))
        # Chi-squared K x 2
        table = np.array([
            [k, n - k] for (k, n) in per_tier.values()
        ], dtype=float)
        if table.sum() == 0 or (table.sum(axis=0) == 0).any():
            chi2_stat, chi2_p = float("nan"), float("nan")
        else:
            chi2_stat, chi2_p, _, _ = chi2_contingency(table)
        pair_stats = {}
        for a, b in pairs:
            ka, na = per_tier[a]
            kb, nb = per_tier[b]
            z, p = _proportion_z_test(ka, na, kb, nb)
            pa = ka / na if na else float("nan")
            pb = kb / nb if nb else float("nan")
            pair_stats[f"{a}_vs_{b}"] = {
                "z": z, "p": p, "p_a": pa, "p_b": pb,
                "diff": pb - pa,
                "sig_bonf": (not np.isnan(p)) and p < bonf_alpha,
            }
        rows.append({
            "field": f,
            "chi2": chi2_stat,
            "chi2_p": chi2_p,
            **{f"{k}_z": v["z"] for k, v in pair_stats.items()},
            **{f"{k}_p": v["p"] for k, v in pair_stats.items()},
            **{f"{k}_diff": v["diff"] for k, v in pair_stats.items()},
            **{f"{k}_sig": v["sig_bonf"] for k, v in pair_stats.items()},
        })
    return pd.DataFrame(rows), bonf_alpha


def render_table(table: pd.DataFrame) -> str:
    """Markdown-style accuracy × tier × field with chance and adjusted."""
    tiers = _infer_tier_order(table)
    header = "| Field | " + " | ".join(f"{t:^14s}" for t in tiers) + " |"
    sep = "|---" * (len(tiers) + 1) + "|"
    lines = [header, sep]
    for f in MASKABLE_FIELDS:
        cells = [f"`{f}`"]
        for tier in tiers:
            row = table[(table["field"] == f) & (table["tier"] == tier)]
            if row.empty or pd.isna(row["raw_acc"].iloc[0]):
                cells.append("-")
                continue
            raw = row["raw_acc"].iloc[0]
            chance = row["chance"].iloc[0]
            adj = row["adj_acc"].iloc[0]
            n = int(row["n"].iloc[0])
            cells.append(
                f"{raw:.0%} (chance {chance:.0%}, adj +{adj:.0%}, n={n})"
            )
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("run_parquets", type=Path, nargs="+",
                   help="one or more run parquets; rows are concatenated "
                        "and de-duplicated on (openalex_id, field) so "
                        "extension runs (--start 500 etc.) compose "
                        "naturally with the original run")
    p.add_argument("--master", type=Path, default=None,
                   help="override master parquet (else use the one "
                        "recorded in the runs)")
    p.add_argument("--plot-dir", type=Path, default=None,
                   help="if set, write a continuous accuracy-vs-citation "
                        "plot (PNG) to this directory")
    p.add_argument("--bin-width", type=float, default=0.3,
                   help="log10 bin width for the binned-accuracy table "
                        "and plot (default 0.3 = ~2x per bin)")
    p.add_argument("--label", type=str, default=None,
                   help="label used in plot filename when multiple runs "
                        "are combined")
    args = p.parse_args()

    parts = []
    for path in args.run_parquets:
        d = pd.read_parquet(path)
        d["_source_run"] = path.name
        parts.append(d)
        print(f"run:    {path.name}  (n={len(d)})")
    df = pd.concat(parts, ignore_index=True)
    before = len(df)
    df = df.drop_duplicates(["openalex_id", "field"], keep="first")
    after = len(df)
    if before != after:
        print(f"  deduplicated on (openalex_id, field): {before} -> {after}")
    master_name = (
        df["master_parquet"].iloc[0] if args.master is None
        else args.master.name
    )
    master_path = (
        args.master if args.master is not None
        else ROOT / "data" / "processed" / master_name
    )
    master = pd.read_parquet(master_path)

    print(f"master: {master_path.name}  (n={len(master)})")
    err_n = df["error"].notna().sum()
    print(f"combined n={len(df)}  errors excluded={err_n}")
    print()

    baselines = chance_baselines(master)
    table = compute_tier_field_table(df, baselines)
    print("=== tier × field (raw / chance / adjusted) ===")
    print(render_table(table))
    print()

    print("=== per-field log-linear slope (LPM, score ~ log10(cit+1)) ===")
    slopes = loglinear_slope(df)
    for _, r in slopes.iterrows():
        if pd.isna(r["slope"]):
            print(f"  {r['field']:14s} (degenerate)")
            continue
        print(
            f"  {r['field']:14s} slope={r['slope']:+.3f}  "
            f"intercept={r['intercept']:+.3f}  r={r['pearson_r']:+.3f}  "
            f"n={int(r['n'])}"
        )
    print()

    print("=== per-field LOGISTIC regression: score ~ log10(cit+1) ===")
    logit = logistic_per_field(df)
    print(
        f"{'field':14s} {'b0':>8s} {'b1':>8s} {'SE(b1)':>8s} "
        f"{'z(b1)':>8s} {'p(b1)':>10s} {'P=0.5 cit':>12s}  n"
    )
    for _, r in logit.iterrows():
        if pd.isna(r.get("b1")):
            print(f"{r['field']:14s} (degenerate)  n={int(r['n'])}")
            continue
        infl = r["inflection_cit"]
        if not np.isnan(infl) and 0 < infl < 1e9:
            infl_str = f"{infl:>12,.0f}"
        else:
            infl_str = f"{'-':>12s}"
        print(
            f"{r['field']:14s} "
            f"{r['b0']:+8.3f} {r['b1']:+8.3f} {r['se_b1']:8.3f} "
            f"{r['z_b1']:+8.2f} {r['p_b1']:>10.2e} {infl_str}  "
            f"{int(r['n'])}"
        )
    print()

    tier_df, bonf_alpha = tier_tests(df)
    print(f"=== tier comparisons (chi-sq + pairwise z; Bonferroni α={bonf_alpha:.4f}) ===")
    print(
        f"{'field':14s} "
        f"{'chi²':>8s} {'chi²p':>10s} | "
        f"{'low→mid':>10s} {'mid→high':>10s} {'low→high':>10s}"
    )
    pairs = ("low_vs_mid", "mid_vs_high", "low_vs_high")
    for _, r in tier_df.iterrows():
        cells = [f"{r['field']:14s}"]
        cells.append(f"{r['chi2']:>8.1f}")
        cells.append(f"{r['chi2_p']:>10.2e}")
        cells.append("|")
        for pk in pairs:
            diff = r[f"{pk}_diff"]
            sig = r[f"{pk}_sig"]
            star = "*" if sig else " "
            cells.append(f"{diff:>+8.0%}{star:>2s}")
        print(" ".join(cells))
    print(f"\n* indicates p < α (Bonferroni) for the proportion z-test")
    print()

    print(
        f"=== binned accuracy (log10 width={args.bin_width}) ==="
    )
    binned = binned_accuracy(df, bin_width=args.bin_width)
    # pivot field x bin_mid for compact display
    pivot_acc = binned.pivot_table(
        index="field", columns=binned["log_mid"].round(2),
        values="accuracy", aggfunc="mean",
    ).reindex(MASKABLE_FIELDS)
    pivot_n = binned.pivot_table(
        index="field", columns=binned["log_mid"].round(2),
        values="n", aggfunc="sum",
    ).reindex(MASKABLE_FIELDS)
    cols = sorted(pivot_acc.columns)
    header = "                "
    for c in cols:
        header += f"  10^{c:.2f}"
    print(header)
    header_n = "  cit ≈         "
    for c in cols:
        header_n += f"  {10 ** c:>6.0f}"
    print(header_n)
    for f in MASKABLE_FIELDS:
        line = f"  {f:14s}"
        for c in cols:
            v = pivot_acc.at[f, c] if c in pivot_acc.columns else None
            n = pivot_n.at[f, c] if c in pivot_n.columns else 0
            if v is None or pd.isna(v) or pd.isna(n):
                line += "        -"
            else:
                line += f"  {v:5.0%}({int(n):>3d})"
        print(line)

    if args.plot_dir is not None:
        args.plot_dir.mkdir(parents=True, exist_ok=True)
        if args.label:
            stem = args.label
        elif len(args.run_parquets) == 1:
            stem = args.run_parquets[0].stem
        else:
            stem = f"combined_{len(args.run_parquets)}runs"
        out_path = args.plot_dir / f"{stem}_continuous.png"
        plot_continuous(df, logit, out_path, bin_width=args.bin_width)
        print(f"\nwrote plot: {out_path}")


if __name__ == "__main__":
    main()
