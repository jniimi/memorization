"""Paper-cluster bootstrap for logistic inflection points and pairwise contrasts.

For every (condition, field) the per-field logistic is paper-IID (each paper
contributes one trial per field at most), so the marginal inflection 95% CI
from a paper-resampling bootstrap matches the naive paper-resampling bootstrap.
For the all-fields-pooled summary the same paper-cluster resampling preserves
within-paper correlation across fields.

Outputs a text table to ``data/analysis/bootstrap_inflections.txt`` and a
keyed JSON to ``data/analysis/bootstrap_inflections.json`` for downstream
table generation.

Run::

    uv run scripts/bootstrap_ci.py --B 1000 --seed 42
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze import _fit_logistic  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MASTER = ROOT / "data/processed/20260511T112255Z_cs-ai_log10_pilot.parquet"
DEFAULT_RUNS: dict[str, str] = {
    "GEM":    "data/runs/1778510977_gemini-3-flash-preview_n200_start0.parquet",
    "GEM-T":  "data/runs/1778542240_gemini-3-flash-preview_think_n200_start0_batch.parquet",
    "mini":   "data/runs/1778506258_gpt-5.4-mini_n200_start0.parquet",
    "mini-T": "data/runs/gpt-5.4-mini_think_log10_merged_n200.parquet",
    "full":   "data/runs/1778508164_gpt-5.4_n200_start0.parquet",
    "DS":     "data/runs/1778505597_deepseek-v4-flash_n200_start0.parquet",
    "DS-T":   "data/runs/1778545800_deepseek-v4-flash_think_n200_start0.parquet",
}
FIELDS = ["first_author", "title", "co_authors", "pages"]


def _infl(x: np.ndarray, y: np.ndarray) -> float:
    f = _fit_logistic(x, y)
    if f is None:
        return np.nan
    b1 = f.get("b1", np.nan)
    if not np.isfinite(b1) or abs(b1) < 1e-9:
        return np.nan
    log_infl = -f["b0"] / b1
    # Guard against 10**(huge) overflow on flat/degenerate logistic fits
    # whose crossing is far outside the sampled citation range.
    if not np.isfinite(log_infl) or log_infl > 10 or log_infl < -3:
        return np.nan
    return float(10 ** log_infl)


def _resolve_runs(runs_json: Path | None) -> dict[str, Path]:
    """Load a runs spec from JSON or fall back to the CS-AI default.

    JSON shape: ``{"GEM": "data/runs/...parquet", ...}`` with paths
    interpreted relative to the repository root if not absolute.
    """
    spec = DEFAULT_RUNS
    if runs_json is not None:
        spec = json.loads(Path(runs_json).read_text())
    out: dict[str, Path] = {}
    for cond, raw in spec.items():
        p = Path(raw)
        out[cond] = p if p.is_absolute() else (ROOT / p)
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--B", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--master", type=Path, default=DEFAULT_MASTER,
                   help="master parquet (default: CS-AI log10 master)")
    p.add_argument("--runs-json", type=Path, default=None,
                   help="JSON file mapping condition label -> parquet "
                        "path; defaults to the seven CS-AI conditions "
                        "released with the paper")
    p.add_argument("--out-dir", type=Path,
                   default=ROOT / "data/analysis")
    p.add_argument("--out-name", type=str, default="bootstrap_inflections.json",
                   help="filename written into --out-dir")
    args = p.parse_args()

    RUNS = _resolve_runs(args.runs_json)
    print(f"loading master + {len(RUNS)} conditions")
    master = pd.read_parquet(args.master)
    papers = master["openalex_id"].unique()
    n_papers = len(papers)

    cond_data: dict[str, pd.DataFrame] = {}
    pid_index: dict[str, dict[str, np.ndarray]] = {}
    for cond, path in RUNS.items():
        d = pd.read_parquet(path)
        d = d[d["error"].isna() & (d["cited_by_count"] > 0)].copy()
        d["logc"] = np.log10(d["cited_by_count"].astype(float))
        d = d.reset_index(drop=True)
        cond_data[cond] = d
        pid_index[cond] = {
            pid: np.asarray(idx) for pid, idx in d.groupby("openalex_id").indices.items()
        }

    # Point estimates
    point: dict[tuple[str, str], float] = {}
    for cond, d in cond_data.items():
        for fld in FIELDS:
            sub = d[d["field"] == fld]
            point[(cond, fld)] = _infl(
                sub["logc"].to_numpy(), sub["score"].astype(float).to_numpy(),
            )
        point[(cond, "overall")] = _infl(
            d["logc"].to_numpy(), d["score"].astype(float).to_numpy(),
        )

    # Bootstrap
    rng = np.random.default_rng(args.seed)
    boot: dict[tuple[str, str], list[float]] = {
        (c, f): [] for c in RUNS for f in FIELDS + ["overall"]
    }
    t0 = time.time()
    for b in range(args.B):
        sampled = rng.choice(papers, size=n_papers, replace=True)
        for cond, d in cond_data.items():
            idx_arr = pid_index[cond]
            row_lists = [idx_arr[p] for p in sampled if p in idx_arr]
            all_idx = np.concatenate(row_lists) if row_lists else np.empty(0, dtype=int)
            sub_all = d.iloc[all_idx]
            for fld in FIELDS:
                s = sub_all[sub_all["field"] == fld]
                boot[(cond, fld)].append(
                    _infl(s["logc"].to_numpy(), s["score"].astype(float).to_numpy())
                )
            boot[(cond, "overall")].append(
                _infl(sub_all["logc"].to_numpy(),
                      sub_all["score"].astype(float).to_numpy())
            )
        if (b + 1) % 100 == 0:
            dt = time.time() - t0
            print(f"  iter {b+1}/{args.B}  ({dt:.0f}s, "
                  f"~{dt/(b+1)*args.B:.0f}s total)")

    # 95% CIs + a simple "% of bootstrap > 1e5 (i.e. extrapolation-dominated)"
    def _ci(vals: list[float]) -> tuple[float, float, int]:
        arr = np.asarray(vals, dtype=float)
        finite = arr[np.isfinite(arr)]
        if len(finite) < 10:
            return (np.nan, np.nan, len(finite))
        return (
            float(np.percentile(finite, 2.5)),
            float(np.percentile(finite, 97.5)),
            len(finite),
        )

    print("\n=== Marginal inflection 95% CI ===")
    print(f"{'cond':8s}{'field':16s}{'point':>10s}{'ci_lo':>10s}{'ci_hi':>10s}  n_finite")
    results: dict[str, dict] = {}
    for cond in RUNS:
        for fld in FIELDS + ["overall"]:
            lo, hi, nf = _ci(boot[(cond, fld)])
            pt = point[(cond, fld)]
            results[f"{cond}|{fld}"] = {
                "point": pt, "ci_lo": lo, "ci_hi": hi,
                "n_finite": nf, "B": args.B,
            }
            def _fmt(x: float) -> str:
                if not np.isfinite(x):
                    return "       NA"
                if abs(x) >= 1e5:
                    return f"{x:>10.2e}"
                return f"{x:>10.0f}"
            print(f"{cond:8s}{fld:16s}{_fmt(pt)}{_fmt(lo)}{_fmt(hi)}  {nf}")

    # Pairwise contrasts (paired Δ from the same bootstrap iter)
    contrasts: list[tuple[str, str, str]] = [
        # The "mini outlier" claim, first_author:
        ("mini", "GEM",   "first_author"),
        ("mini", "DS",    "first_author"),
        ("mini", "full",  "first_author"),
        # Reasoning effect per model, on first_author and overall:
        ("GEM-T",  "GEM",  "first_author"),
        ("DS-T",   "DS",   "first_author"),
        ("mini-T", "mini", "first_author"),
        ("GEM-T",  "GEM",  "overall"),
        ("DS-T",   "DS",   "overall"),
        ("mini-T", "mini", "overall"),
        # And for the headline mini-T rescue, also on title and co_authors:
        ("mini-T", "mini", "title"),
        ("mini-T", "mini", "co_authors"),
        ("mini-T", "mini", "pages"),
    ]
    contrast_out: dict[str, dict] = {}
    print("\n=== Paired contrasts, 95% CI from cluster bootstrap ===")
    print("    raw Δ = a − b citations, log10 Δ = log10(a/b) in decades\n")
    print(f"{'a':>8s} - {'b':<8s} {'field':16s}"
          f"{'Δ raw':>13s}{'  ci(raw)':>22s}"
          f"{'Δ log10':>10s}{'  ci(log10)':>18s}{'p':>9s}")
    for a, b, fld in contrasts:
        va = np.asarray(boot[(a, fld)], dtype=float)
        vb = np.asarray(boot[(b, fld)], dtype=float)
        # Raw difference
        d_raw = va - vb
        mask = np.isfinite(d_raw) & (va > 0) & (vb > 0)
        d_raw = d_raw[mask]
        # Log10 difference (use the same iters that gave both positive finite values)
        d_log = (np.log10(va[mask]) - np.log10(vb[mask]))
        if len(d_raw) < 10:
            print(f"  {a:>8s} - {b:<8s} {fld:16s}  too few finite contrasts")
            continue
        dpt_raw = point[(a, fld)] - point[(b, fld)]
        dpt_log = (np.log10(point[(a, fld)]) - np.log10(point[(b, fld)])
                   if point[(a, fld)] > 0 and point[(b, fld)] > 0 else float("nan"))
        ci_lo_raw = float(np.percentile(d_raw, 2.5))
        ci_hi_raw = float(np.percentile(d_raw, 97.5))
        ci_lo_log = float(np.percentile(d_log, 2.5))
        ci_hi_log = float(np.percentile(d_log, 97.5))
        # Two-sided p from the log scale (sign of Δ is the same on either scale)
        p = float(2 * min((d_log < 0).mean(), (d_log > 0).mean()))
        contrast_out[f"{a}-{b}|{fld}"] = {
            "delta_raw_point": dpt_raw, "ci_lo_raw": ci_lo_raw, "ci_hi_raw": ci_hi_raw,
            "delta_log10_point": dpt_log, "ci_lo_log10": ci_lo_log, "ci_hi_log10": ci_hi_log,
            "p_two_sided": p, "n_finite": int(len(d_raw)),
        }
        def _fmtr(x: float) -> str:
            if not np.isfinite(x): return "         NA"
            if abs(x) >= 1e4:     return f"{x:>13.2e}"
            return f"{x:>13.0f}"
        def _fmtl(x: float) -> str:
            if not np.isfinite(x): return "       NA"
            return f"{x:>+10.2f}"
        print(f"  {a:>8s} - {b:<8s} {fld:16s}"
              f"{_fmtr(dpt_raw)}  [{_fmtr(ci_lo_raw).strip()}, {_fmtr(ci_hi_raw).strip()}]"
              f"  {_fmtl(dpt_log)}  [{_fmtl(ci_lo_log).strip()}, {_fmtl(ci_hi_log).strip()}]"
              f"{p:>9.4f}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / args.out_name
    with out_path.open("w") as f:
        json.dump({"marginal": results, "contrasts": contrast_out,
                   "B": args.B, "seed": args.seed,
                   "master": str(args.master),
                   "runs": {c: str(p) for c, p in RUNS.items()}},
                  f, indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
