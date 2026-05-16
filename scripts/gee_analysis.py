"""Cluster-robust GEE for the cross-condition and reasoning analyses.

For each information-rich field we fit a GEE logistic regression with
exchangeable working correlation, clustered by ``openalex_id``:

    score ~ logc * C(condition)

where the conditions are pooled across all seven runs. The omnibus
Wald test on the ``logc:C(condition)`` interaction asks whether the
citation--recovery slope differs across conditions; the omnibus on
``C(condition)`` asks whether the intercepts differ.

We then run three reasoning-specific GEEs, each pooling one model's
non-reasoning and reasoning runs:

    score ~ logc * reasoning

reporting the main effect of reasoning and its interaction with logc
(slope shift). Cluster is again the paper id.

GEE with an exchangeable working correlation is the natural
Python-only stand-in for a full mixed-effects GLMM; the score equations
are clustered, so the resulting Wald tests are robust to within-paper
correlation across fields and conditions.

Output: a text table to stdout and a JSON to
``data/analysis/gee_results.json``.

Run::

    uv run scripts/gee_analysis.py
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RUNS: dict[str, str] = {
    "GEM":    "data/runs/1778510977_gemini-3-flash-preview_n200_start0.parquet",
    "GEM-T":  "data/runs/1778542240_gemini-3-flash-preview_think_n200_start0_batch.parquet",
    "mini":   "data/runs/1778506258_gpt-5.4-mini_n200_start0.parquet",
    "mini-T": "data/runs/gpt-5.4-mini_think_log10_merged_n200.parquet",
    "full":   "data/runs/1778508164_gpt-5.4_n200_start0.parquet",
    "DS":     "data/runs/1778505597_deepseek-v4-flash_n200_start0.parquet",
    "DS-T":   "data/runs/1778545800_deepseek-v4-flash_think_n200_start0.parquet",
}
INFO_FIELDS = ["first_author", "title", "co_authors", "pages"]


def _resolve_runs(runs_json: Path | None) -> dict[str, Path]:
    spec = DEFAULT_RUNS
    if runs_json is not None:
        spec = json.loads(Path(runs_json).read_text())
    out: dict[str, Path] = {}
    for cond, raw in spec.items():
        p = Path(raw)
        out[cond] = p if p.is_absolute() else (ROOT / p)
    return out


def _load(runs: dict[str, Path]) -> pd.DataFrame:
    frames = []
    for cond, path in runs.items():
        d = pd.read_parquet(path)
        d = d[d["error"].isna() & (d["cited_by_count"] > 0)].copy()
        d["logc"] = np.log10(d["cited_by_count"].astype(float))
        d["condition"] = cond
        frames.append(d[["openalex_id", "field", "logc", "score", "condition"]])
    df = pd.concat(frames, ignore_index=True)
    df["score"] = df["score"].astype(float)
    return df


def _wald_terms(res, terms: list[str]) -> dict[str, float]:
    """Return joint Wald statistic + p for a list of parameter-name patterns."""
    names = list(res.params.index)
    idx = [i for i, n in enumerate(names) if any(t in n for t in terms)]
    if not idx:
        return {"k": 0, "chi2": np.nan, "p": np.nan}
    R = np.zeros((len(idx), len(names)))
    for r, i in enumerate(idx):
        R[r, i] = 1.0
    wt = res.wald_test(R, use_f=False, scalar=True)
    return {
        "k": int(len(idx)),
        "chi2": float(np.asarray(wt.statistic).reshape(-1)[0]),
        "p": float(np.asarray(wt.pvalue).reshape(-1)[0]),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--runs-json", type=Path, default=None,
                   help="JSON file mapping condition label -> parquet "
                        "path; defaults to the CS-AI conditions")
    p.add_argument("--out-dir", type=Path,
                   default=ROOT / "data/analysis")
    p.add_argument("--out-name", type=str, default="gee_results.json",
                   help="filename written into --out-dir")
    args = p.parse_args()

    runs = _resolve_runs(args.runs_json)
    print(f"loading {len(runs)} condition parquets ...")
    df = _load(runs)
    print(f"  total rows: {len(df):,}")
    out: dict[str, dict] = {"per_field_omnibus": {}, "reasoning_per_model": {}}

    fam = sm.families.Binomial()
    cov = sm.cov_struct.Exchangeable()

    print("\n=== Per-field GEE: score ~ logc * C(condition), clustered by paper ===")
    print(f"{'field':16s}  {'main(condition)':>22s}  {'interaction(logc:cond)':>26s}")
    for fld in INFO_FIELDS:
        sub = df[df["field"] == fld].copy()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = smf.gee(
                "score ~ logc * C(condition)",
                groups="openalex_id",
                data=sub,
                family=fam,
                cov_struct=cov,
            )
            res = model.fit()
        # Omnibus joint Wald for the condition main effect and the interaction
        main = _wald_terms(res, ["C(condition)"])  # any term mentioning condition
        # split into intercept-only vs slope-interaction
        names = list(res.params.index)
        cond_intercept_idx = [
            i for i, n in enumerate(names)
            if "C(condition)" in n and ":" not in n
        ]
        cond_slope_idx = [
            i for i, n in enumerate(names) if "logc:C(condition)" in n
        ]
        def _joint(idx):
            if not idx:
                return {"k": 0, "chi2": np.nan, "p": np.nan}
            R = np.zeros((len(idx), len(names)))
            for r, i in enumerate(idx):
                R[r, i] = 1.0
            wt = res.wald_test(R, use_f=False, scalar=True)
            return {
                "k": int(len(idx)),
                "chi2": float(np.asarray(wt.statistic).reshape(-1)[0]),
                "p": float(np.asarray(wt.pvalue).reshape(-1)[0]),
            }
        m = _joint(cond_intercept_idx)
        i = _joint(cond_slope_idx)
        print(f"{fld:16s}  χ²({m['k']:>2d})={m['chi2']:>9.1f}, p={m['p']:.2e}  χ²({i['k']:>2d})={i['chi2']:>9.1f}, p={i['p']:.2e}")
        out["per_field_omnibus"][fld] = {
            "main_condition": m, "interaction_logc_condition": i,
            "n_obs": int(len(sub)), "n_groups": int(sub["openalex_id"].nunique()),
        }

    print("\n=== Per-model reasoning GEE: score ~ logc * thinking ===")
    print(f"{'field':16s}{'model':>10s}{'main(thinking)':>22s}{'interaction':>22s}")
    pairs = [("GEM", "GEM-T", "Gemini"),
             ("mini", "mini-T", "GPT-5.4-mini"),
             ("DS",   "DS-T",  "DeepSeek")]
    for off, on, label in pairs:
        for fld in INFO_FIELDS + ["__all__"]:
            sub = df[df["condition"].isin([off, on])].copy()
            if fld != "__all__":
                sub = sub[sub["field"] == fld]
            sub["thinking"] = (sub["condition"] == on).astype(int)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = smf.gee(
                    "score ~ logc * thinking",
                    groups="openalex_id",
                    data=sub,
                    family=fam,
                    cov_struct=cov,
                )
                res = model.fit()
            params = list(res.params.index)
            def _p(name):
                if name not in params:
                    return None
                return {
                    "beta": float(res.params[name]),
                    "se": float(res.bse[name]),
                    "z": float(res.tvalues[name]),
                    "p": float(res.pvalues[name]),
                }
            mp = _p("thinking")
            ip = _p("logc:thinking")
            def _fmt(d):
                if d is None: return "    n/a              "
                return f"β={d['beta']:+.3f}±{d['se']:.3f} p={d['p']:.1e}"
            shown = fld if fld != "__all__" else "(all fields)"
            print(f"  {shown:14s}{label:>10s}  {_fmt(mp):>22s}  {_fmt(ip):>22s}")
            out["reasoning_per_model"][f"{label}|{fld}"] = {
                "main_thinking": mp, "interaction_logc_thinking": ip,
                "n_obs": int(len(sub)), "n_groups": int(sub["openalex_id"].nunique()),
            }
        print()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / args.out_name
    out["runs"] = {c: str(p) for c, p in runs.items()}
    with out_path.open("w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
