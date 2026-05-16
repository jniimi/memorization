"""Combine the per-condition cloze run parquets into a single release table.

The study uses seven conditions on each of two domains (CS-AI and Oncology).
Each run parquet shares the same schema and already self-identifies its
condition via the `model`, `reasoning`, `seed`, and `start` columns, so the
fourteen files concatenate losslessly into one table. For GPT-5.4-mini with
reasoning we use the merged run (the two start-0/start-100 half-runs joined);
the half-runs themselves are not included here to avoid double-counting.

This script adds a `domain` column and rewrites `master_parquet` to a bare
filename (the source parquets store an absolute machine-local path), then
writes a single `runs_combined.parquet`.

Usage:
  uv run scripts/combine_runs.py --runs-dir data/runs --out zenodo/data/runs_combined.parquet
"""

import argparse
from pathlib import Path

import pandas as pd

# The canonical run files: 7 conditions x 2 domains. Keyed for clarity only.
CANONICAL_RUNS = [
    # CS-AI
    "1778510977_gemini-3-flash-preview_n200_start0.parquet",
    "1778542240_gemini-3-flash-preview_think_n200_start0_batch.parquet",
    "1778506258_gpt-5.4-mini_n200_start0.parquet",
    "gpt-5.4-mini_think_log10_merged_n200.parquet",
    "1778508164_gpt-5.4_n200_start0.parquet",
    "1778505597_deepseek-v4-flash_n200_start0.parquet",
    "1778545800_deepseek-v4-flash_think_n200_start0.parquet",
    # Oncology
    "1778677731_gemini-3-flash-preview_n200_start0.parquet",
    "1778724317_gemini-3-flash-preview_think_n200_start0_batch.parquet",
    "1778674243_gpt-5.4-mini_n200_start0.parquet",
    "gpt-5.4-mini_think_log10_med_merged_n200.parquet",
    "1778677308_gpt-5.4_n200_start0.parquet",
    "1778674443_deepseek-v4-flash_n200_start0.parquet",
    "1778713201_deepseek-v4-flash_think_n200_start0.parquet",
]


def domain_of(master_parquet: str) -> str:
    name = str(master_parquet)
    if "med-onc" in name:
        return "oncology"
    if "cs-ai" in name:
        return "cs-ai"
    raise ValueError(f"cannot infer domain from master_parquet={name!r}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs-dir", type=Path, default=Path("data/runs"))
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    frames = []
    for fname in CANONICAL_RUNS:
        df = pd.read_parquet(args.runs_dir / fname)
        df["master_parquet"] = df["master_parquet"].map(lambda p: Path(p).name)
        df["domain"] = df["master_parquet"].map(domain_of)
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(args.out, index=False)

    print(f"wrote {args.out}: {len(combined):,} rows, {len(frames)} runs")
    summary = (combined.groupby(["domain", "model", "reasoning"])
               .size().rename("rows").reset_index())
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
