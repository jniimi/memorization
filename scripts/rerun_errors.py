"""Rerun rows whose `error` is non-null and write to a `_rerun.parquet`.

Loads a run parquet, finds rows with non-null `error`, regenerates the
cloze prompt from the original master, calls the LLM with the same
settings (model/reasoning recovered from the row), and updates the row
in place. Writes a sibling parquet with `_rerun` suffix so the original
stays immutable for provenance.

Usage:
    uv run scripts/rerun_errors.py path/to/run.parquet
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import httpx
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cloze import create_prompt
from llm import get_client, parse_answer
from scoring import score_response

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("run_parquet", type=Path,
                   help="path to the run parquet to rerun errored rows from")
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--timeout", type=float, default=180.0)
    p.add_argument("--sleep", type=float, default=0.0)
    args = p.parse_args()

    df = pd.read_parquet(args.run_parquet)
    err_idx = df[df["error"].notna()].index
    print(f"loaded {len(df)} rows, {len(err_idx)} errored")
    if len(err_idx) == 0:
        sys.exit("nothing to rerun")

    masters = df.loc[err_idx, "master_parquet"].unique()
    if len(masters) != 1:
        sys.exit(f"expected single master, got {list(masters)}")
    master_path = ROOT / "data" / "processed" / masters[0]
    master = pd.read_parquet(master_path)
    master_by_id = {r["openalex_id"]: r for _, r in master.iterrows()}

    model = df.loc[err_idx, "model"].iloc[0]
    reasoning = bool(df.loc[err_idx, "reasoning"].iloc[0])
    print(f"model={model} reasoning={reasoning} master={masters[0]}")

    new_score: list[int] = []
    new_pred: list[str | None] = []
    new_raw: list[str] = []
    new_thinking: list[str | None] = []
    new_err: list[str | None] = []
    new_elapsed: list[float] = []

    with get_client(model, timeout=args.timeout) as client:
        for i, row_idx in enumerate(err_idx):
            row = df.loc[row_idx]
            field = row["field"]
            oid = row["openalex_id"]
            rec = master_by_id.get(oid)
            if rec is None:
                print(
                    f"  [{i + 1}/{len(err_idx)}] {field}: "
                    f"master record missing, skip"
                )
                new_score.append(0)
                new_pred.append(None)
                new_raw.append("")
                new_thinking.append(None)
                new_err.append("master record missing")
                new_elapsed.append(0.0)
                continue
            item = create_prompt(rec.to_dict(), [field])
            t0 = time.time()
            err: str | None = None
            try:
                resp = client.complete(
                    item.prompt,
                    reasoning=reasoning,
                    max_tokens=args.max_tokens,
                )
                ans = parse_answer(resp)
                scores = score_response(ans, item.ground_truth)
                s = scores.get(field, 0)
                pred = ans.get(field) if isinstance(ans, dict) else None
                raw_text = resp.text
                thinking = resp.thinking
            except (httpx.TimeoutException, httpx.HTTPError) as e:
                err = f"{type(e).__name__}: {e}"
                s = 0
                pred = None
                raw_text = ""
                thinking = None
            dt = time.time() - t0
            new_score.append(s)
            new_pred.append(pred)
            new_raw.append(raw_text)
            new_thinking.append(thinking)
            new_err.append(err)
            new_elapsed.append(round(dt, 2))
            tag = f" ERR={err}" if err else ""
            print(
                f"  [{i + 1}/{len(err_idx)}] {field:14s} score={s} "
                f"({dt:.1f}s){tag}"
            )
            if args.sleep > 0:
                time.sleep(args.sleep)

    df = df.copy()
    df.loc[err_idx, "score"] = new_score
    df.loc[err_idx, "prediction"] = new_pred
    df.loc[err_idx, "raw_text"] = new_raw
    df.loc[err_idx, "thinking"] = new_thinking
    df.loc[err_idx, "error"] = new_err
    df.loc[err_idx, "elapsed_s"] = new_elapsed

    out = args.run_parquet.with_name(
        args.run_parquet.stem + "_rerun.parquet"
    )
    df.to_parquet(out, index=False)
    err_after = df["error"].notna().sum()
    print(f"\nwrote {out}")
    print(f"errors after rerun: {err_after} (was {len(err_idx)})")


if __name__ == "__main__":
    main()
