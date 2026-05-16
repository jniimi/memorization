"""End-to-end cloze runner.

Loads the latest master parquet, deterministically picks N papers per
citation tier, generates per-paper cloze prompts (masking only fields
that are populated for that paper), sends them to the configured LLM,
parses the JSON, scores it, and prints per-field × tier accuracy.

Sampling is deterministic given ``--seed``: each tier's pool is
sorted by ``openalex_id`` and then shuffled with the seed; the run
selects rows ``[start, start + per_tier)`` from the shuffled order.
This makes ``--start`` cleanly composable for extending later runs
(e.g. run 1 with ``--start 0 --per-tier 500``, run 2 with
``--start 500 --per-tier 500`` gives non-overlapping subsets).

Usage:
    uv run scripts/smoke_cloze.py --model qwen3.5:0.8b --per-tier 5
    uv run scripts/smoke_cloze.py --model gemini-3-flash-preview \
        --per-tier 500 --start 0
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import httpx
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cloze import MASKABLE_FIELDS, available_mask_items, create_prompt
from llm import AllKeysExhausted, get_client, parse_answer
from scoring import score_response

ROOT = Path(__file__).resolve().parent.parent


def find_latest_parquet() -> Path:
    files = sorted((ROOT / "data" / "processed").glob("*_pilot.parquet"))
    if not files:
        sys.exit("no master parquet found in data/processed/")
    return files[-1]


def sample_papers(
    df: pd.DataFrame, *, per_tier: int, start: int, seed: int,
) -> pd.DataFrame:
    """Deterministic per-tier shuffle + slice.

    Per-tier pool is sorted by ``openalex_id``, shuffled with ``seed``,
    and rows ``[start, start + per_tier)`` are taken. Used by both the
    realtime runner and the batch runner so the two sample the same
    papers when seed/per_tier/start match.
    """
    tier_names = sorted(df["pilot_tier"].dropna().unique())
    sampled = []
    tier_info: dict[str, int] = {}
    for tier_name in tier_names:
        pool = (
            df[df["pilot_tier"] == tier_name]
              .sort_values("openalex_id")
              .reset_index(drop=True)
        )
        shuffled = pool.sample(frac=1, random_state=seed) \
                       .reset_index(drop=True)
        if start >= len(shuffled):
            print(
                f"⚠ tier {tier_name!r}: --start {start} exceeds pool size "
                f"{len(shuffled)}; skipping this tier",
                flush=True,
            )
            tier_info[tier_name] = 0
            continue
        end = min(start + per_tier, len(shuffled))
        picked = shuffled.iloc[start:end].copy()
        picked["_picked_index"] = range(start, end)
        sampled.append(picked)
        tier_info[tier_name] = len(picked)

    if not sampled:
        # Every tier exhausted under the requested start; return an
        # empty frame with the right columns and the tier_info attrs
        # so callers can detect the no-sample case without crashing.
        empty = df.iloc[0:0].copy()
        empty["_picked_index"] = pd.Series([], dtype=int)
        empty.attrs["tier_info"] = tier_info
        return empty

    sample = (
        pd.concat(sampled)
          .sort_values(["pilot_tier", "cited_by_count"])
          .reset_index(drop=True)
    )
    sample.attrs["tier_info"] = tier_info
    return sample


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="qwen3.5:0.8b")
    p.add_argument("--reasoning", action="store_true",
                   help="enable model thinking (Qwen3 / DeepSeek-R1, "
                        "Gemini thinkingBudget=-1, etc.)")
    p.add_argument("--per-tier", type=int, default=5,
                   help="papers per citation tier")
    p.add_argument("--start", type=int, default=0,
                   help="starting offset within each tier's shuffled "
                        "order. Use --start N --per-tier N to grab the "
                        "next N papers without overlap with a previous "
                        "--start 0 --per-tier N run.")
    p.add_argument("--seed", type=int, default=42,
                   help="seed for the deterministic per-tier shuffle")
    p.add_argument("--master", type=str, default=None,
                   help="path to master parquet (defaults to latest)")
    p.add_argument("--max-tokens", type=int, default=256,
                   help="cap model output length")
    p.add_argument("--timeout", type=float, default=120.0,
                   help="per-request timeout in seconds")
    p.add_argument("--sleep", type=float, default=0.0,
                   help="seconds to sleep between calls (e.g. 6.5 for "
                        "Gemini free tier 10 RPM)")
    p.add_argument("--out-dir", type=str, default=None,
                   help="output directory (defaults to data/runs/)")
    args = p.parse_args()

    pq = Path(args.master) if args.master else find_latest_parquet()
    print(f"loading master: {pq.name}")
    df = pd.read_parquet(pq)

    sample = sample_papers(
        df, per_tier=args.per_tier, start=args.start, seed=args.seed,
    )
    tier_info = sample.attrs.get("tier_info", {})

    print(
        f"papers: {len(sample)} ({tier_info}); "
        f"start={args.start}; seed={args.seed}; "
        f"model={args.model}; reasoning={args.reasoning}"
    )
    print()

    rows: list[dict] = []
    by_field: dict[str, list[int]] = defaultdict(list)
    elapsed = 0.0
    aborted = False
    consec_exhausted = 0
    EXHAUST_LIMIT = 3

    with get_client(args.model, timeout=args.timeout) as client:
        for paper_i, row in sample.iterrows():
            if aborted:
                break
            record = row.to_dict()
            fields_to_mask = available_mask_items(record)
            print(
                f"[{paper_i + 1}/{len(sample)}] tier={record['pilot_tier']} "
                f"cit={record['cited_by_count']} "
                f"avail={len(fields_to_mask)} "
                f"title={(record['title'] or '')[:55]!r}"
            )
            for field_name in fields_to_mask:
                item = create_prompt(record, [field_name])
                t0 = time.time()
                err: str | None = None
                resp = None
                try:
                    resp = client.complete(
                        item.prompt,
                        reasoning=args.reasoning,
                        max_tokens=args.max_tokens,
                    )
                    consec_exhausted = 0
                except AllKeysExhausted as e:
                    err = f"AllKeysExhausted: {e}"
                    consec_exhausted += 1
                except (httpx.TimeoutException, httpx.HTTPError) as e:
                    err = f"{type(e).__name__}: {e}"
                dt = time.time() - t0
                elapsed += dt

                if resp is not None:
                    ans = parse_answer(resp)
                    scores = score_response(ans, item.ground_truth)
                    s = scores.get(field_name, 0)
                    pred_str = (
                        ans.get(field_name) if isinstance(ans, dict) else None
                    )
                    raw_text = resp.text
                    thinking = resp.thinking
                else:
                    s = 0
                    pred_str = None
                    raw_text = ""
                    thinking = None
                by_field[field_name].append(s)

                rows.append({
                    "openalex_id": item.record_id,
                    "picked_index": int(record["_picked_index"]),
                    "field": field_name,
                    "tier": record["pilot_tier"],
                    "cited_by_count": int(record["cited_by_count"]),
                    "ground_truth": item.ground_truth.get(field_name),
                    "prediction": pred_str,
                    "raw_text": raw_text,
                    "thinking": thinking,
                    "score": s,
                    "elapsed_s": round(dt, 2),
                    "error": err,
                    "model": args.model,
                    "reasoning": args.reasoning,
                    "seed": args.seed,
                    "start": args.start,
                    "master_parquet": pq.name,
                })
                gt = item.ground_truth.get(field_name) or ""
                tag = f" ERR={err}" if err else ""
                print(
                    f"    {field_name:14s} score={s} "
                    f"pred={str(pred_str)[:50]!r} "
                    f"gt={gt[:50]!r}  ({dt:.1f}s){tag}"
                )
                if args.sleep > 0:
                    time.sleep(args.sleep)
                if consec_exhausted >= EXHAUST_LIMIT:
                    print(
                        f"\n!! {consec_exhausted} consecutive "
                        f"AllKeysExhausted; aborting and saving partial "
                        f"results"
                    )
                    aborted = True
                    break

    n_calls = len(rows)
    print(f"\ntotal inference time: {elapsed:.1f}s "
          f"(avg {elapsed / max(1, n_calls):.2f}s/call, n={n_calls})")
    print(f"per-field accuracy:")
    for f in MASKABLE_FIELDS:
        v = by_field.get(f, [])
        if v:
            print(f"  {f:14s} {sum(v):>4}/{len(v):<4} = {sum(v)/len(v):.0%}")

    df_rows = pd.DataFrame(rows)
    if not df_rows.empty:
        print("\naccuracy by tier × field (n in parens):")
        pivot_acc = df_rows.pivot_table(
            index="field", columns="tier", values="score", aggfunc="mean"
        ).reindex(MASKABLE_FIELDS)
        pivot_n = df_rows.pivot_table(
            index="field", columns="tier", values="score", aggfunc="count"
        ).reindex(MASKABLE_FIELDS)
        # Order tiers by their numeric lower bound when names follow the
        # log10 scheme (sN_<lo>-<hi>), else by the canonical low/mid/high.
        def _tier_sort_key(t: str):
            canon = {"low": 0, "mid": 1, "high": 2}
            if t in canon:
                return (0, canon[t])
            # log10 scheme: 's3_101-316' -> 101
            m = re.search(r"_(\d+)", t)
            return (1, int(m.group(1)) if m else 1 << 30)
        tier_order = sorted(pivot_acc.columns, key=_tier_sort_key)
        for f in MASKABLE_FIELDS:
            cells = []
            for t in tier_order:
                a = pivot_acc.at[f, t] if t in pivot_acc.columns else None
                n = pivot_n.at[f, t] if t in pivot_n.columns else 0
                if pd.isna(a):
                    cells.append(f"{'-':>10s}")
                else:
                    cells.append(f"{a:>5.0%} ({int(n):>3d})")
            print(f"  {f:14s} " + "  ".join(cells))

    out_dir = Path(args.out_dir) if args.out_dir else (ROOT / "data" / "runs")
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_model = args.model.replace(":", "-").replace("/", "-")
    suffix = "_think" if args.reasoning else ""
    if aborted:
        suffix += "_partial"
    out_path = out_dir / (
        f"{int(time.time())}_{safe_model}{suffix}"
        f"_n{args.per_tier}_start{args.start}.parquet"
    )
    pd.DataFrame(rows).to_parquet(out_path, index=False)
    print(f"\nwrote {len(rows)} items to {out_path}")


if __name__ == "__main__":
    main()
