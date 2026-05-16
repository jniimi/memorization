"""Gemini Batch API runner for cloze experiments.

Three subcommands:

    submit   Build a JSONL of cloze requests from the master parquet
             (same sampling as smoke_cloze.py for the given
             --master/--per-tier/--start/--seed), upload it to the Files
             API, create a batch job, and persist the job state.

    status   Poll a previously submitted batch and print state +
             completed/total counts.

    fetch    Once the batch is in JOB_STATE_SUCCEEDED, download the
             result JSONL, score each response with scoring.score_response,
             join back to ground truth via the per-request key, and
             write a parquet under data/runs/ in the same schema as
             smoke_cloze.py (so analyze.py/compare_models.py just work).

Why a separate runner: Gemini's daily realtime quota gets in the way of
multi-thousand-call sweeps. The Batch API has a separate quota pool,
costs 50% less, and returns within 24h (often much faster). One API key
is sufficient.

A single batch state file at
``data/batches/<ts>_<safe_model>_<suffix>.state.json`` carries every
parameter needed to reproduce sampling at fetch time, so submit and
fetch can run in different processes (or different days).

Usage:
    uv run scripts/gemini_batch.py submit \\
        --model gemini-3-flash-preview --reasoning \\
        --per-tier 200 --start 0 \\
        --master data/processed/<...>_log10_pilot.parquet

    uv run scripts/gemini_batch.py status --state <state.json>

    uv run scripts/gemini_batch.py fetch --state <state.json>
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cloze import MASKABLE_FIELDS, available_mask_items, create_prompt
from llm import _resolve_gemini_keys, parse_answer, LLMResponse
from scoring import score_response
from smoke_cloze import find_latest_parquet, sample_papers

ROOT = Path(__file__).resolve().parent.parent
BATCH_DIR = ROOT / "data" / "batches"
RUNS_DIR = ROOT / "data" / "runs"


# ---------------------------------------------------------------------------
# SDK client
# ---------------------------------------------------------------------------

def _get_genai_client(key_var: str | None = None):
    """Return a google-genai client.

    By default uses the first available Gemini key (same resolver order
    as the realtime client). Pass ``key_var`` to pin a specific env var
    — useful when one key's daily batch/file quota is exhausted.
    """
    load_dotenv(ROOT / ".env")
    if key_var:
        value = os.environ.get(key_var)
        if not value:
            sys.exit(f"env var {key_var} is not set")
        name = key_var
    else:
        keys = _resolve_gemini_keys()
        if not keys:
            sys.exit("no Gemini API key found in env or .env")
        name, value = keys[0]
    print(f"using Gemini key: {name}")
    # Lazy import so the script imports cleanly even before the SDK is
    # installed.
    from google import genai  # type: ignore
    return genai.Client(api_key=value)


# ---------------------------------------------------------------------------
# Request construction
# ---------------------------------------------------------------------------

def _request_for(prompt: str, *, reasoning: bool, max_tokens: int) -> dict:
    """Build a GenerateContentRequest body that matches GeminiClient.complete."""
    gen_cfg: dict = {
        "temperature": 0.0,
        "thinkingConfig": {"thinkingBudget": -1 if reasoning else 0},
        "maxOutputTokens": max_tokens,
        "responseMimeType": "application/json",
    }
    return {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": gen_cfg,
    }


def _build_input_jsonl(
    sample: pd.DataFrame, *, reasoning: bool, max_tokens: int,
) -> tuple[list[dict], dict[str, dict]]:
    """Build one JSONL line per (paper, available field).

    Returns (lines, key_index) where key_index maps the per-request key
    back to {openalex_id, picked_index, tier, cited_by_count, field,
    ground_truth} for fetch-time scoring.
    """
    lines: list[dict] = []
    key_index: dict[str, dict] = {}
    for _, row in sample.iterrows():
        record = row.to_dict()
        fields = available_mask_items(record)
        for field_name in fields:
            item = create_prompt(record, [field_name])
            key = f"{item.record_id}__{field_name}"
            lines.append({
                "key": key,
                "request": _request_for(
                    item.prompt,
                    reasoning=reasoning,
                    max_tokens=max_tokens,
                ),
            })
            key_index[key] = {
                "openalex_id": item.record_id,
                "picked_index": int(record["_picked_index"]),
                "tier": record["pilot_tier"],
                "cited_by_count": int(record["cited_by_count"]),
                "field": field_name,
                "ground_truth": item.ground_truth.get(field_name),
            }
    return lines, key_index


# ---------------------------------------------------------------------------
# submit
# ---------------------------------------------------------------------------

def _chunk(seq: list, size: int) -> list[list]:
    return [seq[i:i + size] for i in range(0, len(seq), size)]


def _create_batch_with_retry(
    client, *, model: str, src: str, display_name: str,
    max_wait_s: float = 900.0,
):
    """Create a batch job; on 429 RESOURCE_EXHAUSTED, back off and retry.

    ``gemini-3-flash-preview`` rejects batches above ~a few thousand
    requests with 429 even on paid tiers. We chunk below that, but the
    *cumulative* enqueued pool across in-flight chunks can still 429
    transiently; waiting lets earlier chunks drain. Re-raises after
    ``max_wait_s`` of cumulative backoff.
    """
    from google.genai import errors as genai_errors  # type: ignore
    waited = 0.0
    backoff = 60.0
    while True:
        try:
            return client.batches.create(
                model=model, src=src, config={"display_name": display_name},
            )
        except genai_errors.ClientError as e:
            # We only retry the specific 4xx that means "wait and try
            # again" (RESOURCE_EXHAUSTED / 429). All other 4xx errors
            # (auth, malformed request, etc.) are non-retriable and
            # re-raised so the caller sees them.
            msg = str(e)
            if "RESOURCE_EXHAUSTED" not in msg and "429" not in msg:
                raise
            if waited >= max_wait_s:
                raise
            sleep_for = min(backoff, max_wait_s - waited)
            print(f"  429 on create; waiting {sleep_for:.0f}s "
                  f"(cumulative {waited:.0f}s) ...")
            time.sleep(sleep_for)
            waited += sleep_for
            backoff = min(backoff * 1.5, 180.0)


def cmd_submit(args: argparse.Namespace) -> None:
    BATCH_DIR.mkdir(parents=True, exist_ok=True)
    from google.genai import types  # type: ignore

    # ---- resume an existing partially-submitted state, or start fresh ----
    if args.resume:
        state_path = Path(args.resume)
        state = _load_state(state_path)
        stem = state["stem"]
        key_index_path = Path(state["key_index"])
        with key_index_path.open() as f:
            key_index = json.load(f)
        pq = Path(state["master_path"])
        df = pd.read_parquet(pq)
        sample = sample_papers(
            df, per_tier=state["per_tier"], start=state["start"],
            seed=state["seed"],
        )
        lines, _ = _build_input_jsonl(
            sample, reasoning=state["reasoning"],
            max_tokens=state["max_tokens"],
        )
        print(f"resuming {stem}: {len(lines)} total requests, "
              f"{len(state.get('chunks', []))} chunks already submitted")
    else:
        pq = Path(args.master) if args.master else find_latest_parquet()
        print(f"loading master: {pq.name}")
        df = pd.read_parquet(pq)
        sample = sample_papers(
            df, per_tier=args.per_tier, start=args.start, seed=args.seed,
        )
        tier_info = sample.attrs.get("tier_info", {})
        print(f"papers sampled: {len(sample)} ({tier_info})")
        lines, key_index = _build_input_jsonl(
            sample, reasoning=args.reasoning, max_tokens=args.max_tokens,
        )
        print(f"total requests: {len(lines)}")
        ts = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
        safe_model = args.model.replace(":", "-").replace("/", "-")
        suffix = "_think" if args.reasoning else ""
        stem = f"{ts}_{safe_model}{suffix}_n{args.per_tier}_start{args.start}"
        key_index_path = BATCH_DIR / f"{stem}.keys.json"
        with key_index_path.open("w", encoding="utf-8") as f:
            json.dump(key_index, f, ensure_ascii=False)
        state = {
            "stem": stem,
            "key_var": args.key_var,
            "model": args.model,
            "reasoning": args.reasoning,
            "master": pq.name,
            "master_path": str(pq),
            "per_tier": args.per_tier,
            "start": args.start,
            "seed": args.seed,
            "max_tokens": args.max_tokens,
            "n_requests": len(lines),
            "tier_info": tier_info,
            "submitted_at": ts,
            "max_per_batch": args.max_per_batch,
            "key_index": str(key_index_path),
            "chunks": [],
        }
    state_path = BATCH_DIR / f"{stem}.state.json"
    max_per_batch = state.get("max_per_batch", args.max_per_batch)

    chunks = _chunk(lines, max_per_batch)
    n_done = len(state.get("chunks", []))
    print(f"{len(chunks)} chunks of <= {max_per_batch} requests; "
          f"{n_done} already submitted, {len(chunks) - n_done} to go")

    client = _get_genai_client(state.get("key_var"))

    def _save() -> None:
        with state_path.open("w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)

    _save()
    for idx in range(n_done, len(chunks)):
        chunk = chunks[idx]
        chunk_path = BATCH_DIR / f"{stem}.chunk{idx:02d}.input.jsonl"
        with chunk_path.open("w", encoding="utf-8") as f:
            for line in chunk:
                f.write(json.dumps(line, ensure_ascii=False) + "\n")
        print(f"[chunk {idx}] {len(chunk)} reqs -> uploading "
              f"{chunk_path.name} ({chunk_path.stat().st_size/1e6:.2f} MB)")
        uploaded = client.files.upload(
            file=chunk_path,
            config=types.UploadFileConfig(
                display_name=f"{stem}_c{idx:02d}",
                mime_type="application/jsonl",
            ),
        )
        batch = _create_batch_with_retry(
            client, model=state["model"], src=uploaded.name,
            display_name=f"{stem}_c{idx:02d}",
        )
        print(f"[chunk {idx}] batch: {batch.name} state={batch.state.name}")
        state["chunks"].append({
            "idx": idx,
            "batch_name": batch.name,
            "uploaded_file": uploaded.name,
            "jsonl_input": str(chunk_path),
            "n_requests": len(chunk),
            "keys": [c["key"] for c in chunk],
        })
        _save()

    print(f"\nall {len(chunks)} chunks submitted. state: {state_path}")
    print(f"follow up with:\n"
          f"  uv run scripts/gemini_batch.py status --state {state_path}\n"
          f"  uv run scripts/gemini_batch.py fetch  --state {state_path}")


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------

def _load_state(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def _state_chunks(state: dict) -> list[dict]:
    """Normalize: new multi-chunk states have ``chunks``; the original
    single-batch state has ``batch_name``."""
    if state.get("chunks"):
        return state["chunks"]
    if state.get("batch_name"):
        return [{"idx": 0, "batch_name": state["batch_name"]}]
    return []


def _batch_brief(batch) -> dict:
    info = {"name": batch.name, "state": batch.state.name}
    stats = getattr(batch, "batch_stats", None)
    if stats is not None:
        info["completed"] = getattr(stats, "completed_request_count", None)
        info["total"] = getattr(stats, "total_request_count", None)
        info["failed"] = getattr(stats, "failed_request_count", None)
    err = getattr(batch, "error", None)
    if err is not None:
        info["error"] = str(err)
    return info


def cmd_status(args: argparse.Namespace) -> None:
    state = _load_state(Path(args.state))
    client = _get_genai_client(state.get("key_var"))
    chunks = _state_chunks(state)
    n_ok = 0
    for ch in chunks:
        b = client.batches.get(name=ch["batch_name"])
        brief = _batch_brief(b)
        if brief["state"] == "JOB_STATE_SUCCEEDED":
            n_ok += 1
        print(f"chunk {ch.get('idx', '?')}: {json.dumps(brief, default=str)}")
    print(f"\n{n_ok}/{len(chunks)} chunks SUCCEEDED")


# ---------------------------------------------------------------------------
# fetch
# ---------------------------------------------------------------------------

def _extract_text_and_thinking(resp_obj: dict) -> tuple[str, str | None]:
    """Pull main text + thinking trace out of a GenerateContentResponse dict."""
    text_parts: list[str] = []
    thinking_parts: list[str] = []
    for cand in (resp_obj.get("candidates") or []):
        content = cand.get("content") or {}
        for p in (content.get("parts") or []):
            t = p.get("text") or ""
            if p.get("thought"):
                thinking_parts.append(t)
            else:
                text_parts.append(t)
        break
    return "".join(text_parts), ("".join(thinking_parts) or None)


def _collect_chunk_results(client, batch) -> dict[str, dict]:
    """Download one chunk's output JSONL and index it by request key."""
    out: dict[str, dict] = {}
    dest = batch.dest
    out_file = getattr(dest, "file_name", None)
    inlined = getattr(dest, "inlined_responses", None)
    if out_file:
        raw = client.files.download(file=out_file)
        text = raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else str(raw)
        for line in text.splitlines():
            if not line.strip():
                continue
            obj = json.loads(line)
            key = obj.get("key")
            if key:
                out[key] = obj
    elif inlined:
        for r in inlined:
            r_d = r if isinstance(r, dict) else getattr(r, "__dict__", {})
            key = (r_d.get("metadata") or {}).get("key")
            if key:
                out[key] = r_d
    return out


def cmd_fetch(args: argparse.Namespace) -> None:
    state = _load_state(Path(args.state))
    client = _get_genai_client(state.get("key_var"))
    chunks = _state_chunks(state)

    # Load key index for ground-truth lookup
    key_index_path = Path(state["key_index"])
    with key_index_path.open() as f:
        key_index: dict[str, dict] = json.load(f)

    results_by_key: dict[str, dict] = {}
    pending: list[str] = []
    for ch in chunks:
        b = client.batches.get(name=ch["batch_name"])
        st = b.state.name
        idx = ch.get("idx", "?")
        if st != "JOB_STATE_SUCCEEDED":
            print(f"chunk {idx}: {st} (not ready)")
            pending.append(f"chunk {idx} ({ch['batch_name']}): {st}")
            continue
        got = _collect_chunk_results(client, b)
        print(f"chunk {idx}: {st}, {len(got)} responses")
        results_by_key.update(got)

    if pending and not args.partial:
        sys.exit("not all chunks ready:\n  " + "\n  ".join(pending) +
                 "\nre-run when ready, or pass --partial to score what we have")

    print(f"got {len(results_by_key)} responses out of "
          f"{len(key_index)} requests")
    missing = [k for k in key_index if k not in results_by_key]
    if missing:
        print(f"⚠ missing {len(missing)} keys (first 5): {missing[:5]}")

    rows: list[dict] = []
    by_field: dict[str, list[int]] = defaultdict(list)
    for key, meta in key_index.items():
        field_name = meta["field"]
        gt = meta["ground_truth"]
        result = results_by_key.get(key)
        err: str | None = None
        pred_str: str | None = None
        raw_text = ""
        thinking: str | None = None
        s = 0
        if result is None:
            err = "missing in batch output"
        else:
            err_obj = result.get("error")
            resp = result.get("response")
            if err_obj is not None:
                err = json.dumps(err_obj) if not isinstance(err_obj, str) else err_obj
            elif resp is not None:
                raw_text, thinking = _extract_text_and_thinking(resp)
                fake = LLMResponse(
                    text=raw_text, thinking=thinking,
                    model=state["model"],
                    reasoning_enabled=bool(state["reasoning"]),
                )
                ans = parse_answer(fake)
                scores = score_response(ans, {field_name: gt})
                s = scores.get(field_name, 0)
                if isinstance(ans, dict):
                    pred_str = ans.get(field_name)
        by_field[field_name].append(s)
        rows.append({
            "openalex_id": meta["openalex_id"],
            "picked_index": meta["picked_index"],
            "field": field_name,
            "tier": meta["tier"],
            "cited_by_count": meta["cited_by_count"],
            "ground_truth": gt,
            "prediction": pred_str,
            "raw_text": raw_text,
            "thinking": thinking,
            "score": s,
            "elapsed_s": None,  # not meaningful for batch
            "error": err,
            "model": state["model"],
            "reasoning": bool(state["reasoning"]),
            "seed": state["seed"],
            "start": state["start"],
            "master_parquet": state["master"],
        })

    print(f"\nper-field accuracy:")
    for f in MASKABLE_FIELDS:
        v = by_field.get(f, [])
        if v:
            print(f"  {f:14s} {sum(v):>4}/{len(v):<4} = {sum(v)/len(v):.0%}")

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    suffix = "_think" if state["reasoning"] else ""
    safe_model = state["model"].replace(":", "-").replace("/", "-")
    out = RUNS_DIR / (
        f"{int(time.time())}_{safe_model}{suffix}_n{state['per_tier']}"
        f"_start{state['start']}_batch.parquet"
    )
    pd.DataFrame(rows).to_parquet(out, index=False)
    print(f"wrote {len(rows)} items to {out}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("submit")
    sp.add_argument("--model", default="gemini-3-flash-preview")
    sp.add_argument("--reasoning", action="store_true")
    sp.add_argument("--per-tier", type=int, default=200)
    sp.add_argument("--start", type=int, default=0)
    sp.add_argument("--seed", type=int, default=42)
    sp.add_argument("--master", type=str, default=None)
    sp.add_argument("--max-tokens", type=int, default=4096)
    sp.add_argument("--max-per-batch", type=int, default=1800,
                    help="split into batch jobs of at most this many "
                         "requests (gemini-3-flash-preview 429s above "
                         "~a few thousand per batch)")
    sp.add_argument("--key-var", type=str, default=None,
                    help="env var name of the Gemini key to use "
                         "(default: first available). Persisted to the "
                         "state file so status/fetch reuse it.")
    sp.add_argument("--resume", type=str, default=None,
                    help="path to an existing .state.json to continue "
                         "submitting remaining chunks")

    st = sub.add_parser("status")
    st.add_argument("--state", required=True)

    fe = sub.add_parser("fetch")
    fe.add_argument("--state", required=True)
    fe.add_argument("--partial", action="store_true",
                    help="write a parquet even if some chunks are not "
                         "yet SUCCEEDED (missing keys -> error rows)")

    args = p.parse_args()
    if args.cmd == "submit":
        cmd_submit(args)
    elif args.cmd == "status":
        cmd_status(args)
    elif args.cmd == "fetch":
        cmd_fetch(args)


if __name__ == "__main__":
    main()
