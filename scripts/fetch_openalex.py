"""Pilot fetch from OpenAlex /works.

Stratified random sampling across citation tiers within a research concept.
Saves raw JSON responses (one file per request, including meta) to data/raw/
and a normalized parquet table to data/processed/.

Usage:
    uv run scripts/fetch_openalex.py
    # OPENALEX_API_KEY is read from the environment or from a .env file at
    # the project root.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import httpx
import pandas as pd
from dotenv import load_dotenv

BASE_URL = "https://api.openalex.org"
ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"

# OpenAlex topic-taxonomy filters. We filter on `primary_topic.*.id` so
# that only papers whose *primary* classification matches are kept.
# `concepts.id` matches any concept at any score, which lets interdisciplinary
# papers from other primary disciplines leak in.
#
# Each entry maps a domain key to (filter_field, filter_value).
DOMAINS: dict[str, tuple[str, str]] = {
    # Field-level (broad)
    "cs":       ("primary_topic.field.id", "fields/17"),  # Computer Science
    "medicine": ("primary_topic.field.id", "fields/27"),  # Medicine
    "biochem":  ("primary_topic.field.id", "fields/13"),
    # Subfield-level (recommended for memorization studies — narrower and
    # better-aligned with arXiv-style preprint cultures)
    "cs-ai":    ("primary_topic.subfield.id", "subfields/1702"),  # AI/ML
    "cs-cv":    ("primary_topic.subfield.id", "subfields/1707"),  # CV/PR
    "cs-theory": ("primary_topic.subfield.id", "subfields/1703"),  # CompTheory
    "med-onc":  ("primary_topic.subfield.id", "subfields/2730"),  # Oncology
    "med-card": ("primary_topic.subfield.id", "subfields/2705"),  # Cardiology
}

# Citation tiers used for stratified sampling.
#
# `three`: the original 3-tier scheme. Bounds rounded so we don't
#   over-anchor to ICAART's GPT-4.1-derived thresholds (~90 inflection,
#   ~1200 saturation). `high` floor at 800 (not 1200) for pool size.
#
# `log10`: ~half-decade-wide bins on log10(citation), so the per-bin
#   sample gives roughly uniform coverage of the log-citation axis. Used
#   as a complement to the 3-tier sample to smooth the continuous
#   accuracy curves at the tier boundaries.
TIER_SCHEMES: dict[str, list[tuple[str, str]]] = {
    "three": [
        ("low",  "10-99"),
        ("mid",  "100-799"),
        ("high", "800-10000000"),
    ],
    "log10": [
        ("s1_10-32",    "10-32"),
        ("s2_33-100",   "33-100"),
        ("s3_101-316",  "101-316"),
        ("s4_317-1000", "317-1000"),
        ("s5_1001-3162", "1001-3162"),
        ("s6_3163plus", "3163-10000000"),
    ],
}
TIERS: list[tuple[str, str]] = TIER_SCHEMES["three"]  # default, overridden in main()


def get_api_key() -> str:
    load_dotenv(ROOT / ".env")
    key = os.environ.get("OPENALEX_API_KEY")
    if not key:
        sys.exit(
            "OPENALEX_API_KEY is not set. Put it in .env at the project root "
            "or export it in the shell."
        )
    return key


def detect_source(locations: list[dict] | None, needle: str) -> bool:
    if not locations:
        return False
    needle = needle.lower()
    for loc in locations:
        src = loc.get("source") or {}
        name = (src.get("display_name") or "").lower()
        if needle in name:
            return True
    return False


def reconstruct_abstract(inv_index: dict | None) -> str | None:
    """Rebuild abstract text from OpenAlex `abstract_inverted_index`.

    OpenAlex stores abstracts as a word -> [positions] mapping (a workaround
    for licensing constraints around full-text). Reconstruction is exact for
    word ordering but punctuation/hyphenation can be slightly lossy.
    """
    if not inv_index:
        return None
    positions: dict[int, str] = {}
    for word, pos_list in inv_index.items():
        for p in pos_list:
            positions[p] = word
    if not positions:
        return None
    return " ".join(positions[i] for i in sorted(positions))


def normalize_work(w: dict) -> dict:
    primary = w.get("primary_location") or {}
    primary_src = primary.get("source") or {}
    biblio = w.get("biblio") or {}
    oa = w.get("open_access") or {}
    pct = w.get("cited_by_percentile_year") or {}
    locs = w.get("locations") or []
    authorships = w.get("authorships") or []
    first_author = authorships[0] if authorships else {}
    author_names = [
        ((a.get("author") or {}).get("display_name") or "")
        for a in authorships
    ]
    abstract = reconstruct_abstract(w.get("abstract_inverted_index"))

    first_page = biblio.get("first_page")
    last_page = biblio.get("last_page")
    if first_page and last_page:
        pages = f"{first_page}--{last_page}"
    elif first_page:
        pages = str(first_page)
    else:
        pages = None

    return {
        "openalex_id": w.get("id"),
        "doi": (w.get("ids") or {}).get("doi"),
        "title": w.get("title") or w.get("display_name"),
        "abstract": abstract,
        "has_abstract": abstract is not None,
        "type": w.get("type"),
        "type_crossref": w.get("type_crossref"),
        "publication_year": w.get("publication_year"),
        "publication_date": w.get("publication_date"),
        "language": w.get("language"),
        "is_retracted": w.get("is_retracted"),
        "is_paratext": w.get("is_paratext"),
        # citation signals
        "cited_by_count": w.get("cited_by_count"),
        "fwci": w.get("fwci"),
        "cited_by_percentile_min": pct.get("min"),
        "cited_by_percentile_max": pct.get("max"),
        "referenced_works_count": w.get("referenced_works_count"),
        # venue
        "primary_venue": primary_src.get("display_name"),
        "primary_venue_type": primary_src.get("type"),
        "primary_venue_id": primary_src.get("id"),
        "primary_version": primary.get("version"),
        # access / corpus footprint
        "is_oa": oa.get("is_oa"),
        "oa_status": oa.get("oa_status"),
        "n_locations": len(locs),
        "has_arxiv": detect_source(locs, "arxiv"),
        "has_pmc": detect_source(locs, "pubmed central"),
        "has_biorxiv": detect_source(locs, "biorxiv"),
        # authorship (cloze targets)
        "n_authors": len(authorships),
        "authors": author_names,  # full ordered list
        "first_author": (first_author.get("author") or {}).get("display_name"),
        "first_author_orcid": (first_author.get("author") or {}).get("orcid"),
        # biblio numerics (cloze targets)
        "volume": biblio.get("volume"),
        "issue": biblio.get("issue"),
        "first_page": first_page,
        "last_page": last_page,
        "pages": pages,
    }


def _build_filter(
    domain_filter: tuple[str, str],
    citation_range: str,
    from_year: int,
    require_abstract: bool,
) -> str:
    key, value = domain_filter
    parts = [
        f"{key}:{value}",
        "type:article",
        f"cited_by_count:{citation_range}",
        f"publication_year:{from_year}-9999",
        "is_paratext:false",
        "is_retracted:false",
        "language:en",
    ]
    if require_abstract:
        parts.append("has_abstract:true")
    return ",".join(parts)


def probe_pool_size(
    client: httpx.Client,
    domain_filter: tuple[str, str],
    citation_range: str,
    from_year: int,
    require_abstract: bool,
) -> tuple[int, float]:
    """One-shot count call; uses per-page=1 to keep response small."""
    params = {
        "filter": _build_filter(
            domain_filter, citation_range, from_year, require_abstract
        ),
        "per-page": 1,
    }
    r = client.get(f"{BASE_URL}/works", params=params)
    r.raise_for_status()
    data = r.json()
    meta = data.get("meta") or {}
    return int(meta.get("count") or 0), float(meta.get("cost_usd") or 0.0)


def fetch_sample(
    client: httpx.Client,
    domain_filter: tuple[str, str],
    citation_range: str,
    from_year: int,
    require_abstract: bool,
    sample_size: int,
    seed: int,
) -> dict:
    """Fetch approximately ``sample_size`` random works.

    Strategy: OpenAlex ``sample`` is hard-capped at 200 per call and the
    cursor-with-sort path biases toward high-citation papers (default
    sort is ``cited_by_count:desc``). To get an unbiased ``> 200`` random
    sample within a tier we issue multiple ``sample=200&seed=N`` calls
    with rolling seeds and dedupe by ``id``.

    Returns once we hit ``sample_size`` unique papers, or once the last
    page added fewer than 5 new papers (heavy duplication suggests the
    pool is exhausted), or after a hard cap of ``ceil(sample_size/200)
    + 10`` calls.
    """
    base_params = {
        "filter": _build_filter(
            domain_filter, citation_range, from_year, require_abstract
        ),
        "per-page": 200,
    }
    seen: dict[str, dict] = {}
    cost = 0.0
    last_meta: dict = {}
    max_calls = (sample_size + 199) // 200 + 10
    for i in range(max_calls):
        params = dict(
            base_params,
            sample=200,
            seed=seed + i,
        )
        r = client.get(f"{BASE_URL}/works", params=params)
        r.raise_for_status()
        data = r.json()
        last_meta = data.get("meta") or {}
        cost += float(last_meta.get("cost_usd") or 0.0)
        page_results = data.get("results") or []
        added = 0
        for w in page_results:
            wid = w.get("id")
            if wid and wid not in seen:
                seen[wid] = w
                added += 1
        if len(seen) >= sample_size:
            break
        if added < 5 and i > 0:
            # Pool nearly exhausted; further seeds would mostly duplicate.
            break
    results = list(seen.values())[:sample_size]
    return {
        "meta": {**last_meta, "cost_usd_total": cost},
        "results": results,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--domain", default="cs-ai", choices=sorted(DOMAINS))
    p.add_argument("--tier-scheme", default="three",
                   choices=sorted(TIER_SCHEMES),
                   help="'three' = low/mid/high (default); 'log10' = "
                        "half-decade-wide log10(citation) bins")
    p.add_argument("--per-tier", type=int, default=30,
                   help="papers to sample per citation tier")
    p.add_argument("--from-year", type=int, default=2010,
                   help="lower bound on publication_year (inclusive). "
                        "Default 2010 keeps the high-citation pool sizable "
                        "while staying within the arXiv-mainstream era.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--require-abstract", action="store_true", default=True,
                   help="restrict the sample to papers that have a non-empty "
                        "abstract_inverted_index in OpenAlex. Required for "
                        "title-mask cloze. Default on.")
    p.add_argument("--no-require-abstract", dest="require_abstract",
                   action="store_false",
                   help="disable the has_abstract filter (allows comparing "
                        "pool sizes with/without the abstract requirement)")
    p.add_argument("--probe-pool", action="store_true",
                   help="issue one extra count call per tier to record the "
                        "true pool size (sample-mode meta.count returns the "
                        "sample size, not the underlying pool)")
    args = p.parse_args()

    api_key = get_api_key()
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    domain_filter = DOMAINS[args.domain]
    tiers = TIER_SCHEMES[args.tier_scheme]
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    all_works: list[dict] = []
    total_cost = 0.0
    pool_sizes: dict[str, int] = {}

    with httpx.Client(timeout=60.0, params={"api_key": api_key}) as client:
        for tier_name, tier_range in tiers:
            if args.probe_pool:
                n_pool, probe_cost = probe_pool_size(
                    client, domain_filter, tier_range, args.from_year,
                    args.require_abstract,
                )
                pool_sizes[tier_name] = n_pool
                total_cost += probe_cost
            else:
                pool_sizes[tier_name] = -1

            abs_tag = "abs" if args.require_abstract else "noabs"
            print(
                f"[{args.domain}/{tier_name}/{abs_tag}] "
                f"pool={pool_sizes[tier_name]}, "
                f"sampling {args.per_tier} from cited_by_count:{tier_range} "
                f"(year>={args.from_year})",
                flush=True,
            )
            data = fetch_sample(
                client,
                domain_filter=domain_filter,
                citation_range=tier_range,
                from_year=args.from_year,
                require_abstract=args.require_abstract,
                sample_size=args.per_tier,
                seed=args.seed,
            )
            meta = data.get("meta") or {}
            cost = float(meta.get("cost_usd_total") or meta.get("cost_usd") or 0.0)
            total_cost += cost
            results = data.get("results") or []
            print(
                f"  returned: {len(results)}, sample-call cost: ${cost:.6f}"
            )

            raw_path = RAW_DIR / (
                f"{timestamp}_{args.domain}_tier-{tier_name}.json"
            )
            with raw_path.open("w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            for w in results:
                w["_pilot_tier"] = tier_name
                w["_pilot_domain"] = args.domain
                all_works.append(w)

    print(f"\nTotal cost: ${total_cost:.6f}")

    rows = []
    for w in all_works:
        row = normalize_work(w)
        row["pilot_tier"] = w["_pilot_tier"]
        row["pilot_domain"] = w["_pilot_domain"]
        row["domain_filter_key"] = domain_filter[0]
        row["domain_filter_value"] = domain_filter[1]
        row["from_year"] = args.from_year
        row["fetched_at"] = timestamp
        rows.append(row)

    df = pd.DataFrame(rows)
    scheme_tag = "" if args.tier_scheme == "three" else f"_{args.tier_scheme}"
    parquet_path = PROCESSED_DIR / (
        f"{timestamp}_{args.domain}{scheme_tag}_pilot.parquet"
    )
    df.to_parquet(parquet_path, index=False)
    print(f"Wrote {len(df)} rows to {parquet_path}")

    # quick sanity printout
    if not df.empty:
        cols = ["pilot_tier", "cited_by_count", "type", "has_arxiv",
                "primary_venue"]
        with pd.option_context("display.max_colwidth", 60,
                               "display.width", 160):
            print("\nHead by tier:")
            print(
                df.sort_values(["pilot_tier", "cited_by_count"])[cols]
                  .groupby("pilot_tier", group_keys=False)
                  .head(3)
                  .to_string(index=False)
            )
            print("\nTier summary:")
            print(
                df.groupby("pilot_tier")
                  .agg(n=("openalex_id", "size"),
                       cit_med=("cited_by_count", "median"),
                       arxiv_rate=("has_arxiv", "mean"),
                       oa_rate=("is_oa", "mean"))
                  .to_string()
            )


if __name__ == "__main__":
    main()
