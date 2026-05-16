"""Cloze prompt builder for the field-level memorization study.

Given a normalized OpenAlex record (one row of the master parquet) and a
list of fields to mask, build the prompt sent to the LLM and return the
ground-truth values for scoring.

Design:
- Pure function, no I/O. Caller is responsible for loading the parquet
  and for filtering records that lack any required field.
- Output schema is dict-form even for single mask, for evaluator
  uniformity:  {"answer": {"first_author": "..."}}.
- Abstract is always included as context when available; toggle via the
  `with_abstract` flag if you want to probe its contribution.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

# Fields the model can be asked to recover. Aligned with ICAART's seven
# manually-scored fields, with `issue` separated from `volume`.
MASKABLE_FIELDS: tuple[str, ...] = (
    "title",
    "first_author",
    "co_authors",
    "year",
    "venue",
    "volume",
    "issue",
    "pages",
)

MASK_TOKEN = "[MASK]"


@dataclass
class ClozeItem:
    """A single cloze instance: prompt + ground truth + provenance."""

    prompt: str                     # rendered instruction sent to the LLM
    ground_truth: dict[str, str]    # {masked_field: original_value}
    record_id: str                  # openalex_id for tracing
    mask_items: list[str]           # echoes the input for round-tripping


# ---------------------------------------------------------------------------
# Field extraction
# ---------------------------------------------------------------------------

def _coerce_str(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, float) and math.isnan(v):
        return ""
    return str(v).strip()


def _co_authors_str(authors: Iterable[str] | None) -> str:
    """APA-style join of authors after the first one.

    Author display names come straight from OpenAlex (e.g. "Geoffrey E.
    Hinton"); we do not attempt to reformat into "Surname, F. M." here.
    The scorer is responsible for cross-format name matching.
    """
    if authors is None:
        return ""
    items = [a for a in (str(x).strip() for x in authors) if a]
    co = items[1:]
    if not co:
        return ""
    if len(co) == 1:
        return co[0]
    if len(co) == 2:
        return f"{co[0]} & {co[1]}"
    return ", ".join(co[:-1]) + ", & " + co[-1]


def _format_field_value(record: dict, field_name: str) -> str:
    """Convert record[field] to its prompt-display string."""
    if field_name == "title":
        return _coerce_str(record.get("title"))
    if field_name == "first_author":
        return _coerce_str(record.get("first_author"))
    if field_name == "co_authors":
        return _co_authors_str(record.get("authors"))
    if field_name == "year":
        return _coerce_str(record.get("publication_year"))
    if field_name == "venue":
        return _coerce_str(record.get("primary_venue"))
    if field_name == "volume":
        return _coerce_str(record.get("volume"))
    if field_name == "issue":
        return _coerce_str(record.get("issue"))
    if field_name == "pages":
        return _coerce_str(record.get("pages"))
    raise KeyError(f"unknown maskable field: {field_name}")


def available_mask_items(record: dict) -> list[str]:
    """Return MASKABLE_FIELDS whose values are present (non-empty)."""
    return [
        f for f in MASKABLE_FIELDS
        if _format_field_value(record, f)
    ]


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

_INSTRUCTION = (
    "You are an academic assistant. The bibliographic record below has "
    "one or more fields replaced with the placeholder \"[MASK]\". Recover "
    "the original value of each masked field, using the unmasked fields "
    "and the abstract as your only context.\n\n"
    "Output a single JSON object with exactly one key, \"answer\", whose "
    "value is an object mapping each masked field name to your best "
    "guess as a string. Fill every masked field. Do not output any text "
    "outside the JSON.\n\n"
    "Example output: {\"answer\": {\"first_author\": \"Hinton, G.\"}}"
)


def _render_prompt(shown: dict[str, str], mask_items: list[str]) -> str:
    record_json = json.dumps(shown, ensure_ascii=False, indent=2)
    masked = ", ".join(mask_items)
    return (
        f"{_INSTRUCTION}\n\n"
        f"Masked fields: {masked}\n\n"
        f"Record:\n{record_json}"
    )


def create_prompt(
    record: dict,
    mask_items: list[str] | tuple[str, ...],
    *,
    with_abstract: bool = True,
) -> ClozeItem:
    """Build a cloze prompt from a record dict.

    Parameters
    ----------
    record:
        A dict with the keys produced by `fetch_openalex.normalize_work`
        (one row of the master parquet, converted via `.to_dict()`).
    mask_items:
        Fields to mask. Must be a subset of `MASKABLE_FIELDS`. Order is
        preserved when listing masked fields back to the model.
    with_abstract:
        If True (default), include the reconstructed abstract in the
        shown context. Setting False lets you probe how much of the
        retrieval signal lives in the abstract vs. the bibliographic
        scaffold.

    Raises
    ------
    ValueError if `mask_items` is empty, contains an unknown field, or
    asks to mask a field whose ground-truth value is empty in this
    record. Use `available_mask_items(record)` upstream to filter.
    """
    mask_items = list(mask_items)

    if not mask_items:
        raise ValueError("mask_items must be non-empty")
    unknown = [f for f in mask_items if f not in MASKABLE_FIELDS]
    if unknown:
        raise ValueError(f"unknown mask fields: {unknown}")

    shown: dict[str, str] = {}
    ground_truth: dict[str, str] = {}
    for f in MASKABLE_FIELDS:
        original = _format_field_value(record, f)
        if f in mask_items:
            if not original:
                raise ValueError(
                    f"cannot mask {f!r}: value is empty in record "
                    f"{record.get('openalex_id')!r}"
                )
            shown[f] = MASK_TOKEN
            ground_truth[f] = original
        else:
            shown[f] = original

    if with_abstract:
        shown["abstract"] = _coerce_str(record.get("abstract"))

    prompt = _render_prompt(shown, mask_items)
    return ClozeItem(
        prompt=prompt,
        ground_truth=ground_truth,
        record_id=_coerce_str(record.get("openalex_id")),
        mask_items=mask_items,
    )
