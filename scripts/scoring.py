"""Field-level binary scoring for cloze responses.

For each masked field, score the model's prediction against the
ground-truth value with a field-specific normalization rule. Returns
{0, 1}. Aggregate metrics (per-field accuracy, citation-tier curves,
etc.) are computed downstream.

Normalization summary
---------------------
title         case + punctuation + whitespace normalized; exact match
first_author  surname (case-folded) + first-initial match across formats
co_authors    same-length list, each pair name-matched, in order
year          first 4-digit substring; exact match
venue         alias-table normalization for major CS venues; exact match
volume        strip "vol." / "volume"; alphanumeric exact match
issue         strip "no." / "issue"; alphanumeric exact match
pages         strip "pp."; unify dash variants; exact match

The alias table starts small. Extend it as we observe miscalls; the
goal is enough coverage to keep false negatives well below the
memorization signal we care about.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any

# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------

def _nfkc(s: str) -> str:
    return unicodedata.normalize("NFKC", s)


def _norm_text(s: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    s = _nfkc(s).lower()
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def _norm_alphanum(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9]", "", s).lower()


# ---------------------------------------------------------------------------
# Title
# ---------------------------------------------------------------------------

def _score_title(pred: str, gt: str) -> int:
    return int(_norm_text(pred) == _norm_text(gt))


# ---------------------------------------------------------------------------
# Author names
# ---------------------------------------------------------------------------

_NAME_SUFFIXES = {"jr", "jr.", "sr", "sr.", "ii", "iii", "iv"}


def _parse_name(s: str) -> tuple[str, list[str]] | None:
    """Return ``(surname_lower, [given_initials_lower])`` or None.

    Tolerates both common formats:
      "Geoffrey E. Hinton"   ← OpenAlex display
      "Hinton, G. E."        ← APA / typical LLM output
    """
    s = _nfkc(s).strip()
    if not s:
        return None

    if "," in s:
        # APA-style: "Surname, F. M."
        surname, given = s.split(",", 1)
        surname = surname.strip()
        if not surname:
            return None
        initials = [c.lower() for c in re.findall(r"[A-Za-z]", given)]
        return surname.lower(), initials

    # Display-style: "First M. Last"
    tokens = s.split()
    while tokens and tokens[-1].lower() in _NAME_SUFFIXES:
        tokens.pop()
    if not tokens:
        return None
    if len(tokens) == 1:
        return tokens[0].lower(), []
    surname = tokens[-1]
    given = tokens[:-1]
    initials: list[str] = []
    for g in given:
        g = g.strip(".")
        if g and g[0].isalpha():
            initials.append(g[0].lower())
    return surname.lower(), initials


def _names_match(pred: str, gt: str) -> bool:
    """Same person if surnames match and the first initial agrees
    (when both sides supply one)."""
    p = _parse_name(pred)
    g = _parse_name(gt)
    if not p or not g:
        return False
    if p[0] != g[0]:
        return False
    if p[1] and g[1] and p[1][0] != g[1][0]:
        return False
    return True


def _score_first_author(pred: str, gt: str) -> int:
    return int(_names_match(pred, gt))


# ---------------------------------------------------------------------------
# Co-authors
# ---------------------------------------------------------------------------

# Initials regex used to recognize APA-style "Surname, Initials" pairs.
# Matches "G", "G.", "G E", "G. E.", etc. (1-4 capital letters, possibly
# dotted, possibly space-separated).
_INITIALS_RE = re.compile(r"[A-Z]\.?(?:\s*[A-Z]\.?){0,3}")


def _split_apa_chunk(s: str) -> list[str]:
    """Split a single chunk that may itself contain APA pairs.

    Examples
    --------
    "Hinton, G."                          → ["Hinton, G."]
    "Hinton, G., Vinyals, O."             → ["Hinton, G.", "Vinyals, O."]
    "Oriol Vinyals"                       → ["Oriol Vinyals"]
    "Vinyals, Dean"                       → ["Vinyals", "Dean"]   (rare)
    """
    s = s.strip().rstrip(",").strip()
    if not s:
        return []
    if "," not in s:
        return [s]
    chunks = [c.strip() for c in s.split(",") if c.strip()]
    if len(chunks) >= 2 and _INITIALS_RE.fullmatch(chunks[1]):
        # APA: pair (surname, initials)+. If a tail chunk has no
        # following initials, treat it as a display-style name.
        out: list[str] = []
        i = 0
        while i < len(chunks):
            if i + 1 < len(chunks) and _INITIALS_RE.fullmatch(chunks[i + 1]):
                out.append(f"{chunks[i]}, {chunks[i + 1]}")
                i += 2
            else:
                out.append(chunks[i])
                i += 1
        return out
    return chunks


def _split_authors(s: str) -> list[str]:
    """Split a co-author string into individual author names.

    Tolerates "&", "and", ";" as strong separators and "," as either an
    APA-internal separator or a list separator. The two-stage parse
    avoids the ambiguity of "Vinyals, O. & Dean, J." where the inner
    comma is APA-internal but the outer "&" is the list separator.
    """
    s = _nfkc(s).strip()
    if not s:
        return []
    s = re.sub(r"\s+and\s+", " & ", s, flags=re.IGNORECASE)
    parts = re.split(r"\s*(?:&|;)\s*", s)
    out: list[str] = []
    for p in parts:
        for sub in _split_apa_chunk(p):
            norm = sub.lower().rstrip(".").strip()
            if norm in {"et al", "et al."}:
                continue
            out.append(sub)
    return out


def _score_co_authors(pred: str, gt: str) -> int:
    pred_names = _split_authors(pred)
    gt_names = _split_authors(gt)
    if len(pred_names) != len(gt_names):
        return 0
    for p, g in zip(pred_names, gt_names):
        if not _names_match(p, g):
            return 0
    return 1


# ---------------------------------------------------------------------------
# Year
# ---------------------------------------------------------------------------

def _score_year(pred: str, gt: str) -> int:
    p = re.search(r"\d{4}", pred)
    g = re.search(r"\d{4}", gt)
    if not p or not g:
        return 0
    return int(p.group() == g.group())


# ---------------------------------------------------------------------------
# Venue
# ---------------------------------------------------------------------------

# canonical_lower → set of accepted normalized aliases.
# All keys / values must already be _norm_text()'d (lowercase,
# punctuation-free).
_VENUE_ALIASES: dict[str, frozenset[str]] = {
    "advances in neural information processing systems": frozenset({
        "advances in neural information processing systems",
        "neurips", "nips",
        "proc neurips", "proceedings of neurips",
        "proc nips", "proceedings of nips",
    }),
    "international conference on machine learning": frozenset({
        "international conference on machine learning",
        "icml", "proc icml", "proceedings of icml",
        "proceedings of the international conference on machine learning",
    }),
    "international conference on learning representations": frozenset({
        "international conference on learning representations",
        "iclr", "proc iclr", "proceedings of iclr",
    }),
    "proceedings of the aaai conference on artificial intelligence": frozenset({
        "proceedings of the aaai conference on artificial intelligence",
        "aaai", "aaai conference",
        "proc aaai", "proceedings of aaai",
        "the aaai conference on artificial intelligence",
    }),
    "ieee cvf conference on computer vision and pattern recognition": frozenset({
        "ieee cvf conference on computer vision and pattern recognition",
        "ieee conference on computer vision and pattern recognition",
        "cvpr", "ieee cvpr",
        "proc cvpr", "proceedings of cvpr",
        "computer vision and pattern recognition",
    }),
    "ieee cvf international conference on computer vision": frozenset({
        "ieee cvf international conference on computer vision",
        "ieee international conference on computer vision",
        "iccv", "ieee iccv",
        "proc iccv", "proceedings of iccv",
    }),
    "annual meeting of the association for computational linguistics": frozenset({
        "annual meeting of the association for computational linguistics",
        "acl", "proc acl", "proceedings of acl",
        "proceedings of the annual meeting of the association for computational linguistics",
    }),
    "conference on empirical methods in natural language processing": frozenset({
        "conference on empirical methods in natural language processing",
        "emnlp", "proc emnlp", "proceedings of emnlp",
    }),
    "journal of machine learning research": frozenset({
        "journal of machine learning research", "jmlr",
    }),
    "transactions on machine learning research": frozenset({
        "transactions on machine learning research", "tmlr",
    }),
}


def _venue_canonical(s: str) -> str:
    n = _norm_text(s)
    for canonical, aliases in _VENUE_ALIASES.items():
        if n == canonical or n in aliases:
            return canonical
    return n


def _score_venue(pred: str, gt: str) -> int:
    return int(_venue_canonical(pred) == _venue_canonical(gt))


# ---------------------------------------------------------------------------
# Volume / Issue / Pages
# ---------------------------------------------------------------------------

def _norm_volume(s: str) -> str:
    s = _nfkc(s).lower()
    s = re.sub(r"\bvol(\.|ume)?\s*", "", s)
    return _norm_alphanum(s)


def _norm_issue(s: str) -> str:
    s = _nfkc(s).lower()
    s = re.sub(r"\b(no|num|number|iss|issue)\.?\s*", "", s)
    return _norm_alphanum(s)


def _norm_pages(s: str) -> str:
    """Normalize 'pp. 123-145', '123--145', '123–145' to '123-145'."""
    s = _nfkc(s).lower()
    s = re.sub(r"\bpp?\.?\s*", "", s)
    s = re.sub(r"\bpages?\s*", "", s)
    # Unicode dashes (en-dash, em-dash, hyphen-minus, etc.) → '-'
    s = re.sub(r"[‐-―−]", "-", s)
    s = re.sub(r"-+", "-", s)
    s = re.sub(r"\s+", "", s)
    return s


def _score_volume(pred: str, gt: str) -> int:
    return int(_norm_volume(pred) == _norm_volume(gt))


def _score_issue(pred: str, gt: str) -> int:
    return int(_norm_issue(pred) == _norm_issue(gt))


def _score_pages(pred: str, gt: str) -> int:
    return int(_norm_pages(pred) == _norm_pages(gt))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_SCORERS = {
    "title": _score_title,
    "first_author": _score_first_author,
    "co_authors": _score_co_authors,
    "year": _score_year,
    "venue": _score_venue,
    "volume": _score_volume,
    "issue": _score_issue,
    "pages": _score_pages,
}


def score_field(pred: Any, ground_truth: Any, field_name: str) -> int:
    """Return 1 if `pred` matches `ground_truth` for the field, else 0.

    Both arguments are coerced to str (None/NaN treated as empty). Empty
    `pred` against any non-empty truth always scores 0.
    """
    pred_s = "" if pred is None else str(pred).strip()
    gt_s = "" if ground_truth is None else str(ground_truth).strip()
    if not pred_s:
        return 0

    scorer = _SCORERS.get(field_name)
    if scorer is None:
        raise KeyError(f"unknown field: {field_name}")
    return scorer(pred_s, gt_s)


def score_response(
    answer: Any,
    ground_truth: dict[str, str],
) -> dict[str, int]:
    """Score every field in `ground_truth`. Returns ``{field: 0 | 1}``.

    If `answer` is not a dict (e.g. malformed model output), every field
    scores 0. Missing keys in a valid `answer` dict also score 0.
    """
    if not isinstance(answer, dict):
        return {f: 0 for f in ground_truth}
    out: dict[str, int] = {}
    for field_name, gt in ground_truth.items():
        out[field_name] = score_field(answer.get(field_name), gt, field_name)
    return out
