# Data and code for "Cross-Model Memorization Thresholds in Citation Generation"

This archive contains the data and analysis code for the study *Cross-Model
Memorization Thresholds in Citation Generation: Evidence from Field-Level
Cloze on Bibliographic Records*.

The study measures how reliably large language models reproduce bibliographic
records. Papers are sampled from OpenAlex and stratified by citation count;
within each paper one bibliographic field at a time is masked, and a model is
asked to recover it (a field-level cloze task). Recovery is scored by strict
binary match. The protocol is run across seven model conditions and two
research domains (computer-science AI and oncology).

## Contents

```
data/
  runs_combined.parquet       all cloze trial results (every condition)
  processed/                  the two fixed paper samples ("master" files)
    20260511T112255Z_cs-ai_log10_pilot.parquet
    20260513T094345Z_med-onc_log10_pilot.parquet
  analysis/                   analysis outputs (bootstrap CIs, GEE results)
  batches/                    Gemini Batch API job state files
scripts/                      Python pipeline (sampling, cloze, analysis)
pyproject.toml, uv.lock        software environment specification
```

## Master samples (`data/processed/`)

The two master parquets are the fixed, pre-registered paper samples; all
results derive from them. Each row is one paper drawn from OpenAlex, with its
bibliographic metadata (title, authors, venue, volume, issue, pages, year,
citation count, etc.). Sampling is stratified into six half-decade
`log10(citation count)` bands.

- `...cs-ai...` — 1,118 computer-science (AI) papers
- `...med-onc...` — 1,079 oncology papers

The cloze ground-truth values are taken from these columns. See
`scripts/fetch_openalex.py` for the full master schema and how the sample is
drawn.

## `data/runs_combined.parquet`

The main results table: one row per cloze trial, i.e. one (paper, masked
field) pair evaluated under one model condition. 111,279 rows. It is the
concatenation of the fourteen per-condition run files (see
`scripts/combine_runs.py`).

| column | description |
|---|---|
| `openalex_id` | OpenAlex work ID of the paper |
| `picked_index` | deterministic selection index within the master sample |
| `field` | masked bibliographic field: `title`, `first_author`, `co_authors`, `year`, `venue`, `volume`, `issue`, `pages` |
| `tier` | citation-count stratum (half-decade `log10` band) |
| `cited_by_count` | citation count of the paper (OpenAlex) |
| `ground_truth` | true value of the masked field |
| `prediction` | value recovered by the model |
| `raw_text` | raw model response |
| `thinking` | reasoning trace (reasoning-enabled conditions only; else empty) |
| `score` | 1 if `prediction` strictly matches `ground_truth`, else 0 |
| `elapsed_s` | wall-clock seconds for the model call |
| `error` | error message if the API call failed, else null |
| `model` | model identifier |
| `reasoning` | whether reasoning/"thinking" mode was enabled (boolean) |
| `seed` | RNG seed for deterministic paper selection |
| `start` | start offset of the deterministic sample slice |
| `master_parquet` | filename of the master sample this trial draws from |
| `domain` | `cs-ai` or `oncology` |

### Conditions

Seven conditions per domain (fourteen in total), as `model` x `reasoning`:

| model | reasoning off | reasoning on |
|---|:---:|:---:|
| `gemini-3-flash-preview` | yes | yes |
| `gpt-5.4-mini` | yes | yes |
| `gpt-5.4` (full-tier size control) | yes | — |
| `deepseek-v4-flash` | yes | yes |

Select a single condition by filtering, e.g.:

```python
import pandas as pd
df = pd.read_parquet("data/runs_combined.parquet")
cond = df[(df.domain == "cs-ai") &
          (df.model == "gpt-5.4-mini") &
          (df.reasoning == True)]
```

The `gpt-5.4-mini` reasoning condition was executed as two non-overlapping
half-runs (sample offsets 0 and 100) and merged; it appears here as a single
condition. Errored API calls are retained with `score = 0` and a non-null
`error`; they were under 0.1% of trials in every condition.

## Analysis outputs (`data/analysis/`)

- `bootstrap_inflections.json`, `medicine_bootstrap.json` — paper-cluster
  bootstrap estimates of the per-(condition, field) logistic inflection point
  (the citation count at which recovery probability crosses 0.5), with 95%
  confidence intervals, for CS-AI and oncology respectively.
- `gee_results.json`, `medicine_gee.json` — cluster-robust generalised
  estimating equation (GEE) coefficients.
- `runs_cs-ai.json`, `runs_medicine.json` — manifests mapping each condition
  to its source run file.

## Software environment

The pipeline targets Python 3.12+ and is managed with
[uv](https://docs.astral.sh/uv/). To recreate the environment:

```
uv sync
```

`scripts/` contains the full pipeline: `fetch_openalex.py` (sampling),
`cloze.py` (prompt construction), `smoke_cloze.py` (the cloze runner),
`llm.py` (model backends), `combine_runs.py` (assembly of
`runs_combined.parquet`), and the analysis scripts `analyze.py`,
`bootstrap_ci.py`, `gee_analysis.py`, `compare_models.py`, and
`cross_domain_figure.py`. The analysis scripts operate on per-condition run
files; `runs_combined.parquet` is provided as a single analysis-ready table
and can be split back by `(domain, model, reasoning)` if needed.

## Data source

Bibliographic metadata is derived from [OpenAlex](https://openalex.org), a
public, openly licensed (CC0) scholarly database.

## License

The data files in `data/` are released under the Creative Commons
Attribution 4.0 International (CC-BY-4.0) license. The code in `scripts/` is
released under the MIT license.
