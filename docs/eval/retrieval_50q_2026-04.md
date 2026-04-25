# 50-Query Retrieval Evaluation — 2026-04

> Status: **template** — populated once `section_embeddings` is filled by the
> upstream encode pipeline (PRD `embedding-pipeline`). Until then, the
> baseline numbers stand alone and section/fused rows show the
> `section_embeddings_empty` skip marker.

## Overview

This evaluation compares three retrieval modes against a 50-query
hand-curated gold set (`eval/retrieval_50q.jsonl`, sibling unit
`eval-queries-curate`). The goal is to quantify whether the new
`section_retrieval` MCP tool — which retrieves over per-section embeddings
plus per-section BM25 — adds value on top of the existing paper-level
`hybrid_search` baseline, and whether fusing the two further improves
ranking quality.

Modes evaluated:

1. **baseline** — `scix.search.hybrid_search` (NASA INDUS dense over
   `paper_embeddings` + `papers.search_tsv` BM25 + `papers.body` GIN BM25),
   fused via Reciprocal Rank Fusion (k=60).
2. **section** — the new `section_retrieval` MCP tool: nomic-embed-text-v1.5
   dense over `section_embeddings` (halfvec(1024) HNSW) plus
   `papers_fulltext.sections_tsv` BM25, RRF k=60. Multiple matching sections
   from the same paper collapse to one bibcode at the earliest hit.
3. **fused** — the ranked bibcode lists from (1) and (2) fused with RRF
   k=60.

Buckets (annotated on each query in the gold set):

- `title_matchable` — sparse retrieval should already win.
- `concept` — dense / semantic retrieval should help.
- `method` — methods-section coverage should benefit `section`.
- `author_specific` — first-author / collaboration queries.

Disciplines: `astrophysics`, `planetary`, `earth`, `heliophysics`.

## Methodology

For each query in the gold set we:

1. Run each mode and take the top-50 unique bibcodes.
2. Score the ranking against `gold_bibcodes` (binary relevance) with:
   - **nDCG@10** — `log2(rank+1)` discount; ideal = 1's at the top capped at
     `min(|gold|, 10)`.
   - **MRR@10** — `1 / rank` of the first relevant bibcode in the top-10,
     `0` if none.
   - **Recall@50** — `|gold ∩ top-50| / |gold|`.
3. Queries with `gold_bibcodes == []` are excluded from the per-mode
   averages rather than scored as zero, so the "n_scored" counts reported
   alongside each metric reflect the eligible subset.

The driver script is `scripts/eval_retrieval_50q.py`. To regenerate this
report's data:

```bash
# Schema-only — no DB or model load (useful while corpus encode is pending).
python scripts/eval_retrieval_50q.py --dry-run

# Live, all three modes (default).
python scripts/eval_retrieval_50q.py \
    --queries eval/retrieval_50q.jsonl \
    --output  docs/eval/retrieval_50q_2026-04.json \
    --modes   baseline,section,fused \
    --k       10
```

Local-only encoding is enforced: the baseline uses NASA INDUS via
`scix.embed.load_model("indus")`, and `section_retrieval` uses
`nomic-embed-text-v1.5` via `scix.embeddings.section_pipeline._load_model`.
No paid-API SDKs are imported.

## Results

> Numbers below are **placeholders** — they will be filled in from
> `docs/eval/retrieval_50q_2026-04.json` once the live run is executed. The
> shape of the table is fixed so downstream comparisons (e.g. ADASS 2026
> writeup §4.4) can wire against it directly.

### Overall

| Mode     | nDCG@10 | MRR@10 | Recall@50 | n_scored |
|----------|--------:|-------:|----------:|---------:|
| baseline | TBD     | TBD    | TBD       | TBD      |
| section  | TBD     | TBD    | TBD       | TBD      |
| fused    | TBD     | TBD    | TBD       | TBD      |

### By bucket

#### title_matchable

| Mode     | nDCG@10 | MRR@10 | Recall@50 |
|----------|--------:|-------:|----------:|
| baseline | TBD     | TBD    | TBD       |
| section  | TBD     | TBD    | TBD       |
| fused    | TBD     | TBD    | TBD       |

#### concept

| Mode     | nDCG@10 | MRR@10 | Recall@50 |
|----------|--------:|-------:|----------:|
| baseline | TBD     | TBD    | TBD       |
| section  | TBD     | TBD    | TBD       |
| fused    | TBD     | TBD    | TBD       |

#### method

| Mode     | nDCG@10 | MRR@10 | Recall@50 |
|----------|--------:|-------:|----------:|
| baseline | TBD     | TBD    | TBD       |
| section  | TBD     | TBD    | TBD       |
| fused    | TBD     | TBD    | TBD       |

#### author_specific

| Mode     | nDCG@10 | MRR@10 | Recall@50 |
|----------|--------:|-------:|----------:|
| baseline | TBD     | TBD    | TBD       |
| section  | TBD     | TBD    | TBD       |
| fused    | TBD     | TBD    | TBD       |

## Discussion

> Populate once results land. Expected hypotheses:

- **`title_matchable`** — baseline should win or tie; the gold set is
  authored so titles match. Section adds little because each section
  duplicates body-tsv signal.
- **`concept`** — section should outperform baseline because per-section
  embeddings localize semantically dense passages that title+abstract
  averaging dilutes.
- **`method`** — the largest expected win for section, since methods
  sections are typically where the relevant prose lives but rarely show up
  in title/abstract terms.
- **`author_specific`** — neither mode should move much; first-author /
  collaboration filters are out of scope for both retrievers and would be
  better served by structured filters in `hybrid_search`.
- **fused** — should not be worse than the better of (baseline, section)
  on any bucket. RRF is rank-only, so the rare case of fused < max
  indicates highly disjoint top-10 sets where neither has a stable winner.

Once numbers are populated, also state which differences exceed a
sensible noise floor for n=50 (paired-bootstrap CI suggested).

## Limitations

- **Full-corpus section encode is upstream-blocked.** The full-corpus pass
  over `section_embeddings` depends on the parser PRD shipping. Until then,
  `section_retrieval` runs against an empty (or partial) table and the
  driver script emits the `skipped_reason: section_embeddings_empty`
  marker on `section` and `fused` modes. Re-run the script after the
  encode lands to populate this report.
- **Gold set size (n=50) is small.** Per-bucket aggregates are at most
  ~12 queries each. Treat per-bucket deltas as directional unless paired
  bootstrap shows them outside ±0.05.
- **Binary relevance only.** No graded labels, so nDCG@10 here is
  essentially a position-discounted hit rate.
- **No author-graph signal.** `author_specific` queries are evaluated
  through pure text retrieval, which is by design — adding structured
  author filtering is a separate work unit.
- **Encoder coverage.** The baseline encodes queries with the same INDUS
  model that produced `paper_embeddings`; the section mode uses the local
  nomic model that produced `section_embeddings`. Cross-mode ranking
  comparison is meaningful precisely because each mode encodes against the
  same model used to populate its index — but absolute score scales
  differ, which is fine because RRF is rank-only.

## Pointers

- Driver: `scripts/eval_retrieval_50q.py`
- Tests: `tests/test_eval_retrieval_50q.py`
- Gold set: `eval/retrieval_50q.jsonl`
- JSON output schema: see `--output` (defaults to
  `docs/eval/retrieval_50q_2026-04.json`)
- Sibling units in this PRD: `schema-section-embeddings`,
  `embedding-pipeline`, `mcp-section-retrieval`, `eval-queries-curate`,
  `tool-audit`.
