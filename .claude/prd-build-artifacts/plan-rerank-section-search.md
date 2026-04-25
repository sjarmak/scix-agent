# Plan — rerank-section-search (M5)

## New `search_within_paper` signature

```python
def search_within_paper(
    conn: psycopg.Connection,
    bibcode: str,
    query: str,
    *,
    top_k: int = 20,
    use_rerank: bool = True,
) -> SearchResult:
```

Positional `(conn, bibcode, query)` retained to satisfy existing callers and
tests. New keyword-only `top_k` and `use_rerank`.

## Algorithm

1. Single SQL pass: fetch `body, title, ts_headline(body, query)` for the
   bibcode (current behaviour) AND fetch `plainto_tsquery('english', query)`
   so we can score sections in Python via per-section `ts_rank`.
   - Actually cheaper: do the existing query (returns body + headline), then
     in Python compute per-section ts_rank by issuing one SQL call:
     `SELECT ts_rank(to_tsvector('english', sec_text), plainto_tsquery('english', %s))`
     for each section. With ≤8 sections per paper this is fine, but better:
     batch via `SELECT ts_rank(to_tsvector('english', x), plainto_tsquery('english', %s))
     FROM unnest(%s::text[]) AS t(x)` returning one row per section.
   - Even simpler and entirely deterministic: compute a Python-side BM25-ish
     score per section by counting query token occurrences (case-insensitive,
     word boundaries) and weighting by `1/log(section_length+1)`. This is
     *not* a semantic judgement — it's a transparent mechanical proxy for
     ts_rank. Acceptable per ZFC ("deterministic ranking with explicit
     tiebreaker rules").
   - Decision: use SQL `ts_rank` via the unnest batch query so we get real
     PostgreSQL-grade BM25-ish scoring. One extra round-trip per call,
     negligible vs the rerank-model latency.
2. Parse sections via `parse_sections(body)`.
3. For each section, compute `ts_rank` against the query.
4. Take top-K (default 20) by ts_rank — these are the BM25 candidates.
5. If `use_rerank=True` AND the default reranker resolves (env != 'off'),
   call `CrossEncoderReranker(query, candidates_as_paper_dicts, top_n=3)`.
   Otherwise just take top-3 by ts_rank.
6. Build `sections` list of up to 3 entries: `{section_name, score, snippet}`.
   - `snippet` is built in Python: locate first query-token match in the
     section text and slice ±150 chars of context.
7. Top-1 section's snippet (or the full ts_headline output, whichever the
   ADR-006 guard requires) populates `headline`.
8. Preserve ADR-006 guard branch — when LaTeX-derived, still apply
   snippet budget to the final snippet AND to each section's snippet
   in `sections`.

## Reranker construction

The signature has `use_rerank: bool = True` but the *actual* reranker is
constructed via the same `_resolve_default_reranker_model()` env lookup as
`mcp_server`. Since `search.py` shouldn't import from `mcp_server` (layering),
duplicate the small lookup helper inline in `search.py` (or factor it later).
We'll keep it minimal: reuse the existing `_RERANK_MODEL_ALIASES` dict by
defining a private copy in `search.py`.

When env is 'off' (default) and `use_rerank=True`, the function silently
skips the rerank step (just truncates ts_rank order to top-3). This is the
ship-with-default-off behaviour the unit description requires.

## SearchResult shape (AC2)

```python
{
    "bibcode": str,
    "title": str,
    "headline": str,                      # backward-compat: top-1 snippet
    "has_body": True,
    "sections": [                         # NEW: up to 3 entries
        {"section_name": str, "score": float, "snippet": str},
        ...
    ],
}
```

`metadata` gains `{"rerank_used": bool, "candidate_count": int}`.

## Synthetic 20-entry fixture

`tests/fixtures/within_paper_rerank_gold_20.jsonl` — each line:

```json
{"paper_body": "...", "query": "...", "gold_section_idx": 2}
```

Each paper_body is a hand-crafted 2-4KB string with 5-8 sections following
section_parser headers (`Introduction`, `Methods`, `Observations`, `Results`,
`Discussion`, `Conclusions`). The gold section contains query terms in a
focused, dense way; other sections may share some vocabulary but the gold
section is unambiguously the best answer.

20 queries spanning realistic astronomy topics: dark matter halo
mass, exoplanet atmosphere transmission, gravitational lensing time delay,
supernova light curve fit, galaxy cluster X-ray, cosmic microwave background
polarization, neutron star merger gravitational wave, pulsar timing array,
black hole spin measurement, stellar metallicity gradient, redshift survey,
quasar variability, etc.

## Tests

`tests/test_search_within_paper_rerank.py`:

1. `test_signature_has_use_rerank_and_top_k` — `inspect.signature` checks.
2. `test_search_result_has_sections_field` — mock cursor; assert
   `papers[0]["sections"]` is a list and `papers[0]["headline"]` still string.
3. `test_backward_compat_positional_call` — calling
   `search_within_paper(conn, bibcode, query)` still works.
4. `test_ndcg3_baseline_vs_rerank` — runs both modes against the 20-entry
   fixture using a stub conn that builds queries against `parse_sections` of
   the fixture body. Computes nDCG@3 for each. If delta >= 0.05 → assert
   improvement; else log negative result + write `results/within_paper_rerank_eval.md`.
5. `test_p95_latency_under_500ms` — runs `use_rerank=True` (MiniLM) over
   all 20 fixture queries, asserts p95 <= 500ms.

For the ndcg3 / latency tests we need a fake `psycopg.Connection` that
returns the fixture body when queried. Pattern: `MagicMock` for `conn` whose
`cursor()` returns context-managed cursors that return a row with body+title
+headline+(via unnest query) per-section ts_rank. We control the section
ranks by computing ts_rank-equivalent via a Python proxy that mirrors what
PostgreSQL would do (count of query tokens in section, normalized by section
length). This is acceptable — we are testing the *rerank delta* not the
SQL ts_rank fidelity.

## Eval methodology

For each fixture entry:
- Run `search_within_paper(conn, bib, query, use_rerank=False)` → top-3
  sections by BM25 proxy. Compute DCG@3 with rel(i)=1 if section_idx==gold
  else 0. Compute nDCG@3.
- Run `search_within_paper(conn, bib, query, use_rerank=True)` after
  setting `SCIX_RERANK_DEFAULT_MODEL=minilm`. Same nDCG@3.

Average across 20. Delta = rerank − baseline.

Go/no-go: delta >= 0.05 → ship use_rerank=True default ON in code, env still
default 'off'. delta < 0.05 → ship same defaults but explicitly document
negative result.

## Eval doc (results/within_paper_rerank_eval.md)

Markdown with:
- Methodology summary
- Baseline nDCG@3
- Reranked nDCG@3 (MiniLM)
- p95 latency (ms)
- Go/no-go recommendation
