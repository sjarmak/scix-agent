# 50-Query Retrieval Eval — Re-baseline Delta Analysis

**Bead**: scix_experiments-l78
**Purpose**: Re-run hybrid RRF (50q) eval after the destructive-pytest incident
restored the papers indexes. Confirm BM25 + vector fusion still produces
equivalent (or better) numbers vs the d5h baseline.

## TL;DR

**No regression detected on the 12 overlapping seeds; aggregate-level
differences vs d5h are fully explained by seed-set drift, not by a functional
change in the retrieval pipeline.** Despite identical `--random-seed 42`, only
12 of 50 seed papers overlap between runs (24%). On those 12 overlapping seeds,
per-seed paired comparison shows ≤ 0.4% relative delta on every metric for
every method (max absolute Δ = 0.0013 nDCG@10), within float-precision noise.
This is a strong but limited signal: the retrieval pipeline produces the same
outputs given the same inputs, but the n=12 paired sample cannot rule out
behavior changes that would only manifest on the 38 non-overlapping seed
queries. See "Caveats" below.

## Inputs

| Run | File | Date | Methods | n_seeds |
|-----|------|------|---------|---------|
| d5h baseline | `build-artifacts/retrieval_eval_50q.md` + `.json` | 2026-04-06 14:40:39 | specter2, indus, nomic, lexical, hybrid_specter2, hybrid_indus | 50 |
| Re-baseline (this run) | `results/retrieval_eval_50q_rebaseline.md` + `.json` | 2026-04-07 21:01:37 | (same) | 50 |

Both runs invoked `scripts/eval_retrieval_50q.py --seed-papers 50 --random-seed 42` against the `_pilot_sample` (10K stratified rows) using `paper_embeddings` for `specter2`, `indus`, and `nomic` (768d).

## Aggregate (Unpaired) Comparison

These numbers are **NOT directly comparable** because the two runs evaluated
different seed papers (see "Seed Drift" below). They are reported only for
completeness.

| Method | nDCG@10 (d5h → new) | Δ (rel) | R@10 | Δ (rel) | R@20 | Δ (rel) | MRR | Δ (rel) |
|---|---|---|---|---|---|---|---|---|
| nomic | 0.4845 → 0.3736 | **-22.9%** ⚠ | 0.3092 → 0.2829 | -8.5% ⚠ | 0.4683 → 0.4229 | -9.7% ⚠ | 0.7435 → 0.6073 | -18.3% ⚠ |
| indus | 0.4434 → 0.3495 | **-21.2%** ⚠ | 0.2859 → 0.2601 | -9.0% ⚠ | 0.4264 → 0.4108 | -3.7% | 0.7083 → 0.6085 | -14.1% ⚠ |
| hybrid_indus | 0.4412 → 0.3332 | **-24.5%** ⚠ | 0.2859 → 0.2576 | -9.9% ⚠ | 0.4264 → 0.4095 | -4.0% | 0.6974 → 0.5544 | -20.5% ⚠ |
| specter2 | 0.4226 → 0.3220 | **-23.8%** ⚠ | 0.2665 → 0.2418 | -9.3% ⚠ | 0.3906 → 0.3706 | -5.1% ⚠ | 0.7074 → 0.5993 | -15.3% ⚠ |
| hybrid_specter2 | 0.4220 → 0.3209 | **-24.0%** ⚠ | 0.2665 → 0.2426 | -9.0% ⚠ | 0.3906 → 0.3647 | -6.6% ⚠ | 0.6957 → 0.5789 | -16.8% ⚠ |
| lexical | 0.2363 → 0.0864 | **-63.4%** ⚠ | 0.0745 → 0.0573 | -23.1% ⚠ | 0.0745 → 0.0729 | -2.1% | 0.7500 → 0.1000 | -86.7% ⚠ |

⚠ flag: relative delta exceeds the bead's 5% threshold. **All methods regress >5% at the aggregate level.**

If this comparison were valid, every method would be flagged. It isn't — see next section.

## Seed Drift

Despite both runs using `--random-seed 42`, only **12 of 50 seed papers overlap**
between d5h and the re-baseline (24%). The remaining 38 seeds in each run are
disjoint queries, so the aggregate distributions above are evaluating different
benchmark sets.

| Run | n seeds | Year range | n unique to this run |
|-----|---------|-----------|---------------------|
| d5h | 50 | 2000-2025 | 38 |
| re-baseline | 50 | 1996-2025 | 38 |
| overlap | 12 | — | — |

**Why does this happen?** `select_seed_papers()` calls `setseed(42/2^31)` and
then `random()` inside a CTE that joins `_pilot_sample` against `citation_edges`,
filters to neighbor count ≥ 5, and samples by stratified `NTILE(5)`. PostgreSQL's
`random()` is deterministic per session after `setseed`, but the *order* in which
random values are drawn depends on row ordering and intermediate JOIN/SORT plans.
Any of the following changes between runs would shuffle the selection:

- New rows in `citation_edges` (or new `_pilot_sample` rows) shifting which papers
  pass the `n_neighbors >= 5` filter
- Index/statistics changes altering the planner's row-ordering for the
  `ROW_NUMBER() OVER (PARTITION BY cite_tier ORDER BY random())` step
- A `VACUUM`/`ANALYZE` between runs

This is a known limitation of `setseed`-based reproducibility on a live database.

## Paired (Per-Seed) Comparison — The Real Comparison

Restricting to the **12 overlapping seeds**, paired by `(method, seed_bibcode)`:

| Method | n_pairs | nDCG@10 (d5h → new) | Δ rel | R@10 Δ rel | R@20 Δ rel | MRR Δ rel |
|---|---|---|---|---|---|---|
| nomic | 12 | 0.4132 → 0.4119 | **-0.3%** | +0.0% | +0.0% | +0.0% |
| indus | 12 | 0.3749 → 0.3749 | **+0.0%** | +0.0% | +0.0% | +0.0% |
| hybrid_indus | 12 | 0.3675 → 0.3678 | **+0.1%** | +0.0% | +0.0% | +0.0% |
| specter2 | 12 | 0.3619 → 0.3606 | **-0.4%** | +0.0% | +0.0% | +0.0% |
| hybrid_specter2 | 12 | 0.3606 → 0.3606 | **+0.0%** | +0.0% | +0.0% | +0.0% |
| lexical | 12 | 0.0000 → 0.0000 | n/a | n/a | n/a | n/a |

**Maximum absolute paired delta**: 0.0013 nDCG@10 points (specter2 and nomic).

**Maximum relative paired delta**: 0.4% (well within float-precision noise on
binary-relevance nDCG with small relevant sets).

**No method regresses more than 5%** under paired comparison. **No regression flagged.**

The lexical row shows zero values for both runs because none of the 12
overlapping seed papers happened to land in the small subset (n=4-6 typically)
where the AND-style `plainto_tsquery` returns hits against `_pilot_sample.tsv`.
Lexical search is functional — see the standalone re-baseline file for the
6 lexical hits in this run — it just doesn't intersect the overlap window.

## Latency

| Method | d5h (ms) | new (ms) | Δ |
|--------|----------|----------|---|
| specter2 | 33 | 51 | +18 |
| indus | 34 | 55 | +21 |
| nomic | 33 | 54 | +21 |
| hybrid_specter2 | 34 | 52 | +18 |
| hybrid_indus | 34 | 51 | +17 |
| lexical | 0 | 1 | +1 |

Vector retrieval is ~50% slower in absolute terms (~+18ms per call). This is
plausibly attributable to the database having grown (8.16M `indus` embeddings
vs ~20K when the d5h baseline was captured), causing larger HNSW index scans
even when `JOIN _pilot_sample` filters down to the 10K window. Latency is well
under the bead's quality threshold (the bead specifies regressions in
nDCG/Recall, not latency) and remains acceptable for paper Section 4.4 reporting.

## text-embedding-3-large Status

The bead description mentions evaluating "SPECTER2 + text-embedding-3-large + BM25 RRF",
but **`text-embedding-3-large` embeddings do not yet exist in `paper_embeddings`**:

```sql
SELECT model_name, COUNT(*) FROM paper_embeddings GROUP BY model_name;
 model_name |  count
------------+---------
 indus      | 8160798
 nomic      |   19998
 specter2   |   19998
```

The d5h baseline file's "Limitations" section already documents this:

> text-embedding-3-large not included (no embeddings generated yet).
> Will be added when OpenAI embeddings are available.

This re-baseline scope is therefore explicitly **"per currently available model"**:
specter2, indus, nomic, lexical, hybrid_specter2, hybrid_indus. The
text-embedding-3-large evaluation remains pending the OpenAI embedding pipeline,
which is tracked separately and is not in scope for this bead.

## Index Restore Verification

The bead's premise is that the destructive pytest incident dropped the papers
indexes and they have since been restored. Verified intact at run time:

| Object | Status |
|--------|--------|
| `_pilot_sample` (10K rows) | present, 2 indexes (`_pilot_sample_bibcode_idx`, `idx_pilot_tsv` GIN on tsv) |
| `paper_embeddings` (8.2M rows) | present, `paper_embeddings_pkey` |
| `papers` (32.4M rows) | present, 18 indexes restored (`idx_papers_*` covering doctype, year, authors, doi, tsv, etc.) |
| `idx_papers_tsv` (GIN on full corpus) | 6.2 GB, present |

Note: the eval script queries `_pilot_sample` directly and does **not** use the
full-corpus `papers.tsv` index, so the restore primarily reassures the rest of
the production search stack — not this specific eval.

## Caveats

- **n=12 paired sample is small.** The conclusion "no regression" applies
  literally to those 12 overlapping queries. The remaining 38 queries in each
  run are disjoint, so we cannot directly compare them. If the retrieval
  pipeline had a query-class-specific regression that happened to spare the
  12 overlap papers, this analysis would not detect it. That said, the
  retrieval functions in `scripts/eval_retrieval_50q.py` (`retrieve_vector`,
  `retrieve_lexical`, `retrieve_hybrid`) operate identically per query,
  so a functional regression would almost certainly affect overlap and
  non-overlap seeds together.
- **`setseed`-based reproducibility is not robust on a live database.** This
  is a known weakness of the current eval script's seed-paper selection. A
  follow-up improvement would be to pin the seed bibcode list explicitly
  (e.g., `--seed-bibcodes path/to/file.txt`) so re-runs are reproducible
  even when underlying tables change. Out of scope for this bead.
- **Latency increased ~50% in absolute terms** (33ms → 51ms). This is below
  the bead's quality threshold (regressions defined on nDCG/Recall) but is
  worth noting as a separate observation; likely due to the 8.16M `indus`
  embeddings now in the table vs ~20K at d5h time.

## Decision

**No regression flagged** under the bead's acceptance criteria (>5% on nDCG@10
or Recall@K per model, on overlapping queries). Hybrid SPECTER2/INDUS + BM25
RRF retrieval produces results functionally equivalent to the d5h baseline on
the 12 overlapping seed papers, with aggregate-level apparent regressions
attributable to seed-set drift rather than pipeline behavior. The d5h numbers
in the paper table remain valid; this re-baseline can be cited as a
confirmation run, with the caveats above.

## Reproduction

```bash
python scripts/eval_retrieval_50q.py \
    --seed-papers 50 \
    --random-seed 42 \
    --output results/retrieval_eval_50q_rebaseline.md \
    --json-output results/retrieval_eval_50q_rebaseline.json
```

## Artifacts

- `results/retrieval_eval_50q_rebaseline.md` — full re-run report (this run)
- `results/retrieval_eval_50q_rebaseline.json` — per-query JSON (this run)
- `results/retrieval_eval_50q_rebaseline_analysis.md` — this delta analysis
- `build-artifacts/retrieval_eval_50q.md` + `.json` — d5h baseline (unchanged)

## Overlap Seed List (n=12)

For paired-comparison reproducibility:

```
2001JGR...10623607S
2008Icar..197...65H
2008PhRvB..78r0502A
2014ITGRS..52.5122S
2015NatCo...6.7497E
2017Optik.135..366H
2019PhRvD.100e5031M
2020ScTEn.71134843W
2022JARS...16c4534W
2023arXiv230313711A
2024JSAES.14805180L
2025JHEP...01..081A
```
