# Lexical candidate-pool cap — broad-term stress eval (bead zsou)

**Date:** 2026-05-29 · **Bead:** `scix_experiments-zsou` (a2fp follow-up) ·
**Raw data:** `docs/eval/lexical_recall_pool_stress_2026-05.json` ·
**Gold set:** `eval/lexical_stress_16q.jsonl`

## Why this exists

a2fp (`docs/eval/lexical_recall_pool_2026-05.json`) found the
`lexical_search` candidate-pool cap met its `<=2pp nDCG@10` acceptance at the
default — but **vacuously**. `plainto_tsquery` ANDs all tokens, so the curated
multi-term 50q gold set's largest match set is 2192 rows; the cap (then 5000)
engaged on **0** queries at the default. The eval could not measure the cap's
recall cost because the gold set contained no broad single-token queries — the
exact case the cap exists for.

This eval closes that gap with a 16-query stress set of high-frequency
single/low-token terms (`galaxy` ~344K matches, `spectroscopy` ~852K, …), every
one of which exceeds the cap.

## Gold methodology (read before interpreting)

`gold_bibcodes` for each query is the **uncapped (`SCIX_LEXICAL_POOL=INF`)
`lexical_search` top-20** — the recall *ceiling*. The eval measures how much of
that ceiling each finite pool preserves. This is the correct, non-vacuous
reference for a recall **cap**: the cap's job is to approximate the uncapped
ranking cheaply, so divergence from it *is* the cap's cost. INF reproduced the
gold at exactly Recall@20 = 1.000, confirming the reference is stable (no
tie-shuffle noise).

**Caveat — this is not a citation-relevance judgment.** On broad single-token
terms, `ts_rank_cd` ranks short documents (press releases `*pres*`, NSF
proposals `*nsf*`/`*prop*`, theses `*PhDT*`) above seminal high-citation
articles, because term density is higher in short text. So the ceiling itself is
low-quality. That is a separate `ts_rank_cd`/corpus-composition problem, **out of
scope for the cap** — but it means "recovering more of the ceiling" is not the
same as "better retrieval." See follow-up below.

## Results

16 broad queries, paired delta vs the INF ceiling:

| pool        | nDCG@10 | Recall@20 | nDCG drop | Recall drop | max ms | cap hits |
|------------:|--------:|----------:|----------:|------------:|-------:|---------:|
| 5000 (old)  | 0.418   | 0.153     | +58.18 pp | +84.69 pp   | ~0.7s  | 16/16    |
| 15000       | 0.683   | 0.350     | +31.65 pp | +65.00 pp   | ~1.2s  | 16/16    |
| **30000 (new)** | **0.860** | **0.628** | **+14.05 pp** | **+37.19 pp** | **~1.5s** | **15/16** |
| 50000       | 0.928   | 0.762     | +7.16 pp  | +23.75 pp   | ~3.5s  | 10/16    |
| INF         | 1.000   | 1.000     | 0         | 0           | ~82s   | 0/16     |

(max-ms is worst-case across the 16 queries and varies with cache warmth; INF
hit ~82–90s, far over the 30s prod `statement_timeout` — uncapped is not
viable in prod.)

## Decision: raise `_LEXICAL_POOL_DEFAULT` 5000 → 30000

The old 5000 default recovered only ~15% of the uncapped top-20 — it samples
<1.5% of `galaxy`'s match set in bitmap-heap (TID/ingestion) order, the worst
ordering for recall. **30000 is the knee:** Recall@20 jumps 0.153→0.628 (4×) and
the nDCG@10 drop falls 58→14 pp, at ~1.5s worst-case. 50000 buys only +13 pp
recall for +40% latency.

Latency lands only on broad queries (narrow queries whose match set is below the
cap are untouched), but `hybrid_search` runs its lanes **serially**, so the
~0.8s→1.5s increase adds directly to hybrid cost on broad queries.
`SCIX_LEXICAL_POOL` remains a live env override for ops retuning without redeploy.

## On the "ACCEPTANCE [FAIL]" line

The script prints `FAIL` because the 14.05 pp drop exceeds the **2 pp** threshold
inherited from a2fp. That threshold is **structurally unmeetable on this
ceiling-gold stress set for any finite cap** — meeting it would require
reproducing the full uncapped top-20, i.e. effectively no cap. It is not a
signal that this change is incomplete. zsou's own acceptance — *cap engaged > 0
at the default, and a real (non-vacuous) measured drop* — is **met**, and the
default now sits at the evidence-based knee.

## Follow-ups

- **`ts_rank_cd` ranks short non-article docs above seminal papers on broad
  terms.** Independent of the cap; affects lexical-lane quality directly. Worth
  a separate bead (e.g. length-normalization / doctype down-weighting, or
  leaning on the vector lane for broad-concept recall in RRF).
- The cap is a blunt instrument (LIMIT-before-rank in TID order). A relevance-
  aware first pass (rank-then-truncate, or a cheaper pre-rank) would beat raising
  a flat cap, but is out of scope here.
