# ts_rank_cd normalization flag — broad-term lexical quality eval (bead q9k5)

**Date:** 2026-06-14 · **Bead:** `scix_experiments-q9k5` (zsou follow-up) ·
**Raw data:** `docs/eval/lexical_rank_flag_2026-06.json` ·
**Harness:** `scripts/eval_lexical_rank_flag.py` ·
**Gold set:** `eval/lexical_stress_16q.jsonl`

## Why this exists

`zsou` (`docs/eval/lexical_recall_pool_stress_2026-05.md`) found that on broad
single-token terms the `lexical_search` top-20 is dominated by short non-article
docs — press releases (`*pres*`), NSF proposals (`*nsf*`/`*prop*`), theses
(`*PhDT*`) — because `ts_rank_cd` rewards term *density* and short text has high
density. Seminal high-citation articles rank below or outside the top-20. That
is **orthogonal to the candidate-pool cap** zsou measured (cap = which slice
gets ranked; this = how ranking scores short vs long docs).

q9k5 asks: does a length-aware `ts_rank_cd` normalization flag fix it? The
default flag `32 = rank/(rank+1)` is a monotonic squash into `[0,1)` that does
**not** length-normalize — verified synthetically in q9k5 comment gc-351512: a
short dense doc still outranks a long one on the same term. The length-aware
bits OR onto 32: `33 = 32|1` adds `/(1+log(length))`, `48 = 32|16` adds
`/(1+log(unique words))`.

## Methodology (read before interpreting)

For each flag the harness sets `SCIX_LEXICAL_RANK_FLAG` and runs every query
through the real `scix.search.lexical_search` (same code path the MCP serves),
with the candidate pool pinned at `INF` (uncapped) so `ts_rank_cd` ranks the
**full** match set under every flag. The candidate set is then identical and
complete across flags, so the rank flag is the only variable — this removes the
cap confound (at the prod cap of 30000 the TID-ordered slice may not even
contain the seminal articles, so a null result there could not distinguish
"flag does not help" from "seminal docs were capped out before ranking").

There is **no relevance gold** for these broad terms (the `gold_bibcodes` in the
stress set are the short-doc recall *ceiling*, not a relevance judgment — see
zsou), so quality is measured by proxies over the prod corpus:

- `article_fraction` — share of top-20 with `doctype='article'` (↑ better)
- `short_doc_fraction` — share in pressrelease/proposal/phdthesis/mastersthesis (↓ better)
- `n_seminal` — count with `citation_count >= 500` (↑ better: seminal articles climbing in)
- `median_citation` — over-correction guard (swapping short junk for long *low-citation* junk shows no lift)

**Decision rule (closes the bead either way):** a candidate flag is adopted iff
vs flag 32 it raises `article_fraction` AND regresses neither `median_citation`
nor `n_seminal`. Otherwise the negative result stands: keep flag 32 and accept
that broad-concept recall is the vector lane's job in RRF.

## Results

16 broad single-token queries, pool=`INF`, seminal = `citation_count >= 500`:

| flag | article_frac | short_frac | seminal/q | median_cit |
|------|-------------:|-----------:|----------:|-----------:|
| **32** (base) | 0.237 | 0.603 | **0.38** | 1.1 |
| 33 (`32\|1`)   | 0.297 | 0.434 | **0.00** | 1.4 |
| 48 (`32\|16`)  | 0.328 | 0.422 | **0.00** | 1.4 |

Deltas vs baseline: flag 33 `article +0.059, short −0.169, seminal −0.375, med_cit +0.3`;
flag 48 `article +0.091, short −0.181, seminal −0.375, med_cit +0.3`.

## Verdict: KEEP flag 32 (negative result)

The length-aware flags do exactly the **first half** of what was hoped — they
cut `short_doc_fraction` (0.60 → ~0.43) and raise `article_fraction` (0.24 →
0.30–0.33). But they fail the second half: **`n_seminal` collapses from 0.38/q
to 0.00/q**. Per-query detail (`per_query` in the JSON) makes the mechanism
plain — under flag 32, six seminal articles surface across the set (cosmic
microwave background ×1, accretion disk ×1, stellar evolution ×2, interstellar
medium ×2); under flags 33/48 **every** query drops to zero seminal. Seminal
papers tend to have longer title+abstract text, so log-length damping evicts the
exact documents we wanted to promote. The flags swap short non-article junk for
long *low-citation* junk: `median_citation` stays ~1 (and is 0.0 on most
queries) under all three flags — broad single-token lexical top-20 is
citation-poor regardless of normalization.

So no flag clears the bar. The flag stays at `32` in `scix.search`
(`_LEXICAL_RANK_FLAG_DEFAULT`); no production behavior changes.

**Conclusion (the bead's stated fallback):** broad-concept recall is the vector
lane's job in RRF, not the lexical lane's. `ts_rank_cd` cannot be tuned to
surface seminal articles on broad single-token terms without a length signal
that simultaneously demotes them — the two goals are in direct tension under a
density-based scorer. The lexical lane's value is precise multi-term and exact
phrase matching; broad single concepts route to the INDUS dense lane, and RRF
fuses the two. No lexical change is warranted.

## What shipped

- `src/scix/search.py`: a `SCIX_LEXICAL_RANK_FLAG` env hook
  (`_resolve_lexical_rank_flag`, mirrors `SCIX_LEXICAL_POOL`) wired into both
  `lexical_search` and `lexical_search_body`. Default unchanged at `32`; the
  hook exists so this A/B (or a future one) re-runs without a code change and so
  this negative result can be cheaply re-litigated if the corpus composition
  shifts.
- `scripts/eval_lexical_rank_flag.py`: the A/B harness (reuses
  `eval_retrieval_50q.load_queries`; prod-health preflight gates on postgres
  liveness, a MemAvailable floor, and cgroup `memory.pressure`; bounds the
  connection at `work_mem=256MB`, `max_parallel_workers_per_gather=0`, and a
  180s `statement_timeout`).
- This doc + `docs/eval/lexical_rank_flag_2026-06.json`.
