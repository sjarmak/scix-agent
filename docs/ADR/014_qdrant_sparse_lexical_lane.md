# ADR-014: Evaluate a Qdrant Sparse BM25 Lexical Lane to Replace the Postgres `scix_english` tsvector Lane

- **Status**: Proposed — **pilot ran 2026-06-11, gate FAILED** (`results/sparse_pilot_eval.md`). Plain BM25 is **not adopted**. The result is promising enough overall, and the failure mode (scientific-token tokenization) addressable enough, that the ADR stays Proposed pending a confound-corrected Phase 2 rather than falling back to the RUM alternative. See "Pilot result" below.
- **Deciders**: Stephanie Jarmak (operator approval at every external step); investigation in the 2026-06-11 retrieval session.
- **Scope**: The **lexical (title/abstract/keywords) lane** of `hybrid_search()` (`src/scix/search.py::lexical_search`). Out of scope for this ADR: the body BM25 lane (`lexical_search_body` over `ix_papers_body_tsv`), the section BM25 lane (`papers_fulltext.sections_tsv`), and the dense lane (already on Qdrant per ADR-013). Those may follow if the lexical pilot succeeds, under their own decisions.
- **Related beads**: `i5oa` (this work — ADR + pilot), `rpjj` (RUM cutover — the alternative fix this would obsolete for the title/abstract lane), `zpm4` (section-BM25 structural-ceiling findings that motivate moving off GIN-ranked-top-k).
- **Related ADRs**: ADR-013 (dense lane on Qdrant — establishes the Qdrant substrate, the pilot discipline, and the named-vector/payload patterns this builds on), ADR-008 (Qdrant payload schema), ADR-010 (`sections_tsv` expression index — the GIN lane this pattern would eventually challenge).
- **Supersedes**: nothing yet. If accepted, narrows migration 003's `idx_papers_tsv` / `scix_english` lane to a fallback role for the title/abstract lexical lane only.

## Context

Today all three lexical lanes are PostgreSQL full-text search: `tsvector`
matched with `@@`, ranked with `ts_rank_cd`/`ts_rank` over GIN indexes. This
is **not** Okapi BM25 — `ts_rank_cd` has no document-length normalization
(BM25's `b`) and no IDF saturation (`k1`). Two structural costs follow:

1. **GIN cannot return ranked top-k from the index.** A GIN index answers
   *membership* (which rows match) but stores nothing rank-relevant, so
   `ORDER BY ts_rank_cd(...) LIMIT k` must fetch *every* matching row, score
   it on the heap, sort, and truncate. For broad queries (`galaxy`, ~344k
   matches) this is unworkable, so the code caps the candidate pool at 30k
   (`SCIX_LEXICAL_POOL` / `SCIX_SECTIONS_POOL`) — a recall ceiling that is a
   workaround for an index limitation, not a relevance decision. The standing
   fix for this is the RUM cutover (`rpjj`); a Qdrant sparse lane is an
   alternative that also removes the cap.
2. **No length normalization** hurts a corpus whose lexical fields range from
   a 10-word title to a multi-thousand-word abstract+keywords blob.

ADR-013 put a Qdrant server (1.18.2, 127.0.0.1, NVMe) in the stack for the
dense lane and proved rebuild-from-Postgres as a recovery strategy. Qdrant
1.18.2 supports sparse vectors (since 1.7.0), the Query API with
prefetch+fusion and the IDF modifier (since 1.10.0), and **configurable RRF
`k` (since 1.16.0)** — the last is load-bearing: below 1.16 the fusion `k` is
hardcoded to 2 and we could not reproduce our `k=60`. We are clear of that.

A Qdrant sparse lane (FastEmbed `Qdrant/bm25`, client-side tokenization +
server-side IDF via `Modifier.IDF`, Okapi `k1=1.2`/`b=0.75`) would give true
BM25 with length normalization, exact ranked top-k from the inverted index
(no pool cap), a tiny on-disk index (~5–6 nonzeros/doc), and — via the Query
API — the option to **collapse dense+lexical RRF into a single Qdrant
request**, retiring app-side fusion for those two lanes. All models are local
open-weight (FastEmbed/ONNX), satisfying `feedback_no_paid_apis`.

Qdrant's own guidance is explicit that for **long documents** BM25 remains
preferable to its learned-sparse alternatives (BM42, miniCOIL), which are
tuned for short RAG chunks. So BM25 — not BM42/miniCOIL — is the like-for-like
candidate; SPLADE++ (Apache-2.0 `prithivida/Splade_PP_en_v1`) is a possible
later quality-upgrade experiment, not the default.

### The risk that gates this decision

Our Postgres lane uses a **custom `scix_english`** text-search config
(migration 003) that routes numbers and hyphenated/compound tokens to
`simple_nostem` — deliberately, for scientific tokens (`z=2.5`, `Fe II`,
instrument and survey names, catalog IDs) — and applies field weighting
(title=A, abstract=B, keywords=C). FastEmbed's BM25 tokenizer (lowercase,
Snowball stem, stopword-strip, punctuation split) will **not** reproduce that
scientific tokenization, and a single BM25 sparse vector does not reproduce
field weighting. Whether plain BM25 nonetheless matches or beats the tuned,
weighted tsvector lane — **especially on `title_matchable` scientific-token
queries** — is an empirical question. This ADR is gated on answering it.

## Decision (proposed — pending pilot)

If the pilot gate passes, then:

1. **The title/abstract/keywords lexical lane serves BM25 from Qdrant**, as a
   named sparse vector (`bm25`, `Modifier.IDF`, `on_disk=True`) alongside the
   `indus` dense vector on a **named-vector v2-lineage collection**, keyed on
   the same UUID-v5-from-bibcode point IDs as the dense lane.
2. **dense+lexical fusion collapses into one Qdrant Query-API request**
   (`prefetch` × 2 + `RrfQuery(rrf=Rrf(k=60))`), replacing the app-side RRF
   for these two lanes. Body BM25 stays a Postgres lane fused in app code
   until separately decided.
3. **Postgres `papers.tsv` / `idx_papers_tsv` remain** as the rollback path
   (unset a gate, re-route to `lexical_search`), for a soak period mirroring
   ADR-013, before any consideration of dropping them.

If the gate **fails**, we keep the Postgres lane and pursue the RUM cutover
(`rpjj`) for the pool-cap problem instead; SPLADE++ becomes the next thing to
pilot for relevance.

## Pilot (the evidence this ADR needs)

`scripts/qdrant_sparse_pilot.py` + `scripts/eval_sparse_pilot.py`, results in
`results/sparse_pilot_eval.{json,md}`. Design:

- **Universe (fair, identical for both systems)**: the union of each 50q
  query's top-100 Postgres lexical candidates ∪ all `gold_bibcodes` ∪ a
  ~50k-paper `TABLESAMPLE` draw (so collection-level IDF approximates the
  32M-corpus IDF rather than a topically-collapsed subset). Both systems then
  rank within the same universe; Postgres scoring is restricted to pilot
  membership.
- **Index**: FastEmbed `Qdrant/bm25` + `Modifier.IDF` over
  `title + ' ' + abstract + ' ' + keywords` (matching the Postgres lane's
  fields, no field weighting — the conservative case), collection
  `scix_sparse_pilot_v1`, sparse index `on_disk`. Heavy build runs under
  `scix-batch`; data on DS NVMe, never NAS.
- **Comparison**: per query, Qdrant BM25 top-50 vs `lexical_search` top-50,
  scored with the **existing** `scripts/eval_retrieval_50q.py` metric
  functions (nDCG@10 / MRR@10 / Recall@50), overall and per bucket
  (`title_matchable`, `concept`, `method`, `author_specific`).

### Gate (pre-registered, before seeing numbers)

**PASS** if Qdrant BM25 is within **−0.02 nDCG@10 overall** of the Postgres
lane **and** does not regress the `title_matchable` bucket by more than
**−0.03 nDCG@10** (the scientific-tokenization risk bucket). A clear overall
win obviously passes. Anything worse is a **FAIL** → keep Postgres + RUM.

### Honest limitations of the pilot

- IDF over a 50k sample approximates but does not equal 32M-corpus IDF (log
  scaling makes this acceptable for a signal; production would re-measure).
- No field weighting and no `scix_english` tokenizer tuning — this is the
  *conservative* BM25 case. A near-miss on `title_matchable` points to
  tokenizer/field-weighting tuning as the next lever, not to abandoning the
  approach.
- The pilot validates lexical ranking quality only. The single-query-fusion
  and 32M-scale build claims inherit from ADR-013's already-measured Qdrant
  behavior and are not re-proven here.

## Pilot result (2026-06-11)

`scix_sparse_pilot_v1`, 52,443 papers, 48/50 queries scored. Full table in
`results/sparse_pilot_eval.md`.

| lane | nDCG@10 | MRR@10 | Recall@50 |
|---|---|---|---|
| Postgres `scix_english` | 0.0648 | 0.0667 | 0.0833 |
| Qdrant BM25 | 0.1516 | 0.1247 | 0.4688 |

Per pre-registered gate: **Δ overall nDCG@10 = +0.0868** (would pass alone),
but **Δ `title_matchable` = −0.1244** (Postgres 0.2256 → BM25 0.1012),
violating the −0.03 title-bucket bound. **Gate = FAIL.**

What it means, with the confounds stated honestly:

1. **The title_matchable regression is the robust, low-confound signal** and
   it confirms the pre-registered risk: on exact scientific-token queries
   (`PRIMER`, `M87`, `EHT`), `scix_english`'s `simple_nostem` tokenization +
   exact-token matching beats FastEmbed BM25's stem/lowercase tokenizer. This
   is the failure we gated on. It is a *tokenizer* problem, not a *BM25*
   problem.
2. **The overall/recall win overstates BM25's true edge**, for two reasons
   the harness does not yet control: (a) the Postgres lane uses
   `plainto_tsquery` (AND-of-all-terms) while BM25 is additive (OR) — so part
   of the win is query-parsing semantics, not the scoring function; (b)
   Qdrant searched the 52k pilot universe (gold guaranteed present) while
   Postgres retrieved over the full 32M corpus — a recall bias toward Qdrant.
3. Net: lexical-only title/abstract retrieval is weak for *both* lanes on
   this hybrid-tuned gold set (it normally rides RRF fusion); BM25 is more
   robust on natural-language multi-word queries but loses the scientific
   exact-token bucket that is the whole reason `scix_english` exists.

**Decision from the pilot: do not adopt plain BM25.** Keep the Postgres lane.
Phase 2 (new bead, gated the same way) before re-deciding vs RUM (`rpjj`):

- Tune the FastEmbed BM25 tokenizer / `Bm25Config` to preserve scientific
  tokens (min/max token len, disable aggressive stemming), or pre-tokenize to
  mirror `scix_english`'s `simple_nostem` on numbers/hyphenated tokens.
- Add field weighting (separate sparse vectors for title vs abstract, weighted
  fusion) to recover the A/B/C signal.
- Fix the harness confounds: an OR-semantics Postgres baseline
  (`websearch_to_tsquery`) **and** restricting Postgres retrieval to the same
  52k universe, so the comparison isolates the scoring function.
- Only then, if `title_matchable` recovers to within −0.03, re-open this ADR.

## Consequences (if accepted)

- A full 32M sparse re-index and a **collection rebuild as named vectors**
  (the current v2 dense collection uses a single *unnamed* vector; mixing in
  a named sparse vector requires recreating it). Cost is on the order of the
  3.2 h dense load plus a CPU-bound BM25 tokenization pass.
- `qdrant-client` and `fastembed` become hard dependencies — **pin both** in
  `pyproject.toml` (ADR-013 already flagged the unpinned `qdrant-client` as a
  contract-test gap; the gRPC break is the precedent).
- The freshness/outbox path (`8m0a`) must carry the sparse vector too, or the
  lexical lane drifts at the same ~1.3k papers/day as the dense lane.
- Native payload filtering (year/OA/doctype) becomes attractive to push into
  Qdrant (ADR-008 pattern) rather than the current 10×-over-fetch +
  Postgres post-filter, to realize the single-query-fusion win fully.

## Validation rules inherited from ADR-013 (binding here)

1. No index trusted until one query has returned from it — the pilot *is*
   that smoke test at scratch scale, before any 32M build.
2. Benchmark/pilot config must match the eventual production config or be
   declared fiction — deviations (no field weighting, sample-IDF) are
   documented above as conservative, not hidden.
3. Never drop the serving (Postgres) lexical index before the Qdrant
   replacement is validated at scale and soaked.
