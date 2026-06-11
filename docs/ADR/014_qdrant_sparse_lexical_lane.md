# ADR-014: Evaluate a Qdrant Sparse BM25 Lexical Lane to Replace the Postgres `scix_english` tsvector Lane

- **Status**: Proposed — **pilot ran 2026-06-11; system-level result is positive** (`results/sparse_hybrid_pilot_eval{,_fair}.md`). Swapping the lexical lane to Qdrant BM25 *improves* the fused hybrid by **+0.06–0.10 nDCG@10**, robust to the haystack confound. Not yet Accepted: the magnitude is still inflated by an AND-vs-OR query-semantics confound, and the result needs confirmation at full-corpus scale. See "Pilot result" below. (An earlier isolated-lane test in `results/sparse_pilot_eval.md` showed a `title_matchable` regression and was initially read as a FAIL — that was the wrong unit of analysis: the lexical lane never serves un-fused. The fused result corrects it.)
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
- **Comparison**: two harnesses, both reusing the **existing**
  `scripts/eval_retrieval_50q.py` metric functions (nDCG@10 / MRR@10 /
  Recall@50), overall and per bucket. (1) `scripts/eval_sparse_pilot.py` —
  isolated lexical lanes (component check). (2)
  `scripts/eval_sparse_hybrid_pilot.py` — **the decision**: dense+lexical
  fused, dense held fixed, lexical source varied; `--restrict-universe` runs
  the haystack-fair variant.

### The decision-grade test is system-level, not isolated-lane

The lexical lane never serves un-fused; it is RRF-fused (k=60) with the dense
INDUS lane (and the body BM25 lane). So the question that decides this ADR is
**"does swapping the lexical source change the fused hybrid?"** — measured by
holding the dense lane fixed and varying only the lexical lane. An
isolated-lane comparison is a useful *component* check (it surfaces tokenizer
behavior) but is not the decision.

**System-level gate (pre-registered):** PASS if `bm25+dense` ≥ `pg_lex+dense`
on overall nDCG@10 (i.e. the swap does not hurt the fused two-lane hybrid),
with per-bucket effects reported, not gated. Confirmed against the canonical
`hybrid_search` (`baseline_search` in `eval_retrieval_50q.py`) so absolute
numbers are trustworthy.

## Pilot result (2026-06-11)

`scix_sparse_pilot_v1`, 52,443 papers, 48/50 queries scored.

### System-level — fused hybrid (the decision)

`results/sparse_hybrid_pilot_eval_fair.md` — **all lanes restricted to the
same 52k universe** (removes the BM25 small-haystack confound; dense/body lose
their 32M reach, so this is the conservative fair A/B). Dense lane identical
across arms, so each Δ isolates the lexical swap.

| arm | nDCG@10 | MRR@10 | Recall@50 |
|---|---|---|---|
| dense_only | 0.1812 | 0.1366 | 0.3750 |
| `pg_lex+dense` ("lexical + qdrant") | 0.1424 | 0.1135 | 0.3958 |
| **`bm25+dense` ("qdrant hybrid")** | **0.1997** | 0.1509 | 0.5521 |
| `pg_lex+body+dense` (current prod shape) | 0.1500 | 0.1220 | 0.4583 |
| **`bm25+body+dense`** | **0.2471** | 0.2237 | 0.5312 |

**Headline: swapping lexical → Qdrant BM25 improves the fused hybrid by
+0.0573 nDCG@10 (two-lane) / +0.0971 (three-lane). GATE = PASS.** The
production-realistic run (dense/body over full 32M, `…_eval.md`) shows the same
direction at +0.0742 / +0.0763. Validation: the controlled `pg_lex+body+dense`
arm (0.0935, 32M run) tracks canonical prod `hybrid_search` (0.0831).

Per-bucket (fair run), where the nuance lives:

- **`author_specific`: the big win** — 0.21 → 0.57. Robust to the haystack
  fix. Largely BM25's additive (OR) scoring beating `plainto_tsquery`'s
  AND-of-all-terms on many-token queries (see confound below).
- **`method`**: roughly flat (slight BM25 edge).
- **`concept`**: flat, low for both.
- **`title_matchable`: BM25 loses ~0.04** (0.2478 → 0.2061) — the
  scientific-token tokenization weakness is real and survives fusion, but it
  is now a small, localized cost, not the headline. Confirms the isolated-lane
  signal in `results/sparse_pilot_eval.md` while showing it is dominated by the
  gains elsewhere.

### Component check — lexical lanes in isolation (context, not decision)

`results/sparse_pilot_eval.md`: Qdrant BM25 0.1516 vs Postgres 0.0648
nDCG@10 overall, but BM25 −0.1244 on `title_matchable`. Read alone this looks
like a failure; fused, that bucket's cost shrinks to −0.04 and is outweighed.
This is exactly why the isolated lane is the wrong unit of analysis.

### The remaining confound (why this is PASS-pending, not Accepted)

The Postgres lane uses `plainto_tsquery` (AND-of-all-terms); BM25 is additive
(OR). Part of BM25's win — especially `author_specific` — is query *parsing*,
not the BM25 *scoring function*. This is not strictly unfair (AND-semantics is
what we run in production today, so BM25 beating it is a real improvement over
the status quo), but it means we cannot yet attribute the gain to BM25 per se.
A `websearch_to_tsquery` (OR) Postgres arm is needed to separate the two.

**Decision from the pilot: directionally adopt-positive; proceed to Phase 2
before committing the 32M rebuild** (new bead). Phase 2:

- Confirm magnitude at full-corpus scale (BM25 over 32M, so the BM25 lane
  also searches 32M — the only way to retire the haystack question for good).
- Add an OR-semantics (`websearch_to_tsquery`) Postgres arm to attribute the
  win between scoring vs query parsing.
- Recover `title_matchable`: tune the FastEmbed BM25 tokenizer / `Bm25Config`
  to preserve scientific tokens (min/max token len, less aggressive stemming)
  or pre-tokenize to mirror `scix_english`'s `simple_nostem`; optionally field
  weighting (separate title/abstract sparse vectors).
- If magnitude holds and `title_matchable` recovers, flip this ADR to Accepted
  and supersede the RUM cutover (`rpjj`) for the title/abstract lane.

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
