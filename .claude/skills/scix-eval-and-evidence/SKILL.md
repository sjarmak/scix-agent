---
name: scix-eval-and-evidence
description: >
  How retrieval and extraction quality is measured and what counts as evidence
  in SciX: the gold sets (50q curated, 1200q recall gold, claim extraction,
  lexical stress), the fusion-calibration sweep and its honest verdict
  (dense_only < bm25_only), nDCG@10 / Recall@K / MRR, Wilson 95% CIs, the OAuth
  persona/UMBRELA judges, the claim_blame gold-set plan (bead 6ajy), and the
  reporting rules (read-only harness, lane provenance, per-bucket numbers, null
  results stated plainly). Load when running or interpreting an eval, adding a
  gold set, judging relevance, choosing an acceptance threshold, or writing up
  a result. NOT for RRF fusion internals (scix-retrieval-architecture), Qdrant
  mechanics (scix-vector-serving-qdrant), CI/pytest (scix-build-test-ci),
  whether a change needs an ADR (scix-change-control), or query_log telemetry
  (scix-db-safety-and-telemetry).
---

# SciX evaluation and evidence discipline

This skill is the acceptance bar: which gold sets exist, which harnesses
produce numbers, which numbers are already banked (including the uncomfortable
ones), and the rules a result must follow before anyone repeats it as fact.

Provenance pin: verified read-only on 2026-07-07 against working copy
`/home/ds/projects/scix_experiments`, branch `bd/0yp5-external-copy-accuracy-audit`
(NOT main), HEAD `452ab86`. Bead IDs (`6ajy`, `isve`, `9sv1`, `h2sj`, `q9k5`,
`4skc`, `9na0`, `dfba`) are references into the project's bead store — use
them to look up history (`bd show <id>`), never as machinery this skill
depends on.

## When NOT to use this skill

| You are trying to…                                    | Use instead                          |
| ----------------------------------------------------- | ------------------------------------ |
| Understand/modify RRF fusion, lanes, `hybrid_search`  | scix-retrieval-architecture          |
| Operate Qdrant, the dense collection, payloads        | scix-vector-serving-qdrant           |
| Fix the embed/ingest pipeline (incl. the s7cy fire)   | scix-embedding-pipeline              |
| Run pytest/CI, env extras, `make check` vs `check-ci` | scix-build-test-ci                   |
| Decide if a change needs an ADR / operator sign-off   | scix-change-control                  |
| Analyse `query_log`, avoid the prod DSN               | scix-db-safety-and-telemetry         |
| Query papers as an end user via MCP tools             | scix-mcp (existing query-side skill) |
| Frame the ADASS paper claim / research campaign       | scix-research-frontier               |

## Jargon (defined once)

| Term            | Meaning here                                                                                                                 |
| --------------- | ---------------------------------------------------------------------------------------------------------------------------- |
| **gold set**    | JSONL file of queries (or paragraphs) with hand- or citation-derived ground truth, under `eval/`                             |
| **nDCG@10**     | normalized discounted cumulative gain over the top 10; the project's primary ranking metric                                  |
| **Recall@K**    | fraction of gold bibcodes present in the top K                                                                               |
| **MRR**         | mean reciprocal rank of the first relevant hit                                                                               |
| **RRF**         | reciprocal rank fusion; `RRF_K = 60` in `src/scix/search.py`                                                                 |
| **lane**        | one retrieval signal (INDUS dense via Qdrant, title/abstract BM25, body BM25) before fusion                                  |
| **Wilson CI**   | 95% binomial confidence interval (`src/scix/eval/wilson.py`); reported with every precision-style proportion                 |
| **UMBRELA**     | published LLM-relevance-judge rubric (arXiv:2406.06519), 0–3 ordinal scale; the default judge persona                        |
| **OAuth judge** | a `claude -p` subprocess acting as judge — no paid Anthropic API, no `anthropic` SDK import                                  |
| **bucket**      | a query stratum (`title_matchable` / `concept` / `method` / `author_specific` in the 50q set; `decile` 0–9 in the 1200q set) |

## 1. Gold-set inventory (as of 2026-07-07)

All retrieval gold sets are JSONL under `eval/`, one query per line.

| File                                        | Size                   | Ground truth                                                                                                                                                          | Use for                                                                                                                                             |
| ------------------------------------------- | ---------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| `eval/retrieval_50q.jsonl`                  | 50 q                   | hand-curated `gold_bibcodes`, stratified by `bucket` (`title_matchable`, `concept`, `method`, `author_specific`)                                                      | per-bucket diagnostics ONLY. Has documented findability gaps; cannot answer fusion questions (see §3 verdict)                                       |
| `eval/recall_gold_v1.jsonl`                 | 1200 q                 | citation-based: gold = paper itself + strongest citation-graph neighbours; binned by community-size `decile` 0–9 (bead h2sj; operator-confirmed GT choice 2026-06-14) | the recall/fusion instrument; per-decile gating                                                                                                     |
| `eval/lexical_stress_16q.jsonl`             | 16 q                   | broad single-token stress terms                                                                                                                                       | `SCIX_LEXICAL_RANK_FLAG` A/B (`scripts/eval_lexical_rank_flag.py`, bead q9k5)                                                                       |
| `eval/retrieval_30q_non_smd.jsonl`          | 35 lines               | non-SMD queries                                                                                                                                                       | **no consumer found** in `scripts/`, `src/`, or `tests/` as of 2026-07-07; verify before relying on it (note: filename says 30q, file has 35 lines) |
| `eval/claim_extraction_gold_standard.jsonl` | 15 entries             | hand-curated atomic claims with char-span anchors; schema in `eval/claim_extraction_gold_standard_README.md`                                                          | claim-extraction shape regression, NOT statistical benchmarking (intentionally small)                                                               |
| `tests/eval/red_team_v1.jsonl`              | —                      | adversarial deep-search cases                                                                                                                                         | `scripts/run_red_team_eval.py` (`--mock` default for CI; `--no-mock` is operator-only and burns OAuth budget)                                       |
| `tests/eval/claim_blame_gold_v1.jsonl`      | **does not exist yet** | planned: 30–50 known-origin claims                                                                                                                                    | bead 6ajy — see §5                                                                                                                                  |

Rules for gold sets:

- Schema for the claim-extraction set is enforced by
  `pytest tests/test_gold_standard_format.py` (enums, per-discipline balance,
  span integrity). Run it before merging any edit to that file.
- `eval/build_gold_standard.py` is the claim-extraction entry builder (computes
  char offsets for you). It is NOT a retrieval-gold generator.
- The 50q and 1200q files share a superset schema
  (`{query, gold_bibcodes[], bucket, discipline, notes, [decile]}`) so one
  loader reads both (`load_queries` in `scripts/eval_retrieval_50q.py`).
- Placeholder bibcodes in the claim set are prefixed `GOLD` — they are not
  real ADS records; never resolve them against the corpus.

## 2. The metric stack (where numbers come from)

| Module                             | What it owns                                                                                                                                                                                                                  |
| ---------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `src/scix/ir_metrics.py`           | canonical nDCG@K, Recall@K, Precision@K, MRR; `RetrievalScore` / `EvalReport` DTOs. Do not reimplement metrics — import these                                                                                                 |
| `src/scix/eval/metrics.py`         | three-way eval runner (§M4) + `INHOUSE_DISCLAIMER` constant (see §6 rule 2)                                                                                                                                                   |
| `src/scix/eval/wilson.py`          | `wilson_95_ci(successes, total)`. Anchor: `wilson_95_ci(95, 100) → (0.887, 0.978)`; `total == 0 → (0.0, 1.0)`                                                                                                                 |
| `src/scix/eval/persona_judge.py`   | ordinal 0–3 relevance judge (§4)                                                                                                                                                                                              |
| `src/scix/eval/llm_judge.py`       | binary entity-link judge (`correct/incorrect/ambiguous`) + hand-rolled `cohens_kappa()`. Legacy: new relevance work uses `persona_judge`                                                                                      |
| `src/scix/eval/audit.py`           | stratified sampler over `document_entities` (default 125/tier) for the M9 link audit; reports per-tier precision with Wilson CIs                                                                                              |
| `src/scix/eval/real_data.py`       | read-only prod-DB bridges for the three-way/lane-consistency evals (all SELECTs, parameterised)                                                                                                                               |
| `src/scix/eval/lane_delta.py`      | lane-consistency Jaccard arithmetic; p90 divergence gate `≤ 0.05`. **`compute_lane_delta_set()` is a stub returning `frozenset()`** until the Wikidata backfill (u07) lands                                                   |
| `src/scix/eval/query_expansion.py` | **deterministic numpy stub, tests/pilot only** — fenced out of production; must be replaced (bead `eecm`) before PRD §S3 promotes. Do not wire it into a serving path                                                         |
| `src/scix/eval/tool_surface/`      | MCP tool-selection eval (runner/scorer/report/stubs); results in `results/tool_surface_eval/summary.json` (e.g. variant v0: tool_accuracy 0.944 over 90 runs × 30 queries). Surface-cap policy lives in scix-mcp-tool-surface |

Trap: `scripts/refresh_fusion_mv.py` refreshes the **entity**
`document_entities_canonical` materialized view. Despite the name it has
nothing to do with retrieval fusion or the fusion sweep.

## 3. The fusion-calibration sweep — the banked result and its caveat

`scripts/fusion_sweep.py` (beads isve → 9sv1) sweeps fusion strategies over
the separate dense and BM25 lanes: `dense_only`, `bm25_only`, `naive_rrf(k)`,
`weighted_sum(w_dense)`, `rank_cutoff_rrf(cutoff)`, `dense_prior(lam)`.

How to exercise it WITHOUT touching the DB or model (safe anywhere):

```bash
python scripts/fusion_sweep.py --dry-run     # validates wiring + fusion math on synthetic lanes
pytest tests/test_fusion_sweep.py            # dry-run-path unit tests
```

Live run — **do not run casually.** It is READ-ONLY against the corpus
(SELECT/kNN only) but loads the INDUS encoder and queries prod PG + Qdrant.
On this installation heavy work must run inside the memory-scoped wrapper
(host operational requirement; see scix-memory-and-batch-discipline):

```bash
# heavy: INDUS model in RAM + 1200 queries against prod lanes
scix-batch python scripts/fusion_sweep.py --queries eval/recall_gold_v1.jsonl \
    --output results/fusion_sweep_1200q.md --json-output results/fusion_sweep_1200q.json
```

Defaults: `--queries eval/retrieval_50q.jsonl`, `--output
results/fusion_sweep_v1.md`, `--json-output results/fusion_sweep_v1.json`,
`--pool 100` per-lane candidates, `--k 10` (nDCG/MRR cutoff).

### Banked numbers (results/, generated 2026-06; re-read the files, do not trust memory)

1200q recall gold (`results/fusion_sweep_1200q.md`), nDCG@10 (canonical home
for the publication-facing interpretation of these numbers is
`scix-research-frontier` §1; this table carries them as the measurement
methodology's worked example):

| Config                 | nDCG@10 | Δ vs dense_only |
| ---------------------- | ------- | --------------- |
| naive_rrf(k=60) — best | 0.4786  | +0.0983         |
| bm25_only              | 0.4291  | +0.0488         |
| dense_only             | 0.3803  | +0.0000         |

50q curated (`results/fusion_sweep_v1.md`): bm25_only 0.0756 is the top
config, dense_only 0.0399 — diagnostic only; the set has findability gaps.

### The verdict you must not soften

Both sweep reports carry the same conclusion, verbatim in the files:
**the premise of a dominant dense lane does NOT reproduce.** `dense_only`
nDCG@10 is _below_ `bm25_only` on both gold sets, so any hybrid lift over
dense-alone partly reflects outrunning a weak dense lane, not a clean fusion
gain. The original contrary numbers (dense ~0.864 vs BM25 ~0.088, bead dfba)
came from a different instrument and did not survive the 1200q re-run.

PROVISIONAL pending Stephanie (discovery Q3): report the headline as
"naive RRF (k=60) lifts +0.098 nDCG@10 over dense-alone, but the dense lane
underperforms BM25, so the lift partly reflects a weak dense lane." Never
state INDUS/dense superiority as fact; a dense lane that beats BM25 is an
open frontier (scix-research-frontier), not a settled gate.

### The 50q three-mode harness

`scripts/eval_retrieval_50q.py` drives `baseline` / `section` / `fused` modes
(RRF k=60 throughout; metrics nDCG@10, MRR@10, Recall@50; per-bucket
averages; default output `docs/eval/retrieval_50q_2026-04.json`). It has the
same `--dry-run` stub mode. The `section` mode zero-stubs with
`skipped_reason = "section_embeddings_empty"` when `section_embeddings` is
unpopulated — a zero there is a coverage fact, not a quality result.

Note (PROVISIONAL pending Stephanie, discovery Q2): the working tree carries
uncommitted embedding-pipeline changes (the s7cy remediation) and a modified
`results/within_paper_rerank_eval.md`. Everything in this skill describes
committed HEAD; treat uncommitted eval artifacts as in-flight, canonize
nothing from them.

## 4. Judges: how relevance gets scored without a paid API

All judging goes through Claude Code OAuth subagents (`claude -p`
subprocesses). No `anthropic` SDK import, no paid API — this is a standing
project rule, not a preference.

### Ordinal relevance (0–3): `src/scix/eval/persona_judge.py`

- Default persona: **`umbrela_judge`** (`.claude/agents/umbrela_judge.md`,
  verbatim Castorini UMBRELA rubric, Apache-2.0, arXiv:2406.06519) — chosen
  because its published Kendall's τ > 0.87 vs TREC human assessors gives a
  benchmarked baseline. Legacy alternate: `in_domain_researcher` (retained
  for A/B on the same seed).
- Scores are ints in [0, 3]; a triple that exhausts retries returns
  `ERROR_SENTINEL = -1` and must surface as a hard error, never as a 0.
- Dispatch: `ClaudeSubprocessDispatcher` (`claude -p`), bounded concurrency 4,
  3 retries, 2.0 s backoff base, 120 s per-call timeout. `StubDispatcher` is
  the deterministic test double.
- Snippets honor the licensing budget (`build_snippet`: title + abstract +
  first 500 chars of body).

### Judge calibration: `scripts/calibrate_judge.py`

Input: CSV `query,bibcode,human_score` (0–3). Reports Spearman ρ and
quadratic-weighted kappa; `trustworthy` iff κ ≥ 0.6 (Landis–Koch
"substantial"). Every run appends to `results/judge_calibration_log.jsonl`
for prompt-drift watch.

**Null-state fact (2026-07-07): `results/judge_calibration_log.jsonl` does
not exist** — no calibration run against a human-labeled seed is recorded in
the repo. Judge-graded numbers are therefore an engineering signal whose
judge is benchmarked only by UMBRELA's published external validation, not by
a local human seed. Say so when you report them.

### Binary link audit: `llm_judge.py` + `scripts/m9_audit_judge.py`

The M9 entity-link audit samples ~125 candidates per tier
(`sample_stratified`), judges `correct/incorrect/ambiguous` via `claude -p`,
writes labels to `entity_link_audits` (annotator `claude_oauth_judge_v1`,
idempotent reruns), and prints per-tier precision **with Wilson 95% CIs**.
Recomputing `tier_weight()` / refreshing the MV is a separately gated stage
(`m9_apply_calibration.py`, operator sign-off — see scix-change-control).

## 5. The claim_blame gold set — designated next eval (bead 6ajy, OPEN)

The 2026-06-22 deep-audit's "smartest addition": **`claim_blame` has no
real-corpus accuracy number.** Every existing `claim_blame` test is a
synthetic-fixture shape test; the PRD MH-1/MH-8b recall gate was never run.
The plan (verbatim from the bead, still open as of 2026-07-07):

- Build a 30–50 case known-origin gold set (Riess+2011 H0, Perlmutter+1999 Λ,
  BICEP2 dust, …) → `tests/eval/claim_blame_gold_v1.jsonl` with
  `claim_text` + verified origin bibcode + acceptable-origin set.
- Run `claim_blame` over it via the `deep_search_investigator` OAuth subagent
  path; emit origin-recall@1/@5, retraction-overlay precision,
  lineage-chronology correctness, and **COVERAGE%** (`v_claim_edges` holds
  ~821K of 299M edges — always report the denominator).
- Output `results/claim_blame_gold_v1.md` in the `fusion_sweep_1200q.md`
  report shape.
- Predefined gates: ≥70% origin recall → adopt the hard intent filter;
  <40% → keep current weights.
- Hand-labeling needs domain judgment: the bead is marked `gc.no_route: true`
  and is NOT to be auto-dispatched — an operator or OAuth-judge labeling pass
  must be surfaced for prioritization first.

Zero new infra: it reuses `claim_blame`, the citation graph, and the
persona-judge harness. No ADR needed, no paid API.

## 6. What counts as evidence here (the reporting rules)

Every banked result in `results/` follows these; a result that skips one gets
challenged in review. Each rule cites a live example you can open.

1. **Read-only harness.** Eval scripts issue SELECT/kNN only
   (`fusion_sweep.py` docstring; `real_data.py` design notes). An eval that
   writes to the corpus is a bug. Write-side judging (M9 labels) targets its
   own audit table, never corpus tables.
2. **In-house disclaimer up front.** Reports open with the provenance
   disclaimer — `INHOUSE_DISCLAIMER` in `src/scix/eval/metrics.py` makes it
   the first lines of the M4 report; the rerank memos carry the same block.
   Self-reported engineering signal, not an external benchmark. Never delete
   it to make a report look stronger.
3. **Lane and config provenance.** Reports pin the full instrument: gold-set
   path + size, lanes, `rrf_k`, pool size, model names AND revisions, device,
   `qdrant_url`, judge persona + prompt version (see the "Provenance details"
   blocks in `results/retrieval_eval_50q_rerank_*.md`). A number without its
   instrument is not reportable.
4. **Per-bucket, not just aggregate.** 50q reports break out the four
   buckets; 1200q gates per decile (that per-decile requirement is WHY the
   1200q set exists — bead h2sj); NER precision is per
   `(type, era, agreement)` bucket with the lower bound used when the bucket
   is unknown (scix-entity-ner-system). Aggregates hide exactly the failures
   this project has actually had.
5. **Uncertainty on every proportion; significance on every delta.** Wilson
   95% CIs for precision-style claims (`wilson_95_ci`); paired Wilcoxon
   signed-rank with Bonferroni correction for per-query metric deltas (see
   `results/retrieval_eval_50q_rerank_local.md`: minilm Δ −0.0453,
   p = 0.042 vs corrected α = 0.025 → reported as NOT significant).
6. **Null and negative results are banked, not buried.** Live examples:
   - Reranker A/B (bead 4skc): baseline `hybrid_indus` beat every local
     cross-encoder; the domain-tuned `nasa-impact/nasa-smd-ibm-ranker`
     regressed nDCG@10 and was worst. Verdict shipped anyway.
   - UMBRELA second opinion (bead 9na0): corroborated it on graded relevance
     (judge nDCG@10 minilm 0.7766 > hybrid 0.7108 > indus_ranker 0.6716;
     disjoint-promotion Δ mean judge score −0.3458) — with the explicit note
     that judged-pool nDCG is NOT comparable to citation-nDCG absolutes.
   - The fusion sweep's "premise does NOT reproduce" verdict (§3).
7. **Gates are predefined, in writing, before the run.** 6ajy fixes ≥70%/<40%
   before any number exists; lane-consistency fixes p90 ≤ 0.05
   (`GATE_THRESHOLD`, `lane_delta.py`); judge trust fixes κ ≥ 0.6. Success is
   measured against the pre-committed threshold, never judged by eye after.
8. **Different relevance notions are labeled, not mixed.** Citation-edge GT
   is a noisy binary proxy; judge grades are 0–3 ordinal; the two are
   reported side by side (9na0 memo), never averaged together.

## 7. Known stubs, gaps, and traps in the eval layer (2026-07-07)

| Item                                   | State                                                                                                                                                           |
| -------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `query_expansion.py`                   | numpy placeholder, fenced to tests/pilot; replace before §S3 promotes (bead eecm). Flagged in the 2026-06-22 audit as a placeholder-code violation              |
| `lane_delta.compute_lane_delta_set`    | always returns `frozenset()` until the Wikidata backfill lands; the p90 gate currently runs unadjusted                                                          |
| `results/judge_calibration_log.jsonl`  | absent — no human-seed judge calibration recorded                                                                                                               |
| `eval/retrieval_30q_non_smd.jsonl`     | no consumer found; 35 lines despite "30q" name                                                                                                                  |
| `tests/eval/claim_blame_gold_v1.jsonl` | not built yet (bead 6ajy OPEN)                                                                                                                                  |
| 50q gold set                           | findability gaps documented in `results/fusion_sweep_v1.md`; per-bucket diagnostics only                                                                        |
| `claim_blame` accuracy                 | NO real-corpus number exists; do not cite one                                                                                                                   |
| Live evals on this host                | must run under `scix-batch` (installation-specific memory discipline; scix-memory-and-batch-discipline). `--dry-run` and `pytest` collection are safe unwrapped |
| DSN                                    | eval scripts that touch the DB default to prod `scix`; read-only or not, set/verify the DSN story first (scix-db-safety-and-telemetry)                          |

## Provenance and maintenance

Pinned: branch `bd/0yp5-external-copy-accuracy-audit`, HEAD `452ab86`,
verified 2026-07-07 by read-only inspection only (source reading, `ls`,
`grep`, `bd show`, `pytest --collect-only`; no eval script was executed, not
even `--dry-run`).

One-line re-verification commands (all read-only):

```bash
git -C /home/ds/projects/scix_experiments branch --show-current && git -C /home/ds/projects/scix_experiments rev-parse --short HEAD
wc -l eval/*.jsonl                                        # gold-set sizes (claim 15, lexical 16, recall 1200, 30q-file 35, 50q 50)
ls src/scix/eval/                                         # eval module inventory
grep -n 'RRF_K = 60' src/scix/search.py                   # fusion constant
grep -n 'DEFAULT_PERSONA' src/scix/eval/persona_judge.py  # umbrela_judge is default
grep -n 'kappa >= 0.6' -ri scripts/calibrate_judge.py     # judge trust gate
ls results/judge_calibration_log.jsonl                    # absent ⇒ still no calibration run
ls tests/eval/claim_blame_gold_v1.jsonl                   # absent ⇒ 6ajy still open
bd show 6ajy                                              # claim_blame gold-set bead status
head -40 results/fusion_sweep_1200q.md                    # banked fusion numbers + verdict
python scripts/fusion_sweep.py --dry-run                  # wiring check, no DB/model
pytest --collect-only -q tests/test_fusion_sweep.py tests/test_ir_metrics.py tests/test_gold_standard_format.py tests/test_persona_judge.py
```

If any re-verification line disagrees with this skill, the repo wins — update
the skill under an explicit bead.
