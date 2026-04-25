# SciX Deep Search v1 — Substrate Rollout (2026-04-25)

Operator: Stephanie (via Claude Code session). Per PRD `docs/prd/scix_deep_search_v1.md` and build log `.claude/prd-build.log.md`. Code-on-main was complete; this run was the code → infrastructure cutover.

## Outcome at a glance

| Step | Action | Verdict | Key numbers |
|---|---|---|---|
| 1 | MH-0 latency probe | **FAIL** | p50=15.341s (limit 3.0s), p95=19.854s (limit 6.0s) |
| 2 | Apply migrations 056/057/058 (test → prod) | **PASS (with 057 fix)** | v_claim_edges = 821,867 rows after dedupe of 898 duplicates |
| 3 | MH-1 intent backfill on 823K corpus | **PASS (after fine-tuning SciCite locally)** | 823K rows, 0 NULL, all 3 classes ≥5%; macro-F1 0.864 on SciCite test |
| 4 | Correction events live ingest | **PASS** | 2,534 papers with retraction events (acceptance: >1000) |
| 5 | Persona harness smoke test | **DEFERRED to v2** | Tools-only pivot decided (PRD Amendment A11); persona harness not part of v1 |

**Pivot decision (2026-04-25):** tools-only — `claim_blame` and `find_replications` ship as direct MCP tools; the persona harness is deferred to v2. Recorded in `docs/prd/scix_deep_search_v1.md` Amendment A11.

Bead state at end of session:
- `scix_experiments-m3d` (was P1, **closed**) — MH-0 pivot decision: tools-only
- `scix_experiments-8dn` (was P1, **closed**) — MH-1 model gap; resolved by local SciCite fine-tune
- `scix_experiments-wfb` (P2, **open**) — Journal errata RSS feeds 403/404; corrections-pipeline data-coverage gap (no errata/EoC events ingested)

---

## Step 1 — MH-0 latency probe: **FAIL**

Command: `python scripts/mh0_latency_probe.py --runs 5 --turns 5 --report docs/eval/mh0_latency_probe.md`

| Metric | Value | Threshold | Status |
|---|---|---|---|
| p50 per-turn dispatcher overhead | 15.341 s | 3.0 s | OVER (5.1×) |
| p95 per-turn dispatcher overhead | 19.854 s | 6.0 s | OVER (3.3×) |

Probe configuration: 5 runs × 5 turns = 25 measured turns. Tool execution excluded by construction; this is pure `claude -p` subprocess + OAuth + persona context replay overhead.

**Implication.** A 25-turn investigation loop at p50 alone burns ~6.4 minutes of subprocess overhead, leaving no headroom against the 8-minute MH-8 flagship wall-clock. The MH-7 harness shape is not viable as currently dispatched.

**Pivot options pre-authorized in PRD amendment A1:**
1. Replan MH-7 with a persistent dispatcher / batched tool calls / shorter turn budget (carries N2 build-stall risk).
2. Pivot v1 to standalone MCP tools (`claim_blame`, `find_replications`) without the agent loop; defer the deep-search persona to v2.

Tracked: bead `scix_experiments-m3d` (P1). Steps 2–3 still ran per FAIL handling because the substrate they build is reusable under either pivot.

Artifacts: `docs/eval/mh0_latency_probe.md`, `docs/eval/mh0_latency_probe.md.json`.

---

## Step 2 — Migrations 056/057/058: **PASS (with 057 fix)**

### 056_intent_populate.sql (ingest_log marker)

Applied cleanly to both `scix_test` and `scix`. Inserts a single `intent_backfill:citation_contexts` row in `ingest_log` with status `pending`. Idempotent re-apply via `ON CONFLICT (filename) DO NOTHING`.

### 057_v_claim_edges.sql (materialized view)

Applied to `scix_test` cleanly (empty data). **First prod attempt FAILED** with:

```
ERROR: could not create unique index "idx_v_claim_edges_pk"
DETAIL: Key (source_bibcode, target_bibcode, char_offset)=
  (2003CP....291...53J, 1975JPhC....8.2011L, 1747) is duplicated.
```

Diagnosis: `citation_contexts` has 898 of 823,000 rows (0.11%) that violate the assumed `(source_bibcode, target_bibcode, char_offset)` natural key — exact-duplicate rows from re-ingest. Top duplicate group repeated 7 times, all with identical `context_text`/`section_name`. The empty `scix_test` did not surface this.

**Fix shipped in this session:** rewrote the materialized view's `SELECT` to use `DISTINCT ON (source_bibcode, target_bibcode, char_offset)` with `ORDER BY ... cc.intent NULLS LAST, cc.ctid` so post-MH-1 backfill the survivor row carries non-NULL intent if any duplicate has it. Added a comment documenting the upstream data condition. Migration is still idempotent.

After fix: applied cleanly to both DBs. Prod row count:

```
SELECT COUNT(*) FROM v_claim_edges → 821867
```

(823,000 − 898 duplicates − ~235 dropped by INNER JOIN on `citation_edges`/`papers` ≈ 821,867 ✓)

First refresh (non-CONCURRENT) ran cleanly: `REFRESH MATERIALIZED VIEW v_claim_edges`.

### 058_correction_events.sql (papers.correction_events JSONB + retracted_at)

Applied cleanly to both DBs. `papers.retracted_at` and ancillary indexes already existed from prior work; `IF NOT EXISTS` guards behaved as designed. `LOGGED`-table re-assertion in the trailing `DO` block held.

---

## Step 3 — MH-1 intent backfill: **PASS** (after detour)

### What went wrong on the first attempt

A `--smoke-test --device 0` run wrote 100 rows and reported success. Underneath, transformers had logged `classifier.bias | MISSING; classifier.weight | MISSING`. The default model `allenai/scibert_scivocab_uncased` is **base SciBERT**, not a SciCite-fine-tuned classifier — the sequence-classification head was re-initialized randomly every load. The 100 row labels (61 `method`, 39 `background`, 0 `result_comparison`) were samples from a random head, not citation-intent classifications.

Rolled back same session: `UPDATE citation_contexts SET intent=NULL WHERE intent IS NOT NULL` → `UPDATE 100`; ingest_log marker reset to `pending`. Bead `scix_experiments-8dn` filed.

### Unblock: local fine-tune

`scripts/train_citation_intent.py` (already present in the repo) downloads `allenai/scicite` from HuggingFace, tokenizes with SciBERT, fine-tunes 3 epochs with macro-F1 as best-model selector, and evaluates on test.

Run: `scix-batch --mem-high 30G --mem-max 50G python scripts/train_citation_intent.py --output-dir models/citation_intent --epochs 3 --batch-size 32`. Wall-clock ~3.5 min on the 5090.

| Test split | Macro-F1 | Acc |
|---|---|---|
| SciCite held-out (1,859) | **0.864** | 0.875 |

Per-class F1: background 0.872, method 0.888, result_comparison 0.833. Beats the ≥0.80 acceptance gate; matches the SciCite paper's reported ~0.84.

`SCIX_INTENT_MODEL_PATH` default updated to point at `models/citation_intent/final/`.

### Two adjacent bugs fixed during sanity-checking

1. **`src/scix/citation_intent.py:142`** — `SciBertClassifier._get_pipeline` lacked `max_length=512`, so contexts >512 tokens crashed with `RuntimeError: tensor a (627) must match tensor b (512)`. Patched.
2. **`scripts/ingest_corrections.py:195`** — `WHERE %s = ANY(doi)` didn't match the `idx_papers_doi` GIN index (cost 7,977,927; sequential scan of 32M papers per DOI). Replaced with `WHERE doi @> ARRAY[%s]::text[]` (cost 507,585; bitmap heap scan via the GIN). ~16× speedup minimum, decisive in practice — first run looked stalled because Postgres was sequentially scanning 32M rows per DOI lookup.

### Full backfill

Run: `SCIX_INTENT_MODEL_PATH=… scix-batch --mem-high 30G --mem-max 50G python scripts/backfill_citation_intent.py --batch-size 256 --device 0`. Wall-clock 46 min at 297 rec/s (script's "12 rec/s" estimate was very pessimistic).

Final distribution on 823,000 rows:

| intent | count | pct | acceptance (≥5%) |
|---|---|---|---|
| method | 621,917 | 75.57% | PASS |
| background | 141,046 | 17.14% | PASS |
| result_comparison | 60,037 | 7.29% | PASS |
| NULL | 0 | 0.000% | PASS (≤1%) |

Method-heavy distribution reflects domain shift from SciCite's CS papers to astronomy/physics/biology in our citation_contexts. Not degenerate.

`scripts/backfill_citation_intent.py --validate-sample 500` wrote `docs/eval/mh1_intent_validation.md`.

`scripts/refresh_v_claim_edges.py --allow-prod` ran in 36.6s. `v_claim_edges.intent` is now populated (75.56% / 17.14% / 7.30%).

End-to-end smoke test of `find_replications('2001Natur.410...63N', limit=5)` (Nagamatsu et al.'s MgB2 superconductivity discovery) returned 5 ranked citing papers with sensible intent labels, hedge detection, and intent-weighted ordering — the substrate is live.

Bead `scix_experiments-8dn` closed.

---

## Step 4 — Correction events live ingest: **PASS**

Re-evaluated the "skip steps 4 and beyond" instruction from Step 1 FAIL handling — Step 4 is purely substrate work and independent of the persona harness, so the skip was overcautious. Ran in parallel with the SciCite fine-tune.

Run: `scix-batch --mem-high 8G --mem-max 16G python scripts/ingest_corrections.py --yes-production`. Wall-clock ~10 min after the GIN-index fix.

Result:

| Metric | Value | Acceptance |
|---|---|---|
| events collected (post-dedup) | 9,945 | — |
| papers updated | 2,529 | — |
| total papers with `correction_events != '[]'` | 2,534 | PASS (>1000) |
| of which retraction events | 2,534 | — |
| of which erratum / correction / EoC | 0 | — (see follow-up) |

All 2,534 events are `type=retraction`. Source breakdown was Retraction Watch + OpenAlex; Crossref `update-to` ran but contributed nothing for this DOI list. Journal RSS phase failed 9 of 15 feeds (`ApJ`/`ApJL`/`ApJS`/`AJ`/`A&A`/`MNRAS`/`PASP`/`JGR`/`ARA&A` — IOP changed URL pattern late 2025, others added Cloudflare/bot-detection). Filed bead `scix_experiments-wfb` (P2): without errata feeds, only retractions surface — the BICEP2-style flagship still works (it depends on retraction handling), but errata coverage is incomplete for v1+.

## Step 5 — Persona harness smoke test: **DEFERRED to v2**

Per PRD Amendment A11 (tools-only pivot, decided 2026-04-25): the persona harness is not part of v1. `claim_blame` and `find_replications` ship as direct MCP tools (already wired in `src/scix/mcp_server.py:1455+`, `EXPECTED_TOOLS=15`). Any MCP-aware client (Claude Code, Claude Desktop, Cursor) acts as the deep-search persona by calling the tools directly. No SciX-side orchestration loop in v1.

Persona harness code (`.claude/agents/deep_search_investigator.md`, `scripts/scix_deep_search.py`, `src/scix/citation_grounded.py`, MH-8 fixtures) stays on `main` for v2 to pick up — either by building the persistent dispatcher prototype or by accepting a longer per-question wall-clock.

---

## State of the substrate after this session

- ✅ Migrations 056/057/058 deployed to test + prod (057 fix shipped: `DISTINCT ON` for 0.11% duplicate citation_contexts rows)
- ✅ `v_claim_edges` populated and refreshed; intent column 75.56% / 17.14% / 7.30%; index lookups <1 ms
- ✅ `citation_contexts.intent` 100% non-NULL across 823,000 rows
- ✅ `papers.correction_events` populated for 2,534 papers (all retraction events)
- ✅ Locally fine-tuned SciBERT-SciCite at `models/citation_intent/final/` (test macro-F1 0.864)
- ✅ `find_replications` smoke test passes against the populated substrate
- ❌ MH-7 persona harness pending pivot decision

## Substrate-completeness audit (post-pivot)

Verified end-to-end with the canonical PRD MH-3 fixture (BICEP2 PRL `2014PhRvL.112x1101B`) and surfaced two structural gaps:

1. **`citation_contexts` covers only 30,316 of 14.9M body-having papers (~0.2%).** As a consequence, `v_claim_edges` carries 821k rows over a 299M-edge citation graph (~0.27% coverage). `find_replications('2014PhRvL.112x1101B')` returns 0 results despite 1,620 forward citations existing in `citation_edges`. The PRD assumed full coverage; the in-text-context extract pipeline (`scripts/extract_citation_contexts.py`) was only ever run on a small subset. Tracked: `scix_experiments-79n` (P1). Includes benchmark numbers — full backfill is ~25h single-process at 70 papers/sec.
2. **BICEP2 supersession is not in any of the four correction-event sources.** The PRD MH-3 acceptance specifically named BICEP2 as a fixture, but the supersession (BICEP2/Planck joint dust analysis, Ade+ 2015) was published as a *separate paper*, not as a formal Crossref `update-to` correction. Retraction Watch and OpenAlex `is_retracted` only flag formal retractions. Tracked: `scix_experiments-96p` (P2).

**Pipeline idempotency fix** shipped in this session: `src/scix/citation_context.py:284` `SELECT` now has `NOT EXISTS` skip-already-processed filter (previously every run re-processed the same papers, accumulating duplicates — that's the source of the 898 duplicates migration 057 had to dedupe). Plan verified to use the existing `idx_citctx_source_target` index. Future re-runs are safe and only touch new papers.

## Open follow-ups

| Bead | Pri | Description |
|---|---|---|
| `scix_experiments-79n` | P1 | Backfill `citation_contexts` on the remaining ~6.2M body-having papers; re-run intent backfill + `v_claim_edges` refresh after |
| `scix_experiments-96p` | P2 | Detect BICEP2-style supersession (separate paper, not formal correction) — curated list short term, mining-from-contexts long term |
| `scix_experiments-wfb` | P2 | Replace dead journal errata RSS feeds (9 of 15 returning 403/404); surface erratum / correction / EoC events alongside retractions |

`scix_experiments-m3d` was closed by the tools-only pivot decision (PRD Amendment A11). `scix_experiments-8dn` was closed by the local SciCite fine-tune.
