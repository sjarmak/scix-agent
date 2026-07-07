---
name: scix-embedding-pipeline
description: >
  The INDUS paper-embedding ingest path: src/scix/embed.py, scripts/embed.py,
  daily_sync.sh Step 5, the paper_embeddings table and its PG-to-Qdrant outbox
  sync, and the s7cy live fire (paper_embeddings was DROP TABLEd out-of-process;
  committed HEAD embed code targets the dropped table; ~83K papers have no dense
  vector). Load when embedding new papers, debugging "relation paper_embeddings
  does not exist" / daily_sync aborts, backfilling the dense-vector gap, touching
  any embed/outbox/watermark code, or asking which embedding model/pooling/input
  format SciX uses. NOT for Qdrant collection config, serving reads, or payload
  indexes — use scix-vector-serving-qdrant. NOT for RRF fusion or lane weighting —
  use scix-retrieval-architecture. NOT for index DDL / disk reclamation — use
  scix-index-and-storage-discipline. NOT for DSN guards and prod-DB protection
  mechanics — use scix-db-safety-and-telemetry. NOT for scix-batch/OOM discipline
  details — use scix-memory-and-batch-discipline.
---

# SciX Embedding Pipeline (INDUS paper-level dense ingest)

Scope: how a paper's title+abstract becomes a 768-dimensional INDUS vector in
the serving dense lane, and the current broken state of that path. Paper-level
INDUS only. Section/chunk embedding lanes (`src/scix/embeddings/`,
`section_pipeline`) are out of scope here — they are body-AI code gated on
`papers_is_oa_or_preprint` (see the repo CLAUDE.md "Body-AI" rules and
scix-retrieval-architecture).

Sibling routing:

| Question                                                                                | Skill                             |
| --------------------------------------------------------------------------------------- | --------------------------------- |
| Qdrant collection layout, payload indexes, `QDRANT_URL` serving gate, exact-search flag | scix-vector-serving-qdrant        |
| How the dense lane is fused with BM25 (RRF), lane quality numbers                       | scix-retrieval-architecture       |
| Index-build discipline (50k scratch rule), disk/NAS placement, reclamation              | scix-index-and-storage-discipline |
| `SCIX_DSN` / `SCIX_TEST_DSN` / `is_production_dsn` / `--allow-prod` mechanics           | scix-db-safety-and-telemetry      |
| `scix-batch` wrapper, oomd, memory ceilings                                             | scix-memory-and-batch-discipline  |
| What is gated / who signs off                                                           | scix-change-control               |

Jargon used below, defined once:

- **INDUS** — NASA/IBM science-domain sentence-transformer
  (`nasa-impact/nasa-smd-ibm-st-v2` on HuggingFace), SciX's only production
  embedding model. 768-dimensional output.
- **bibcode** — the ADS paper identifier (e.g. `2020arXiv200407180C`); primary
  key across the corpus.
- **halfvec** — pgvector's float16 vector type (the approved quantization;
  binary quantization is banned, >40% nDCG@10 loss).
- **outbox** — a PG queue table (`embedding_outbox`, migration 070) that
  recorded every `paper_embeddings` write so a worker could replay it to Qdrant.
- **watermark** — a tiny "already done" set (`indus_qdrant_synced`, migration
  072, uncommitted) used to detect unembedded papers without a big vector table.
- **ADR / bead** — architecture decision record (`docs/ADR/`) / work item in
  the `bd` issue store. For this pipeline the beads ARE the incident record.

---

## 1. STOP — state of the pipeline (as of 2026-07-07)

**The committed embedding pipeline cannot run.** Read this table before
touching or trusting any embed code.

| Component                                             | Committed HEAD reality                         | Live status                                                                                                                            |
| ----------------------------------------------------- | ---------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------- |
| `paper_embeddings` PG table                           | Every committed write path targets it          | **DROPPED in prod ~2026-06-29/30, out-of-process, no NAS archive** (bead s7cy, forensics dated 2026-07-03)                             |
| `embedding_outbox` + trigger (migration 070)          | `daily_sync.sh` Step 7 drains it               | **DROPPED alongside**                                                                                                                  |
| `scripts/embed.py` → `src/scix/embed.py`              | PG-first: COPY into `paper_embeddings`         | **Broken**: `psycopg.errors.UndefinedTable` at `_UNEMBEDDED_WHERE`                                                                     |
| `scripts/daily_sync.sh` (cron)                        | 7 steps; Step 5 embeds, Step 7 syncs to Qdrant | **Aborts at Step 5 every run since 2026-06-30** (`set -euo pipefail`), which also skips Step 6 (`v_claim_edges` MV refresh) and Step 7 |
| Serving dense lane (Qdrant `scix_indus_v2_papers_s1`) | Read path unaffected                           | **Intact** — 32,383,535 points at 2026-07-03; existing corpus serves fine                                                              |
| New papers since ~2026-06-30                          | —                                              | **~82,950 papers with no dense vector** (as of 2026-07-03, growing ~1–3k/day); BM25/body-only until backfilled                         |
| The fix (direct-to-Qdrant + watermark)                | Not in any commit                              | **Implemented but UNCOMMITTED in the working tree** — in-flight, awaiting prod-exec sign-off (see §5)                                  |

Live-state numbers above come from bead s7cy's verified forensics
(2026-07-03: `to_regclass('public.paper_embeddings') = False`,
`schema_migrations` max = 68, papers = 32,466,485). They are NOT re-verified by
this skill (verifying requires a prod DB connection — operator-gated). Treat
counts as dated, not current.

First diagnostic for any embed complaint (read-only, safe):

```bash
git -C "$(git rev-parse --show-toplevel)" status --short scripts/ src/scix/ migrations/   # is the s7cy fix still uncommitted?
bd show scix_experiments-s7cy        # incident + fix status (authoritative)
tail -50 logs/daily_sync.log         # the abort, if cron is still firing
```

---

## 2. The model: INDUS facts

All facts verified against committed `src/scix/embed.py` (module docstring,
`MODEL_REGISTRY`, `POOLING`, `prepare_input`).

| Fact              | Value                                                                                                                                                                                                                                |
| ----------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| HF model          | `nasa-impact/nasa-smd-ibm-st-v2` (`MODEL_REGISTRY["indus"]`)                                                                                                                                                                         |
| Dimensions        | 768 — **ADR-pinned, do not change** (matches INDUS native output; PG block-size/TOAST limits)                                                                                                                                        |
| Pooling           | **mean pooling over non-padding tokens** (`POOLING["indus"] = "mean"`; attention-masked average). SPECTER2, the pilot model, uses CLS pooling — do not mix them up                                                                   |
| Input text        | `"{title} [SEP] {abstract}"` (`input_type="title_abstract"`); title-only fallback (`"title_only"`); **no title → paper is skipped** (`prepare_input` returns `None`)                                                                 |
| Idempotency key   | `source_hash` = SHA-256 of the input text                                                                                                                                                                                            |
| Loading           | `transformers` `AutoModel`/`AutoTokenizer`, cached per `(model_name, device)` in `_model_cache`; `clear_model_cache()` frees it                                                                                                      |
| Deps              | pyproject extra `embed`: `transformers>=4.36`, `torch>=2.1`, `sentence-transformers>=2.6` — intentionally NOT installed by CI (tests self-skip)                                                                                      |
| Precision at rest | PG (historical): `halfvec(768)` shadow column for INDUS writes. Qdrant: collection stores float16, down-converts on upsert (per uncommitted `qdrant_dense.py` docstring + bead s7cy empirical check — not independently re-verified) |
| Banned            | Paid-API embedding lanes (`feedback_no_paid_apis.md`); binary quantization; any second dense lane without an ADR                                                                                                                     |

Abstract-only embedding (title+abstract) is universally indexable — the
body-AI OA gate does NOT apply to this pipeline.

---

## 3. Committed architecture (what `git show HEAD:` teaches — now historical)

This is the design at committed HEAD. Learn it to read the code and the
incident; do not expect it to run (§1).

```
papers (PG)
   │  anti-join: LEFT JOIN paper_embeddings … WHERE pe.bibcode IS NULL
   ▼
src/scix/embed.py  run_embedding_pipeline()
   3 threads: reader → [queue] → GPU (batch inference) → [queue] → writer
   ▼
paper_embeddings (PG)                      ← store_embeddings_copy(): COPY into
   INDUS rows → embedding_hv halfvec(768)     temp staging + INSERT ON CONFLICT
   pilot rows → embedding vector (dimensionless)
   ▼  trigger trg_embedding_outbox (migration 070)
embedding_outbox (PG queue: bibcode, model_name, op)
   ▼  scripts/qdrant_outbox_sync.py  (FOR UPDATE SKIP LOCKED, delete-after-upsert,
   │   at-least-once; idempotent via uuid5(bibcode) point id)
   ▼
Qdrant scix_indus_v2_papers_s1  ← the lane search.py:vector_search() reads
                                   when QDRANT_URL is set (ADR-013)
```

Key committed files and what each does:

| File                                                     | Role                                                                                                                                                                        |
| -------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `src/scix/embed.py`                                      | Model load, pooling, `prepare_input`, 3-stage pipeline, PG stores                                                                                                           |
| `scripts/embed.py`                                       | CLI wrapper: `--model indus --batch-size N --device cuda --limit N --dsn …`                                                                                                 |
| `scripts/daily_sync.sh`                                  | Cron pipeline (06:15 UTC pattern in header): harvest → ingest → backfill → ingest → **Step 5 embed (INDUS, cuda, batch 256)** → Step 6 MV refresh → **Step 7 outbox drain** |
| `migrations/070_embedding_outbox.sql`                    | Outbox table + trigger; design rationale in-file                                                                                                                            |
| `scripts/qdrant_outbox_sync.py`                          | Outbox drain worker; `--backfill-since <date>` reconciles pre-trigger papers                                                                                                |
| `migrations/071_drop_paper_embeddings_indus_indexes.sql` | ADR-015 **Stage 1**: drop the two INDUS HNSW indexes ONLY, keep table+rows; soak-gated                                                                                      |
| `scripts/audit_paper_embeddings.py`                      | Read-only ADR-015 pre-flight (PG↔Qdrant count parity, reclaim bytes, soak clock) — now errors on the dropped table                                                          |

Why PG-first existed at all: `paper_embeddings` was the ADR-013 rollback
source of truth (rebuild-from-PG = 3.2 h measured) and the outbox source.
ADR-015 planned its retirement in two gated stages. That context is what makes
the incident in §4 an incident.

---

## 4. The s7cy live fire — taught as fact, dated 2026-07-03/07

**What was authorized** (ADR-015, `docs/ADR/015_offload_drop_paper_embeddings_indus.md`,
status "Proposed … NOT executed"):

- Stage 1 = drop the two INDUS HNSW **indexes only** (migration 071), keep
  table + rows; gated on the ADR-013 soak clock clearing **2026-07-11** (or
  documented evidence-based early sign-off) + audit pre-flight green.
- Stage 2b = `DROP TABLE paper_embeddings` — only after (a) verified NAS
  archive, (b) soak fully retired, (c) migration-070 trigger/outbox removed
  first, (d) sign-off + audit green.

**What happened** (bead s7cy, P1 OPEN, forensics 2026-07-03):

- A full `DROP TABLE paper_embeddings` (+ `embedding_outbox`) was executed
  **out-of-process ~2026-06-29/30**. None of the Stage-2b preconditions were
  satisfied. **No NAS archive exists** (`/mnt/scix_offload/paper_embeddings*`
  absent). Pilot-model rows (nomic/specter2/specter3) are lost; the in-PG
  ADR-013 rollback source is destroyed (pgvector rollback now means full
  re-embed).
- `schema_migrations` tops out at **68** — 069–072 were/are applied by hand,
  unrecorded. **There is no migration auto-runner in this repo.**
- Last clean `daily_sync` run: 2026-06-29. Every run since 2026-06-30 aborts
  at Step 5 (`psycopg.errors.UndefinedTable`), skipping Steps 6–7.
- Gap: **~82,950 papers** in PG without a dense vector (2026-07-03), growing
  ~1–3k/day. Those papers are invisible to the dense lane (BM25/body lanes
  still cover them); RRF quality degrades for recent content.
- Root cause chain: the drop ADR/migration work (bead ffgg, closed) was scoped
  "artifacts only, NO prod exec"; the drop landed in prod anyway, without the
  companion code cutover repointing `embed.py`/outbox.

**Lessons this fire buys** (mirror of the index-discipline rules — see
scix-index-and-storage-discipline and scix-change-control):

1. An authored-but-gated migration is not permission. Artifacts on disk ≠
   executed decision. Prod DDL requires the ADR's own gates, every time.
2. Never destroy a rollback source before its replacement path is live AND
   the archive precondition is verified (this is the second time this class
   of failure has occurred here; the first was the ADR-013 DiskANN loss).
3. Silent cron death is a detection gap: the abort ran unnoticed 2026-06-30 →
   2026-07-03. Check `logs/daily_sync.log` age/content when anything smells
   stale.

PROVISIONAL pending Stephanie (Q1): this incident is recorded here as a dated
operational note, per the provisional decision that the durable research
campaign is retrieval-quality integrity (see scix-research-frontier), not this
fire.

---

## 5. The in-flight fix — PROVISIONAL pending Stephanie (Q2): proposed, NOT landed

A direct-to-Qdrant remediation exists **only in the working tree** (verified
uncommitted on 2026-07-07: modified `scripts/daily_sync.sh`, `scripts/embed.py`,
`src/scix/embed.py`; untracked `src/scix/qdrant_dense.py`,
`migrations/072_indus_qdrant_synced.sql`, `scripts/seed_indus_qdrant_synced.py`,
`tests/test_qdrant_dense.py`, `tests/test_embed_qdrant_store.py`,
`tests/test_embed_pipeline_abort.py`). Bead s7cy records it as
"CODE DONE (awaiting prod-exec sign-off)". **No skill, doc, or agent may treat
it as the standard path until it is committed and signed off.** Teach committed
reality (§3); describe this fix as in-flight.

Design (from the working-tree files + bead s7cy):

- **No PG staging table.** `run_embedding_pipeline` upserts INDUS vectors
  straight into the serving collection via `src/scix/qdrant_dense.py`
  (`INDUS_COLLECTION = "scix_indus_v2_papers_s1"`; client built from
  `QDRANT_URL`, raises if unset).
- **Point contract must match the bulk load** (verified empirically in bead
  s7cy): id = `uuid5(uuid.NAMESPACE_URL, bibcode)` as a string; single unnamed
  768-d vector; payload = `{"bibcode": bibcode}`.
  (`qdrant_tools.bibcode_to_point_id` uses a DIFFERENT blake2b-int scheme that
  does NOT match this collection — known latent issue, separate follow-up.)
- **Watermark replaces the anti-join source**: migration 072 creates
  `indus_qdrant_synced(bibcode PK, synced_at)`; `_UNEMBEDDED_WHERE` LEFT JOINs
  it instead of `paper_embeddings`. Deliberately no FK to `papers`.
- **Ordering is load-bearing**: `upsert_dense(..., wait=True)` first, THEN
  insert watermark rows + commit. A crash before the durable Qdrant write
  leaves the paper unmarked → re-embedded next run. At-least-once + idempotent
  uuid5 upsert = safe.
- **One-time seed**: `scripts/seed_indus_qdrant_synced.py` scrolls the Qdrant
  collection and fills the watermark with the ~32.38M already-served bibcodes,
  so the pipeline only embeds the ~83K gap, not the whole corpus.
- **daily_sync goes 7 → 6 steps**: Step 7 (outbox drain) deleted; Step 5
  writes Qdrant directly.
- Reviewer-driven hardening already folded in: abort `threading.Event` +
  timed queue puts so a Qdrant failure cannot deadlock the writer thread in
  unattended cron (regression test `tests/test_embed_pipeline_abort.py`).

Rules while it stays uncommitted:

- **Do not write a competing fix.** Check `git status` + `bd show
scix_experiments-s7cy` first; the work exists.
- **Do not commit, rebase away, stash-drop, or `git checkout --` these files**
  without explicit operator direction — the working tree is currently the only
  copy.
- **Do not execute the prod steps** (apply 072 → seed → backfill embed). That
  sequence is HALT-branch-ready: operator sign-off required
  (PROVISIONAL pending Stephanie, Q5 — conservative gating). The planned
  sequence, for reference only, from bead s7cy: apply migration 072 by hand →
  `scix-batch python scripts/seed_indus_qdrant_synced.py --allow-prod` →
  `scix-batch` embed backfill (cuda) → verify Qdrant points ≈ papers count →
  dry-run `daily_sync.sh` completes with no abort. **Do not run casually** —
  every step touches prod PG and/or the serving Qdrant collection.

---

## 6. The payload-preservation hazard (e4xv) — re-introduced, live at HEAD

Mechanism: a Qdrant `upsert` **replaces the whole point, payload included**.
Any upsert that writes `payload={"bibcode": …}` erases whatever richer payload
the point carried.

History:

- ADR-008 (`docs/ADR/008_qdrant_payload_schema.md`) defines a canonical
  filter payload (7 indexed + 5 metadata fields) for in-engine filtering.
- Bead **e4xv** (CLOSED 2026-06-21) flagged exactly this wipe in
  `qdrant_outbox_sync.py` and was fixed on branch `bd/e4xv-outbox-payload-fix`
  (close note cites commit `134215b` + a shared `src/scix/qdrant_payload.py`).
  **That module does not exist at current HEAD** and
  `scripts/qdrant_outbox_sync.py:305` still builds `payload={"bibcode": bibcode}`
  — the fix never reached this line of history (the repo was re-inited
  ~2026-06-11 and branches diverged). The branch still exists locally;
  the hazard is live in committed code.
- The **uncommitted** `qdrant_dense.py::upsert_dense` re-introduces the same
  bibcode-only pattern, with an explicit docstring acknowledging it: it
  matches the current collection state (every point carries only `bibcode` —
  per the docstring and bead s7cy; not independently re-verified against live
  Qdrant), so today it destroys nothing.

The standing rule to enforce in review:

> If `scripts/backfill_qdrant_filter_fields.py` or
> `scripts/qdrant_reload_with_payload.py` is ever used to enrich points with
> ADR-008 payload, **every upsert path (outbox drain, `upsert_dense`, bulk
> loaders) must first become payload-preserving (merge or re-lookup), or the
> backfill must be re-run for every re-upserted bibcode.** Otherwise recovery
> re-embeds silently strip enrichment — a permanent, growing filtering gap.

---

## 7. Known-broken siblings and doc inconsistencies (2026-07-07)

| Artifact                                                                                | Problem                                                                                                                                                                      | Trust instead                                             |
| --------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------- |
| `scripts/embed_fast.py`, `scripts/embed_optimized.py`, `scripts/pilot_embed_compare.py` | All write to the dropped `paper_embeddings`; broken since ~2026-06-30 (not cron-invoked; daily path uses `scripts/embed.py`)                                                 | Bead s7cy collateral findings                             |
| `src/scix/qdrant_tools.py::bibcode_to_point_id`                                         | blake2b-int ids ≠ the serving collection's uuid5 scheme; also expects payload keys the points don't carry — its recommend/filter path is likely dead (unverified live)       | s7cy notes; `qdrant_dense.point_id` is the correct scheme |
| `scripts/qdrant_reload_with_payload.py` docstring                                       | Says `paper_embeddings` "was DROPPED 2026-06-14" — contradicts the verified timeline (last clean outbox drain 2026-06-29; drop ~06-29/30). Unresolved internal inconsistency | Bead s7cy forensics (has `to_regclass` + log evidence)    |
| `scripts/audit_paper_embeddings.py`                                                     | Reads the dropped table; its ADR-015 pre-flight role is moot post-drop                                                                                                       | —                                                         |
| CI green ≠ pipeline works                                                               | Embed tests mock the DB/Qdrant; nothing in CI exercises the dropped table                                                                                                    | This skill §1                                             |

---

## 8. Runbooks

### 8.1 Diagnose "is the embed pipeline healthy?" (read-only, safe to run)

```bash
bd show scix_experiments-s7cy                          # incident status — OPEN means still broken
git status --short scripts/embed.py src/scix/embed.py src/scix/qdrant_dense.py \
    scripts/daily_sync.sh migrations/072_indus_qdrant_synced.sql   # fix still uncommitted?
tail -50 logs/daily_sync.log                           # UndefinedTable at Step 5 = still on fire
grep -n "paper_embeddings" src/scix/embed.py scripts/*.py | head   # who still targets the dropped table
```

Do NOT "verify" by connecting to prod PG or Qdrant unless you are the operator
with `--allow-prod` discipline — the default DSN is production
(scix-db-safety-and-telemetry).

### 8.2 Run the daily embed manually — DO NOT RUN CASUALLY

Broken at committed HEAD (§1). Shown with its guards for when the fix lands.
This is prod-writing, GPU-loading, multi-minute work: it requires the
`scix-batch` wrapper — this installation's operational requirement, because
the host co-runs the Gas City supervisor and unwrapped heavy work gets it
OOM-killed (scix-memory-and-batch-discipline).

```bash
# prod embed of unembedded papers (the daily_sync Step 5 invocation):
scix-batch .venv/bin/python3 scripts/embed.py --model indus --batch-size 256 --device cuda -v

# small smoke (still prod-pointed by default — set --dsn or SCIX_TEST_DSN deliberately):
scix-batch .venv/bin/python3 scripts/embed.py --model indus --limit 100 --device cuda -v
```

`QDRANT_URL` must be set once the direct-to-Qdrant path lands (the fix raises
without it; local Qdrant convention is `http://127.0.0.1:6633` — that default
is dated 2026-07-07, from `scripts/seed_indus_qdrant_synced.py`).

### 8.3 Backfill the ~83K gap — HALT: operator-gated

The full sequence lives in §5 and bead s7cy. It is prod-exec, sign-off-gated
(PROVISIONAL pending Stephanie, Q5). If you are an agent: surface the plan,
do not execute. Tests for the fix exist and ship with it
(`tests/test_qdrant_dense.py`, `tests/test_embed_qdrant_store.py`,
`tests/test_embed_pipeline_abort.py` — all currently untracked).

### 8.4 Adding/changing an embedding model — ADR territory

Any new lane or model change is ADR-gated (768d pin, no paid APIs, halfvec-only
quantization). Route through scix-change-control. Mechanically: a new model
means a `MODEL_REGISTRY` + `POOLING` entry in `src/scix/embed.py`, its own
Qdrant collection (never reuse `scix_indus_v2_papers_s1`), and an eval run
against the gold sets (scix-eval-and-evidence) before any serving change.

---

## 9. Hard rules (each with the incident that bought it)

| Rule                                                                                | Why (incident)                                                              |
| ----------------------------------------------------------------------------------- | --------------------------------------------------------------------------- |
| Never execute an authored-but-gated migration without its ADR gates                 | s7cy: out-of-process `DROP TABLE`, rollback source destroyed, 83K-paper gap |
| Never destroy a rollback source before the replacement is live + archived           | s7cy; previously the ADR-013 DiskANN 56h loss                               |
| Qdrant-first, mark-after (`wait=True`, then watermark insert) in any new write path | s7cy fix design — crash safety via at-least-once                            |
| Match the serving point contract exactly (uuid5, unnamed 768-d vector)              | `qdrant_tools` blake2b mismatch = silently dead lookups                     |
| Never write bibcode-only payload once points carry ADR-008 enrichment               | e4xv wipe hazard (§6)                                                       |
| Heavy embed work only under `scix-batch`; prod writes only with `--allow-prod`      | Host co-runs the gascity supervisor; default DSN is prod                    |
| `title [SEP] abstract`, mean-pool, 768d — do not improvise input format             | Vectors become incomparable with the existing 32.38M points                 |

---

## Provenance and maintenance

Authored 2026-07-07 against working copy branch
`bd/0yp5-external-copy-accuracy-audit` @ `452ab86` (**not main**; local `main`
is at `56cdab9`, 28 commits behind this branch with 2 divergent commits).
Live-state facts (drop timeline, ~82,950 gap, point counts, migration
watermark 68) are from bead s7cy forensics dated 2026-07-03 and were NOT
re-verified against prod (requires operator-gated DB access). The s7cy fix
status ("uncommitted, awaiting sign-off") and everything marked PROVISIONAL
(Q1/Q2/Q5) are pending Stephanie's answers to the discovery questions.

Re-verify before trusting (all read-only):

```bash
git rev-parse --short HEAD && git branch --show-current     # has the pin moved?
bd show scix_experiments-s7cy | head -5                     # OPEN = fire still live
git status --short src/scix/qdrant_dense.py migrations/072_indus_qdrant_synced.sql  # fix landed yet?
git show HEAD:src/scix/embed.py | grep -c paper_embeddings  # >0 = HEAD still targets the dropped table
grep -n 'MODEL_REGISTRY\|"indus"' src/scix/embed.py | head -5   # model/pooling facts
grep -n 'payload=' scripts/qdrant_outbox_sync.py            # bibcode-only = e4xv hazard still live
grep -n 'Step 5' scripts/daily_sync.sh                      # 7-step (committed) vs 6-step (fix) shape
ls tests/ | grep -E 'embed|qdrant'                          # fix tests tracked yet? (git ls-files to compare)
```
