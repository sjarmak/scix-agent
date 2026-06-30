# ADR-015: Offload and Retire the INDUS Footprint in `paper_embeddings` to Unblock Prod Disk

- **Status**: Proposed (artifacts authored 2026-06-22; NOT executed — gated on operator/ADR sign-off + audit pre-flight)
- **Deciders**: Stephanie Jarmak (operator approval at every external/destructive step); SciX maintainers
- **Scope**: the `model_name='indus'` footprint inside the multi-model `paper_embeddings` table — specifically the two INDUS HNSW indexes (`idx_embed_hnsw_indus`, `idx_embed_hnsw_indus_hv`) and, in a later stage, the INDUS row/column data. Postgres remains authoritative for everything else; the pilot-model rows (`nomic`, `specter2`, `specter3`) in the same table are explicitly out of scope.
- **Supersedes**: nothing. **Implements** the never-authored `058_drop_paper_embeddings_indus_indexes.sql` named by `docs/prd/qdrant_nas_migration.md` MH-17 (slot 058 on disk is `correction_events`; the index-drop migration was never written).
- **Related ADRs**: ADR-013 (dense lane serves from Qdrant; `paper_embeddings` is the rollback source of truth through the soak), ADR-011 (drop-and-reclaim pattern — its `VACUUM FULL`/450 GB-free caveat applies to **row/column** reclaim, NOT to the index drop; see Decision §Stage 1).
- **Related beads**: `scix_experiments-ffgg` (this authoring task), `q4z8` (DEEP_AUDIT 2026-06-22 disk follow-up), `uy40` (RAM-relief HOLD — `results/qdrant_ram_relief_options.md`), `8m0a` (outbox sync).

## Context

The entire scix-experiments rig is walled by prod NVMe at **~98 % (≈42 GB free / 1.9 TB)**. The documented relief valve — offload `paper_embeddings`'s INDUS footprint now that Qdrant (`scix_indus_v2_papers_s1`, 32.4 M points, green) serves the dense lane — has **no implementation on `main`**. The PRD (`qdrant_nas_migration.md`) names `scripts/qdrant_backfill_indus.py`, `scripts/audit_paper_embeddings.py`, and `migrations/058_drop_paper_embeddings_indus_indexes.sql`; none exist. The Qdrant re-ingest is effectively done (the collection is loaded and canaried, `results/qdrant_canary_v1.md`), so what is missing is the **safe disk-freeing path**, not the cutover.

Three facts reshape the naïve "drop `paper_embeddings` (~195 GB)" framing and are the core of this ADR:

1. **`paper_embeddings` is a multi-model table.** It carries `model_name='indus'` (the full 32 M corpus, the bulk of the bytes) alongside small pilot-model rows (`nomic`, `specter2`, `specter3`, ~20 K rows each — eval-only). A blanket `DROP TABLE` would destroy the pilots and the outbox trigger source. The unit of offload is the **INDUS footprint**, not the table.

2. **The INDUS rows are still the rollback source of truth AND the outbox source.** ADR-013 keeps `paper_embeddings` as the authoritative store through the soak; rebuild-from-PG is the measured 3.2 h recovery path (vs 56 h for the failed DiskANN build). Migration 070's trigger writes new INDUS vectors here and the outbox (`scripts/qdrant_outbox_sync.py`) ships them to Qdrant. Deleting the rows removes both the rollback source and the outbox source — that is a separate, heavier decision than freeing the index bytes.

3. **The disk weight is dominated by the two INDUS HNSW indexes, and `DROP INDEX` reclaims to the OS immediately.** `idx_embed_hnsw_indus` is ~120 GB (RAM-resident by design, ADR-013); `idx_embed_hnsw_indus_hv` (halfvec, migration 054) adds tens of GB more. `DROP INDEX` unlinks the index relation files at commit — the space returns to the OS directly, with **no `VACUUM FULL`, no `pg_repack`, and no ~450 GB-free precondition**. The ADR-011 reclaim caveat (which the bead description inherited) applies only to in-table `DROP COLUMN`/`DELETE` reclaim — i.e. our Stage 2 — not to the index drop.

The consequence: the heaviest, immediate, low-risk disk win is **dropping the INDUS HNSW indexes**, which keeps the rollback source intact. That is the relief valve. The full row/column reclaim is a smaller, later, more conditional win.

## The soak hold — is it satisfied? (No, not yet.)

ADR-013's dense-lane cutover flipped to production on **2026-06-11**. The PRD is unambiguous (lines 108, 120, 243):

> Keep `paper_embeddings` and the existing pgvector HNSW indexes untouched for **≥30 days post-cutover**. No destructive cleanup until rollback can be retired with evidence. … Day 31 post-flip, a runbook step prompts the operator to run the index-drop migration … Until day 31 fires, `VACUUM FULL` on `paper_embeddings` is forbidden and a CI lint flags any PR that adds it.

- **Cutover date:** 2026-06-11.
- **Day 31 (soak clears):** **2026-07-11**.
- **Today:** 2026-06-22 — **day 11; 19 days of soak remain.**

So as of authoring, **the relief valve is hold-blocked.** The pgvector INDUS indexes ARE the rollback serving path; dropping them mid-soak forfeits the one-env-flag rollback that the whole migration was designed around. This ADR therefore authors the artifacts but pins execution behind the soak clock. Two paths to unblock execution:

- **(a) Wait for 2026-07-11** and execute Stage 1 then (the PRD-sanctioned default).
- **(b) Evidence-based early retirement** — an explicit operator decision to shorten the soak, justified by banked canary evidence: G1 is Δ=0.0000 on nDCG@10/MRR/recall@50 on-disk (`qdrant_canary_v1.md`); the only failing gate is G3 (10-thread p95 latency *variance*, a serving-RAM issue, not a correctness or recall issue — `results/qdrant_ram_relief_options.md`). Early retirement is permitted **only** with per-`feedback_no_destructive_cleanup_without_evidence.md` sign-off and the audit pre-flight green. This ADR does not pre-authorize (b); it documents it as the operator's lever.

The pre-flight that informs either path is `scripts/audit_paper_embeddings.py` (this ADR's companion artifact).

## Decision

A staged offload, each stage independently gated. **Author all artifacts now; execute nothing without sign-off.**

### Stage 0 — Preconditions (read-only, do now)

1. **Run the audit** (`scripts/audit_paper_embeddings.py --allow-prod`, under `scix-batch`). It is read-only and verifies, as a single pre-flight:
   - PG INDUS row count (`count(*) … WHERE model_name='indus'`) vs the Qdrant collection `points_count` — parity is the precondition that the derived cache fully covers the source.
   - presence + freshness of a NAS archive of the INDUS rows, if one has been taken.
   - presence of a Qdrant snapshot on NAS (`scripts/qdrant_snapshot_to_nas.sh`).
   - reclaimable bytes per index and per the row/column footprint (`pg_relation_size`/`pg_total_relation_size`), so the operator sees the exact disk yield of each stage.
   - the soak-clock status (days since 2026-06-11 cutover) so the go/no-go is explicit.
2. **NAS archive precondition for Stage 1 is light:** because Stage 1 keeps the INDUS rows, the rollback source of truth never leaves Postgres — no row archive is strictly required to drop the *indexes*. A Qdrant snapshot on NAS is still recommended (cheap insurance for the derived cache) but is not a Stage-1 blocker. The heavy NAS archive precondition belongs to **Stage 2**.

### Stage 1 — Drop the INDUS HNSW indexes (the relief valve; soak-gated)

`migrations/071_drop_paper_embeddings_indus_indexes.sql` — drops `idx_embed_hnsw_indus` and `idx_embed_hnsw_indus_hv` with `DROP INDEX CONCURRENTLY IF EXISTS` (no long lock; not inside a txn). Both are **partial, INDUS-only** indexes (`WHERE model_name='indus'`), so the pilot-model indexes (`idx_embed_hnsw_nomic`, `idx_embed_hnsw_specter2`) and rows are untouched.

- **Disk yield:** ~120 GB (vector index) + the halfvec index, reclaimed **to the OS at commit** — no `VACUUM FULL`, no 450 GB-free precondition. This alone clears the prod-disk wall.
- **Rollback preserved:** the INDUS rows remain. Rollback to pgvector serving = rebuild the index (`CREATE INDEX CONCURRENTLY`, the migration 054 procedure) from the still-present rows, then unset `QDRANT_URL`. Qdrant rebuild-from-PG (3.2 h) also remains available. The outbox source is intact.
- **Gate:** soak clear (2026-07-11) OR documented evidence-based early-retirement sign-off; audit pre-flight green (PG↔Qdrant count parity).

### Stage 2 — Reclaim the INDUS row/column bytes (later, separate decision; NOT pre-authored as an applied migration)

The PRD retains the **row data for one further calendar quarter** after the index drop (line 243). Reclaiming it is a distinct decision whose *mechanism* depends on facts not yet fixed, so this ADR deliberately does **not** ship a Stage-2 migration file (that would be placeholder code committing to an undecided shape). The two candidate mechanisms, to be chosen at sign-off time with the audit output in hand:

- **Stage 2a — keep pilots:** archive INDUS rows to NAS (`pg_dump`/`COPY … WHERE model_name='indus'` → `/mnt/scix_offload/paper_embeddings_indus_archive/…`, row count must equal the live count at archive time), then `DELETE FROM paper_embeddings WHERE model_name='indus'` followed by `pg_repack -t paper_embeddings` (online) or a `VACUUM FULL` maintenance window. **This is where the ADR-011 caveat bites:** `DELETE` returns pages to the free-space map, not the OS; only the repack/rewrite shrinks the relation file, and a `VACUUM FULL` transiently needs free disk ≈ the live table size. Sequence it **after** Stage 1 has already freed the index bytes, so the repack has headroom.
- **Stage 2b — pilots also disposable:** if the eval-only pilot rows are confirmed retired, `DROP TABLE paper_embeddings` (after the trigger/outbox in migration 070 is removed) frees table + TOAST + all indexes to the OS at commit, no repack. Cleanest, but forfeits the pilot embeddings and the in-PG rollback source entirely — only acceptable once the NAS archive + Qdrant snapshot are both verified and the soak is fully retired.

Stage 2's NAS archive is the **hard** precondition (it becomes the only rollback/rebuild source once the rows leave PG). The decision and chosen mechanism get their own bead, their own migration file, and their own sign-off.

## Consequences

### Positive
- Stage 1 alone clears the prod-disk wall (~120 GB+ to the OS, immediately), unblocking the dense-lane RAM-relief work (`uy40`), the PDS/IAU ingests (`dbl.16`), and any reclaim. It needs neither `VACUUM FULL` nor 450 GB free.
- The rollback contract ADR-013 depends on survives Stage 1 untouched: rows stay, rebuild-from-PG and the env-flag rollback both still work.
- Pilot-model embeddings and the outbox are unaffected (partial INDUS-only indexes; row drop deferred to Stage 2).
- Corrects the bead's premise (table-drop + mandatory `VACUUM FULL`) to the accurate, lower-risk, higher-yield index-drop-first path.

### Negative / Caveats
- **Stage 1 must respect the soak.** Executing before 2026-07-11 without explicit evidence-based sign-off violates the PRD rollback guarantee. The artifacts are inert until that gate clears; the migration carries the precondition in its header and the audit prints the clock.
- **Dropping the INDUS indexes makes pgvector rollback *slower*, not impossible.** Post-Stage-1, reverting to pgvector serving requires rebuilding the HNSW index (~45–90 min, migration 054 procedure) before the env-flag flip is meaningful. The rollback stops being instantaneous. Operators must treat Stage 1 as the point where "instant rollback" degrades to "rebuild-then-rollback" — acceptable only once Qdrant has soaked.
- **Stage 2 deletes the in-PG rollback source.** Once INDUS rows leave `paper_embeddings`, the only rebuild source is the NAS archive (or re-embedding the corpus through INDUS). This is why Stage 2 gates on a verified NAS archive and a fully-retired soak, and is intentionally not pre-authored here.
- **Outbox interaction.** Stage 2b's `DROP TABLE` requires first removing migration 070's trigger and quiescing `qdrant_outbox_sync.py`; Stage 2a's `DELETE` leaves the trigger firing `DELETE` outbox events — the sync worker must tolerate or be paused. Audit/sign-off must check outbox lag is zero before either.

## Rollback Plan

- **Undo Stage 1:** rebuild the indexes from the still-present rows —
  ```sql
  CREATE INDEX CONCURRENTLY idx_embed_hnsw_indus_hv
      ON paper_embeddings USING hnsw (embedding_hv halfvec_cosine_ops)
      WITH (m = 32, ef_construction = 256)
      WHERE model_name = 'indus';
  -- legacy vector index, only if reverting off halfvec:
  CREATE INDEX CONCURRENTLY idx_embed_hnsw_indus
      ON paper_embeddings USING hnsw ((embedding::vector(768)) vector_cosine_ops)
      WITH (m = 16, ef_construction = 64)
      WHERE model_name = 'indus';
  ```
  then unset `QDRANT_URL` and restart the MCP. No data movement.
- **Undo Stage 2:** re-ingest INDUS rows from the NAS archive (`COPY paper_embeddings FROM …`) or re-embed via `scripts/embed.py`/the INDUS pipeline; then rebuild the index as above. Cost: dominated by the re-embed/restore, hours not minutes.

## Open Questions

1. **Soak path:** wait to 2026-07-11 (default), or seek evidence-based early-retirement sign-off given the disk emergency? The audit's clock + parity output is the input to that call.
2. **Pilot disposability (decides Stage 2a vs 2b):** are `nomic`/`specter2`/`specter3` rows still needed for any eval? If retired, Stage 2b (`DROP TABLE`) is materially cleaner. Confirm before authoring the Stage-2 migration.
3. **Halfvec serving state:** `idx_embed_hnsw_indus` (legacy `vector`) vs `idx_embed_hnsw_indus_hv` (halfvec) — is the legacy index already redundant (serving fully cut to Qdrant)? If so, dropping it is pure win even pre-soak-end as a *non-rollback* index, pending confirmation it is not the rollback target. The audit reports both indexes' sizes and last-scan stats to inform this.
