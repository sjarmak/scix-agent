# Runbook — paper_embeddings halfvec migration

**Bead**: `scix_experiments-0vy`
**Branch**: `storage/halfvec-migration`
**Goal**: migrate the 32.4M INDUS rows in `paper_embeddings.embedding`
  (float32 `vector(768)`) onto a `halfvec(768)` column + matching HNSW
  index, cutting index footprint ≥40% while preserving nDCG@10/Recall@10
  on the 50-query eval.

---

## 0. Architectural decision — Option B (shadow column) **with mandatory pre-drop of idx_embed_hnsw_indus**

We considered two paths:

| Option | Approach | Pros | Cons |
|---|---|---|---|
| A | `ALTER TABLE ... ALTER COLUMN embedding TYPE halfvec(768) USING ...` | Single DDL; no code changes beyond cast literal. | ACCESS EXCLUSIVE on the table for the full rewrite (~hours on 32M rows / 125 GB TOAST), blocks `scripts/embed.py` daily cron, single-shot failure rolls the whole thing back. |
| B | Add `embedding_hv halfvec(768)` shadow column, backfill in batches, build new HNSW, cut code over, drop old column later. | Mostly online (only INDUS dense queries see an outage; lexical and DB writes remain available). Batches checkpoint. Rollback = don't deploy code. Survives OOM kills. | More code (column-aware casts in `search.py`, dual write in `embed.py`). Doubles storage during transition. **Backfill speed requires `idx_embed_hnsw_indus` to be dropped first** — see warning below. |

**Chose B.** Online-ness matters more than code simplicity on a 253 GB table
with a daily embed pipeline. The added TOAST during transition (~125 GB)
is recovered in phase 5 when the legacy column is dropped.

> **Critical (verified 2026-04-29 against prod scix):** the original Option B
> assumption — "per-row UPDATEs of `embedding_hv` skip `idx_embed_hnsw_indus`
> because the indexed expression `embedding::vector(768)` is unchanged" — does
> **not** hold on this table. `pg_stat_user_tables` after a 1000-row probe:
> `n_tup_hot_upd=3`, `n_tup_newpage_upd=29959` (i.e. HOT optimization fails on
> 99.99% of rows). Heap fragmentation forces new tuple versions onto fresh
> pages, which in turn forces a full HNSW index entry rewrite per row.
>
> Measured cost: ~39 ms/row UPDATE with `idx_embed_hnsw_indus` present →
> 14 days for 32.4M rows. Non-viable.
>
> **Therefore phase 4 (backfill) MUST be preceded by dropping
> `idx_embed_hnsw_indus`.** Local MCP INDUS dense search is degraded for the
> backfill+rebuild window (~2-3 h total — see phase 5 timing). RRF in
> `src/scix/search.py` falls back to lexical-only during the window.
> Public `mcp.sjarmak.ai` is intentionally DOWN per ops state, so external
> impact is nil; coordinate with the local-MCP user before starting.

---

## 1. Pre-flight checks

Run these as `ds` on the scix host, **before** starting the migration:

```bash
# 1. M3 community recompute has completed.
#    Must see count(community_id_fine) >= 19M AND no live
#    recompute_citation_communities.py process.
psql "$SCIX_DSN" -c "SELECT count(community_id_fine) FROM paper_metrics;"
pgrep -af recompute_citation_communities.py || echo "M3 idle ✓"

# 2. pgvector >= 0.8.2 and halfvec opclass available.
psql "$SCIX_DSN" -c "SELECT extversion FROM pg_extension WHERE extname='vector';"
psql "$SCIX_DSN" -c "SELECT opcname FROM pg_opclass WHERE opcname='halfvec_cosine_ops';"

# 3. Disk headroom on /var/lib/postgresql (or wherever PGDATA lives).
#    Need ~130 GB free during backfill (shadow column TOAST) + ~60 GB for new index.
df -h /var/lib/postgresql

# 4. scix-batch wrapper is on PATH.
command -v scix-batch

# 5. Current embed.py cron is quiet (won't contend on writes).
systemctl list-timers | grep -i embed || crontab -l | grep embed

# 6. Baseline index size captured.
psql "$SCIX_DSN" -c "\
  SELECT indexname, pg_size_pretty(pg_relation_size(indexname::regclass)) \
    FROM pg_indexes WHERE tablename='paper_embeddings';"
```

Save the pre-migration sizes to
`results/halfvec_migration/sizes_before.txt`.

---

## 2. Baseline eval (pre-migration)

```bash
scix-batch python scripts/eval_retrieval_50q.py \
    --full-corpus \
    --seed-papers 50 \
    --random-seed 42 \
    --json-output results/halfvec_migration/baseline.json \
    --output      results/halfvec_migration/baseline.md
```

This runs the citation-based 50-query harness (Section 4.4 of the paper)
against the full 32.4M corpus with `random_seed=42` so pre/post use the
same seed papers + ground-truth citation sets. Methods include
`indus`, `lexical`, and `hybrid_indus` by default under `--full-corpus`.

Capture `nDCG@10`, `Recall@10`, `Recall@20`, `MRR` per method plus the
timing histograms the harness prints. Paste them into
`results/halfvec_migration/baseline.json` (the harness writes it).

---

## 3. Schema prep — migration 053

Apply the "add shadow column" migration:

```bash
scix-batch psql "$SCIX_DSN" -v ON_ERROR_STOP=1 \
    -f migrations/053_paper_embeddings_halfvec.sql
```

This is metadata-only: `ALTER TABLE ... ADD COLUMN ... halfvec(768)`
doesn't rewrite data. Takes < 1 s even on 32M rows.

**Verify**:
```bash
psql "$SCIX_DSN" -c "\\d paper_embeddings" | grep embedding_hv
psql "$SCIX_DSN" -c "SELECT count(*) FROM halfvec_backfill_progress;"
```

---

## 4. Backfill — scripts/backfill_halfvec.py

> **MUST drop `idx_embed_hnsw_indus` before this phase** (see §0). Without
> the drop, per-row UPDATE cost is ~39 ms/row → 14 days. Once the index is
> gone the cost drops back to the ~4 µs/row range that the runbook originally
> assumed.

```bash
# Step 0 (pre-backfill): drop the legacy HNSW partial index.
# This is what makes the backfill feasible. INDUS dense search is offline
# from this point until phase 5 finishes.
scix-batch psql "$SCIX_DSN" -v ON_ERROR_STOP=1 \
    -c "DROP INDEX CONCURRENTLY IF EXISTS idx_embed_hnsw_indus;"
```

Populates `embedding_hv` for all `model_name='indus'` rows. Idempotent,
batched (default 20k/batch), signal-safe (SIGTERM finishes current batch).

```bash
scix-batch --mem-high 10G --mem-max 15G \
    python scripts/backfill_halfvec.py \
        --dsn "$SCIX_DSN" \
        --model indus \
        --batch-size 20000 \
        2>&1 | tee logs/halfvec_backfill.log
```

**Estimated runtime**: 32.4M rows / 20k batch = ~1600 batches. After the
HNSW drop, per-batch cost is dominated by the toast rewrite (~30 MB WAL/batch)
rather than HNSW graph maintenance, so wall clock returns to the ~60-90 min
range. With the index still present, batches stalled at ~6+ minutes apiece
on this host (verified 2026-04-29).

**Monitoring** (from another tmux pane):
```bash
watch -n 30 'psql "$SCIX_DSN" -c "
    SELECT rows_updated, last_bibcode, updated_at
      FROM halfvec_backfill_progress
     ORDER BY started_at DESC LIMIT 1;"'
```

**Resumption**: if the script is killed, re-run the same command. It
reads the open progress row and continues from `last_bibcode`.

**Completion check**:
```bash
psql "$SCIX_DSN" -c "
    SELECT count(*) FILTER (WHERE embedding_hv IS NULL) AS missing
      FROM paper_embeddings WHERE model_name='indus';"
# Expect 0
```

---

## 5. New HNSW index — migration 054

Built with `CREATE INDEX CONCURRENTLY` so reads + writes continue.

```bash
scix-batch --mem-high 20G --mem-max 30G \
    psql "$SCIX_DSN" -v ON_ERROR_STOP=1 \
        -f migrations/054_paper_embeddings_halfvec_index.sql \
        2>&1 | tee logs/halfvec_index_build.log
```

**Estimated runtime**: 45-90 minutes on this host
(maintenance_work_mem=8GB, max_parallel_maintenance_workers=7).

> **Build-time gotcha (verified 2026-04-30):** with the default 8 GB
> `maintenance_work_mem`, the HNSW build hits "graph no longer fits into
> maintenance_work_mem after 3870730 tuples" and spills to disk. Spilled
> builds become 10×+ slower and risk getting killed by systemd-oomd (the
> first 2026-04-29 build ran 21 hours and was terminated mid-stream,
> leaving an INVALID index — `indisvalid=false`/`indisready=false` —
> which the planner ignores. Symptom: every INDUS dense query falls back
> to a Parallel Seq Scan over 32M rows / ~5-10 min wall-clock per query,
> verified via `EXPLAIN`).
>
> Recovery from an INVALID index:
> ```sql
> -- The terminated CREATE INDEX backend may still be alive holding locks;
> -- terminate it first or DROP INDEX CONCURRENTLY will block forever.
> SELECT pg_terminate_backend(pid)
>   FROM pg_stat_activity
>  WHERE query LIKE 'CREATE INDEX%idx_embed_hnsw_indus_hv%';
> DROP INDEX CONCURRENTLY IF EXISTS idx_embed_hnsw_indus_hv;
> ```
>
> For the rebuild, prefer **non-concurrent** `CREATE INDEX` when no
> concurrent writers to `paper_embeddings` exist (verified via
> `crontab -l` + `pg_stat_activity`). Non-concurrent is significantly
> faster because it avoids the 2-phase coordination overhead. The
> ACCESS EXCLUSIVE lock is fine since INDUS dense search is already
> offline at this stage anyway.

**Monitor progress** (from another pane):
```bash
psql "$SCIX_DSN" -c "
    SELECT phase, round(100 * blocks_done::numeric / blocks_total, 1) AS pct
      FROM pg_stat_progress_create_index;"
```

**Expected final size**: ~60 GB (half of the 120 GB `vector_cosine_ops`
index — halfvec is 2 bytes/dim instead of 4).

**If the build fails partway** (leaves an INVALID index):
```bash
psql "$SCIX_DSN" -c "DROP INDEX CONCURRENTLY IF EXISTS idx_embed_hnsw_indus_hv;"
# Then re-run migration 054.
```

---

## 6. Code cutover

Deploy this PR (`storage/halfvec-migration`). The only runtime changes:

- `src/scix/search.py` — `vector_search` / `_filter_first_vector_search`
  route INDUS queries to `pe.embedding_hv <=> $1::halfvec(768)` against
  `idx_embed_hnsw_indus_hv`. Pilot models unaffected.
- `src/scix/embed.py` — INDUS writes populate `embedding_hv`
  (`embedding` stays NULL for new INDUS rows); pilots unchanged.

Verify in dev:

```bash
python -c "
from scix.db import get_connection
from scix.search import vector_search
conn = get_connection()
import random
vec = [random.random() for _ in range(768)]
r = vector_search(conn, vec, model_name='indus', limit=5)
for p in r.papers: print(p['bibcode'], p['score'])
"
```

Then restart the MCP server so it picks up the new code:

```bash
cd ~/projects/scix_experiments
./deploy/run.sh  # rebuild image + restart
curl -sf https://mcp.sjarmak.ai/health
```

---

## 7. Post-migration eval

Re-run the 50-query eval against the new index and diff. The `--random-seed
42` flag is critical — it guarantees the same seed papers so paired stats
are valid:

```bash
scix-batch python scripts/eval_retrieval_50q.py \
    --full-corpus \
    --seed-papers 50 \
    --random-seed 42 \
    --json-output results/halfvec_migration/post.json \
    --output      results/halfvec_migration/post.md

python - <<'PY'
import json, pathlib
base = json.loads(pathlib.Path('results/halfvec_migration/baseline.json').read_text())
post = json.loads(pathlib.Path('results/halfvec_migration/post.json').read_text())
def summarize(doc):
    return {m['method']: m for m in doc.get('summaries', [])}
bs, ps = summarize(base), summarize(post)
for method in sorted(set(bs) | set(ps)):
    b, p = bs.get(method, {}), ps.get(method, {})
    for k in ('ndcg_at_10','recall_at_10','recall_at_20','mrr'):
        bv, pv = b.get(k), p.get(k)
        if bv is None or pv is None: continue
        print(f"{method:18s} {k:15s} base={bv:.4f} post={pv:.4f} delta={pv - bv:+.4f}")
PY
```

**Acceptance gates**:
- `ΔnDCG@10`  ≥ -0.005  (≤ 0.5 pt drop)
- `ΔRecall@10`  ≥ -0.01
- `p50_ms_post` ≤ `p50_ms_base * 1.20` (expect improvement — smaller index = better cache locality)

If any gate fails: **do not drop the old index**. Fall back by reverting
the PR; the old `idx_embed_hnsw_indus` and `embedding` column are still
present and queries revert seamlessly.

---

## 8. Index footprint verification

```bash
psql "$SCIX_DSN" -c "
    SELECT indexname, pg_size_pretty(pg_relation_size(indexname::regclass)) AS size
      FROM pg_indexes
     WHERE tablename='paper_embeddings'
       AND indexname LIKE 'idx_embed_hnsw_indus%';"
```

Write the comparison to `results/halfvec_migration/sizes_after.txt` and
confirm `idx_embed_hnsw_indus_hv` is ≥40% smaller than the pre-migration
`idx_embed_hnsw_indus`.

**Result (2026-04-30):** 120 GB → 24 GB = **80.0% reduction** (96 GB freed).
Acceptance gate (≥40%) passes by a wide margin. See
`results/halfvec_migration/sizes_after.txt`.

---

## 9. Cleanup (later, gated on stability)

`idx_embed_hnsw_indus` is already dropped in phase 4 (the original "later"
DROP ran early, by necessity). Remaining cleanup:

1. `VACUUM (ANALYZE, VERBOSE) paper_embeddings;` to reclaim TOAST space
   left over from the now-unused `embedding` values in INDUS rows.
2. Eventual `DROP COLUMN embedding` (separate follow-up once
   `paper_embeddings_nomic` / `_specter2` pilots are retired — see bead
   `scix_experiments-2xe` successors).

---

## 10. Rollback

| Stage reached | Rollback |
|---|---|
| Migration 053 only | `ALTER TABLE paper_embeddings DROP COLUMN embedding_hv; DROP TABLE halfvec_backfill_progress;` |
| `idx_embed_hnsw_indus` dropped, backfill partially complete | INDUS dense is already offline. To exit early: drop column as above, then `CREATE INDEX CONCURRENTLY idx_embed_hnsw_indus ON paper_embeddings USING hnsw ((embedding::vector(768)) vector_cosine_ops) WITH (m=16, ef_construction=64) WHERE model_name='indus';` (~6 h to rebuild against the `vector` column). |
| Index 054 built, code not deployed | Set `SCIX_USE_HALFVEC=0` in env so queries fall back to the legacy path. NOTE: `idx_embed_hnsw_indus` is gone, so legacy fallback is seq-scan-tier slow (~44 s/query) until the old index is rebuilt or the new halfvec index is wired up. |
| Code deployed, eval regression | `SCIX_USE_HALFVEC=0` cuts queries back to the legacy `embedding` column, but with `idx_embed_hnsw_indus` gone, that path is seq-scan slow. Practical fallback is to drop `idx_embed_hnsw_indus_hv` and rebuild `idx_embed_hnsw_indus` (~6 h). |

---

## 11. Known risks

- **TOAST bloat during backfill.** Every `UPDATE` writes a new TOAST row
  for `embedding_hv`. `paper_embeddings` hasn't been VACUUM-FULLed in
  ages; partial vacuums happen via autovacuum. If bloat becomes a
  concern mid-run, pause the backfill and run a targeted
  `VACUUM (ANALYZE) paper_embeddings` between batches.
- **HNSW build memory.** `maintenance_work_mem=8GB` × 7 workers can
  theoretically hit 56 GB peak. Keep the scix-batch memory limits in
  phase 5; the per-worker allocation is usually far below the limit
  because vectors are only 768×2 bytes = 1.5 KB each.
- **Concurrent M3 recompute.** Do not start at the same time as
  `scripts/recompute_citation_communities.py` — they compete for I/O
  and M3 is the OOM-kill priority target at ~34 GB RSS.
- **Query regression in iterative_scan.** pgvector 0.8's iterative scan
  is validated on `vector_cosine_ops`; halfvec_cosine_ops uses the same
  code path but we have no historical data. The 50-query eval in
  phase 7 is the guardrail.
- **LHS cast on `embedding_hv` defeats HNSW match (verified 2026-04-30).**
  Migration 054 indexes `embedding_hv halfvec_cosine_ops` with NO
  expression wrapper — but the legacy code in `src/scix/search.py` was
  templated against the pilot indexes `((embedding)::vector(768))` and
  applied the same `({vec_col})::{vec_cast}` to the halfvec column.
  Result: planner can't match the indexed expression and falls back to
  Seq Scan over 32M rows (verified ~12 min wall-clock vs sub-second).
  Fixed by gating on `use_halfvec`: bare `pe.embedding_hv` for the
  halfvec path, `(pe.embedding)::vector(768)` for the pilot path.
  Unit test `test_indus_query_uses_embedding_hv_no_lhs_cast` is the
  regression guard.
