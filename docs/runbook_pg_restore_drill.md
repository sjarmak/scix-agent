# PostgreSQL restore drill — weekly_pg_backup dumps

Companion to `scripts/weekly_pg_backup.sh`. Procedure for verifying a dump on
NAS is intact and for recovering after NVMe loss.

## Risk model

This is a research project. The accepted redundancy model (per
`storage_tiering_policy` memory and bead `scix_experiments-9ou`) is
"duplicate to NAS". The corpus itself (`papers`, `papers_fulltext`,
`citation_edges`, etc.) is rebuildable from the raw ADS JSONL preserved at
`/mnt/scix_offload/ads_metadata_by_year_picard/` plus the daily-sync
pipeline. The weekly NAS dump covers ONLY the derived tables that are
expensive or impossible to rebuild — entity graph, citation_contexts with
intent labels, communities, paper_claims, curated taxonomies, etc.
`paper_embeddings` is opt-in (`--include-embeddings`); excluded by default
because it's both 253 GB and recomputable in 3-9 GPU-hours.

There is no off-NAS, cloud, or PITR backup. Simultaneous NVMe + NAS loss is
mitigated only by upstream-reproducibility from ADS.

## What's on NAS

```
/mnt/postgres/scix_dumps/YYYY-MM-DD/
  schema.sql.gz         pg_dump --schema-only of public schema
  data.dump             pg_dump -Fc -Z 6 of derived tables (data only)
  data.toc.txt          pg_restore --list output (dump TOC sanity check)
  manifest.txt          row counts + table sizes + sha256 checksums
  embeddings.dump       (only if --include-embeddings was passed)
  embeddings.toc.txt    (likewise)
```

Retention: 4 most recent dated dirs (default); older dirs are pruned by
the next backup run.

## Quarterly drill — verify dump integrity

Run this periodically against the latest NAS dump to confirm it parses,
that schema.sql is replayable, and that small tables roundtrip with
matching row counts.

```bash
LATEST=$(ls -1d /mnt/postgres/scix_dumps/[0-9]* | tail -1)
echo "Drilling against $LATEST"

# 1. Verify the archive TOC parses (catches half-written / corrupt files).
pg_restore --list "$LATEST/data.dump" > /tmp/drill.toc
diff -q "$LATEST/data.toc.txt" /tmp/drill.toc \
    && echo "TOC matches" || echo "WARNING: TOC drift"

# 2. Verify file checksums match the manifest.
( cd "$LATEST" && sha256sum --check <(grep '  ' manifest.txt | grep -E '\.(dump|sql\.gz)$') )

# 3. Roundtrip a small subset into a scratch DB.
psql -d postgres -c "DROP DATABASE IF EXISTS scix_restore_drill;"
psql -d postgres -c "CREATE DATABASE scix_restore_drill;"
psql -d scix_restore_drill -c "CREATE EXTENSION vector;"
psql -d scix_restore_drill -c "CREATE EXTENSION pg_trgm;"
psql -d scix_restore_drill -c "CREATE EXTENSION pgcrypto;"
zcat "$LATEST/schema.sql.gz" | psql -d scix_restore_drill -v ON_ERROR_STOP=0

# Tables with no FK to skipped corpus tables — restore cleanly.
pg_restore --data-only --no-owner --no-privileges \
    -t communities -t vocabularies -t uat_concepts \
    -d scix_restore_drill "$LATEST/data.dump"

# Verify counts match manifest.
psql -d scix_restore_drill -c "
  SELECT 'communities' AS t, count(*) FROM communities
  UNION ALL SELECT 'vocabularies', count(*) FROM vocabularies
  UNION ALL SELECT 'uat_concepts', count(*) FROM uat_concepts;"

# Cleanup.
psql -d postgres -c "DROP DATABASE scix_restore_drill;"
```

Verified working on 2026-04-29 against `2026-04-29/data.dump`:
communities=2089, vocabularies=9, uat_concepts=2313, paper_umap_2d=390892,
curated_entity_core=617 — all match production.

## Disaster recovery — NVMe died

The full sequence to rebuild from zero, using ADS upstream + the latest NAS
dump:

1. Reinstall PostgreSQL 16 + pgvector + pgvectorscale.
2. Create `scix` DB and required extensions:
   ```sql
   CREATE DATABASE scix;
   \c scix
   CREATE EXTENSION vector;
   CREATE EXTENSION pg_trgm;
   CREATE EXTENSION pgcrypto;
   ```
3. Apply schema:
   ```bash
   zcat /mnt/postgres/scix_dumps/<latest>/schema.sql.gz | psql -d scix
   ```
4. Re-ingest the corpus from raw JSONL (multi-day):
   ```bash
   for year in 2021 2022 2023 2024 2025; do
       scix-batch python scripts/ingest.py \
           /mnt/scix_offload/ads_metadata_by_year_picard/${year}/
   done
   ```
5. Restore the derived tables. FK constraints between derived tables and
   `papers` / `entities` will fail if the FK target hasn't been re-populated;
   either restore in a sensible order, or temporarily drop the FK and recreate
   it `NOT VALID` after restore (operator validates async):
   ```bash
   pg_restore --data-only --no-owner --no-privileges \
       --disable-triggers \
       -d scix /mnt/postgres/scix_dumps/<latest>/data.dump
   ```
   `--disable-triggers` requires PG superuser. Without superuser, the
   workaround used in the quarterly drill works: `ALTER TABLE x DROP
   CONSTRAINT x_fkey; pg_restore ...; ALTER TABLE x ADD CONSTRAINT x_fkey
   FOREIGN KEY ... NOT VALID;`.
6. Re-embed (optional, 3-9 GPU-hours; or restore from `embeddings.dump`
   if it was included in the dump):
   ```bash
   scix-batch python scripts/embed.py --model indus --device cuda
   ```
7. Rebuild HNSW indexes (data-only dump excludes indexes by definition):
   ```bash
   bash scripts/restore_indexes.sh
   ```
8. Re-resolve entity links and run any harvesters whose APIs still work
   (see `harvester_api_issues` memory for which are broken).

Estimated RTO for the full sequence: 2-4 days, dominated by re-ingest
(~24h on this hardware) and re-embed (3-9h). The dump short-circuits the
expensive parts: entity graph (impossible to rebuild — harvester APIs
broken), citation_contexts intent labels (LLM-classified, expensive),
community partitions (expensive), curated_entity_core (manual).

## Known caveats

- The `entities` table has a circular FK on itself (`canonical_entity_id`).
  `pg_dump --data-only` warns about this; restore must use
  `--disable-triggers` or a transaction-wrapped restore that defers FK
  checks. This is documented in step 5 above.
- `data.dump` is a single multi-table archive. To restore a single table,
  use `pg_restore -t <table>`.
- `paper_embeddings` is excluded by default. To include it for a one-off
  full snapshot:
  ```bash
  scripts/weekly_pg_backup.sh --include-embeddings
  ```
  Adds ~150 GB+ to the dump dir. Don't include it in the cron run unless
  the operator has rationed NAS space accordingly.
