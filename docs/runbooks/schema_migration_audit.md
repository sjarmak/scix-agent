# Schema Migration Audit

> Bead: `scix_experiments-l0ub`. Last reconciled 2026-05-03.

`scripts/audit_schema_migrations.py` checks whether each file under
`migrations/` has both (a) a row in `schema_migrations` and (b) its DDL
effects present in the database. It catches three classes of drift the
manual `psql -f` workflow can produce:

- **MISSING_ROW** — effects are in the DB but no `schema_migrations` row.
- **MISSING_EFFECTS** — row recorded but the table/column/view is gone.
- **FILENAME_MISMATCH** — the row at version `N` references a different
  filename than the one currently on disk for `N`.
- **MISSING_BOTH** — file present but neither row nor effects (i.e.
  pending application).

## Run it

```bash
# Default: prod scix
.venv/bin/python scripts/audit_schema_migrations.py

# Other DB
SCIX_DSN="dbname=scix_test" .venv/bin/python scripts/audit_schema_migrations.py

# CI mode — exit nonzero on any drift
.venv/bin/python scripts/audit_schema_migrations.py --exit-on-drift
```

## How probes work

For each migration the script tries `auto_probe()` on the SQL — it strips
`-- line comments` and matches the first `CREATE TABLE`, `CREATE VIEW`, or
`CREATE FUNCTION`. Migrations whose primary effect is a column add or
marker INSERT need a `MANUAL_PROBES` entry — the script asserts on these
in unit tests so they don't silently regress to `UNKNOWN`. Add new
manual probes in `scripts/audit_schema_migrations.py` next to the
existing ones (v=25, 58, 60, 66 today).

## Recording reconciliation

When the audit finds `MISSING_ROW` for a migration whose effects are
already present, INSERT a row by hand using the file's git-add timestamp
as a defensible `applied_at`:

```bash
git log --all --diff-filter=A --format='%aI' -- migrations/057_v_claim_edges.sql | head -1
# 2026-04-25T09:19:11-04:00

psql -d scix -c "INSERT INTO schema_migrations (version, filename, applied_at) \
                 VALUES (57, '057_v_claim_edges.sql', '2026-04-25T09:19:11-04:00') \
                 ON CONFLICT (version) DO NOTHING;"
```

For brand-new applies, `applied_at = now()` is correct.

## Numbering collisions

If two beads pick the same migration version, rename the loser to the
next free slot **above the in-DB max** (not the on-disk max — the DB
might have versions whose files were lost). Update header comments to
note the rename and the bead that owns it.

The 2026-05-03 reconciliation renamed three files this way:

| Old name | New name | Reason |
|---|---|---|
| `055_paper_umap_2d.sql` | `065_paper_umap_2d.sql` | collided with `055_agent_entity_context_rewrite.sql` |
| `056_intent_populate.sql` | `066_intent_populate.sql` | collided with `056_concepts_vocabularies.sql` |
| `063_section_entities.sql` | `067_section_entities.sql` | collided with zpm4 `063_section_bm25.sql` (in DB but file lost from main) |

The zpm4 files (`063_section_bm25.sql`, `064_section_bm25_index.sql`)
were restored from commit `fae4027` so on-disk numbering is contiguous
again. Both are idempotent (`CREATE INDEX IF NOT EXISTS`,
`ADD COLUMN IF NOT EXISTS`).

## What "deferred" means

Some `MISSING_ROW` entries are intentional. Migrations 053 and 054
(paper_embeddings halfvec + HNSW) were superseded by the pgvectorscale
DiskANN path — see closed bead `scix_experiments-l1t3` for the call.
The audit will flag them as drift; that's expected. Add to a known-
deferred list rather than INSERTing rows for migrations that were
never applied.
