# Entity Linking Kickoff Runbook

**Date:** 2026-04-18
**Goal:** Populate `document_entities` by running the deterministic linking tiers, then refresh the fusion materialized view so MCP entity tools return linked papers.

## Prerequisites

- PostgreSQL running with `scix` database
- All migrations applied (49 as of 2026-04-18)
- 1.58M entities loaded (1.49M targets from SsODNet bulk harvest, plus datasets/methods/missions/instruments/observables/software)
- 9,208 papers with v1 LLM extractions in `extractions` table
- `document_entities` table exists but is empty
- Virtual environment at `.venv/`

## Current State (2026-04-18)

| Table | Count |
|-------|-------|
| entities | 1,577,567 |
| entity_aliases | 885,717 |
| entity_identifiers | 1,579,897 |
| document_entities | 0 |
| extractions | 16,845 (9,208 unique papers) |
| link_runs | 0 |

## Steps

### Step 1: Tier 1 — Keyword Match (minutes)

Pure SQL join: matches `papers.keywords` against `entities.canonical_name` and `entity_aliases.alias`. Writes `tier=1, link_type='keyword_match', confidence=1.0`.

```bash
.venv/bin/python scripts/link_tier1.py --allow-prod -v
```

Dry run first if you want to see counts:
```bash
.venv/bin/python scripts/link_tier1.py --dry-run -v
```

### Step 2: Tier 2 — Aho-Corasick Abstract Match (longer, CPU-bound)

Scans all paper abstracts for entity name mentions using an Aho-Corasick automaton. Boundary-aware (won't match "ACT" inside "ACTION"). Fixed confidence 0.85. Per-entity cap of 25K papers to prevent common-word flooding.

```bash
.venv/bin/python scripts/link_tier2.py --allow-prod --workers 8 -v
```

**Note:** With 1.49M target entities, the automaton build may be large. Monitor memory. If it OOMs, reduce to a subset by entity_type:
```bash
# Check if the script supports entity_type filtering — if not, consider
# running with fewer workers first:
.venv/bin/python scripts/link_tier2.py --allow-prod --workers 4 -v
```

Produces a summary at `build-artifacts/tier2_summary.md`.

### Step 3: Link Existing LLM Extractions (fast)

Resolves the 9,208 papers with v1 extractions against the entity table. Writes to `document_entities` with match details.

```bash
.venv/bin/python scripts/link_entities.py -v
```

### Step 4: Refresh Fusion Materialized View

Merges all tiers into `document_entities_canonical` using noisy-OR confidence fusion.

```bash
.venv/bin/python scripts/refresh_fusion_mv.py -v
```

If this script doesn't exist as a standalone CLI, refresh via psql:
```bash
psql scix -c "UPDATE fusion_mv_state SET dirty = true WHERE id = 1;"
.venv/bin/python -c "
from scix.db import get_connection
from scix.fusion_mv import refresh_if_due
conn = get_connection()
refresh_if_due(conn, min_interval_seconds=0)
conn.close()
print('Refreshed')
"
```

### Step 5: Verify

```bash
# Counts by tier
psql scix -c "SELECT tier, link_type, count(*) FROM document_entities GROUP BY tier, link_type ORDER BY tier;"

# Total unique paper-entity links
psql scix -c "SELECT count(*) FROM document_entities;"

# Check fusion MV
psql scix -c "SELECT count(*) FROM document_entities_canonical;"

# Spot-check a known paper
psql scix -c "
  SELECT e.canonical_name, e.entity_type, de.tier, de.confidence
  FROM document_entities de
  JOIN entities e ON e.id = de.entity_id
  WHERE de.bibcode = (SELECT bibcode FROM document_entities LIMIT 1)
  ORDER BY de.confidence DESC;
"
```

### Step 6: Test via MCP

After linking, the MCP `entity` tool should return papers:
```
entity(action="search", query="JWST")
entity_context(entity_id=<id>)
```

## Potential Issues

| Symptom | Cause | Fix |
|---------|-------|-----|
| Tier 2 OOM | 1.49M target entities in automaton | Filter to high-value entity_types first (mission, instrument, method, dataset) or reduce workers |
| Tier 1 returns 0 | papers.keywords column empty/null | Check `SELECT count(*) FROM papers WHERE keywords IS NOT NULL AND keywords != '{}'` |
| link_tier2.py refuses to run | Production DSN guard | Add `--allow-prod` flag |
| Fusion MV refresh fails | MV not yet created | Run `psql scix -f migrations/033_fusion_mv.sql` |

## What Comes After

- **Incremental sync**: `scripts/link_incremental.py` for daily updates (reads watermark from `link_runs`)
- **v3 extraction**: Cost-efficient option is to run Haiku on high-value papers only (not full 32M corpus). Use `scripts/seed_query_log.py` to identify high-demand papers from MCP query logs, then extract only those.
- **Confidence calibration**: Tier 2 confidence is fixed at 0.85. Future work (u11) will calibrate via logistic regression against human labels.
- **JIT resolution (u10)**: Wire live Haiku calls for on-demand entity resolution during MCP queries — budget-capped per-day.
