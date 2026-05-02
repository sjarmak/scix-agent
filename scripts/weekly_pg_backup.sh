#!/usr/bin/env bash
# weekly_pg_backup.sh — Weekly pg_dump to NAS of the *irreplaceable subset* of
# derived tables. Excludes the corpus (papers, papers_fulltext, citation_edges,
# paper_embeddings) by design — see scope block below for what's excluded and why.
#
# Scope (per scix_experiments-9ou): research-project risk model. The corpus
# itself (papers, papers_fulltext, citation_edges) is reproducible from the
# raw ADS JSONL preserved at /mnt/scix_offload/ads_metadata_by_year_picard/
# plus upstream APIs. This backup covers ONLY the derived tables that are
# expensive or impossible to rebuild — entity graph (harvester APIs are
# unstable per harvester_api_issues memory), citation_contexts with intent
# labels, community partitions, paper_claims/replication tooling, curated
# taxonomies, and small calibration/audit logs.
#
# Excluded by design:
#   - papers, papers_fulltext, papers_*           — re-harvestable from ADS
#   - citation_edges, s2_citations, paper_metrics — recomputable
#   - paper_embeddings                            — 253 GB, 3-9 GPU-hr to
#                                                   regenerate; opt-in via
#                                                   --include-embeddings, will
#                                                   leave PG once the Qdrant
#                                                   migration completes
#   - query_log, ingest_log, harvest_runs, *_staging — telemetry/working state
#
# Output: /mnt/postgres/scix_dumps/YYYY-MM-DD/
#   schema.sql.gz       — pg_dump --schema-only of public schema
#   data.dump           — pg_dump -Fc of derived tables (data only)
#   embeddings.dump     — pg_dump -Fc of paper_embeddings (only if requested)
#   manifest.txt        — row counts + checksums + pg_restore --list output
#
# Retention: keeps 4 most recent successful dump dirs in /mnt/postgres/scix_dumps/.
#
# Usage:
#   scripts/weekly_pg_backup.sh                        # default: schema + derived tables
#   scripts/weekly_pg_backup.sh --include-embeddings   # also dump paper_embeddings (~150 GB compressed)
#   scripts/weekly_pg_backup.sh --keep N               # retention count (default 4)

set -euo pipefail

REPO_DIR="/home/ds/projects/scix_experiments"
cd "$REPO_DIR"

# ─── Config ──────────────────────────────────────────────────────────────────

DSN="${SCIX_DSN:-dbname=scix}"
DEST_ROOT="/mnt/postgres/scix_dumps"
LOCK_FILE="/tmp/scix_weekly_pg_backup.lock"
KEEP=4
INCLUDE_EMBEDDINGS=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --include-embeddings) INCLUDE_EMBEDDINGS=1; shift ;;
        --keep) KEEP="$2"; shift 2 ;;
        -h|--help)
            sed -n '2,30p' "$0"
            exit 0 ;;
        *) echo "Unknown arg: $1" >&2; exit 2 ;;
    esac
done

# ─── Tables ──────────────────────────────────────────────────────────────────

DERIVED_TABLES=(
    # Entity graph (harvester APIs unstable; rebuild requires re-running
    # multiple harvesters that have since broken — see harvester_api_issues).
    entities
    entity_aliases
    entity_identifiers
    entity_dictionary
    entity_relationships
    entity_link_audits
    entity_link_disputes
    entity_merge_log
    entity_split_log
    document_entities
    extraction_entity_links
    extractions
    curated_entity_core
    core_promotion_log

    # Citation contexts with intent labels (823K rows; LLM-classified).
    citation_contexts

    # Community partitions (Leiden multi-resolution + semantic).
    communities

    # Claims/replication tooling.
    paper_claims

    # UMAP 2D projections (rebuildable but expensive).
    paper_umap_2d

    # Curated taxonomies and concept graph.
    concepts
    concept_relationships
    uat_concepts
    uat_relationships
    paper_uat_mappings
    spdf_spase_crosswalk
    vocabularies

    # Datasets and dataset/entity links.
    datasets
    dataset_entities
    document_datasets

    # Small calibration/audit/diagnostic tables.
    tier_weight_calibration_log
    fusion_mv_state
    citation_diff
)

# ─── Helpers ─────────────────────────────────────────────────────────────────

ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
log() { echo "[$(ts)] $*"; }
fail() { echo "[$(ts)] ERROR: $*" >&2; exit 1; }

# ─── Preconditions ───────────────────────────────────────────────────────────

[[ -d "$DEST_ROOT" ]] || fail "Backup root $DEST_ROOT does not exist (NAS mounted?)"
[[ -w "$DEST_ROOT" ]] || fail "Backup root $DEST_ROOT not writable"

# Single-instance lock.
exec 9>"$LOCK_FILE"
flock -n 9 || fail "Another weekly_pg_backup.sh is running (lock $LOCK_FILE held)"

DATE_STAMP="$(date -u +%Y-%m-%d)"
DEST_DIR="$DEST_ROOT/$DATE_STAMP"
PARTIAL_DIR="$DEST_ROOT/${DATE_STAMP}.partial"

# Clean any stale .partial from a prior failed run, then start fresh.
[[ -d "$PARTIAL_DIR" ]] && rm -rf "$PARTIAL_DIR"
[[ -d "$DEST_DIR" ]] && fail "Today's dump dir $DEST_DIR already exists"
mkdir -p "$PARTIAL_DIR"

log "Starting weekly backup → $PARTIAL_DIR"
log "DSN: $DSN"
log "Tables: ${#DERIVED_TABLES[@]} derived tables"
log "Include paper_embeddings: $INCLUDE_EMBEDDINGS"

# ─── Verify all tables exist before dumping anything ─────────────────────────

MISSING=()
for tbl in "${DERIVED_TABLES[@]}"; do
    if ! psql -d "$DSN" -tAc "SELECT to_regclass('public.$tbl')" | grep -q "^$tbl$"; then
        MISSING+=("$tbl")
    fi
done
if (( ${#MISSING[@]} > 0 )); then
    log "WARN: tables not present, will be skipped: ${MISSING[*]}"
fi

# Filter to tables that actually exist.
TABLES_PRESENT=()
for tbl in "${DERIVED_TABLES[@]}"; do
    if [[ ! " ${MISSING[*]} " =~ " $tbl " ]]; then
        TABLES_PRESENT+=("$tbl")
    fi
done

# ─── Schema dump (full public schema, not just dumped tables) ────────────────

log "Dumping public schema → schema.sql.gz"
pg_dump -d "$DSN" --schema-only --schema=public \
    --no-owner --no-privileges \
    | gzip -c > "$PARTIAL_DIR/schema.sql.gz"
SCHEMA_BYTES=$(stat -c %s "$PARTIAL_DIR/schema.sql.gz")
log "  schema.sql.gz: $(numfmt --to=iec --suffix=B "$SCHEMA_BYTES")"

# ─── Data dump (custom format, all derived tables in one file) ───────────────

log "Dumping ${#TABLES_PRESENT[@]} derived tables → data.dump (custom format, -Z6)"
PG_DUMP_ARGS=(-d "$DSN" -Fc -Z 6 --data-only --no-owner --no-privileges)
for tbl in "${TABLES_PRESENT[@]}"; do
    PG_DUMP_ARGS+=(--table="public.$tbl")
done

pg_dump "${PG_DUMP_ARGS[@]}" -f "$PARTIAL_DIR/data.dump"
DATA_BYTES=$(stat -c %s "$PARTIAL_DIR/data.dump")
log "  data.dump: $(numfmt --to=iec --suffix=B "$DATA_BYTES")"

# Validate the dump archive parses cleanly. pg_restore --list reads the TOC
# without restoring; failure here means the archive is corrupt.
log "Validating data.dump TOC..."
pg_restore --list "$PARTIAL_DIR/data.dump" > "$PARTIAL_DIR/data.toc.txt"
TOC_ENTRIES=$(grep -c '^[0-9]' "$PARTIAL_DIR/data.toc.txt" || true)
log "  TOC entries: $TOC_ENTRIES"

# ─── Optional: paper_embeddings ──────────────────────────────────────────────

if (( INCLUDE_EMBEDDINGS == 1 )); then
    log "Dumping paper_embeddings → embeddings.dump (LARGE, ~150 GB+)"
    pg_dump -d "$DSN" -Fc -Z 6 --data-only --no-owner --no-privileges \
        --table=public.paper_embeddings \
        -f "$PARTIAL_DIR/embeddings.dump"
    EMB_BYTES=$(stat -c %s "$PARTIAL_DIR/embeddings.dump")
    log "  embeddings.dump: $(numfmt --to=iec --suffix=B "$EMB_BYTES")"
    pg_restore --list "$PARTIAL_DIR/embeddings.dump" > "$PARTIAL_DIR/embeddings.toc.txt"
fi

# ─── Manifest ────────────────────────────────────────────────────────────────

log "Writing manifest.txt with row counts + sha256 checksums"
{
    echo "# scix_experiments weekly_pg_backup manifest"
    echo "# generated: $(ts)"
    echo "# host: $(hostname)"
    echo "# dsn: $DSN"
    echo "# pg_dump: $(pg_dump --version)"
    echo
    echo "## Tables included"
    for tbl in "${TABLES_PRESENT[@]}"; do
        rows=$(psql -d "$DSN" -tAc "SELECT count(*) FROM public.$tbl")
        bytes=$(psql -d "$DSN" -tAc "SELECT pg_total_relation_size('public.$tbl')")
        printf "  %-32s rows=%-12s size=%s\n" "$tbl" "$rows" "$(numfmt --to=iec --suffix=B "$bytes")"
    done
    if (( ${#MISSING[@]} > 0 )); then
        echo
        echo "## Tables missing (skipped)"
        for tbl in "${MISSING[@]}"; do
            echo "  $tbl"
        done
    fi
    echo
    echo "## Files"
    (cd "$PARTIAL_DIR" && sha256sum -- *.dump *.sql.gz 2>/dev/null || true)
    echo
    echo "## Restore drill"
    echo "  See docs/runbook_pg_restore_drill.md"
} > "$PARTIAL_DIR/manifest.txt"

# ─── Promote .partial → final ────────────────────────────────────────────────

mv "$PARTIAL_DIR" "$DEST_DIR"
log "Promoted to $DEST_DIR"

# ─── Retention ───────────────────────────────────────────────────────────────

log "Pruning old dumps (keep $KEEP most recent)"
mapfile -t OLD_DUMPS < <(
    find "$DEST_ROOT" -mindepth 1 -maxdepth 1 -type d \
        -regextype posix-extended -regex '.*/[0-9]{4}-[0-9]{2}-[0-9]{2}$' \
        | sort -r | tail -n +"$((KEEP + 1))"
)
for old in "${OLD_DUMPS[@]:-}"; do
    [[ -z "$old" ]] && continue
    log "  removing $old"
    rm -rf "$old"
done

TOTAL_BYTES=$(du -sb "$DEST_DIR" | awk '{print $1}')
log "Backup complete: $DEST_DIR ($(numfmt --to=iec --suffix=B "$TOTAL_BYTES"))"
