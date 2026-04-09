#!/usr/bin/env bash
# Restore missing indexes on papers table (from migrations 001/003/012/018)
# Run after the destructive test incident that dropped all non-PK indexes.
# Uses CONCURRENTLY to avoid blocking reads on 32M row table.
# Must run each statement separately (CONCURRENTLY can't be in a transaction).
set -euo pipefail

DB="${1:-scix}"

echo "=== Restoring papers indexes on database: $DB ==="
echo "Started: $(date)"

# Step 1: Drop any invalid indexes from interrupted builds
echo "[1/3] Dropping invalid indexes..."
for idx in $(psql -d "$DB" -t -c "
  SELECT i.relname FROM pg_class t
  JOIN pg_index ix ON t.oid = ix.indrelid
  JOIN pg_class i ON i.oid = ix.indexrelid
  WHERE t.relname = 'papers' AND NOT ix.indisvalid;"); do
  echo "  Dropping invalid: $idx"
  psql -d "$DB" -c "DROP INDEX CONCURRENTLY IF EXISTS $idx;"
done

# Step 2: Recreate all expected indexes
echo "[2/3] Creating missing indexes..."

declare -a INDEXES=(
  "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_papers_authors ON papers USING GIN (authors)"
  "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_papers_keywords ON papers USING GIN (keywords)"
  "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_papers_arxiv ON papers USING GIN (arxiv_class)"
  "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_papers_doi ON papers USING GIN (doi)"
  "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_papers_year ON papers (year)"
  "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_papers_doctype ON papers (doctype)"
  "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_papers_tsv ON papers USING GIN (tsv)"
  "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_papers_first_author ON papers (first_author)"
  "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_papers_data ON papers USING GIN (data)"
  "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_papers_facility ON papers USING GIN (facility)"
  "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_papers_esources ON papers USING GIN (esources)"
  "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_papers_nedid ON papers USING GIN (nedid)"
  "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_papers_simbid ON papers USING GIN (simbid)"
  "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_papers_keyword_norm ON papers USING GIN (keyword_norm)"
  "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_papers_author_norm ON papers USING GIN (author_norm)"
  "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_papers_author_count ON papers (author_count)"
  "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_papers_openalex_id ON papers(openalex_id)"
)

for sql in "${INDEXES[@]}"; do
  idx_name=$(echo "$sql" | grep -oP 'idx_papers_\w+')
  echo "  Creating: $idx_name ..."
  psql -d "$DB" -c "$sql;" && echo "    OK" || echo "    FAILED (will retry)"
done

# Step 3: Verify
echo "[3/3] Verifying..."
psql -d "$DB" -c "
  SELECT i.relname AS index_name, ix.indisvalid AS valid
  FROM pg_class t
  JOIN pg_index ix ON t.oid = ix.indrelid
  JOIN pg_class i ON i.oid = ix.indexrelid
  WHERE t.relname = 'papers'
  ORDER BY i.relname;"

echo "Completed: $(date)"
