#!/usr/bin/env bash
# Restore a subset dump onto the demo VPS's compose Postgres.
#
# Assumes:
#   - docker compose stack is up (postgres container is healthy)
#   - migrations/ directory was copied onto the VPS alongside this script
#   - DUMP_DIR contains the four files produced by export_subset.sh:
#       01_papers.dump 02_embeddings.dump 03_citations.dump 04_metrics.dump
#
# The dumps contain views renamed as tables (_demo_*_v). This script
# restores them into a staging schema, then INSERTs into the real tables
# defined by migrations.
#
# Usage on VPS:
#   DUMP_DIR=/opt/scix-demo/dump MIGRATIONS_DIR=/opt/scix-demo/migrations \
#     ./restore_subset.sh

set -euo pipefail

DUMP_DIR="${DUMP_DIR:-./dump}"
MIGRATIONS_DIR="${MIGRATIONS_DIR:-./migrations}"
# DSN for the containerised Postgres, as seen from the VPS host.
# docker compose exec is the safest way — avoids exposing a port.
COMPOSE_FILE="${COMPOSE_FILE:-docker-compose.yml}"
PG_EXEC=(docker compose -f "${COMPOSE_FILE}" exec -T postgres)
PSQL=("${PG_EXEC[@]}" psql -v ON_ERROR_STOP=1 -U scix -d scix)
PG_RESTORE=("${PG_EXEC[@]}" pg_restore -U scix -d scix)

echo "[1/6] Applying migrations from ${MIGRATIONS_DIR}..."
for f in $(ls "${MIGRATIONS_DIR}"/*.sql | sort); do
  echo "  -> $(basename "$f")"
  "${PSQL[@]}" < "$f" >/dev/null
done

echo "[2/6] Ensuring pgvector extension..."
"${PSQL[@]}" <<'SQL' >/dev/null
CREATE EXTENSION IF NOT EXISTS vector;
SQL

echo "[3/6] Restoring dump files into staging tables..."
# Each dump was created FROM a view, so pg_restore will recreate them as
# tables named _demo_*_v in the public schema.
for f in 01_papers 02_embeddings 03_citations 04_metrics; do
  echo "  -> ${f}.dump"
  cat "${DUMP_DIR}/${f}.dump" | "${PG_RESTORE[@]}" --no-owner --no-privileges
done

echo "[4/6] Moving rows from staging into real tables..."
"${PSQL[@]}" <<'SQL'
-- Defer FK checks during bulk insert.
SET session_replication_role = replica;

INSERT INTO papers (
  bibcode, title, abstract, year, doctype, pub, volume, issue, page,
  authors, first_author, affiliations, keywords, arxiv_class, doi,
  identifier, alternate_bibcode, bibstem, bibgroup, property,
  pubdate, entry_date, citation_count, read_count, reference_count
)
SELECT
  bibcode, title, abstract, year, doctype, pub, volume, issue, page,
  authors, first_author, affiliations, keywords, arxiv_class, doi,
  identifier, alternate_bibcode, bibstem, bibgroup, property,
  pubdate, entry_date, citation_count, read_count, reference_count
FROM _demo_papers_v
ON CONFLICT (bibcode) DO NOTHING;

INSERT INTO paper_embeddings (bibcode, model_name, embedding, input_type, source_hash)
SELECT bibcode, model_name, embedding, input_type, source_hash
FROM _demo_embeddings_v
ON CONFLICT (bibcode, model_name) DO NOTHING;

INSERT INTO citation_edges (source_bibcode, target_bibcode, edge_attrs)
SELECT source_bibcode, target_bibcode, edge_attrs
FROM _demo_citations_v
ON CONFLICT (source_bibcode, target_bibcode) DO NOTHING;

INSERT INTO paper_metrics (
  bibcode, pagerank, hub_score, authority_score,
  community_id_coarse, community_id_medium, community_id_fine,
  community_taxonomic,
  community_semantic_coarse, community_semantic_medium, community_semantic_fine,
  updated_at
)
SELECT
  bibcode, pagerank, hub_score, authority_score,
  community_id_coarse, community_id_medium, community_id_fine,
  community_taxonomic,
  community_semantic_coarse, community_semantic_medium, community_semantic_fine,
  updated_at
FROM _demo_metrics_v
ON CONFLICT (bibcode) DO NOTHING;

SET session_replication_role = DEFAULT;

DROP TABLE _demo_papers_v;
DROP TABLE _demo_embeddings_v;
DROP TABLE _demo_citations_v;
DROP TABLE _demo_metrics_v;

ANALYZE papers;
ANALYZE paper_embeddings;
ANALYZE citation_edges;
ANALYZE paper_metrics;
SQL

echo "[5/6] Regenerating papers.tsv for BM25 (trigger was disabled during restore)..."
"${PSQL[@]}" < "$(dirname "$0")/post_restore_bm25.sql"

echo "[6/6] Building HNSW index for INDUS embeddings (this is the slow part)..."
"${PSQL[@]}" <<'SQL'
SET maintenance_work_mem = '512MB';
SET max_parallel_maintenance_workers = 2;
CREATE INDEX IF NOT EXISTS idx_embed_hnsw_indus
ON paper_embeddings
USING hnsw ((embedding::vector(768)) vector_cosine_ops)
WITH (m = 16, ef_construction = 64)
WHERE model_name = 'indus';
SQL

echo
echo "Row counts:"
"${PSQL[@]}" <<'SQL'
SELECT 'papers'           AS t, count(*) FROM papers
UNION ALL SELECT 'paper_embeddings', count(*) FROM paper_embeddings
UNION ALL SELECT 'citation_edges',   count(*) FROM citation_edges
UNION ALL SELECT 'paper_metrics',    count(*) FROM paper_metrics;
SQL

echo "Done. Smoke-test: curl -sf https://<your-demo-host>/health"
