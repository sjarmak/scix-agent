#!/usr/bin/env bash
# Export a year-range subset of scix for workshop demo deployment.
#
# Produces a directory of pg_dump custom-format files that the demo VPS
# restores into its own Postgres. Skips heavy columns (body, tsv, raw) so
# the dump stays small and fast to transfer.
#
# Defaults cover years 2022-2026 (~3.9M papers with INDUS embeddings).
#
# Usage:
#   ./export_subset.sh                         # 2022-2026 by default
#   YEAR_MIN=2023 YEAR_MAX=2025 ./export_subset.sh
#   OUT_DIR=/tmp/scix-demo-dump ./export_subset.sh

set -euo pipefail

YEAR_MIN="${YEAR_MIN:-2022}"
YEAR_MAX="${YEAR_MAX:-2026}"
OUT_DIR="${OUT_DIR:-/tmp/scix-demo-dump}"
SRC_DSN="${SRC_DSN:-dbname=scix}"

mkdir -p "${OUT_DIR}"
echo "[1/6] Building bibcode shortlist for years ${YEAR_MIN}-${YEAR_MAX} in source DB..."

# Shortlist = papers in the year range that have an INDUS embedding and a
# title+abstract. Keeping it to embedded papers means the demo semantic
# search never returns a result that can't be ranked.
psql "${SRC_DSN}" -v ON_ERROR_STOP=1 \
  -v year_min="${YEAR_MIN}" -v year_max="${YEAR_MAX}" <<'SQL'
DROP TABLE IF EXISTS _demo_bibcodes;
CREATE UNLOGGED TABLE _demo_bibcodes AS
SELECT p.bibcode
FROM papers p
WHERE p.year BETWEEN :year_min AND :year_max
  AND p.title IS NOT NULL
  AND p.abstract IS NOT NULL
  AND EXISTS (
    SELECT 1 FROM paper_embeddings e
    WHERE e.bibcode = p.bibcode AND e.model_name = 'indus'
  );

CREATE UNIQUE INDEX ON _demo_bibcodes (bibcode);
ANALYZE _demo_bibcodes;
SELECT count(*) AS shortlist_size FROM _demo_bibcodes;
SQL

echo "[2/6] Dumping papers subset (metadata only, no body/tsv/raw)..."
# Use a view so pg_dump streams only the filtered rows. Heavy columns
# (body, tsv, raw) are excluded to keep the dump compact — demo uses
# title + abstract + lightweight metadata only.
psql "${SRC_DSN}" -v ON_ERROR_STOP=1 <<SQL
DROP VIEW IF EXISTS _demo_papers_v;
CREATE VIEW _demo_papers_v AS
SELECT
  bibcode, title, abstract, year, doctype, pub, volume, issue, page,
  authors, first_author, affiliations, keywords, arxiv_class, doi,
  identifier, alternate_bibcode, bibstem, bibgroup, property,
  pubdate, entry_date, citation_count, read_count, reference_count
FROM papers
WHERE bibcode IN (SELECT bibcode FROM _demo_bibcodes);
SQL

pg_dump "${SRC_DSN}" \
  --format=custom --compress=9 \
  --table=_demo_papers_v \
  --file="${OUT_DIR}/01_papers.dump"

echo "[3/6] Dumping INDUS embeddings for subset..."
psql "${SRC_DSN}" -v ON_ERROR_STOP=1 <<SQL
DROP VIEW IF EXISTS _demo_embeddings_v;
CREATE VIEW _demo_embeddings_v AS
SELECT bibcode, model_name, embedding, input_type, source_hash
FROM paper_embeddings
WHERE model_name = 'indus'
  AND bibcode IN (SELECT bibcode FROM _demo_bibcodes);
SQL

pg_dump "${SRC_DSN}" \
  --format=custom --compress=9 \
  --table=_demo_embeddings_v \
  --file="${OUT_DIR}/02_embeddings.dump"

echo "[4/6] Dumping citation edges within subset..."
# Only edges where both endpoints are in the subset. Prevents dangling FKs
# to papers we didn't export.
psql "${SRC_DSN}" -v ON_ERROR_STOP=1 <<SQL
DROP VIEW IF EXISTS _demo_citations_v;
CREATE VIEW _demo_citations_v AS
SELECT source_bibcode, target_bibcode, edge_attrs
FROM citation_edges
WHERE source_bibcode IN (SELECT bibcode FROM _demo_bibcodes)
  AND target_bibcode IN (SELECT bibcode FROM _demo_bibcodes);
SQL

pg_dump "${SRC_DSN}" \
  --format=custom --compress=9 \
  --table=_demo_citations_v \
  --file="${OUT_DIR}/03_citations.dump"

echo "[5/6] Dumping paper_metrics for subset..."
psql "${SRC_DSN}" -v ON_ERROR_STOP=1 <<SQL
DROP VIEW IF EXISTS _demo_metrics_v;
CREATE VIEW _demo_metrics_v AS
SELECT
  bibcode, pagerank, hub_score, authority_score,
  community_id_coarse, community_id_medium, community_id_fine,
  community_taxonomic,
  community_semantic_coarse, community_semantic_medium, community_semantic_fine,
  updated_at
FROM paper_metrics
WHERE bibcode IN (SELECT bibcode FROM _demo_bibcodes);
SQL

pg_dump "${SRC_DSN}" \
  --format=custom --compress=9 \
  --table=_demo_metrics_v \
  --file="${OUT_DIR}/04_metrics.dump"

echo "[6/6] Cleaning up temp objects in source DB..."
psql "${SRC_DSN}" -v ON_ERROR_STOP=1 <<SQL
DROP VIEW IF EXISTS _demo_papers_v;
DROP VIEW IF EXISTS _demo_embeddings_v;
DROP VIEW IF EXISTS _demo_citations_v;
DROP VIEW IF EXISTS _demo_metrics_v;
DROP TABLE IF EXISTS _demo_bibcodes;
SQL

du -sh "${OUT_DIR}"/*.dump
echo
echo "Done. Transfer ${OUT_DIR}/ to the demo VPS and run restore_subset.sh there."
