#!/usr/bin/env bash
# Ingest all ADS metadata files into PostgreSQL.
# Processes files in reverse chronological order (most recent first)
# for maximum citation edge resolution early.
#
# The ingest pipeline skips files already in ingest_log with status=complete.
# Safe to re-run — idempotent.
#
# Usage: nohup bash scripts/ingest_all.sh > ingest_all.log 2>&1 &

set -euo pipefail

DATA_DIR="ads_metadata_by_year_picard"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_PREFIX="$(date +%Y%m%d_%H%M%S)"

echo "=== SciX Full Corpus Ingestion ==="
echo "Start: $(date)"
echo "Data dir: $DATA_DIR"
echo ""

# List all JSONL files, sort reverse chronological (2020 before 2019, etc.)
# --no-drop-indexes: we're doing incremental ingestion, don't touch indexes
files=$(ls -1 "$DATA_DIR"/ads_metadata_*_full.jsonl* | sort -t_ -k3 -rn)

total=$(echo "$files" | wc -l)
current=0

for file in $files; do
    current=$((current + 1))
    basename=$(basename "$file")

    # Extract year from filename
    year=$(echo "$basename" | grep -oP '\d{4}')

    echo ""
    echo "[$current/$total] $basename (year $year) — $(date +%H:%M:%S)"

    # Run ingestion for this single file
    python3 "$SCRIPT_DIR/ingest.py" \
        --file "$file" \
        --no-drop-indexes \
        --batch-size 10000 \
        2>&1 | tail -5

    echo "  Done: $(date +%H:%M:%S)"
done

echo ""
echo "=== Ingestion Complete ==="
echo "End: $(date)"
echo ""

# Print final stats
python3 -c "
import psycopg
conn = psycopg.connect('dbname=scix')
cur = conn.cursor()
cur.execute('SELECT COUNT(*) FROM papers')
papers = cur.fetchone()[0]
cur.execute('SELECT COUNT(*) FROM citation_edges')
edges = cur.fetchone()[0]
cur.execute('SELECT MIN(year), MAX(year) FROM papers')
ymin, ymax = cur.fetchone()
print(f'Final: {papers:,} papers, {edges:,} edges, years {ymin}-{ymax}')
conn.close()
"
