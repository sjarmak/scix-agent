#!/usr/bin/env bash
# daily_sync.sh — Daily ADS harvest → ingest → embed pipeline
#
# Cron example (06:15 UTC daily, after ADS nightly index refresh):
#   15 6 * * * /home/ds/scix_experiments/scripts/daily_sync.sh >> /home/ds/scix_experiments/logs/daily_sync.log 2>&1
#
# Prerequisites:
#   - .env file with ADS_API_KEY, SCIX_DSN
#   - Python venv at .venv/ with all deps installed
#   - CUDA available for GPU embedding

set -euo pipefail

REPO_DIR="/home/ds/projects/scix_experiments"
cd "$REPO_DIR"

# ─── Environment ──────────────────────────────────────────────────────────────

if [ -f .env ]; then
    set -a
    # shellcheck disable=SC1091
    source .env
    set +a
fi

# Activate venv
source .venv/bin/activate

HARVEST_DIR="data/daily_harvest"
LOG_PREFIX="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

echo "═══════════════════════════════════════════════════════════════"
echo "[$LOG_PREFIX] Daily sync starting"
echo "═══════════════════════════════════════════════════════════════"

# ─── Step 1: Harvest new records from ADS ─────────────────────────────────────

echo "[$LOG_PREFIX] Step 1/3: Harvesting new records from ADS..."
python scripts/harvest_daily.py --output-dir "$HARVEST_DIR" -v

# Find today's harvest file
TODAY=$(date -u +%Y-%m-%d)
HARVEST_FILE="$HARVEST_DIR/ads_daily_${TODAY}.jsonl.gz"

if [ ! -f "$HARVEST_FILE" ]; then
    echo "[$LOG_PREFIX] No harvest file produced — no new records today. Done."
    exit 0
fi

RECORD_COUNT=$(zcat "$HARVEST_FILE" | wc -l)
echo "[$LOG_PREFIX] Harvested $RECORD_COUNT records"

if [ "$RECORD_COUNT" -eq 0 ]; then
    echo "[$LOG_PREFIX] Empty harvest file — nothing to ingest. Done."
    exit 0
fi

# ─── Step 2: Ingest into PostgreSQL (upsert papers + citation edges) ──────────

echo "[$LOG_PREFIX] Step 2/3: Ingesting into PostgreSQL..."
python scripts/ingest.py --file "$HARVEST_FILE" --no-drop-indexes -v

# ─── Step 3: Embed new papers with INDUS ─────────────────────────────────────

echo "[$LOG_PREFIX] Step 3/3: Embedding new papers (INDUS)..."
python scripts/embed.py --model indus --batch-size 256 --device cuda -v

# ─── Done ─────────────────────────────────────────────────────────────────────

END_PREFIX="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "═══════════════════════════════════════════════════════════════"
echo "[$END_PREFIX] Daily sync complete ($RECORD_COUNT records)"
echo "═══════════════════════════════════════════════════════════════"
