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

echo "[$LOG_PREFIX] Step 1/5: Harvesting new records from ADS..."
python3 scripts/harvest_daily.py --output-dir "$HARVEST_DIR" -v

# Find today's harvest file
TODAY=$(date -u +%Y-%m-%d)
HARVEST_FILE="$HARVEST_DIR/ads_daily_${TODAY}.jsonl.gz"

if [ ! -f "$HARVEST_FILE" ]; then
    echo "[$LOG_PREFIX] No harvest file produced — no new records today."
    RECORD_COUNT=0
else
    RECORD_COUNT=$(zcat "$HARVEST_FILE" | wc -l)
    echo "[$LOG_PREFIX] Harvested $RECORD_COUNT records"
fi

# ─── Step 2: Ingest new records into PostgreSQL ───────────────────────────────

if [ "$RECORD_COUNT" -gt 0 ]; then
    echo "[$LOG_PREFIX] Step 2/5: Ingesting into PostgreSQL..."
    python3 scripts/ingest.py --file "$HARVEST_FILE" --no-drop-indexes -v
else
    echo "[$LOG_PREFIX] Step 2/5: Skipped (no new records)"
fi

# ─── Step 3: Backfill body/refs for papers ADS has since processed ────────────
# When arxiv papers are first indexed, ADS often hasn't finished extracting
# full text or reference lists yet. This step re-fetches recent papers from
# ADS to pick up body text or references that became available after initial
# harvest. Only records that actually gained body or edges are re-ingested.

echo "[$LOG_PREFIX] Step 3/5: Backfilling body/references from ADS..."
python3 scripts/backfill_recent_from_ads.py --output-dir "$HARVEST_DIR" --days 7 -v

BACKFILL_FILE="$HARVEST_DIR/ads_backfill_${TODAY}.jsonl.gz"

# ─── Step 4: Ingest backfill file (if any records gained body or edges) ──────

if [ -f "$BACKFILL_FILE" ]; then
    BACKFILL_COUNT=$(zcat "$BACKFILL_FILE" | wc -l)
    echo "[$LOG_PREFIX] Step 4/5: Ingesting $BACKFILL_COUNT enriched records..."
    python3 scripts/ingest.py --file "$BACKFILL_FILE" --no-drop-indexes -v
else
    echo "[$LOG_PREFIX] Step 4/5: Skipped (no records gained body or edges)"
fi

# ─── Step 5: Embed new papers with INDUS ─────────────────────────────────────

if [ "$RECORD_COUNT" -gt 0 ]; then
    echo "[$LOG_PREFIX] Step 5/5: Embedding new papers (INDUS)..."
    python3 scripts/embed.py --model indus --batch-size 256 --device cuda -v
else
    echo "[$LOG_PREFIX] Step 5/5: Skipped (no new records to embed)"
fi

# ─── Done ─────────────────────────────────────────────────────────────────────

END_PREFIX="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "═══════════════════════════════════════════════════════════════"
echo "[$END_PREFIX] Daily sync complete ($RECORD_COUNT records)"
echo "═══════════════════════════════════════════════════════════════"
