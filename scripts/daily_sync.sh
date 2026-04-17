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

# ─── Log rotation ─────────────────────────────────────────────────────────────
# Rotate logs/daily_sync.log when it exceeds 5 MB. Keeps 7 generations.
# The cron entry appends via `>> logs/daily_sync.log`; this block renames the
# current file *before* the run writes much. The already-open FD inherited from
# cron continues writing to the renamed inode, so today's output lands in .log.1
# while tomorrow's run starts fresh in .log.
LOG_FILE="logs/daily_sync.log"
mkdir -p logs
if [ -f "$LOG_FILE" ] && [ "$(stat -c %s "$LOG_FILE" 2>/dev/null || echo 0)" -gt 5242880 ]; then
    for i in 6 5 4 3 2 1; do
        [ -f "${LOG_FILE}.$i" ] && mv "${LOG_FILE}.$i" "${LOG_FILE}.$((i+1))"
    done
    mv "$LOG_FILE" "${LOG_FILE}.1"
fi

# ─── Environment ──────────────────────────────────────────────────────────────

if [ -f .env ]; then
    set -a
    # shellcheck disable=SC1091
    source .env
    set +a
fi

# Use venv python directly — the activate script has a stale VIRTUAL_ENV
# path (/home/ds/scix_experiments/.venv) from before the projects/ move.
PYTHON=".venv/bin/python3"

HARVEST_DIR="data/daily_harvest"
BACKFILL_DAYS="${BACKFILL_DAYS:-21}"

# Fresh timestamp per log line (not captured once at script start).
ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }

echo "═══════════════════════════════════════════════════════════════"
echo "[$(ts)] Daily sync starting"
echo "═══════════════════════════════════════════════════════════════"

# ─── Step 1: Harvest new records from ADS ─────────────────────────────────────

echo "[$(ts)] Step 1/5: Harvesting new records from ADS..."
$PYTHON scripts/harvest_daily.py --output-dir "$HARVEST_DIR" -v

# Find today's harvest file
TODAY=$(date -u +%Y-%m-%d)
HARVEST_FILE="$HARVEST_DIR/ads_daily_${TODAY}.jsonl.gz"

if [ ! -f "$HARVEST_FILE" ]; then
    echo "[$(ts)] No harvest file produced — no new records today."
    RECORD_COUNT=0
else
    RECORD_COUNT=$(zcat "$HARVEST_FILE" | wc -l)
    echo "[$(ts)] Harvested $RECORD_COUNT records"
fi

# ─── Step 2: Ingest new records into PostgreSQL ───────────────────────────────

if [ "$RECORD_COUNT" -gt 0 ]; then
    echo "[$(ts)] Step 2/5: Ingesting into PostgreSQL..."
    $PYTHON scripts/ingest.py --file "$HARVEST_FILE" --no-drop-indexes -v
else
    echo "[$(ts)] Step 2/5: Skipped (no new records)"
fi

# ─── Step 3: Backfill body/refs for papers ADS has since processed ────────────
# When arxiv papers are first indexed, ADS often hasn't finished extracting
# full text or reference lists yet. This step re-fetches recent papers from
# ADS to pick up body text or references that became available after initial
# harvest. Window is wider than the harvest because body extraction at ADS can
# lag reference extraction by weeks. Only records that actually gained body or
# edges are re-ingested.

echo "[$(ts)] Step 3/5: Backfilling body/references from ADS (last ${BACKFILL_DAYS}d)..."
$PYTHON scripts/backfill_recent_from_ads.py --output-dir "$HARVEST_DIR" --days "$BACKFILL_DAYS" -v

BACKFILL_FILE="$HARVEST_DIR/ads_backfill_${TODAY}.jsonl.gz"
BACKFILL_COUNT=0

# ─── Step 4: Ingest backfill file (if any records gained body or edges) ──────

if [ -f "$BACKFILL_FILE" ]; then
    BACKFILL_COUNT=$(zcat "$BACKFILL_FILE" | wc -l)
    echo "[$(ts)] Step 4/5: Ingesting $BACKFILL_COUNT enriched records..."
    $PYTHON scripts/ingest.py --file "$BACKFILL_FILE" --no-drop-indexes -v
else
    echo "[$(ts)] Step 4/5: Skipped (no records gained body or edges)"
fi

# ─── Step 5: Embed new papers with INDUS ─────────────────────────────────────
# Run whenever harvest OR backfill produced rows. embed.py filters to
# unembedded papers internally, so it's a cheap no-op when there's nothing new.

if [ "$RECORD_COUNT" -gt 0 ] || [ "$BACKFILL_COUNT" -gt 0 ]; then
    echo "[$(ts)] Step 5/5: Embedding new papers (INDUS)..."
    $PYTHON scripts/embed.py --model indus --batch-size 256 --device cuda -v
else
    echo "[$(ts)] Step 5/5: Skipped (no new records to embed)"
fi

# ─── Done ─────────────────────────────────────────────────────────────────────

echo "═══════════════════════════════════════════════════════════════"
echo "[$(ts)] Daily sync complete (harvest=$RECORD_COUNT, backfill=$BACKFILL_COUNT)"
echo "═══════════════════════════════════════════════════════════════"
