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
#
# Failure model (bead dxa): a step failure is recorded, not fatal. Steps that do
# not depend on the failed one still run — in particular Step 6 (v_claim_edges
# refresh) is independent of everything before it, and used to be skipped by
# `set -e` whenever Step 5 died. The run still exits non-zero if any step
# failed; the failure is reported, never swallowed.
#
# Every run writes $STATUS_FILE from an EXIT trap, so an aborted run leaves a
# truthful record of which steps ran. scripts/check_pipeline_health.py reads it.

set -euo pipefail

# SCIX_REPO_DIR is a test seam (tests/test_daily_sync_steps.py runs this script
# against a sandbox tree). Production and cron use the default.
REPO_DIR="${SCIX_REPO_DIR:-/home/ds/projects/scix_experiments}"
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
# SCIX_PYTHON is a test seam; cron uses the default.
PYTHON="${SCIX_PYTHON:-.venv/bin/python3}"

HARVEST_DIR="data/daily_harvest"
BACKFILL_DAYS="${BACKFILL_DAYS:-21}"
STATUS_FILE="${DAILY_SYNC_STATUS_FILE:-logs/daily_sync_status.json}"

# Fresh timestamp per log line (not captured once at script start).
ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }

# ─── Step bookkeeping ─────────────────────────────────────────────────────────

TOTAL_STEPS=6
declare -A STEP_STATUS=()
FAILED_STEPS=()
RECORD_COUNT=0
BACKFILL_COUNT=0
HEALTH_RC="null"
STARTED_AT="$(ts)"

record_step() {  # record_step <step-number> <ok|skipped|failed>
    STEP_STATUS["$1"]="$2"
    if [ "$2" = "failed" ]; then
        FAILED_STEPS+=("$1")
    fi
}

step_ok() {  # step_ok <step-number> — true only if that step ran and succeeded
    [ "${STEP_STATUS[$1]:-}" = "ok" ]
}

# Run one step, record its outcome, and keep going. Always returns 0 so that
# `set -e` cannot take the rest of the pipeline down with it; the recorded
# failure is what makes the script exit non-zero at the end.
run_step() {  # run_step <step-number> <command...>
    local n="$1"
    shift
    local rc=0
    "$@" || rc=$?
    if [ "$rc" -eq 0 ]; then
        record_step "$n" ok
    else
        record_step "$n" failed
        echo "[$(ts)] Step $n/$TOTAL_STEPS FAILED (exit $rc) — independent steps continue"
    fi
    return 0
}

# ─── Run-status file ──────────────────────────────────────────────────────────
# Written atomically (tmp + mv) so the health gate never reads a half-file.

write_status_file() {  # write_status_file <exit-code-so-far>
    local rc="$1"
    local tmp="${STATUS_FILE}.tmp"
    local steps_json="" sep="" n
    for n in $(seq 1 "$TOTAL_STEPS"); do
        steps_json+="${sep}\"${n}\": \"${STEP_STATUS[$n]:-missing}\""
        sep=", "
    done
    local failed_json=""
    if [ "${#FAILED_STEPS[@]}" -gt 0 ]; then
        failed_json=$(printf '%s,' "${FAILED_STEPS[@]}")
        failed_json="${failed_json%,}"
    fi
    mkdir -p "$(dirname "$STATUS_FILE")"
    cat > "$tmp" <<EOF
{
  "script": "daily_sync.sh",
  "started_at": "$STARTED_AT",
  "finished_at": "$(ts)",
  "total_steps": $TOTAL_STEPS,
  "steps": {$steps_json},
  "failed_steps": [$failed_json],
  "harvest_records": $RECORD_COUNT,
  "backfill_records": $BACKFILL_COUNT,
  "health_exit_code": $HEALTH_RC,
  "exit_code": $rc
}
EOF
    mv "$tmp" "$STATUS_FILE"
}

on_exit() {
    local rc=$?
    write_status_file "$rc"
}
trap on_exit EXIT

echo "═══════════════════════════════════════════════════════════════"
echo "[$(ts)] Daily sync starting"
echo "═══════════════════════════════════════════════════════════════"

# ─── Step 1: Harvest new records from ADS ─────────────────────────────────────
# No -v: neither harvest_daily.py nor the ADS client emits DEBUG of its own, so
# -v only turned on urllib3's per-request dumps (1707 lines of CloudFront
# headers in the 2026-07 log, some >2000 chars) which buried the real traceback.
# Bead dxa.

echo "[$(ts)] Step 1/6: Harvesting new records from ADS..."
run_step 1 $PYTHON scripts/harvest_daily.py --output-dir "$HARVEST_DIR"

# Find today's harvest file
TODAY=$(date -u +%Y-%m-%d)
HARVEST_FILE="$HARVEST_DIR/ads_daily_${TODAY}.jsonl.gz"

if ! step_ok 1; then
    # A failed harvest can leave a partially written .jsonl.gz (harvest_daily.py
    # streams into the final path), so ingesting it would silently import a
    # truncated day. Skip Step 2 instead.
    echo "[$(ts)] Harvest failed — not ingesting a possibly partial harvest file."
elif [ ! -f "$HARVEST_FILE" ]; then
    echo "[$(ts)] No harvest file produced — no new records today."
else
    RECORD_COUNT=$(zcat "$HARVEST_FILE" | wc -l)
    echo "[$(ts)] Harvested $RECORD_COUNT records"
fi

# ─── Step 2: Ingest new records into PostgreSQL ───────────────────────────────

if [ "$RECORD_COUNT" -gt 0 ]; then
    echo "[$(ts)] Step 2/6: Ingesting into PostgreSQL..."
    run_step 2 $PYTHON scripts/ingest.py --file "$HARVEST_FILE" --no-drop-indexes
else
    echo "[$(ts)] Step 2/6: Skipped (no new records)"
    record_step 2 skipped
fi

# ─── Step 3: Backfill body/refs for papers ADS has since processed ────────────
# When arxiv papers are first indexed, ADS often hasn't finished extracting
# full text or reference lists yet. This step re-fetches recent papers from
# ADS to pick up body text or references that became available after initial
# harvest. Window is wider than the harvest because body extraction at ADS can
# lag reference extraction by weeks. Only records that actually gained body or
# edges are re-ingested.
#
# Independent of Steps 1-2: it queries ADS by date window, not by harvest file.

echo "[$(ts)] Step 3/6: Backfilling body/references from ADS (last ${BACKFILL_DAYS}d)..."
run_step 3 $PYTHON scripts/backfill_recent_from_ads.py --output-dir "$HARVEST_DIR" --days "$BACKFILL_DAYS"

BACKFILL_FILE="$HARVEST_DIR/ads_backfill_${TODAY}.jsonl.gz"

# ─── Step 4: Ingest backfill file (if any records gained body or edges) ──────

if ! step_ok 3; then
    echo "[$(ts)] Step 4/6: Skipped (backfill failed — file may be partial)"
    record_step 4 skipped
elif [ -f "$BACKFILL_FILE" ]; then
    BACKFILL_COUNT=$(zcat "$BACKFILL_FILE" | wc -l)
    echo "[$(ts)] Step 4/6: Ingesting $BACKFILL_COUNT enriched records..."
    run_step 4 $PYTHON scripts/ingest.py --file "$BACKFILL_FILE" --no-drop-indexes
else
    echo "[$(ts)] Step 4/6: Skipped (no records gained body or edges)"
    record_step 4 skipped
fi

# ─── Step 5: Embed new papers with INDUS ─────────────────────────────────────
# Run whenever harvest OR backfill produced rows. embed.py filters to
# unembedded papers internally, so it's a cheap no-op when there's nothing new.
# -v is kept here: scix.embed does emit DEBUG worth having, and the third-party
# HTTP/model loggers are clamped to WARNING inside the pipeline (bead dxa).

if [ "$RECORD_COUNT" -gt 0 ] || [ "$BACKFILL_COUNT" -gt 0 ]; then
    echo "[$(ts)] Step 5/6: Embedding new papers (INDUS)..."
    run_step 5 $PYTHON scripts/embed.py --model indus --batch-size 256 --device cuda -v
else
    echo "[$(ts)] Step 5/6: Skipped (no new records to embed)"
    record_step 5 skipped
fi

# ─── Step 6/6: Refresh v_claim_edges materialized view (MH-2) ────────────────
# Concurrent refresh keeps reads online. Wrapped in $SCIX_BATCH per CLAUDE.md
# memory-isolation rule (PATH fallback so cron works on hosts w/o the wrapper).
#
# Runs unconditionally: it refreshes a view over the citation graph and shares
# no state with Steps 1-5. Coupling it to Step 5 via `set -e` is what left the
# view 12 days stale during the 2026-07 GPU outage (bead dxa).
#
# NOTE (bead s7cy): Step 5 now upserts INDUS vectors straight into the Qdrant
# serving collection (scix_indus_v2_papers_s1, ADR-013) — paper_embeddings and
# the migration-070 PG→Qdrant outbox are retired (ADR-015). The former Step 7
# outbox drain is therefore gone; there is no PG staging lane to sync.
echo "[$(ts)] Step 6/6: Refreshing v_claim_edges materialized view..."
run_step 6 ${SCIX_BATCH:-} $PYTHON scripts/refresh_v_claim_edges.py --allow-prod

# ─── Post-run health gate (bead tdl) ─────────────────────────────────────────
# The status file must be on disk before the gate runs — the gate asserts
# against it. The EXIT trap rewrites it afterwards to record health_exit_code.

EXIT_CODE=0
if [ "${#FAILED_STEPS[@]}" -gt 0 ]; then
    EXIT_CODE=1
fi
write_status_file "$EXIT_CODE"

echo "[$(ts)] Health gate: scripts/check_pipeline_health.py"
HEALTH_RC=0
$PYTHON scripts/check_pipeline_health.py --allow-prod --status-file "$STATUS_FILE" || HEALTH_RC=$?

# ─── Done ─────────────────────────────────────────────────────────────────────

echo "═══════════════════════════════════════════════════════════════"
echo "[$(ts)] Daily sync complete (harvest=$RECORD_COUNT, backfill=$BACKFILL_COUNT)"
if [ "${#FAILED_STEPS[@]}" -gt 0 ]; then
    echo "[$(ts)] FAILED steps: ${FAILED_STEPS[*]} — exiting non-zero"
elif [ "$HEALTH_RC" -ne 0 ]; then
    echo "[$(ts)] Health gate reported a breach (exit $HEALTH_RC) — exiting non-zero"
    EXIT_CODE="$HEALTH_RC"
fi
echo "═══════════════════════════════════════════════════════════════"

exit "$EXIT_CODE"
