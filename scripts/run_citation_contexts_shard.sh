#!/usr/bin/env bash
# Citation_contexts shard run + post-shard chain (bead scix_experiments-6hr7).
#
# Single-day workflow invoked by the systemd timer in
# deploy/systemd/scix-citation-contexts-backfill.{service,timer}:
#
#   1. extract_citation_contexts.py --shard $SHARD/4 (extracts ~58 GB of
#      citation contexts; aborts if free disk drops below 50 GB).
#   2. backfill_citation_intent.py --resume (SciBERT-SciCite labels
#      intent for newly-inserted rows; resumable via WHERE intent IS NULL).
#   3. REFRESH MATERIALIZED VIEW CONCURRENTLY v_claim_edges (rolls the
#      new contexts forward into the operator-facing claim-edge view).
#
# Each step propagates its exit code; a failure short-circuits the chain
# so partial state is preserved for the next run.
#
# Usage (manual):
#   bash scripts/run_citation_contexts_shard.sh 0
#   bash scripts/run_citation_contexts_shard.sh 2
#
# Usage (systemd): the unit computes shard from $(date +%j) % 4 so each
# day rotates to the next shard; see the .service file for details.

set -euo pipefail

SHARD_INDEX="${1:?shard index required (0..3)}"
SHARD_TOTAL="${SHARD_TOTAL:-4}"

if (( SHARD_INDEX < 0 || SHARD_INDEX >= SHARD_TOTAL )); then
    echo "ERROR: shard index $SHARD_INDEX out of range [0, $SHARD_TOTAL)" >&2
    exit 64
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# scix-batch is at ~/.local/bin/scix-batch per CLAUDE.md §Memory isolation;
# fall back to bare invocation if it's not on PATH (e.g. inside CI).
SCIX_BATCH="${SCIX_BATCH:-scix-batch}"
if ! command -v "$SCIX_BATCH" >/dev/null 2>&1; then
    echo "WARN: $SCIX_BATCH not found on PATH — running without memory cgroup. " \
         "On the prod host this risks oomd collateral-killing gascity." >&2
    SCIX_BATCH=""
fi

PYTHON="${PYTHON:-$REPO_ROOT/.venv/bin/python}"
if [[ ! -x "$PYTHON" ]]; then
    PYTHON="$(command -v python3)"
fi

PSQL="${PSQL:-psql}"
DSN="${SCIX_DSN:-dbname=scix}"

log() { printf '%s [shard %s/%s] %s\n' "$(date -u +%FT%TZ)" "$SHARD_INDEX" "$SHARD_TOTAL" "$*"; }

run_step() {
    local label="$1"; shift
    log "step BEGIN: $label"
    if "$@"; then
        log "step OK:    $label"
    else
        local rc=$?
        log "step FAIL:  $label (exit $rc)"
        exit "$rc"
    fi
}

# Step 1 — extract citation contexts for this shard.
run_step "extract_citation_contexts" \
    $SCIX_BATCH "$PYTHON" scripts/extract_citation_contexts.py \
        --allow-prod \
        --shard "$SHARD_INDEX/$SHARD_TOTAL" \
        --batch-size 1000

# Step 2 — backfill intent for newly-inserted rows. The classifier filters
# WHERE intent IS NULL, so this is naturally idempotent across shard runs.
run_step "backfill_citation_intent" \
    $SCIX_BATCH "$PYTHON" scripts/backfill_citation_intent.py \
        --resume \
        --batch-size 256

# Step 3 — refresh the v_claim_edges matview. CONCURRENTLY requires the
# unique idx_v_claim_edges_pk to exist (migration 057); without it this
# would fall back to AccessExclusiveLock-holding non-concurrent refresh.
run_step "refresh_v_claim_edges" \
    "$PSQL" -d "$DSN" -v ON_ERROR_STOP=1 -c \
        "REFRESH MATERIALIZED VIEW CONCURRENTLY v_claim_edges;"

log "shard $SHARD_INDEX/$SHARD_TOTAL complete"
