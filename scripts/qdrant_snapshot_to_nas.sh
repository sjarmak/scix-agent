#!/usr/bin/env bash
# qdrant_snapshot_to_nas.sh — periodic Qdrant collection snapshot to NAS.
#
# Bead scix_experiments-q000 (dec-3jq). Snapshots/backups go to NAS (/mnt);
# the live Qdrant runtime stays on local NVMe — NFSv3 is unsafe for live
# vector writes but fine for one-shot sequential snapshot output (the NAS
# duplicate/backup tier). See docs/prd/qdrant_nas_migration.md and the
# storage_tiering_policy.
#
# How it works (single, disk-pressure-safe mode):
#   The container's /qdrant/snapshots dir MUST be bind-mounted to a NAS
#   path (/mnt/...). Qdrant then writes the snapshot tar straight to NAS,
#   so we never stage 78G on the NVMe — the exact constraint that deferred
#   the baseline snapshot in khug (78G snapshot, 47G free locally).
#
#   We deliberately do NOT download-then-move via the API: Qdrant always
#   materialises the tar in its snapshots dir first, so a local snapshots
#   dir would still need the full 78G locally. Requiring the NAS mount is
#   the only mode that works under disk pressure, so it is the only mode.
#
# Retention is driven entirely through the Qdrant API (DELETE removes the
# NAS file too), keeping the API listing and the NAS files as one source
# of truth — no separate find-on-NAS pruning that could drift.
#
# This script is a thin trigger: the heavy 78G write happens inside the
# Qdrant container's own cgroup, not here. Its own RSS is a few MB (curl +
# jq), so it creates no memory pressure and is intentionally NOT wrapped in
# scix-batch — that scope would bound this process, not Qdrant, exactly as
# the CLAUDE.md note observes that the scix-batch cgroup does not bound
# Postgres-side memory.
#
# Usage:
#   scripts/qdrant_snapshot_to_nas.sh --dry-run   # report readiness, no writes
#   scripts/qdrant_snapshot_to_nas.sh             # create + verify + prune
#
# Flags (all also settable via the env var in parens):
#   --url URL            Qdrant REST endpoint   (QDRANT_URL,   default :6633)
#   --container NAME     container to inspect   (QDRANT_CONTAINER, scix-qdrant-v2)
#   --collection NAME    collection to snapshot (QDRANT_COLLECTION, scix_indus_v2_papers_s1)
#   --keep N             snapshots to retain    (QDRANT_SNAPSHOT_KEEP, 4)
#   --min-bytes N        truncation-guard floor (QDRANT_SNAPSHOT_MIN_BYTES, 1 GiB)
#   --dry-run            preflight only; exits 0 with a READY/NOT-READY verdict
#
# Exit codes: 0 success (or dry-run report); non-zero on any real-run
# failure — nothing is swallowed.

set -euo pipefail

QDRANT_URL="${QDRANT_URL:-http://127.0.0.1:6633}"
CONTAINER="${QDRANT_CONTAINER:-scix-qdrant-v2}"
COLLECTION="${QDRANT_COLLECTION:-scix_indus_v2_papers_s1}"
KEEP="${QDRANT_SNAPSHOT_KEEP:-4}"
MIN_BYTES="${QDRANT_SNAPSHOT_MIN_BYTES:-1073741824}"   # 1 GiB
CREATE_TIMEOUT="${QDRANT_SNAPSHOT_CREATE_TIMEOUT:-7200}" # 2h; 78G over NFS
DRY_RUN=0

while [ $# -gt 0 ]; do
    case "$1" in
        --dry-run)      DRY_RUN=1 ;;
        --url)          QDRANT_URL="$2"; shift ;;
        --container)    CONTAINER="$2"; shift ;;
        --collection)   COLLECTION="$2"; shift ;;
        --keep)         KEEP="$2"; shift ;;
        --min-bytes)    MIN_BYTES="$2"; shift ;;
        -h|--help)      sed -n '2,55p' "$0"; exit 0 ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
    shift
done

log()  { printf '%s qdrant-snapshot: %s\n' "$(date -Is)" "$*"; }
die()  { printf '%s qdrant-snapshot: ERROR: %s\n' "$(date -Is)" "$*" >&2; exit 1; }

for bin in curl jq docker stat mountpoint; do
    command -v "$bin" >/dev/null 2>&1 || die "required tool not found: $bin"
done

qget() { curl -fsS --max-time 30 "$QDRANT_URL$1"; }

# --- Readiness gates (shared by dry-run and the real run) -------------------
# Returns the host-side snapshots dir on stdout, or a reason on failure.
readiness_reason=""
snap_src=""
check_readiness() {
    snap_src="$(docker inspect "$CONTAINER" \
        --format '{{range .Mounts}}{{if eq .Destination "/qdrant/snapshots"}}{{.Source}}{{end}}{{end}}' \
        2>/dev/null || true)"
    if [ -z "$snap_src" ]; then
        readiness_reason="container '$CONTAINER' has no /qdrant/snapshots mount. Enable the NAS snapshots volume in deploy/qdrant-compose.yml (uncomment '-v /mnt/qdrant_snapshots:/qdrant/snapshots'), create the NAS dir, and recreate the container."
        return 1
    fi
    case "$snap_src" in
        /mnt/*) ;;
        *) readiness_reason="snapshots mount source '$snap_src' is not under /mnt (NAS). Per qdrant_nas_migration, snapshots go to NAS; runtime stays on NVMe."; return 1 ;;
    esac
    if ! mountpoint -q /mnt; then
        readiness_reason="/mnt is not a mounted filesystem — NAS unavailable; refusing to write to a local stub."
        return 1
    fi
    if [ ! -d "$snap_src" ]; then
        readiness_reason="snapshots dir '$snap_src' does not exist on NAS (create it before enabling the mount)."
        return 1
    fi
    if ! qget "/collections/$COLLECTION" >/dev/null 2>&1; then
        readiness_reason="collection '$COLLECTION' not reachable at $QDRANT_URL."
        return 1
    fi
    return 0
}

list_json() { qget "/collections/$COLLECTION/snapshots"; }

# --- Dry run: preflight report only, never writes ---------------------------
if [ "$DRY_RUN" -eq 1 ]; then
    log "DRY RUN — no snapshot will be created or deleted."
    log "url=$QDRANT_URL container=$CONTAINER collection=$COLLECTION keep=$KEEP"
    if check_readiness; then
        log "snapshots dir (NAS): $snap_src"
        snaps="$(list_json | jq -r '.result | sort_by(.creation_time) | reverse')"
        n="$(printf '%s' "$snaps" | jq 'length')"
        log "existing snapshots: $n"
        printf '%s' "$snaps" | jq -r '.[] | "  \(.name)  \(.size // 0) bytes  \(.creation_time // "?")"' || true
        prune="$(printf '%s' "$snaps" | jq -r ".[$((KEEP-1)):] | .[].name" || true)"
        if [ -n "$prune" ]; then
            log "with a new snapshot, retention (keep $KEEP) would prune:"
            printf '%s\n' "$prune" | sed 's/^/  /'
        else
            log "retention (keep $KEEP) would prune nothing."
        fi
        log "VERDICT: READY"
    else
        log "VERDICT: NOT READY — $readiness_reason"
    fi
    exit 0
fi

# --- Real run ---------------------------------------------------------------
check_readiness || die "$readiness_reason"
log "snapshots dir (NAS): $snap_src"

log "creating snapshot of '$COLLECTION' (wait=true; up to ${CREATE_TIMEOUT}s)…"
create_out="$(curl -fsS --max-time "$CREATE_TIMEOUT" -X POST \
    "$QDRANT_URL/collections/$COLLECTION/snapshots?wait=true")" \
    || die "snapshot create call failed"

name="$(printf '%s' "$create_out" | jq -r '.result.name // empty')"
api_size="$(printf '%s' "$create_out" | jq -r '.result.size // 0')"
[ -n "$name" ] || die "snapshot create returned no name: $create_out"
log "created: $name (api-reported size: $api_size bytes)"

# Verify the file actually landed on NAS at the expected path and is not
# truncated — Qdrant can report ok while an NFS write silently short-writes.
host_file="$snap_src/$COLLECTION/$name"
[ -f "$host_file" ] || die "snapshot file not found on NAS: $host_file"
fsize="$(stat -c%s "$host_file")"
if [ "$api_size" -gt 0 ] && [ "$fsize" -ne "$api_size" ]; then
    die "NAS file size $fsize != api-reported $api_size for $host_file (short write?)"
fi
[ "$fsize" -ge "$MIN_BYTES" ] || die "snapshot $host_file is $fsize bytes, below truncation floor $MIN_BYTES"

# Cross-check it appears in the API listing.
list_json | jq -e --arg n "$name" '.result | any(.name == $n)' >/dev/null \
    || die "snapshot '$name' missing from API listing after create"
log "verified on NAS: $host_file ($fsize bytes), present in API listing"

# Retention via the API (DELETE also removes the NAS file).
prune="$(list_json | jq -r "
    .result | sort_by(.creation_time) | reverse | .[$KEEP:] | .[].name" || true)"
if [ -n "$prune" ]; then
    while IFS= read -r old; do
        [ -n "$old" ] || continue
        log "retention: deleting old snapshot $old"
        curl -fsS --max-time 300 -X DELETE \
            "$QDRANT_URL/collections/$COLLECTION/snapshots/$old" >/dev/null \
            || die "failed to delete old snapshot $old"
    done <<< "$prune"
else
    log "retention: nothing to prune (keep=$KEEP)"
fi

remaining="$(list_json | jq '.result | length')"
log "done. snapshot=$name retained=$remaining keep=$KEEP"
