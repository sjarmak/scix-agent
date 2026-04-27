#!/usr/bin/env bash
# deploy/qdrant-run.sh — plain `docker run` launcher for scix-qdrant,
# mirroring `deploy/qdrant-compose.yml` for hosts without the `docker
# compose` plugin (matches the deploy/run.sh fallback pattern).
#
# The compose file is the canonical spec; this script just executes it
# imperatively. Keep them in sync — if you add a port, mount, or env
# var to the compose file, mirror it here.
#
# Usage:
#   ./deploy/qdrant-run.sh                # start (or restart)
#   ./deploy/qdrant-run.sh stop           # stop + remove
#   ./deploy/qdrant-run.sh logs           # tail logs

set -euo pipefail

DEPLOY_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$DEPLOY_DIR/.." && pwd)"

CTR="scix-qdrant"
IMAGE="qdrant/qdrant:latest"
STORAGE="$REPO_DIR/.qdrant_storage"

action="${1:-up}"

case "$action" in
    stop|down)
        docker rm -f "$CTR" 2>/dev/null || true
        echo "scix-qdrant stopped."
        exit 0
        ;;
    logs)
        exec docker logs -f --tail 100 "$CTR"
        ;;
    up|start|"")
        ;;
    *)
        echo "Usage: $0 [up|stop|logs]" >&2
        exit 2
        ;;
esac

mkdir -p "$STORAGE"

# Recreate so port bindings always match what's in this script — no
# stale 0.0.0.0 from a previous run can survive.
docker rm -f "$CTR" 2>/dev/null || true

# Bindings MUST be 127.0.0.1-prefixed. Empty HostIp ("6333:6333") maps
# to 0.0.0.0 and exposes the API to the LAN — the exact failure mode
# bead s1a was opened to fix. The companion preflight script enforces
# this on every start.
docker run -d \
    --name "$CTR" \
    --restart unless-stopped \
    -p 127.0.0.1:6333:6333 \
    -p 127.0.0.1:6334:6334 \
    -v "$STORAGE:/qdrant/storage" \
    -e RUN_MODE=production \
    "$IMAGE" >/dev/null

# Snapshots NAS mount is intentionally omitted here; uncomment once
# /mnt/qdrant_snapshots is provisioned, in lock-step with the compose
# file:
#   -v /mnt/qdrant_snapshots:/qdrant/snapshots \

echo "scix-qdrant up. Verifying binding policy..."
"$REPO_DIR/scripts/preflight_qdrant_security.sh" "$CTR"
