#!/usr/bin/env bash
# deploy/run.sh — launch the hardened SciX MCP stack via plain docker,
# no `docker compose` plugin required.
#
# Usage:
#   cp deploy/.env.example deploy/.env    # edit with your values
#   ./deploy/run.sh                       # build image + start stack
#   ./deploy/run.sh stop                  # stop + remove containers
#   ./deploy/run.sh logs                  # tail logs from both containers
#
# If you have `docker compose` installed, prefer that — this is a fallback.

set -euo pipefail

DEPLOY_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$DEPLOY_DIR/.." && pwd)"
cd "$DEPLOY_DIR"

ENV_FILE="$DEPLOY_DIR/.env"
if [ ! -f "$ENV_FILE" ]; then
    echo "ERROR: $ENV_FILE not found. Copy .env.example and fill it in." >&2
    exit 1
fi

# shellcheck disable=SC1090
set -a; . "$ENV_FILE"; set +a

: "${MCP_AUTH_TOKEN:?MCP_AUTH_TOKEN must be set in deploy/.env}"
: "${SCIX_DSN:?SCIX_DSN must be set in deploy/.env}"
: "${HF_CACHE_DIR:?HF_CACHE_DIR must be set in deploy/.env}"
: "${CF_TUNNEL_TOKEN:?CF_TUNNEL_TOKEN must be set in deploy/.env}"

IMAGE="scix-mcp:latest"
NET="scix-mcp-net"
PY_CTR="scix-mcp"
CF_CTR="scix-cloudflared"

action="${1:-up}"

case "$action" in
    stop|down)
        docker rm -f "$CF_CTR" "$PY_CTR" 2>/dev/null || true
        docker network rm "$NET" 2>/dev/null || true
        echo "Stack stopped."
        exit 0
        ;;
    logs)
        docker logs -f --tail 100 "$PY_CTR" &
        docker logs -f --tail 100 "$CF_CTR" &
        wait
        exit 0
        ;;
    up|"")
        ;;
    *)
        echo "Unknown action: $action" >&2
        exit 1
        ;;
esac

echo "[$(date +%H:%M:%S)] Building $IMAGE..."
docker build -f "$DEPLOY_DIR/Dockerfile" -t "$IMAGE" "$REPO_DIR"

# Make sure we start from a clean slate.
docker rm -f "$CF_CTR" "$PY_CTR" 2>/dev/null || true
docker network inspect "$NET" >/dev/null 2>&1 || docker network create "$NET" >/dev/null

echo "[$(date +%H:%M:%S)] Starting $PY_CTR..."
docker run -d \
    --name "$PY_CTR" \
    --network "$NET" \
    --add-host=host.docker.internal:host-gateway \
    --read-only \
    --tmpfs /tmp:size=64m,mode=1777 \
    --cap-drop ALL \
    --security-opt no-new-privileges:true \
    --cpus 2.0 \
    --memory 4g \
    --restart unless-stopped \
    -e MCP_AUTH_TOKEN="$MCP_AUTH_TOKEN" \
    -e MCP_PORT=8000 \
    -e MCP_HOST=0.0.0.0 \
    -e SCIX_DSN="$SCIX_DSN" \
    -v "$HF_CACHE_DIR":/home/scix/.cache/huggingface:ro \
    "$IMAGE" >/dev/null

echo "[$(date +%H:%M:%S)] Waiting for $PY_CTR healthcheck..."
for i in $(seq 1 60); do
    status=$(docker inspect -f '{{.State.Health.Status}}' "$PY_CTR" 2>/dev/null || echo "unknown")
    if [ "$status" = "healthy" ]; then
        echo "[$(date +%H:%M:%S)] $PY_CTR healthy."
        break
    fi
    sleep 2
done

echo "[$(date +%H:%M:%S)] Starting $CF_CTR..."
docker run -d \
    --name "$CF_CTR" \
    --network "$NET" \
    --read-only \
    --tmpfs /tmp:size=16m \
    --cap-drop ALL \
    --security-opt no-new-privileges:true \
    --restart unless-stopped \
    cloudflare/cloudflared:latest \
    tunnel --no-autoupdate run --token "$CF_TUNNEL_TOKEN" >/dev/null

echo "[$(date +%H:%M:%S)] Stack up."
echo "  Python:     docker logs -f $PY_CTR"
echo "  Cloudflare: docker logs -f $CF_CTR"
