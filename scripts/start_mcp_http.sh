#!/usr/bin/env bash
# start_mcp_http.sh — Launch the SciX MCP HTTP server with Cloudflare Tunnel
#
# Usage:
#   # Generate a token and start:
#   ./scripts/start_mcp_http.sh
#
#   # With a specific token:
#   MCP_AUTH_TOKEN=my-secret ./scripts/start_mcp_http.sh
#
#   # Local dev (no auth, no tunnel):
#   MCP_NO_AUTH=1 ./scripts/start_mcp_http.sh --no-tunnel

set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_DIR"

PYTHON=".venv/bin/python3"
MCP_PORT="${MCP_PORT:-8000}"
CLOUDFLARED="${CLOUDFLARED:-/tmp/cloudflared}"
NO_TUNNEL=false

for arg in "$@"; do
    case "$arg" in
        --no-tunnel) NO_TUNNEL=true ;;
    esac
done

# Generate a token if not set
if [ -z "${MCP_AUTH_TOKEN:-}" ] && [ -z "${MCP_NO_AUTH:-}" ]; then
    MCP_AUTH_TOKEN=$(python3 -c "import secrets; print(secrets.token_urlsafe(32))")
    export MCP_AUTH_TOKEN
    echo "════════════════════════════════════════════════════════════"
    echo "  Generated MCP_AUTH_TOKEN: $MCP_AUTH_TOKEN"
    echo "════════════════════════════════════════════════════════════"
fi

# Use read-only DB role for HTTP connections
export SCIX_DSN="host=localhost dbname=scix user=scix_reader password=scix_reader_local_only"

echo "[$(date -u +%H:%M:%S)] Starting MCP HTTP server on port $MCP_PORT..."
export MCP_PORT
MCP_HOST="127.0.0.1" $PYTHON -m scix.mcp_server_http &
MCP_PID=$!

# Wait for server to be ready
for i in $(seq 1 30); do
    if curl -sf http://127.0.0.1:$MCP_PORT/health > /dev/null 2>&1; then
        echo "[$(date -u +%H:%M:%S)] MCP server ready (pid=$MCP_PID)"
        break
    fi
    sleep 1
done

if ! curl -sf http://127.0.0.1:$MCP_PORT/health > /dev/null 2>&1; then
    echo "ERROR: MCP server failed to start"
    kill $MCP_PID 2>/dev/null
    exit 1
fi

# Start Cloudflare Tunnel
if [ "$NO_TUNNEL" = false ] && [ -x "$CLOUDFLARED" ]; then
    echo "[$(date -u +%H:%M:%S)] Starting Cloudflare Tunnel..."
    $CLOUDFLARED tunnel --url http://127.0.0.1:$MCP_PORT --no-autoupdate 2>&1 &
    CF_PID=$!

    # Wait for tunnel URL
    sleep 5
    echo ""
    echo "════════════════════════════════════════════════════════════"
    echo "  Share this Claude Desktop / Claude Code config:"
    echo ""
    echo '  {'
    echo '    "mcpServers": {'
    echo '      "scix": {'
    echo "        \"url\": \"<TUNNEL_URL>/mcp/\","
    echo "        \"headers\": {\"Authorization\": \"Bearer ${MCP_AUTH_TOKEN:-<no-auth>}\"}"
    echo '      }'
    echo '    }'
    echo '  }'
    echo ""
    echo "  (Replace <TUNNEL_URL> with the https://...trycloudflare.com URL above)"
    echo "════════════════════════════════════════════════════════════"
elif [ "$NO_TUNNEL" = false ]; then
    echo "WARNING: cloudflared not found at $CLOUDFLARED — skipping tunnel"
    echo "  Server available locally at http://127.0.0.1:$MCP_PORT/mcp/"
fi

# Wait for either process to exit
trap "kill $MCP_PID ${CF_PID:-} 2>/dev/null" EXIT INT TERM
wait $MCP_PID
