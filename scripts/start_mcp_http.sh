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

# Generate a token if not set. Written to a 0600 file, NEVER echoed to
# stdout — stdout typically ends up in world-readable log files (/tmp/*.out,
# journald, CI artifacts) and has a way of leaking into chat transcripts.
TOKEN_FILE="${MCP_TOKEN_FILE:-${HOME}/.config/scix-mcp/token}"
if [ -z "${MCP_AUTH_TOKEN:-}" ] && [ -z "${MCP_NO_AUTH:-}" ]; then
    MCP_AUTH_TOKEN=$(python3 -c "import secrets; print(secrets.token_urlsafe(32))")
    export MCP_AUTH_TOKEN
    mkdir -p "$(dirname "$TOKEN_FILE")"
    (umask 077 && printf '%s\n' "$MCP_AUTH_TOKEN" > "$TOKEN_FILE")
    chmod 600 "$TOKEN_FILE"
    echo "════════════════════════════════════════════════════════════"
    echo "  Generated new MCP_AUTH_TOKEN and wrote it to:"
    echo "    $TOKEN_FILE   (mode 0600)"
    echo "  Read it with:  cat $TOKEN_FILE"
    echo "════════════════════════════════════════════════════════════"
fi

# Use read-only DB role for HTTP connections
export SCIX_DSN="host=localhost dbname=scix user=scix_reader password=scix_reader_local_only"

# Use local HuggingFace cache only — no network calls on startup (faster,
# avoids rate limits). Model is already downloaded to ~/.cache/huggingface/.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

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
    CF_LOG="/tmp/cloudflared-mcp.log"
    $CLOUDFLARED tunnel --url http://127.0.0.1:$MCP_PORT --no-autoupdate > "$CF_LOG" 2>&1 &
    CF_PID=$!

    # Wait for tunnel URL to appear in the log
    TUNNEL_URL=""
    for i in $(seq 1 30); do
        TUNNEL_URL=$(grep -oE 'https://[a-z0-9-]+\.trycloudflare\.com' "$CF_LOG" 2>/dev/null | head -1)
        if [ -n "$TUNNEL_URL" ]; then break; fi
        sleep 1
    done

    echo ""
    echo "════════════════════════════════════════════════════════════"
    if [ -n "$TUNNEL_URL" ]; then
        echo "  Tunnel URL: $TUNNEL_URL"
        echo ""
        # Write the full MCP client config to a 0600 file instead of echoing
        # to stdout. stdout is world-readable once redirected to /tmp/*.out.
        CONFIG_FILE="${MCP_CONFIG_FILE:-${HOME}/.config/scix-mcp/client-config.json}"
        mkdir -p "$(dirname "$CONFIG_FILE")"
        (umask 077 && cat > "$CONFIG_FILE" <<EOF
{
  "mcpServers": {
    "scix": {
      "url": "${TUNNEL_URL}/mcp/",
      "headers": {"Authorization": "Bearer ${MCP_AUTH_TOKEN:-<no-auth>}"}
    }
  }
}
EOF
        )
        chmod 600 "$CONFIG_FILE"
        echo "  Client config written to: $CONFIG_FILE  (mode 0600)"
        echo "  View it with:             cat $CONFIG_FILE"
    else
        echo "  Cloudflare Tunnel did not produce a URL in 30s."
        echo "  Check $CF_LOG for details."
    fi
    echo "════════════════════════════════════════════════════════════"
elif [ "$NO_TUNNEL" = false ]; then
    echo "WARNING: cloudflared not found at $CLOUDFLARED — skipping tunnel"
    echo "  Server available locally at http://127.0.0.1:$MCP_PORT/mcp/"
fi

# Wait for either process to exit
trap "kill $MCP_PID ${CF_PID:-} 2>/dev/null" EXIT INT TERM
wait $MCP_PID
