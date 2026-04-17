# SciX MCP — Client Setup

Instructions for connecting to a running SciX MCP server. If you *operate* the
server, see `deploy/README.md` instead.

## What you need from the server operator

Two things, shared via a secure channel (1Password shared vault, Signal, SFTP —
**never** email, Slack, Discord, Twitter DM):

1. **URL** — looks like `https://mcp.<domain>/mcp/`
2. **Bearer token** — a 40+ character random string

The token is equivalent to a password. If it leaks, ask the operator to
rotate it.

## Step 1 — Prove the connection works (no client install required)

```bash
TOKEN='<paste-the-token-here>'
URL='https://mcp.<domain>/mcp/'

curl -X POST \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -H "Accept: application/json, text/event-stream" \
  -d '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"test","version":"0"}}}' \
  "$URL"
```

A healthy response starts with:

```
event: message
data: {"jsonrpc":"2.0","id":1,"result":{"protocolVersion":"2024-11-05", ...,"serverInfo":{"name":"scix","version":"1.27.0"}}}
```

If you see `401` — the token is wrong. If `503` — the server is down on the
operator's side; ping them. If `curl` hangs — your local DNS can't resolve the
hostname yet; try `dig @1.1.1.1 mcp.<domain>`.

## Step 2a — Claude Code setup

CLI (easiest):

```bash
claude mcp add --scope user --transport http scix "https://mcp.<domain>/mcp/" \
  --header "Authorization: Bearer <TOKEN>"
```

Or manually edit `~/.claude.json`:

```json
{
  "mcpServers": {
    "scix": {
      "url": "https://mcp.<domain>/mcp/",
      "headers": { "Authorization": "Bearer <TOKEN>" }
    }
  }
}
```

Test: start a Claude Code session, type `/mcp`, pick **scix**. You should see
13 tools — `search`, `citation_graph`, `read_paper`, `concept_search`, and so
on.

## Step 2b — Claude Desktop setup

Config file location:

| OS | Path |
|---|---|
| macOS | `~/Library/Application Support/Claude/claude_desktop_config.json` |
| Windows | `%APPDATA%\Claude\claude_desktop_config.json` |

Add (or merge into existing `mcpServers`):

```json
{
  "mcpServers": {
    "scix": {
      "url": "https://mcp.<domain>/mcp/",
      "headers": { "Authorization": "Bearer <TOKEN>" }
    }
  }
}
```

**Restart Claude Desktop.** The hammer icon in the compose box should show
`scix` and its tools.

Requires Claude Desktop **0.8+** for HTTP-MCP support. Update if older.

## Step 3 — Try a real call

In a conversation with the server active, ask:

> Use the scix server to search for "infrared asteroid spectroscopy"
> published in the last five years. Limit to 5 results.

The model will call `scix.search` and return papers with bibcodes, titles,
and citation counts.

## Available tools (reference)

| Tool | Purpose |
|---|---|
| `search` | Hybrid semantic+keyword search across ~32M papers |
| `concept_search` | Papers tagged with a Unified Astronomy Thesaurus concept |
| `get_paper` | Metadata + abstract for one bibcode |
| `read_paper` | Full-text body of a paper (paginated) |
| `citation_graph` | Forward/backward citations around a paper |
| `citation_similarity` | Co-cited and bibliographically-coupled papers |
| `citation_chain` | Shortest path between two papers |
| `entity` | Look up named methods / datasets / instruments |
| `entity_context` | Where an entity appears in the literature |
| `graph_context` | Local entity-relationship graph |
| `facet_counts` | Aggregate counts by year, author, etc. |
| `temporal_evolution` | How a topic evolves over time |
| `find_gaps` | Under-researched topics in a given area |

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| `/health` returns 200 but `/mcp/` always 401 | Wrong token — re-paste or ask operator to rotate |
| Every endpoint returns 503 | Server container or tunnel is down on operator's side |
| DNS fails to resolve `mcp.<domain>` | Local resolver stale — wait or try `dig @1.1.1.1` |
| `curl` works but Claude Desktop shows "failed to connect" | Desktop version too old — update to 0.8+ |
| Tool calls time out | Server CPU maxed out — operator can check with `docker stats scix-mcp` |

If stuck, send the operator:
- the exact HTTP status from `curl -v`
- a screenshot of the MCP list in your client
- the error message from your client's logs
