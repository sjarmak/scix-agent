# SciX MCP — Hardened Deployment

This directory contains the production-grade deployment for the SciX MCP
HTTP server. It replaces the `scripts/start_mcp_http.sh` quick-tunnel
setup with:

- **Containerized Python server** running as unprivileged uid 1001,
  read-only root filesystem, all Linux capabilities dropped, no new
  privileges, CPU/memory limits.
- **Containerized cloudflared** running a **named tunnel** (stable URL
  that survives restarts, Cloudflare-account-owned).
- **Cloudflare Access** sits in front of the tunnel and does per-user
  auth (email, Google, GitHub, etc.) before any request reaches your
  machine. Bearer-token check becomes defense-in-depth, not the only
  lock on the door.

## Threat model this setup addresses

| Threat from the quick-tunnel setup | How this deploy addresses it |
|---|---|
| RCE in MCP server → full user compromise (ssh keys, claude tokens, DB superuser) | Container, unprivileged user, read-only FS, cap-drop=ALL, no-new-privileges |
| Quick-tunnel URL rotates on every restart | Named Cloudflare Tunnel → stable DNS hostname |
| One shared bearer token = no per-user revocation, no audit trail | Cloudflare Access → per-user auth via SSO, per-user audit logs |
| Bearer token in launch-script stdout / world-readable /tmp logs | Token only in `deploy/.env` (chmod 600) + container env |
| Cloudflare quick-tunnels have no uptime guarantee | Named tunnels are production-supported |
| INDUS CPU flood DoS from any token holder | Per-container CPU/memory limits; rate-limit stays in place |

## One-time setup (Cloudflare side)

You need a free Cloudflare account and any domain pointed at CF's
nameservers. Work done in the CF dashboard; nothing to automate here.

1. **Sign up**: https://dash.cloudflare.com/ (free tier is enough).
2. **Add your domain** (Cloudflare Dashboard → Add site → Free plan).
3. **Zero Trust → Networks → Tunnels → Create a tunnel**
   - Name: `scix-mcp`
   - Connector: **Cloudflared**
   - Copy the **tunnel token** it shows (single opaque string).
     This goes into `deploy/.env` as `CF_TUNNEL_TOKEN`.
   - **Public Hostnames** tab → Add a public hostname:
     - Subdomain: e.g. `mcp`
     - Domain: your domain
     - Service: `HTTP` → `scix-mcp:8000`
       (service name matches the compose/run.sh container name)
4. **Zero Trust → Access → Applications → Add an application**
   - Type: **Self-hosted**
   - Application name: `SciX MCP`
   - Session Duration: `24h` (or shorter)
   - Application domain: `mcp.yourdomain.com` (matches step 3)
   - Identity providers: enable Google / GitHub / one-time PIN / etc.
   - **Policies** → Add policy
     - Action: `Allow`
     - Rule: `Emails → <your email>` (or `Emails ending in @yourorg.com`)
5. **Copy the Access Team domain** from Zero Trust → Settings → Custom
   Pages → you'll see `https://<your-team>.cloudflareaccess.com`.
   Users log in here before the tunnel forwards them anywhere.

After step 5 your public URL is `https://mcp.yourdomain.com/mcp/` and
Cloudflare enforces SSO **before** any request hits your machine.

## One-time setup (host side)

```bash
cp deploy/.env.example deploy/.env
chmod 600 deploy/.env
$EDITOR deploy/.env          # fill in MCP_AUTH_TOKEN, CF_TUNNEL_TOKEN, HF_CACHE_DIR
```

Generate a fresh bearer token:

```bash
python3 -c "import secrets; print(secrets.token_urlsafe(32))"
```

Paste it into `MCP_AUTH_TOKEN=` in the `.env`.

## Launch

Either Compose (if installed):

```bash
cd deploy
docker compose up -d --build
docker compose logs -f
```

or the plain-docker wrapper (no compose needed):

```bash
./deploy/run.sh          # builds + starts
./deploy/run.sh logs     # tail both containers
./deploy/run.sh stop     # tear down
```

## Client config

```json
{
  "mcpServers": {
    "scix": {
      "url": "https://mcp.yourdomain.com/mcp/",
      "headers": { "Authorization": "Bearer <MCP_AUTH_TOKEN>" }
    }
  }
}
```

First request from each new device will redirect through Cloudflare
Access SSO. After that, Access issues a session cookie and the bearer
token provides the second factor.

## Token rotation

```bash
# Generate + write to .env, then restart the server container only
# (cloudflared doesn't care about MCP_AUTH_TOKEN).
python3 -c "import secrets; print(f'MCP_AUTH_TOKEN={secrets.token_urlsafe(32)}')" \
    >> deploy/.env.new
# Replace the line in .env manually, then:
docker restart scix-mcp
```

Distribute the new token via the same out-of-band channel you use for
your Cloudflare Access app.

## Ongoing hygiene

- `git log --oneline deploy/.env*` — make sure the real `.env` is never
  committed. It's in `.gitignore`.
- `docker logs scix-mcp | grep -iE 'error|unauthor'` — quick audit for
  brute-force attempts.
- Cloudflare Zero Trust → Logs → Access → per-user access log.
- Rebuild image monthly to pick up base-image CVE fixes:
  `docker compose build --no-cache scix-mcp && docker compose up -d`.
