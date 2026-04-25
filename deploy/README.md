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

### Postgres on the host

The containers expect postgres to be reachable from the docker bridge
gateway (`host.docker.internal`, 172.17.0.1 on Linux). Two configurations
are supported:

1. **Recommended: postgres listens on the bridge gateway.** Set
   `listen_addresses = 'localhost,172.17.0.1'` in `postgresql.conf` and add
   a `pg_hba.conf` entry for `172.17.0.0/16` with `scram-sha-256`. Restart
   postgres. The container connects directly.
2. **Fallback: pg_docker_proxy.** If you can't edit postgres config (e.g.
   no sudo), `./deploy/run.sh` auto-starts `scripts/pg_docker_proxy.py`
   which listens on 172.17.0.1:5432 and forwards to 127.0.0.1:5432, making
   postgres reachable to containers. The proxy is a foreground python
   process owned by the invoking user; it does not survive reboot unless
   `./deploy/run.sh` is re-run.

Either way, the `SCIX_DSN` in `deploy/.env` stays the same:
`host=host.docker.internal dbname=scix user=scix_reader password=...`

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

## Rerank rollout

The `mcp__scix__search` tool exposes a `use_rerank: bool = True` flag
which, when honored, runs the retrieved candidates through a cross-
encoder reranker selected by the `SCIX_RERANK_DEFAULT_MODEL` environment
variable (wired in commit `1ea69b7`).

**Active value: `SCIX_RERANK_DEFAULT_MODEL=off` (the shipped default).**

This default exists because the M1 ablation (commit `06a6cc3`,
[`results/retrieval_eval_50q_rerank_local.md`](../results/retrieval_eval_50q_rerank_local.md))
showed both candidate rerankers regress nDCG@10 vs the unranked
INDUS-hybrid baseline at the Bonferroni-corrected α=0.025 threshold:

| Config             | nDCG@10 | Δ vs baseline | Wilcoxon p | Verdict             |
|--------------------|---------|---------------|------------|---------------------|
| `hybrid_indus`     | 0.3255  | —             | —          | baseline (winner)   |
| `minilm`           | 0.2802  | −0.0453       | 0.042      | regression          |
| `bge-large`        | 0.2699  | −0.0556       | 0.026      | regression          |

Neither candidate clears the gate, so M4 closed with rerank disabled.
The negative result is documented above so future operators don't re-run
the same A/B with the same models expecting a different answer.

### Flipping rerank on (operator opt-in)

A fresh-eyes operator who wants to experiment can set the env var to one
of the two known model paths and restart the MCP server:

```bash
# Option A: MiniLM (~80 MB, fast — the default operator-flip target)
echo 'SCIX_RERANK_DEFAULT_MODEL=minilm' >> deploy/.env

# Option B: BAAI/bge-reranker-large (~1.3 GB, slower; weights expected
# under models/bge-reranker-large or HF cache)
echo 'SCIX_RERANK_DEFAULT_MODEL=bge-large' >> deploy/.env

docker restart scix-mcp
```

Implications:

- Per-call latency grows by the rerank stage. M1 measurements: MiniLM
  p95 ≈ 70 ms (CPU, 50 candidates); bge-reranker-large p95 ≈ 570 ms (GPU)
  / 4+ s (CPU). On the deployed server (CPU-only) bge-reranker-large is
  in 4-second territory per call.
- Retrieval quality is *expected to drop* on the 50-query gold set per
  M1. If the operator's experiment depends on a different metric (e.g.
  per-query Top-1 quality on a domain-specific subset) that's a valid
  reason to flip, but the headline nDCG@10 number will look worse.
- Roll back by restoring `SCIX_RERANK_DEFAULT_MODEL=off` (or removing
  the line entirely; `off` is the hard-coded fallback) and restarting
  the container.

## Daily canaries

Even with rerank disabled by default, we run a small daily smoke test
against the MiniLM checkpoint. The canary catches model behaviour drift
(e.g. an upstream HuggingFace re-upload that silently changes weights)
and verifies the `CrossEncoderReranker` integration surface still loads
+ scores end-to-end, so when a follow-up reranker lands we have
continuity.

| Canary             | Script                              | Purpose                                                    |
|--------------------|-------------------------------------|------------------------------------------------------------|
| MiniLM rerank      | [`scripts/canary_rerank.py`](../scripts/canary_rerank.py) | 20 fixed (query, paper) pairs; alert on >5% score drift |
| NER drift          | [`scripts/canary_ner.py`](../scripts/canary_ner.py)       | 10 fixed snippets; alert on >5% per-entity F1 drift     |

Sample cron entries (operators add their own — nothing is auto-installed):

```cron
# Daily MiniLM rerank drift check at 06:00 local
0 6 * * * cd /home/ds/projects/scix_experiments && .venv/bin/python scripts/canary_rerank.py >> logs/canary_rerank/cron.log 2>&1

# Daily NER drift check at 03:00 local
0 3 * * * cd /home/ds/projects/scix_experiments && .venv/bin/python scripts/canary_ner.py >> logs/canary_ner/cron.log 2>&1
```

Both canaries exit non-zero on drift, so any cron-mail or
log-monitoring wrapper picks up the failure for free.
