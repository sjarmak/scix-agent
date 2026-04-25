# SciX MCP Demo Stack (Workshop Stop-Gap)

A self-contained, public-accessible SciX MCP server for workshop attendees.
Runs on a €14/mo Hetzner VPS. **No inbound ports open on the home network** —
the stack lives entirely off-premises, and Cloudflare Tunnel dials out from
the VPS.

## What it is

- Postgres 16 + pgvector (containerised, internal to compose — no host exposure)
- SciX MCP HTTP server (same image as prod `deploy/Dockerfile`)
- Cloudflare named tunnel (outbound-only)
- Subset of scix data: papers + INDUS embeddings + citation edges + metrics
  for publication years 2022-2026 (~3.9M papers, ~50-80 GB on disk)

## What it is not

- Not a full mirror: no full-text bodies, no fusion-MV state, no entity
  resolution tables, no alternate embedding models
- Not HA — single VPS, single Postgres, no backups beyond the compose
  volume. Tear down and re-provision if corrupted.

## One-time setup

### 1. Provision the VPS

Hetzner CX42 recommended (€14/mo, 8 GB RAM, 160 GB NVMe).
After `cloud-init`, install Docker:

```bash
ssh root@<vps-ip>
apt-get update && apt-get install -y ca-certificates curl
install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
  -o /etc/apt/keyrings/docker.asc
echo "deb [arch=$(dpkg --print-architecture) \
  signed-by=/etc/apt/keyrings/docker.asc] \
  https://download.docker.com/linux/ubuntu $(. /etc/os-release; echo $VERSION_CODENAME) stable" \
  > /etc/apt/sources.list.d/docker.list
apt-get update && apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin
mkdir -p /opt/scix-demo
```

### 2. Create the Cloudflare tunnel

In the Cloudflare dashboard: **Zero Trust → Networks → Tunnels → Create
tunnel**. Name it `scix-demo`, pick **Cloudflared**, copy the token (this is
`CF_TUNNEL_TOKEN`).

Add a public hostname:
- **Subdomain**: `demo-mcp` (or whatever)
- **Domain**: `sjarmak.ai`
- **Service**: `http://scix-mcp:8000`

Save. Leaves DNS CNAME `demo-mcp.sjarmak.ai → <uuid>.cfargotunnel.com`
behind.

### 3. Transfer code, migrations, and HF cache to the VPS

From the local machine:

```bash
# code + migrations + compose files
rsync -avz --exclude='.venv' --exclude='__pycache__' --exclude='ads_metadata_by_year_picard' \
  ~/projects/scix_experiments/ root@<vps-ip>:/opt/scix-demo/repo/

# HF cache (only INDUS snapshot is needed)
rsync -avz ~/.cache/huggingface/hub/models--nasa-impact--nasa-smd-ibm-st-v2 \
  root@<vps-ip>:/opt/scix-demo/hf-cache/hub/
```

### 4. Configure `.env` on the VPS

```bash
ssh root@<vps-ip>
cd /opt/scix-demo/repo/deploy/demo
cp .env.example .env
chmod 600 .env
# edit .env — fill in MCP_AUTH_TOKEN, POSTGRES_PASSWORD, CF_TUNNEL_TOKEN
```

### 5. Start Postgres (first), then export the subset, then finish

From the local machine, produce the dump:

```bash
cd ~/projects/scix_experiments/deploy/demo
./export_subset.sh
# produces /tmp/scix-demo-dump/{01_papers,02_embeddings,03_citations,04_metrics}.dump
# ~15-25 GB total compressed
```

Transfer the dump to the VPS:

```bash
rsync -avz --progress /tmp/scix-demo-dump/ root@<vps-ip>:/opt/scix-demo/dump/
```

On the VPS, bring up Postgres only (it must be healthy before restore):

```bash
cd /opt/scix-demo/repo/deploy/demo
docker compose up -d postgres
# wait ~20s for healthcheck to pass
docker compose ps
```

Run the restore (applies migrations, loads dumps, builds HNSW index):

```bash
DUMP_DIR=/opt/scix-demo/dump \
MIGRATIONS_DIR=/opt/scix-demo/repo/migrations \
./restore_subset.sh
# HNSW build for 3.9M vectors takes ~30-60 min on a CX42
```

Bring up the rest:

```bash
docker compose up -d
docker compose ps   # all three services should be healthy
docker compose logs -f cloudflared | grep -i "Registered tunnel"
```

### 6. Smoke test

```bash
curl -sf https://demo-mcp.sjarmak.ai/health
# {"status":"ok"}

curl -s -H "Authorization: Bearer $(grep ^MCP_AUTH_TOKEN .env | cut -d= -f2)" \
  https://demo-mcp.sjarmak.ai/mcp/ \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","id":1,"method":"tools/list"}'
```

## Workshop handout

Give attendees:

```json
{
  "mcpServers": {
    "scix-demo": {
      "type": "http",
      "url": "https://demo-mcp.sjarmak.ai/mcp/",
      "headers": {
        "Authorization": "Bearer <shared-workshop-token>"
      }
    }
  }
}
```

Rotate the token after each workshop (edit `.env`, `docker compose up -d
scix-mcp`).

## Teardown

```bash
# stop stack, keep data
docker compose down

# nuke everything including the Postgres volume
docker compose down -v

# or delete the VPS entirely
hcloud server delete scix-demo
```

## Costs

- Hetzner CX42: **€14/mo**
- Cloudflare Tunnel: **free**
- Domain: already paid for `sjarmak.ai`
- Egress: Hetzner includes 20 TB/mo; workshop traffic is a rounding error

**Total: ~€14/mo**. Can be torn down to €0 between workshops.

## Known limits / design choices

- **Data freshness**: static snapshot at export time. To refresh, re-run
  `export_subset.sh`, rsync the new dump, drop/recreate the Postgres
  volume, re-run `restore_subset.sh`. No incremental sync — not worth the
  complexity for a demo.
- **CPU-only embeddings**: `SCIX_EMBED_DEVICE=cpu` in compose. INDUS
  query-time encoding takes ~200-400 ms per query on CX42 CPUs, which is
  fine for a demo workload.
- **HNSW ef_search**: default 100 is used. If recall feels weak during
  workshops, bump via `SET LOCAL hnsw.ef_search = 200` in search code.
- **BM25 via regenerated tsv**: `papers.tsv` is not dumped (it would
  roughly double the dump size). Instead, `restore_subset.sh` calls
  `post_restore_bm25.sql` to regenerate it from title/abstract/keywords
  using the same `scix_english` config and weighting as production. Adds
  ~3-5 min to the restore on a CX42.
- **Token auth only**: workshop attendees share one bearer token. Fine
  for a 45-min session; don't leave it running for weeks with the same
  token. Cloudflare Access (SSO) is a future upgrade if we keep the demo
  running long-term.
