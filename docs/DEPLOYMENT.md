# SciX MCP — Deployment guide

Operator guide for standing up the SciX MCP HTTP server from a clean
environment. Written for the ADS ops team; assumes familiarity with the
BeeHive deployment conventions (`tailor:<name>-v<X>` image tags, Kafka
pipelines, kops-managed `v126.kube.adslabs.org` cluster).

Two deployment targets are supported:

- **Kubernetes** on the AWS cluster (microservice style) — see
  [`deploy/k8s/`](../deploy/k8s/README.md).
- **Docker Compose** on a backoffice host (adsnest / adsnull style) —
  see [`deploy/compose/backoffice.yaml`](../deploy/compose/backoffice.yaml).

Both run the same image from `deploy/Dockerfile`. Pick whichever matches
where the other SciX services for this workflow live.

---

## Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| PostgreSQL | 16.x | Any modern 15+ works; 16 is what the pilot ran |
| `pgvector` extension | 0.8.2+ | Halfvec, iterative scan, parallel HNSW builds |
| `pg_search` extension | optional | ParadeDB BM25. Migration 004 is a no-op if absent. |
| Disk | ~600 GB | Full corpus (32M papers + embeddings + indexes) |
| RAM (DB host) | 64 GB+ | HNSW builds and Leiden jobs are memory-hungry |
| GPU | optional | INDUS runs on CPU; GPU (A10G/RTX+) speeds batch embedding 10–20× |
| HuggingFace cache | ~1.5 GB | For `nasa-impact/nasa-smd-ibm-st-v2` (INDUS) |

The MCP server container itself is modest: 500m CPU / 2 GiB RAM request,
2 CPU / 4 GiB limit. It's the DB that needs headroom.

---

## 1. Bootstrap the database

```bash
createdb scix
psql scix -c "CREATE EXTENSION vector;"           # required
psql scix -c "CREATE EXTENSION IF NOT EXISTS pg_search;"   # optional (ParadeDB)
```

Apply all migrations in order:

```bash
for f in migrations/*.sql; do
  echo "=== $f ==="
  psql scix -v ON_ERROR_STOP=1 -f "$f" || exit 1
done
```

Migrations are **append-only and sequentially numbered** — see
[`docs/UPGRADING.md`](./UPGRADING.md). Running them on an empty DB takes
well under a minute; the big cost is filling them with data (§2) and
building indexes (§3).

Create the read-only role the MCP uses at runtime:

```sql
CREATE ROLE scix_reader LOGIN PASSWORD '<random>';
GRANT CONNECT ON DATABASE scix TO scix_reader;
GRANT USAGE ON SCHEMA public TO scix_reader;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO scix_reader;
ALTER DEFAULT PRIVILEGES IN SCHEMA public
  GRANT SELECT ON TABLES TO scix_reader;
```

The MCP never writes — all writes happen via ingest / embed jobs run by
operators, not by user traffic.

---

## 2. Ingest the corpus

The MCP is read-only; to have anything to serve you need to populate
`papers`, `citations`, and related tables. Two paths:

### Path A — JSONL snapshot (what the pilot uses)

ADS already emits ADS-JSON per bibcode via the classic pipelines and via
the modern harvester. Feed a directory of `.jsonl` / `.jsonl.gz` /
`.jsonl.xz` files (sharded by year is fine) into:

```bash
# From repo root, with .venv active
SCIX_DSN='host=<db-host> dbname=scix user=<superuser> password=...' \
  python scripts/ingest.py --data-dir /path/to/ads_metadata
```

Ingest uses `COPY`, not row-by-row inserts — throughput is 2–5k rec/s on
decent hardware. It is resumable: it tracks per-file progress in
`ingest_log` and skips already-loaded files on re-run.

### Path B — Kafka consumer (future work)

Your harvester pipeline already emits to `HarvesterOutput` topic with an
Avro schema. A production integration should add a consumer that writes
the same rows `ingest.py` does, keyed by bibcode, so the MCP's DB stays
in lockstep with the rest of the SciX corpus. This is scoped in
[`docs/ADS_INTEGRATION.md`](./ADS_INTEGRATION.md) but not shipped in
v0.1.0.

---

## 3. Generate embeddings

```bash
# INDUS (NASA SMD embedding, 768d — primary dense signal)
SCIX_DSN='...' SCIX_EMBED_DEVICE=cuda \
  python scripts/embed.py --model indus --batch-size 64
```

On CPU this takes ~a week for the full 32M corpus; on a single RTX 5090
it's ~16 hours. Scale horizontally by running multiple workers with
disjoint bibcode ranges (the script takes `--offset`/`--limit`).

After embeddings land, build the HNSW index:

```bash
psql scix -f migrations/054_paper_embeddings_halfvec_index.sql
# (Index builds in parallel — see max_parallel_maintenance_workers.)
```

`halfvec` (float16) is safe; binary quantization is **not** — it drops
nDCG@10 by >40% on scientific retrieval. See
[`docs/runbooks/halfvec_migration.md`](./runbooks/halfvec_migration.md)
for the quantization tradeoff.

---

## 4. Build and publish the image

```bash
docker build -f deploy/Dockerfile -t scix-mcp:v0.1.0 .
# Re-tag for the ADS registry (match your ECR/honeycomb conventions):
docker tag scix-mcp:v0.1.0 \
  084981688622.dkr.ecr.us-east-1.amazonaws.com/honeycomb:scix-mcp-v0.1.0
docker push 084981688622.dkr.ecr.us-east-1.amazonaws.com/honeycomb:scix-mcp-v0.1.0
```

The image:
- Runs as uid 1001 with a read-only root filesystem
- Drops all Linux caps, `no-new-privileges`
- Ships no HF cache — see §5 for how to provide it

---

## 5. Deploy — Kubernetes target

See [`deploy/k8s/README.md`](../deploy/k8s/README.md) for the full apply
order. Short version:

```bash
kubectl apply -f deploy/k8s/namespace.yaml
kubectl apply -f deploy/k8s/configmap.yaml
kubectl -n scix-mcp create secret generic scix-mcp-secrets \
  --from-literal=MCP_AUTH_TOKEN="$(python3 -c 'import secrets; print(secrets.token_urlsafe(32))')" \
  --from-literal=SCIX_DSN="host=<pg-host> dbname=scix user=scix_reader password=<pw>"
kubectl apply -f deploy/k8s/deployment.yaml -f deploy/k8s/service.yaml
# Edit the host first, then:
kubectl apply -f deploy/k8s/ingress.example.yaml
```

**INDUS model cache** is the one non-trivial decision — see
`deploy/k8s/README.md` for the `emptyDir` (simple) vs `PersistentVolumeClaim`
(prod) tradeoff.

---

## 5. Deploy — docker-compose target

```bash
cp deploy/.env.example deploy/.env
chmod 600 deploy/.env
$EDITOR deploy/.env   # fill MCP_AUTH_TOKEN, SCIX_DSN, HF_CACHE_DIR

# Pre-warm the HF cache on the host if HF_HUB_OFFLINE=1 (default):
python3 -c "from transformers import AutoModel; \
  AutoModel.from_pretrained('nasa-impact/nasa-smd-ibm-st-v2', \
  cache_dir='$HF_CACHE_DIR')"

docker compose -f deploy/compose/backoffice.yaml \
  --env-file deploy/.env up -d
```

The container publishes on `127.0.0.1:8000` by default — put your
existing nginx / traefik / Cloudflare in front for TLS and public exposure.

Health check:
```bash
curl -sf http://127.0.0.1:8000/health
# {"status":"ok"}
```

---

## 6. Verify end-to-end

```bash
TOKEN=<the MCP_AUTH_TOKEN you set>
curl -sS -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -X POST https://<host>/mcp/ \
  -d '{"jsonrpc":"2.0","id":1,"method":"tools/list"}' | jq '.result.tools | length'
# 13  (post-consolidation tool count — see CHANGELOG for details)
```

Then a search:
```bash
curl -sS -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -X POST https://<host>/mcp/ \
  -d '{"jsonrpc":"2.0","id":2,"method":"tools/call",
       "params":{"name":"search","arguments":{"query":"exoplanet atmospheres","limit":3}}}' \
  | jq '.result.content[0].text' | head -40
```

---

## 7. Operations

### Rotating the bearer token

**k8s**:
```bash
NEW_TOKEN=$(python3 -c 'import secrets; print(secrets.token_urlsafe(32))')
kubectl -n scix-mcp create secret generic scix-mcp-secrets \
  --from-literal=MCP_AUTH_TOKEN="$NEW_TOKEN" \
  --from-literal=SCIX_DSN="$(kubectl -n scix-mcp get secret scix-mcp-secrets \
       -o jsonpath='{.data.SCIX_DSN}' | base64 -d)" \
  --dry-run=client -o yaml | kubectl apply -f -
kubectl -n scix-mcp rollout restart deploy/scix-mcp
```

**compose**: edit `deploy/.env`, then `docker compose -f deploy/compose/backoffice.yaml --env-file deploy/.env up -d --force-recreate scix-mcp`.

### Observability

- `/health` → 200 when Postgres pool is reachable (used by k8s probes
  and docker healthcheck).
- Logs are stdout/stderr JSON-ish; wire into Graylog / Fluentd the same
  way the rest of SciX does.
- Per-tool timeouts are tunable via `SCIX_TIMEOUT_*` env (see
  `configmap.yaml`). Tune only if you see 30s+ tool calls under load.

### Scaling

The MCP server is stateless. Scale horizontally with `replicas: N`; each
pod carries its own Postgres pool (`SCIX_POOL_MIN`/`SCIX_POOL_MAX` env).
Watch DB `max_connections` headroom — `pool_max × replicas` is the hard
upper bound on concurrent connections from the MCP tier.

### Upgrading

See [`docs/UPGRADING.md`](./UPGRADING.md) for the migration-and-tag
contract. Short version: pull a new tagged release, apply any new
migrations in order, rebuild + push the image, `kubectl rollout restart`.

---

## 8. Common failures

| Symptom | Likely cause | Fix |
|---|---|---|
| Pod `CrashLoopBackOff`, logs show `connection refused` | `SCIX_DSN` host not reachable from the pod | Verify networking / security groups; use `kubectl run -it --rm debug --image=postgres:16 -- psql "$SCIX_DSN"` to confirm |
| `401 Unauthorized` from every request | Token mismatch | `kubectl -n scix-mcp get secret scix-mcp-secrets -o jsonpath='{.data.MCP_AUTH_TOKEN}' \| base64 -d` vs client header |
| Slow startup (~90 s) on fresh pod | First-time INDUS download with `HF_HUB_OFFLINE=0` | Either accept it, or switch to a pre-seeded PVC (see `deploy/k8s/README.md`) |
| `semantic_search` returns 500s | INDUS can't load — HF cache missing with `HF_HUB_OFFLINE=1` | Seed the cache volume, or flip offline flags to `0` |
| Tool calls hang / time out | Postgres pool exhausted | Raise `SCIX_POOL_MAX`; check for long-running queries via `pg_stat_activity` |
| `pg_search`-specific errors in logs | ParadeDB extension missing | Migration 004 is a no-op without it; ignore or install pg_search |
