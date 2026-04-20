# SciX Agent — Parity Playbook

A step-by-step blueprint for standing up a SciX MCP deployment that
matches the pilot state (32.4M papers, 299.3M citation edges, 14.9M
full-text bodies, 32.4M INDUS embeddings with halfvec HNSW, 1.58M
entities, 28.4M document→entity links, citation-graph metrics computed
at three resolutions).

This document is **ordered** — each phase depends on what comes before.
Durations are wall-clock estimates on hardware comparable to the pilot
(single RTX 5090, 64 GB RAM, fast NVMe). Your cluster with parallel
workers will typically be faster; a CPU-only box will be slower.

Companion docs:
- [`DEPLOYMENT.md`](./DEPLOYMENT.md) — mechanical deploy (k8s, compose)
- [`ADS_INTEGRATION.md`](./ADS_INTEGRATION.md) — positioning + auth options
- [`UPGRADING.md`](./UPGRADING.md) — tag/migration contract

---

## Phase 0 — Prerequisites (0.5–2 days, mostly approval cycles)

| Decision | Who owns it | Our suggestion |
|---|---|---|
| Deployment target | ADS ops | Microservice on `v126.kube.adslabs.org` (follows BeeHive convention) |
| Dedicated Postgres | ADS ops | Yes — 64+ GB RAM, ~600 GB disk, own server. Colocating with `master_pipeline` will have both rough |
| Postgres version | ADS ops | 16.x (pilot runs on 16) |
| pgvector install | DBA | 0.8.2+ required; parameter-group change on RDS + reboot if managed |
| GPU allocation | ADS ops | 1 GPU node for ~1–2 days during initial embed, then releasable |
| Auth pattern | Security + ops | See `ADS_INTEGRATION.md` §Auth — gateway-validated ADS token is lowest-friction |
| Image registry | ADS ops | ECR repo of your choice, e.g. `honeycomb:scix-mcp-v0.1.0` |
| Container platform | ADS ops | k8s manifests in `deploy/k8s/` are ready to apply |

**Acceptance for Phase 0**: a Postgres instance reachable from where
you'll run the ingest/embed jobs, with `CREATE EXTENSION vector;`
tested successfully.

---

## Phase 1 — Clone, build, schema bootstrap (1 hour)

```bash
# 1.1 Clone the tagged release (or mirror it under adsabs/)
git clone https://github.com/sjarmak/scix-agent.git
cd scix-agent
git checkout v0.1.0

# 1.2 Python env for running the pipeline scripts
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,mcp,search,embed,graph]"

# 1.3 Build the image (for later deploy)
docker build -f deploy/Dockerfile -t scix-mcp:v0.1.0 .
docker tag scix-mcp:v0.1.0 <your-registry>/scix-mcp:v0.1.0
docker push <your-registry>/scix-mcp:v0.1.0

# 1.4 Create database + extensions + read-only role
createdb scix
psql scix <<'SQL'
CREATE EXTENSION vector;
CREATE EXTENSION IF NOT EXISTS pg_search;   -- optional (ParadeDB)
CREATE ROLE scix_reader LOGIN PASSWORD '<pw>';
GRANT CONNECT ON DATABASE scix TO scix_reader;
GRANT USAGE ON SCHEMA public TO scix_reader;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO scix_reader;
ALTER DEFAULT PRIVILEGES IN SCHEMA public
  GRANT SELECT ON TABLES TO scix_reader;
SQL

# 1.5 Apply all 54 migrations in order. Idempotent.
export SCIX_DSN='host=<pg-host> dbname=scix user=<superuser> password=<pw>'
for f in migrations/*.sql; do
  echo "=== $f ==="
  psql "$SCIX_DSN" -v ON_ERROR_STOP=1 -f "$f" || exit 1
done
```

**Acceptance**:
```sql
SELECT count(*) FROM information_schema.tables WHERE table_schema='public';
-- expect 60+ tables including papers, citation_edges, paper_embeddings,
-- entities, document_entities, uat_concepts, etc.
SELECT extname, extversion FROM pg_extension WHERE extname IN ('vector','pg_search');
-- vector | 0.8.2 (or higher);  pg_search optional
```

---

## Phase 2 — Corpus ingest: metadata + citation edges (~3 hours for 32M)

You feed a directory of JSONL files. See `DEPLOYMENT.md` §Data format
for the exact field-name contract, but in short: this is the same
ADS-JSON format the master/harvester pipelines already emit.

```bash
# Expect .jsonl / .jsonl.gz / .jsonl.xz files. Sharded by year is fine.
export SCIX_DSN='host=<pg-host> dbname=scix user=<ingest_writer> password=<pw>'
.venv/bin/python scripts/ingest.py --data-dir /path/to/ads_jsonl/

# Resumable: tracks per-file progress in ingest_log, re-running skips
# completed files. Throughput 2–5k rec/s via COPY.
```

**Acceptance**:
```sql
SELECT count(*) FROM papers;              -- should approach 32,400,594
SELECT count(*) FROM citation_edges;      -- should approach 299,329,159
SELECT min(year), max(year) FROM papers;  -- 1800, 2026
```

If your JSONL source does not carry `body` / full-text, you'll see
`SELECT count(body) FROM papers;` return near zero. That's fine for a
partial parity — body ingest is Phase 3, and the MCP works without it.

---

## Phase 3 — Full-text body ingest (optional; ~1–2 days if you want it)

The pilot has 14.9M bodies (46%). You get this by either:

- **3a.** Feeding JSONL that includes the `body` field (our pilot path)
  — no extra step, just re-run `scripts/ingest.py` against the
  body-inclusive JSONL and it `UPDATE`s in place.
- **3b.** Running `scripts/backfill_body_from_ads.py` or
  `scripts/backfill_body_parallel.py` against the ADS API using an
  `ADS_API_KEY` — this is how the pilot pulled recent years. See
  `docs/runbooks/arxiv_s3_ingest.md` for the arXiv bulk source path.
- **3c.** Skipping entirely — `read_paper` MCP tool and full-text
  `search` returns empty for rows without body, but every other tool
  works.

**Acceptance**:
```sql
SELECT count(body) FROM papers;  -- parity target: 14.9M (46%)
SELECT size_pretty FROM pg_size_pretty(pg_total_relation_size('papers'));
```

---

## Phase 4 — INDUS embeddings on the full corpus (~16h on 1× RTX-class GPU, the long pole)

INDUS (`nasa-impact/nasa-smd-ibm-st-v2`, 768d) is the primary dense
retrieval signal. Run on a GPU node; CPU works but is ~10–20× slower.

```bash
# 4.1 Stage unembedded papers (avoids slow anti-join during streaming)
psql "$SCIX_DSN" <<'SQL'
CREATE TABLE _to_embed AS
SELECT p.bibcode, p.title, p.abstract
FROM papers p
LEFT JOIN paper_embeddings pe
  ON p.bibcode = pe.bibcode AND pe.model_name = 'indus'
WHERE pe.bibcode IS NULL AND p.title IS NOT NULL
ORDER BY p.bibcode;
CREATE INDEX ON _to_embed (bibcode);
SQL

# 4.2 Run the optimized multi-worker embedder.
#     On a multi-GPU node, run N workers with disjoint bibcode prefixes
#     (see scripts/embed_optimized.py --help for sharding flags).
.venv/bin/python scripts/embed_optimized.py --device cuda:0
#     For CPU-only or slower: scripts/embed.py --device cpu

# 4.3 When done, drop the staging table.
psql "$SCIX_DSN" -c 'DROP TABLE _to_embed;'
```

**Acceptance**:
```sql
SELECT model_name, count(*) FROM paper_embeddings GROUP BY model_name;
-- indus | ~32,400,000  (parity target)
```

---

## Phase 5 — halfvec backfill + HNSW indexes (~4–8 hours)

Cuts embedding storage by half and builds the production retrieval
index. `docs/runbooks/halfvec_migration.md` has the full runbook.

```bash
# 5.1 Backfill halfvec shadow column (batches, resumable, online).
.venv/bin/python scripts/backfill_halfvec.py --allow-prod

# 5.2 Apply the halfvec HNSW index migration explicitly if you haven't
#     run it already (migration 054 is in the migrations/ pass).
psql "$SCIX_DSN" -f migrations/054_paper_embeddings_halfvec_index.sql
```

**Acceptance**:
```sql
SELECT count(*) FROM paper_embeddings WHERE embedding_hv IS NOT NULL;
-- expect to match the INDUS embedding count
SELECT indexname FROM pg_indexes WHERE tablename='paper_embeddings';
-- expect idx_embed_hnsw_indus_halfvec
```

---

## Phase 6 — Citation graph metrics (~6–12 hours)

Computes PageRank, HITS, and Leiden communities at 3 resolutions.
Writes to `paper_metrics` and community columns on `papers`.

```bash
.venv/bin/python scripts/graph_metrics.py --allow-prod
# Defaults: --res-coarse 5.0 --res-medium 1.0 --res-fine 0.1 --seed 42
# Use --skip-labels if you don't want per-community LLM labels (faster).
```

**Acceptance**:
```sql
SELECT count(*) FROM paper_metrics;                     -- ~32M rows
SELECT count(DISTINCT community_coarse) FROM papers;    -- tens of thousands
SELECT count(DISTINCT community_fine) FROM papers;      -- hundreds of thousands
```

---

## Phase 7 — Entity graph (~2–4 days; can run in parallel with Phase 6)

The pilot ships 1.58M entities (1.49M solar-system bodies from SSODNet,
62k VizieR datasets, 10k GCMD, 9k PwC methods, 4k ASCL software, 4k
PhySH, 229 SPASE, etc.) plus 28.4M document→entity links across tiers
1 and 2. Tier 3 (LLM extraction) is not populated in the pilot.

### 7.1 Ontology harvesters (~1–2 days, parallelizable)

```bash
# Authority-per-script. Each hits an external API and is rate-limited
# by the upstream. Safe to run concurrently in different shells.

# NASA / domain authorities
.venv/bin/python scripts/harvest_ssodnet.py --allow-prod      # ~1.49M solar bodies (largest)
.venv/bin/python scripts/harvest_gcmd.py --allow-prod         # ~10k earth science keywords
.venv/bin/python scripts/harvest_spase.py --allow-prod        # ~230 heliophysics products
.venv/bin/python scripts/harvest_spdf.py --allow-prod
.venv/bin/python scripts/harvest_cmr.py --allow-prod
.venv/bin/python scripts/harvest_pds4.py --allow-prod
.venv/bin/python scripts/harvest_sbdb.py --allow-prod

# Astronomy / datasets
.venv/bin/python scripts/harvest_vizier.py --allow-prod       # ~62k datasets
.venv/bin/python scripts/harvest_aas_facilities.py --allow-prod
.venv/bin/python scripts/harvest_astromlab.py --allow-prod

# Software + methods
.venv/bin/python scripts/harvest_ascl.py --allow-prod         # ~4k astro software
.venv/bin/python scripts/harvest_pwc_methods.py --allow-prod  # ~9k ML methods

# Physics subjects
.venv/bin/python scripts/harvest_physh.py --allow-prod        # ~4k PhySH concepts

# ADS-internal data-field index
.venv/bin/python scripts/harvest_ads_data_field.py --allow-prod
```

**Note**: harvesters write to staging tables; run `scripts/promote_staging_extractions.py --allow-prod` to promote from `*_staging` to `entities`/`entity_aliases`/`entity_identifiers`. Some harvesters handle promotion inline; check each script header for the contract.

### 7.2 UAT concept bootstrap

```bash
.venv/bin/python scripts/load_uat.py  # ~2,300 concepts from the UAT release
```

`paper_uat_mappings` is populated automatically during Phase 2 ingest
(ADS's JSONL carries `keyword_schema=uat`).

### 7.3 Entity relationships from source hierarchies (~2–4 hours)

```bash
.venv/bin/python scripts/populate_entity_relationships.py \
  --allow-prod --sources all
# Add --include-ssodnet-targets for full 1.48M part_of edges (slower);
# omit for the faster ~1.5M relationship baseline.
```

### 7.4 Linking pipeline — tier 1 (keyword match, ~minutes)

```bash
.venv/bin/python scripts/link_tier1.py --allow-prod -v
```

### 7.5 Linking pipeline — tier 2 (Aho-Corasick, ~1–2 days, CPU-bound)

Scans every paper abstract for entity name mentions using an
Aho-Corasick automaton. See `docs/runbooks/entity_linking_kickoff.md`
for tuning (workers, per-entity caps, prefix shards).

```bash
.venv/bin/python scripts/link_tier2.py \
  --allow-prod --workers 8 --entity-source curated -v
```

**Acceptance** (after 7.1–7.5):
```sql
SELECT 'entities', count(*) FROM entities
UNION ALL SELECT 'document_entities', count(*) FROM document_entities
UNION ALL SELECT 'papers_with_entity', count(DISTINCT bibcode) FROM document_entities;
-- Parity targets:
-- entities:             ~1,577,644
-- document_entities:    ~28,442,746
-- papers_with_entity:   ~12,440,705  (38% of corpus)
```

### 7.6 (Optional, not yet run in the pilot) Tier 3 LLM extraction

`src/scix/extract.py` + `staging_ner_extractions` + the promote path
are in place but not yet run at corpus scale. This is the unlock for
fine-grained mentions (specific instruments used, datasets referenced,
methodology spans). **Skip this for parity** — neither side has done it.

---

## Phase 8 — Deploy the MCP server (~2 hours)

See `deploy/k8s/README.md` for the full apply order. Short version:

```bash
kubectl apply -f deploy/k8s/namespace.yaml
kubectl apply -f deploy/k8s/configmap.yaml
kubectl -n scix-mcp create secret generic scix-mcp-secrets \
  --from-literal=MCP_AUTH_TOKEN="$(python3 -c 'import secrets; print(secrets.token_urlsafe(32))')" \
  --from-literal=SCIX_DSN="host=<pg-host> dbname=scix user=scix_reader password=<pw>"
# Edit the image field in deployment.yaml to your registry tag first.
kubectl apply -f deploy/k8s/deployment.yaml -f deploy/k8s/service.yaml
# Skip if internal-only; otherwise edit the hostname first:
kubectl apply -f deploy/k8s/ingress.example.yaml
```

**Acceptance**:
```bash
kubectl -n scix-mcp rollout status deploy/scix-mcp
kubectl -n scix-mcp port-forward svc/scix-mcp 8000:80 &
curl -sf http://127.0.0.1:8000/health
# {"status":"ok"}
```

---

## Phase 9 — Auth (integrate with ADS's auth plane)

The shipped MCP uses a shared `MCP_AUTH_TOKEN`. For production, swap
to ADS-token-via-gateway; see `ADS_INTEGRATION.md` §Auth for three
patterns. The minimal change:

1. Put a Traefik ForwardAuth middleware (or `api_gateway`/`adsws`
   ingress) in front of the MCP Service.
2. Have that middleware validate `Authorization: Bearer <ADS token>`
   against your existing token-validation endpoint.
3. Set `MCP_NO_AUTH=1` on the Deployment ConfigMap so the MCP trusts
   its upstream. **Only do this if the gateway is not bypassable.**

---

## Phase 10 — Parity verification

Run these queries against your DB and confirm numbers are within a few
percent of the pilot. Some drift is expected (corpus is live).

```sql
-- Core corpus
SELECT count(*)                           FROM papers;               -- ~32.4M
SELECT count(body)                        FROM papers;               -- ~14.9M (if Phase 3)
SELECT count(*)                           FROM citation_edges;       -- ~299.3M

-- Embeddings
SELECT model_name, count(*) FROM paper_embeddings GROUP BY 1;
-- indus | ~32.4M
SELECT count(*) FROM paper_embeddings WHERE embedding_hv IS NOT NULL;
-- ~32.4M (halfvec backfill)

-- Graph metrics
SELECT count(*) FROM paper_metrics;                                   -- ~32M
SELECT count(DISTINCT community_medium) FROM papers;                  -- ~10k-100k

-- Entities (if Phase 7 run)
SELECT count(*) FROM entities;                                        -- ~1.58M
SELECT count(*) FROM document_entities;                               -- ~28.4M
SELECT count(DISTINCT bibcode) FROM document_entities;                -- ~12.4M
```

MCP functional smoke test (with a client or curl):
```bash
TOKEN=<your token>
for tool in search concept_search get_paper read_paper \
            citation_graph citation_similarity citation_chain \
            entity entity_context graph_context \
            find_gaps temporal_evolution facet_counts; do
  curl -sS -H "Authorization: Bearer $TOKEN" \
    -H "Content-Type: application/json" \
    -X POST https://<host>/mcp/ \
    -d "{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"tools/call\",
         \"params\":{\"name\":\"$tool\",\"arguments\":{}}}" \
    | jq -r ".result.content[0].text // .error.message" | head -1
done
```

All 13 tools should return something other than "tool not found". Some
will error on empty arguments — that's expected. The goal is to confirm
the server is reachable, authed, and all tool handlers are registered.

---

## Realistic calendar

| Week | What gets done |
|---|---|
| **Week 1** | Phases 0–5 done in parallel where possible. Core MCP serving a full-corpus retrieval + graph stack, with INDUS dense retrieval live. Entity tools return empty. |
| **Week 2** | Phase 7 runs in background (harvesters day 1, linking days 2–5). No MCP redeploy needed when data lands — tools start returning results. |
| **Week 3** | Phase 9 — auth integration with `api_gateway`, security review, production traffic cutover. |
| **Week 4+** | Optional: implement the Kafka consumer from `ADS_INTEGRATION.md` §Keeping the MCP's DB in sync so the corpus stays fresh instead of depending on nightly JSONL re-ingest. |

## When you get stuck

- Migration errors: `migrations/` is idempotent — re-running is safe.
  If one fails, capture the full `psql` error + migration number and
  open an issue against the shared repo.
- Ingest stalls: resumable. Check `SELECT * FROM ingest_log ORDER BY
  started_at DESC LIMIT 10;` — in-progress files resume from where
  they left off.
- Embedding OOMs: lower `--batch-size` on `scripts/embed_optimized.py`.
  Or split the work with bibcode prefix flags.
- Tier 2 linker too slow: see runbook for `--bibcode-prefix` sharding
  and `--max-per-entity` to cap common-word flooding.
- MCP `entity_*` tools return empty: confirm `document_entities` has
  rows. The MCP works cleanly with zero entities — it just doesn't add
  value until Phase 7 completes.

## What's deliberately out of scope (for both of us)

- Tier 3 LLM extraction at corpus scale — infrastructure is in place,
  neither side has run it. Requires an LLM budget decision.
- Kafka consumer for live corpus sync — sketched in
  `ADS_INTEGRATION.md`; v0.2.0 work.
- `pgvectorscale` migration — docs/ops guide exists if you later
  outgrow pgvector HNSW. Not required for parity.
