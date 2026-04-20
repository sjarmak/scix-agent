# Changelog

All notable changes to the SciX MCP server and supporting infrastructure.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
this project uses [Semantic Versioning](https://semver.org/).

Versions tag the container image (`scix-mcp:vX.Y.Z`) and the
corresponding git commit. Database migrations are append-only — the
highest `migrations/NNN_*.sql` applied determines the minimum required
code version (see [`docs/UPGRADING.md`](docs/UPGRADING.md)).

## [Unreleased]

_Nothing yet._

## [0.1.0] — 2026-04-20

First handoff cut for the ADS ops team. Everything below is the state
shipped with this tag.

### Server

- **MCP HTTP server** (`src/scix/mcp_server_http.py`) with streamable
  HTTP transport, bearer-token auth, in-memory token-bucket rate limiter
  (60 req/min sustained, 10 req/s burst), `/health` endpoint, and
  graceful shutdown.
- **MCP stdio server** (`src/scix/mcp_server.py`) for Claude Desktop /
  Claude Code clients that prefer a local subprocess.
- **13 tools** after the 30→13 consolidation pass:
  `search`, `concept_search`, `get_paper`, `read_paper`,
  `citation_graph`, `citation_similarity`, `citation_chain`,
  `entity`, `entity_context`, `graph_context`,
  `find_gaps`, `temporal_evolution`, `facet_counts`.
  Per-tool timeouts are tunable via `SCIX_TIMEOUT_*` env vars.

### Retrieval

- **Hybrid search** combining dense (pgvector HNSW), lexical
  (`ts_rank_cd`, with optional `pg_search` BM25), and structural signals
  via Reciprocal Rank Fusion (k=60).
- **INDUS embeddings** (`nasa-impact/nasa-smd-ibm-st-v2`, 768d) as the
  primary dense signal for the 32M-paper corpus.
- **halfvec shadow column** on `paper_embeddings` for float16 storage —
  ~half the on-disk footprint of float32, with no measurable nDCG@10
  loss on scientific retrieval (binary quantization remains unsupported
  because it drops >40% nDCG).
- **Iterative scan + parallel HNSW builds** enabled via pgvector 0.8.2.

### Graph

- **Citation graph** (299M edges) with PageRank, HITS, and Leiden
  community detection at multiple CPM resolutions (0.001 / 0.01 / 0.1).
- **Entity graph** with harvester integration (UAT, ROR, Wikidata,
  NASA SPDF/SPASE, OpenAlex), staging→promote pipeline, evidence-backed
  relationship storage.

### Schema

- **54 migrations** (`migrations/001_initial_schema.sql` through
  `migrations/054_paper_embeddings_halfvec_index.sql`) — apply in order
  on an empty Postgres 16 + pgvector 0.8.2 database.
- `pg_search` (ParadeDB BM25) extension is **optional**; migration 004
  is a no-op when absent.
- All tables are `LOGGED` — migration 023 fixed an earlier regression
  where `UNLOGGED` embeddings tables lost 32M rows on Postgres restart.

### Deployment

- **Hardened container image** (`deploy/Dockerfile`): Python 3.12-slim,
  multi-stage build, runs as uid 1001, `HF_HUB_OFFLINE=1`, read-only
  root FS, drops all Linux caps, `no-new-privileges`.
- **Kubernetes manifests** (`deploy/k8s/`) matching the BeeHive
  microservice convention: Namespace, ConfigMap, Secret (template),
  Deployment, Service, optional Traefik Ingress.
- **Docker Compose variants**:
  - `deploy/docker-compose.yml` — bundles a Cloudflare named tunnel for
    the pilot on sjarmak's workstation.
  - `deploy/compose/backoffice.yaml` — no tunnel; publishes on
    `127.0.0.1:8000` for use behind an existing reverse proxy.
- **Operator documentation**: [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md)
  (schema bootstrap → corpus ingest → embedding → k8s/compose deploy →
  token rotation) and [`docs/ADS_INTEGRATION.md`](docs/ADS_INTEGRATION.md)
  (positioning next to Solr, BeeHive placement, corpus-sync options).

### Security

- Bearer-token auth on every MCP request, constant-time comparison via
  `hmac.compare_digest`.
- Runtime DB role is `scix_reader` (SELECT-only); writes happen in
  operator-run ingest/embed jobs, never from request traffic.
- Secrets (`MCP_AUTH_TOKEN`, `SCIX_DSN`, `ADS_API_KEY`, Cloudflare
  tokens) sourced from env / Kubernetes Secrets; `.env`,
  `deploy/.env`, and `CLAUDE.local.md` are gitignored; git history is
  clean of committed tokens as of this tag.

### Known gaps (deferred to future versions)

- Corpus sync is manual (nightly JSONL re-ingest). A Kafka consumer on
  the `HarvesterOutput` topic is scoped in `docs/ADS_INTEGRATION.md`
  for v0.2.0.
- `/metrics` Prometheus endpoint is not exposed; add before wiring
  into the existing ADS Grafana dashboards.
- No Helm chart — ADS teams wrap the raw manifests in their own
  GitOps tooling. Add only if multi-tenant or multi-cluster use emerges.
- Horizontal scaling is stateless but the per-pod rate limiter is
  in-memory; add a Redis-backed limiter if the same bearer token will
  be used across replicas.

[Unreleased]: https://github.com/sjarmak/scix-agent/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/sjarmak/scix-agent/releases/tag/v0.1.0
