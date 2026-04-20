# Integrating the SciX MCP with ADS production

Notes for positioning the SciX MCP HTTP server inside the existing ADS
stack. Companion to [`docs/DEPLOYMENT.md`](./DEPLOYMENT.md), which covers
the mechanical steps.

## Where this fits

The MCP is a **complementary retrieval and graph-context layer**, not a
replacement for any existing ADS service. Concretely:

| Concern | ADS today | Where the MCP fits |
|---|---|---|
| Lexical search over abstracts / titles | MontySolr behind `solr-service` | — (Solr stays authoritative) |
| Per-bibcode metadata lookup | `adsws`, `resolver-service` | — |
| Full-text PDF retrieval | `ADSfulltext` pipeline → proj storage | — |
| Citation metrics, export, biblib | dedicated microservices | — |
| **Dense semantic retrieval** (INDUS, full 32M corpus) | none | MCP `search`, `concept_search` |
| **Citation-graph context** (PPR, multi-hop chains, co-citation) | citation counts only | MCP `citation_chain`, `citation_similarity`, `graph_context` |
| **Agent-friendly navigation** over the corpus | — | The 13-tool MCP surface |

The natural consumer of the MCP is an LLM-based agent (Claude, GPT, etc.)
that wants to reason over the corpus — think of it as the retrieval and
graph-walk counterpart to the UI-facing search endpoints. Classic SciX
clients (bumblebee, the search UI) have no reason to call the MCP.

## Placement in BeeHive

Following the convention in
`ads_devops_docs/source/services+software/services/kubernetes.md`:

- **Microservice, on the AWS cluster**. The MCP is a web service with a
  DB dependency — exactly what the microservices tier on
  `v126.kube.adslabs.org` is for. It does **not** belong on the
  bare-metal backoffice hosts with the ingest pipelines (those are
  batch workloads; the MCP is request/response).
- Suggested path in BeeHive:
  `kubectl/v126.kube.adslabs.org/ads-prod/microservices/scix-mcp/`
  with `deployment.yaml`, `service.yaml`, `ingress.yaml`, `configmap.yaml`.
  Copy from `deploy/k8s/` in this repo and rewrite the `image:` field
  to the ECR tag you published.
- Image tag convention: `tailor:scix-mcp-v0.1.0` (matches
  `tailor:solr-service-v1.1.0`, `tailor:backoffice-brain-v0.0.1`, etc.).
- Operator of record (to negotiate): Taylor Jacovich, per the pattern
  in the existing devops docs.

## Dependencies on existing ADS infrastructure

| Dependency | Shared with | Notes |
|---|---|---|
| PostgreSQL | Most modern workflows already use Postgres via BeeHive. The MCP needs its **own** database (`scix`) — do not colocate tables with `master_pipeline` / `fulltext_pipeline` / etc. | 32M papers + embeddings ≈ 600 GB. Halfvec cuts embedding column size in half vs float32. |
| pgvector 0.8.2+ | New extension on the SciX side | Iterative scan, parallel HNSW builds, halfvec all required — migrations assume 0.8.x. |
| pg_search (ParadeDB) | Optional | Migration 004 is a no-op without it; falls back to `ts_rank_cd`. |
| Kafka / harvester topic | `HarvesterOutput` (Avro) | Not used in v0.1.0. See §Live ingest below. |
| Object storage (CephFS / S3) | — | MCP does not need it. Full-text bodies live in Postgres (`papers.body`, GIN-indexed). |
| Graylog / Fluentd / Prometheus | Yes | Pod stdout → existing log pipeline. Add a scrape target later when we expose `/metrics`. |
| GPU nodes | DCGM dashboard | Optional — only needed if you move INDUS embedding from CPU to CUDA. |

## Keeping the MCP's DB in sync with the rest of SciX

The pilot populates `scix.papers` from a JSONL snapshot via
`scripts/ingest.py`. For production, this becomes the weak link: the
Solr index and the MCP DB drift the moment a new bibcode arrives.

Three options, in increasing order of integration:

**1. Scheduled re-ingest (v0.1.0, acceptable for pilot)**
Run `scripts/ingest.py` nightly against a freshly-exported JSONL dump.
The ingest is resumable and skips unchanged files. Latency from
publication → MCP visibility ≈ 1 day.

**2. Kafka consumer on `HarvesterOutput` (v0.2.0, recommended)**
Add a `scix-mcp-ingest` Kafka consumer that reads the same output topic
the master / ingest pipelines consume, writes to `scix.papers` +
related tables, and enqueues newly-ingested bibcodes for the embedding
pipeline. Latency ≈ minutes. Design sketch:

```
HarvesterOutput (Avro) ──► scix-mcp-ingest ──► scix.papers (COPY)
                                            └► scix.embed_queue (LISTEN/NOTIFY)
                                                           │
                         scix-mcp-embed-worker ◄───────────┘
                         (pulls from queue, writes paper_embeddings)
```

This reuses the existing Avro schema and Schema Registry — no new
contract surface. The embed worker can run on a GPU node pool (1 pod
is plenty for incremental traffic; the bulk backfill is a separate
offline job).

**3. Dual-write from ingest pipeline**
Modify the existing master/ingest pipeline to write to `scix.papers`
in parallel with its current destinations. This is the tightest
coupling but hardest to deploy — don't do this unless option 2 proves
insufficient.

## Security posture

### Auth

The MCP ships with a single shared bearer token (`MCP_AUTH_TOKEN`)
checked on every request and rate-limited per token in-memory. That's
the right primitive for the pilot (sjarmak's workstation, internal
testing) but almost certainly not how ADS wants this deployed. A few
things we know about your stack that are relevant:

- ADS already issues per-user API tokens at
  `https://ui.adsabs.harvard.edu/user/settings/token`, and the
  `api_gateway` / `adsws` microservices validate them for every
  existing workflow.
- Traefik — the ingress controller on `v126.kube.adslabs.org` — supports
  ForwardAuth middleware, so an inbound `Authorization: Bearer <ADS
  token>` can be validated upstream before the request ever reaches a
  backend pod.
- The MCP's auth check is a single ASGI middleware
  (`src/scix/mcp_server_http.py::_AuthWrap`) — swapping it, bypassing
  it, or running with `MCP_NO_AUTH=1` behind a trusted ingress are all
  supported.

Plausible integration patterns (pick whichever matches your auth
plane — we don't have a strong opinion):

1. **Gateway-validated ADS tokens.** Put the MCP behind `api_gateway`
   or a Traefik ForwardAuth middleware that validates the inbound ADS
   token against `adsws`, then forwards to the MCP with
   `MCP_NO_AUTH=1`. Clients pass their existing ADS token in the
   `Authorization` header; you get per-user attribution and revocation
   for free. This is the lowest-code path and the one that makes the
   MCP behave like every other SciX microservice.
2. **In-process ADS-token validation.** Add a middleware to the MCP
   that calls an ADS token-validation endpoint per request with a
   short TTL cache. More code on our side, but keeps the MCP
   deployable without the gateway (useful for labs, partners,
   or staging clusters). Happy to add this behind a feature flag if
   option 1 ends up not fitting — it's a small, isolated change.
3. **Shared bearer token.** The current `MCP_AUTH_TOKEN`. Fine for
   internal-only clusters or short-lived sandboxes where per-user
   auditing isn't a requirement.

All three coexist with Cloudflare Access / Okta / any SSO layer in
front of the ingress if you want interactive human auth on top of
programmatic token auth.

### DB credentials

The MCP uses the `scix_reader` role with `SELECT`-only grants. There
is no code path that writes to Postgres at request time; writes only
happen from operator-run ingest / embed jobs. Whatever you set up for
rotating Postgres creds in BeeHive (sealed secrets, vault, whatever
the pattern is) can apply here unchanged.

### Network

The pod drops all Linux caps, runs as uid 1001 with a read-only root
FS. Egress is only needed for INDUS model download on first start
(with `HF_HUB_OFFLINE=0`) — seed the cache volume and you can lock
egress down entirely.

## Handoff summary for ops

- Code + manifests live in a shared GitHub repo; ADS ops pulls tagged
  releases (`v0.1.0`, `v0.2.0`, …) — see `docs/UPGRADING.md` once it
  lands.
- The only two files that need ADS-specific edits are:
  - `deploy/k8s/deployment.yaml` → change `image:` to the ECR tag
  - `deploy/k8s/ingress.example.yaml` → set hostname + TLS secret
- Everything else (ConfigMap keys, env names, Secret shape) is designed
  to match the conventions already in `SciXHarvesterPipeline` and
  `SciXDevelopmentEnvironment`, so the manifests drop into BeeHive
  without re-plumbing.

## Open questions for the ADS team

1. Which ECR / Honeycomb repo should host the `scix-mcp` image? (The
   harvester pipeline uses
   `084981688622.dkr.ecr.us-east-1.amazonaws.com/honeycomb`.)
2. Are there existing TLS-terminating ingresses on
   `v126.kube.adslabs.org` that the MCP should sit behind, or does this
   get its own hostname + cert?
3. Is there an appetite for adding a `scix-mcp-ingest` Kafka consumer
   to the modern workflow (option 2 above)? That's the unlock for
   near-real-time corpus sync.
4. Which Postgres is this pointed at — new dedicated instance, or a
   database inside an existing server? 600 GB + heavy HNSW index
   builds argue for dedicated.
5. Which of the auth patterns above (gateway-validated ADS token,
   in-MCP ADS-token validation, shared bearer) should this target?
   If it's the gateway pattern, is there an existing
   `api_gateway` / `adsws` endpoint we should point a ForwardAuth
   middleware at, or should we write against a specific route?
