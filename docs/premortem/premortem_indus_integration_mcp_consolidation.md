# Premortem: INDUS Integration & MCP Tool Consolidation

## Risk Registry

| #   | Failure Lens             | Severity     | Likelihood | Risk Score | Root Cause                                                                                    | Top Mitigation                                                                           |
| --- | ------------------------ | ------------ | ---------- | ---------- | --------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------- |
| 1   | Technical Architecture   | Critical (4) | High (3)   | **12**     | No pre-filter cardinality check — iterative scan blows up on selective filters at 32M scale   | Add cardinality estimation before HNSW scan; filter-first fallback for <1% selectivity   |
| 2   | Operational              | Critical (4) | High (3)   | **12**     | No automated matview refresh + no staleness monitoring = silent data degradation              | Deploy cron-based refresh after daily_sync; add matview freshness probe with alerting    |
| 3   | Scale & Evolution        | Critical (4) | High (3)   | **12**     | Architecture assumes fixed 32M paper-level vectors; full-text chunks → 10x growth breaks HNSW | Separate chunk_embeddings table; vector count threshold before index rebuild             |
| 4   | Integration & Dependency | Critical (4) | High (3)   | **12**     | No MCP SDK version pin + no UAT URL freshness check = silent total breakage of 2/13 tools     | Pin mcp SDK `>=1.2,<2.0`; vendor UAT SKOS or pin to release tag; add all-tool smoke test |
| 5   | Scope & Requirements     | High (3)     | Medium (2) | **6**      | 50-query eval tests single-turn retrieval, not multi-step agent workflows                     | Add multi-tool workflow scenarios to eval; shadow-mode test consolidated tools           |

## Cross-Cutting Themes

### Theme 1: Materialized View as Single Point of Failure (Technical + Operational + Scale)

Three independent lenses flagged `agent_document_context` matview:

- **Technical**: Refresh takes 45+ min at 200M document_entities, blocking queries
- **Operational**: No automated refresh exists; manual refresh during peak hours caused cascading outage
- **Scale**: Refresh time grows superlinearly with entity count; at 120M rows it locks the system for an hour daily

**Combined severity**: Critical. This is the highest-confidence risk — three independent agents identified it as a failure vector through different causal chains. The matview is load-bearing (`get_paper(include_entities=true)` is the default path) with no operational guardrails.

### Theme 2: Connection Pool Exhaustion Under Concurrent Load (Technical + Operational + Scale)

Three lenses identified the 10-connection pool as a bottleneck:

- **Technical**: 3 sequential HNSW scans per hybrid search + 30s timeout = pool saturation at 4 concurrent users
- **Operational**: Matview refresh holds connections, starving all other tools
- **Scale**: Multi-user access impossible at current pool sizing

**Combined severity**: High. Pool exhaustion cascades into total MCP server unresponsiveness — all tools fail, not just the slow one.

### Theme 3: SessionState Fragility (Technical + Operational + Scale)

Three lenses flagged the in-memory SessionState singleton:

- **Technical**: No eviction policy, O(n) annotation overhead as sessions grow
- **Operational**: Lost on process restart with no persistence or warning
- **Scale**: Not shared across instances, blocking horizontal scaling

**Combined severity**: Medium. Session state loss is disruptive but not data-losing. The `find_gaps` degradation from noisy implicit tracking (Scope lens) compounds this.

### Theme 4: Eval Gap — Single-Turn vs Multi-Step (Scope only, but high impact)

Only the Scope lens flagged this, but it's the most insidious risk:

- The 50-query eval validates retrieval quality (nDCG@10) but never tests multi-tool agent workflows
- Consolidation could pass the eval while degrading real agent task completion rates
- Implicit session tracking ("seen" vs "focused") is a server-side heuristic about agent intent — potentially violates ZFC principle

**Severity**: High. The eval is the gate for Phase 1. If the gate doesn't catch workflow regressions, they ship silently.

## Mitigation Priority List

| Priority | Mitigation                                                                                            | Failure Modes Addressed       | Effort | Impact                                                            |
| -------- | ----------------------------------------------------------------------------------------------------- | ----------------------------- | ------ | ----------------------------------------------------------------- |
| **P0**   | Replace matview with live query + short-TTL app cache for single-bibcode lookups                      | Technical, Operational, Scale | Medium | Eliminates matview refresh as operational burden                  |
| **P0**   | Add pre-filter cardinality estimation in hybrid_search(); filter-first fallback for selective queries | Technical                     | Medium | Prevents 2-8s query latency on filtered searches                  |
| **P0**   | Deploy automated matview refresh (if matview kept) in daily_sync.sh with staleness monitoring         | Operational                   | Low    | Prevents silent data staleness                                    |
| **P1**   | Add multi-step workflow scenarios to 50-query eval                                                    | Scope                         | Medium | Catches workflow regressions that single-turn eval misses         |
| **P1**   | Increase connection pool to 20; add pool isolation (fast vs slow tools)                               | Technical, Operational, Scale | Low    | Prevents pool exhaustion cascading across tools                   |
| **P1**   | Add startup readiness gate (model loaded before accepting requests)                                   | Operational                   | Low    | Prevents errors during cold start                                 |
| **P1**   | Default `include_entities=false` on get_paper; agents opt-in to entity payloads                       | Scope                         | Low    | Prevents context window bloat; cleaner implicit session signal    |
| **P2**   | Separate chunk_embeddings table for future full-text pipeline                                         | Scale                         | Medium | Preserves 32M-row HNSW index when chunks arrive                   |
| **P2**   | Extract INDUS model to embedding microservice (TEI/FastAPI)                                           | Scale                         | High   | Decouples model memory from server scaling                        |
| **P2**   | Add SessionState persistence (Redis or pg table)                                                      | Operational, Scale            | Medium | Survives restarts; enables horizontal scaling                     |
| **P2**   | Replace OpenAI LRU(512) with shared cache (Redis) + cost ceiling alert                                | Scale                         | Medium | Prevents cache thrashing and cost explosion under multi-user load |
| **P3**   | Partition paper_embeddings by year for HNSW on 5M-row partitions                                      | Technical, Scale              | High   | Reduces filtered scan expansion; enables incremental index builds |
| **P3**   | Shadow-mode testing: serve old + new tool surfaces in parallel, compare workflow completion           | Scope                         | High   | Gold standard validation but expensive to implement               |

## Design Modification Recommendations

### 1. Replace matview default path with live query + cache (addresses 3 failure lenses)

**Change**: Instead of routing `get_paper(include_entities=true)` through `agent_document_context` matview, run the live JOIN query with a 60-second application-level LRU cache keyed by bibcode. The JOIN is indexed (bibcode PK + entity FK) and should be <5ms for single-bibcode lookups. Reserve the matview for batch analytics only.

**Why**: The matview is the single biggest operational risk (no refresh automation, 45-min refresh at scale, cascading pool exhaustion). A live query with caching eliminates the refresh problem entirely.

**Effort**: Medium — change the query path in `search.py:get_document_context()`, add app-level cache, keep matview for optional batch use.

### 2. Add cardinality-aware query routing in hybrid_search() (addresses Technical)

**Change**: Before running `vector_search()` with filters, estimate filter selectivity via `EXPLAIN` output or a fast `count(*)` on the filter predicate. If selectivity <1% of corpus, switch to filter-first: run the filter as a CTE returning matching bibcodes, then brute-force cosine similarity on the filtered subset (bypassing HNSW). Add a latency circuit breaker: if any individual HNSW scan exceeds 500ms, abort and fall back to BM25-only results with a metadata flag.

**Why**: Filtered iterative scan on 32M vectors is the primary latency risk. Real agent workloads use progressively narrower filters.

**Effort**: Medium — add ~30 lines to `hybrid_search()` for cardinality check and fallback path.

### 3. Default include_entities=false on get_paper (addresses Scope)

**Change**: Flip the default so agents get lightweight metadata by default and explicitly request entity context when needed. This also makes the implicit session tracking cleaner — only `get_paper(include_entities=true)` signals "focused" intent.

**Why**: The Scope narrative identified a cascade: large default payloads → fewer inspect calls → sparser session state → degraded find_gaps. Defaulting to lightweight responses breaks this chain.

**Effort**: Low — one parameter default change.

### 4. Add multi-step workflow eval scenarios (addresses Scope)

**Change**: Extend the 50-query eval to include 10 multi-tool workflow scenarios: discover → inspect → explore → synthesize chains with 3+ tool calls each. Measure task completion rate and tool selection accuracy, not just nDCG@10.

**Why**: The consolidation's eval gate (Phase 1) only catches single-turn retrieval regressions. Multi-step workflow degradation — the actual risk — passes undetected.

**Effort**: Medium — design scenarios, implement agent simulation harness.

### 5. Increase pool + add pool isolation (addresses 3 failure lenses)

**Change**: Increase `SCIX_POOL_MAX` to 20. Create two logical pools: a "fast" pool (8 connections) for sub-100ms tools (get_paper, facet_counts, entity_context) and a "slow" pool (12 connections) for search, citation_chain, and other potentially expensive queries. Add pool wait-queue monitoring.

**Why**: At max_size=10, 4 concurrent hybrid searches (each holding a connection for 2-3 sequential HNSW scans) saturate the pool and block all tools.

**Effort**: Low — pool configuration change + minor routing logic in `_get_conn()`.

## Full Failure Narratives

### 1. Technical Architecture Failure (Critical/High, Score: 12)

The consolidated `search` tool routed all queries through `hybrid_search()` with iterative scan auto-enabled for filtered queries. Selective filters (<0.5% of 32M rows) caused HNSW graph expansion into sequential-scan territory (2-8s per query). Connection pool (max 10) saturated within 3 concurrent sessions. The `agent_document_context` matview with 200M+ document_entities rows took 45+ min to refresh, blocking all queries. SessionState singleton had no eviction, causing O(n) annotation overhead.

Root cause: No pre-filter cardinality check; iterative scan unconditionally enabled for any filter on 32M rows.

### 2. Operational Failure (Critical/High, Score: 12)

No automated matview refresh was deployed. After 3 weeks, `get_paper(include_entities=true)` returned stale/missing data for new papers. Manual refresh during peak hours held EXCLUSIVE lock for 22 minutes, exhausting the connection pool. Server kill lost all SessionState and OpenAI LRU cache. Cold restart took 45s for INDUS model load, during which search returned errors. Naive cron fix without CONCURRENTLY caused 4x35-min daily outages.

Root cause: No automated off-peak matview refresh and no staleness monitoring.

### 3. Scope & Requirements Failure (High/Medium, Score: 6)

Consolidation passed 50-query eval but agent task completion dropped 18%. Agents didn't explore non-default search modes after hybrid returned poor results. Implicit session tracking ("seen" set) was too noisy for find_gaps. get_paper(include_entities=true) default produced 3-5x larger payloads, causing agents to call it less frequently, starving implicit session state. Agents ended up using only 5 of 13 tools.

Root cause: Eval tested single-turn retrieval, not multi-step workflows.

### 4. Scale & Evolution Failure (Critical/High, Score: 12)

Full-text chunking pipeline produced 320M chunk embeddings in same `paper_embeddings` table. HNSW rebuild on 352M vectors ran for 9 days before running out of disk. Vector searches fell back to sequential scan (12s p95). Matview refresh at 120M document_entities took 45+ min daily. OpenAI LRU(512) cache hit rate dropped below 5% with multiple concurrent users, spiking costs to $2K/month. Single-process server couldn't scale horizontally (SessionState not shared, GPU memory exhausted at 4 instances).

Root cause: Architecture assumed fixed 32M paper-level vectors with single-user access.

### 5. Integration & Dependency Failure (Critical/High, Score: 12)

Unpinned `mcp` SDK (`mcp>=1.0`) pulled breaking 2.0 release — `inputSchema` renamed to `input_schema`, server failed to start. No health check endpoint (eliminated in consolidation) meant 36-hour silent outage. Separately, UAT SKOS vocabulary URL hardcoded to GitHub `master` branch; astrothesaurus org renamed to `main`, returning 404. `concept_search` fell back to stale April snapshot, missing 47 new concepts. 50-query eval didn't cover concept_search, so regression went undetected. Two of 13 tools broken simultaneously during ADASS reviewer testing.

Root cause: No SDK version pin, no UAT URL freshness check, no all-tool smoke test on deployment.

---

_Generated by /premortem as part of the /research-project pipeline. Input: prd_indus_integration_mcp_consolidation.md_
