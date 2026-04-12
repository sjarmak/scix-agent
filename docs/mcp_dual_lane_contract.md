# MCP Dual-Lane Entity Enrichment Contract

**Status:** M12 (PRD §M12). Stable.
**Owner:** entity-enrichment working group.
**Related:** M13 resolver single-entry-point rule (`src/scix/resolve_entities.py`),
`scripts/ast_lint_resolver.py`, migrations 021/028/034/037.

## Problem

The entity graph serves two very different workloads:

1. **Structural / graph analytics** — community detection, citation chains,
   co-citation, bibliographic coupling. These tools touch the _whole graph_
   and need a consistent, slowly-changing snapshot. They can tolerate
   minutes of staleness for orders of magnitude better latency.

2. **Query-time retrieval** — hybrid search, BM25, vector search. These
   tools need entity enrichment _per result_, on demand, against fresh
   link state. They cannot afford a global graph scan and must honour
   tight per-request latency budgets.

One lane cannot satisfy both. This document pins the contract.

## The two lanes

### Static lane (pre-materialised)

Reads from `document_entities_canonical` (the materialised view backed by
`document_entities`) and from graph-centric tables (`citation_edges`,
community tables). Refresh cadence is bounded by the fusion MV refresh
job (see migration 033) and the canonical MV refresh job (migration 028).

**Consumers:**

- `citation_chain`
- `co_citation_analysis`
- `bibliographic_coupling`
- `community_detection`

**Latency budget (u03 / M13):** p95 ≤ 300 ms per tool call for graph
traversals up to depth 2, ≤ 1 s for depth 3+. Freshness window: 24 h.

**Access:** read-only via canonical views. Writes to `document_entities`
are forbidden outside `src/scix/resolve_entities.py` — enforced by
`scripts/ast_lint_resolver.py`.

### JIT lane (just-in-time)

Reads from `document_entities_jit_cache` (partitioned per-session cache,
migration 034). Fresh link state is computed on miss by the M13 resolver
service and cached for the query-log session TTL.

**Consumers:**

- `hybrid_search` when called with `enrich_entities=True`
- `vector_search`
- `bm25_search`

**Latency budget (u03 / M13):** p95 ≤ 50 ms cache hit, ≤ 300 ms cache
miss (resolver round-trip), ≤ 600 ms tail (warm-up). Freshness window:
session TTL (default 15 min).

**Access:** reads go through the JIT cache. Writes to
`document_entities_jit_cache` are forbidden outside
`src/scix/resolve_entities.py` — enforced by `scripts/ast_lint_resolver.py`.

## Why this matters

Mixing lanes breaks either latency or freshness guarantees:

- If graph analytics tools queried the JIT cache, they'd churn the cache
  with entities they don't need and evict fresh entries for interactive
  users.
- If retrieval tools queried `document_entities_canonical` directly,
  they'd see snapshots up to 24 h stale — fatal for the "I just added
  this entity, where is it?" agent workflow.
- If either lane wrote `document_entities` directly, the resolver
  service could not enforce ambiguity resolution, tier promotion, or
  audit trails. Hence the single-entry-point rule.

## M13 resolver contract (reference)

All writes to `document_entities` and `document_entities_jit_cache`
must go through `src/scix/resolve_entities.py`. Reads from
`document_entities_canonical` must likewise go through the resolver's
cached accessors — direct reads bypass the ambiguity-resolution layer.

The AST lint (`scripts/ast_lint_resolver.py`) scans every `.py` file
outside `resolve_entities.py` for forbidden SQL patterns and fails the
build if any are found. Legitimate exceptions (migration tooling,
rollback scripts) use the `# noqa: resolver-lint` escape hatch on the
offending line.

### Allowed direct access

- **Reads** from `document_entities` via SELECT/FROM are allowed
  (citation-consistency computation, audit reports, etc.) — only
  writes are restricted.
- Writes to auxiliary feedback/audit tables such as
  `entity_link_disputes` (migration 037, §S5) and `entity_link_audits`
  (migration 035) are allowed without going through the resolver, as
  they do not affect the canonical link state.

## Per-tool decision table

| Tool                                  | Lane   | Budget (p95)             | Freshness   |
| ------------------------------------- | ------ | ------------------------ | ----------- |
| `citation_chain`                      | static | 300 ms (d≤2) / 1 s (d≥3) | 24 h        |
| `co_citation_analysis`                | static | 300 ms                   | 24 h        |
| `bibliographic_coupling`              | static | 300 ms                   | 24 h        |
| `community_detection`                 | static | 1 s                      | 24 h        |
| `hybrid_search(enrich_entities=True)` | JIT    | 50 ms hit / 300 ms miss  | session TTL |
| `vector_search`                       | JIT    | 50 ms hit / 300 ms miss  | session TTL |
| `bm25_search`                         | JIT    | 50 ms hit / 300 ms miss  | session TTL |

## Change management

Adding a tool to either lane requires:

1. A PRD update referencing this document.
2. An AST-lint-clean implementation (reads via canonical view or JIT
   cache, writes via the resolver).
3. Latency budget attestation — benchmark under the target lane with
   a representative workload before merge.
4. Update the per-tool decision table above.
