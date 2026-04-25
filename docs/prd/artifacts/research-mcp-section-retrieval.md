# Research — `section_retrieval` MCP tool

Bead: prd-build/section-embeddings-mcp-consolidation work unit
**mcp-section-retrieval**.

## Where things live

| Concern | File / Symbol |
| --- | --- |
| MCP server entrypoint | `src/scix/mcp_server.py` |
| `list_tools()` handler | `src/scix/mcp_server.py:1038` |
| Dispatch | `_dispatch_consolidated()` at `src/scix/mcp_server.py:1946` |
| Tool template (similar shape) | `_handle_find_gaps`, `_handle_claim_blame` |
| Filters schema | `_FILTERS_SCHEMA` at `src/scix/mcp_server.py:765` |
| Tool registration enumeration | `EXPECTED_TOOLS` tuple at `src/scix/mcp_server.py:801` |
| Smoke harness for new tools | `_smoke_call_new_tools()` at `src/scix/mcp_server.py:978` |
| Section embedding model loader | `src/scix/embeddings/section_pipeline.py:_load_model` |
| Local nomic prefix for queries | `"search_query: "` (NOMIC_DOC_PREFIX is for documents) |
| Section embeddings table | `migrations/061_section_embeddings.sql` — `section_embeddings(bibcode, section_index, section_heading, section_text_sha256, embedding halfvec(1024))` |
| BM25 index | `papers_fulltext.sections_tsv` (GENERATED tsvector, GIN index `idx_papers_fulltext_sections_tsv`) |
| Section text | `papers_fulltext.sections -> section_index ->> 'text'` (JSONB array) |
| Canonical URL builder | `scix.sources.ar5iv._build_canonical_url(arxiv_id)` returns `https://arxiv.org/abs/{arxiv_id}` |
| arXiv ID lookup | `scix.search` (around line 3197) — looks up first matching identifier from `papers.identifier[]` matching `_ARXIV_ID_RE` |

## Current tool count

Verified at this branch HEAD: 16 tools registered in `list_tools()`. Adding
`section_retrieval` makes 17. `EXPECTED_TOOLS` currently has 15 entries
(13 baseline + claim_blame + find_replications); the 16th visible tool is
`find_similar_by_examples`, which is *optional* and only registered when
`QDRANT_URL` is set. After adding `section_retrieval`, `EXPECTED_TOOLS`
must have 16 entries.

## Schema summary

```sql
-- 061_section_embeddings.sql
CREATE TABLE section_embeddings (
    bibcode             TEXT NOT NULL REFERENCES papers(bibcode),
    section_index       INT  NOT NULL,
    section_heading     TEXT,
    section_text_sha256 TEXT NOT NULL,
    embedding           halfvec(1024) NOT NULL,
    PRIMARY KEY (bibcode, section_index)
);
CREATE INDEX idx_section_embeddings_hnsw
    ON section_embeddings USING hnsw (embedding halfvec_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- generated tsvector on papers_fulltext for BM25
ALTER TABLE papers_fulltext ADD COLUMN sections_tsv tsvector
    GENERATED ALWAYS AS (
        to_tsvector(
            'english',
            coalesce(jsonb_path_query_array(sections, '$[*].text')::text, '')
            || ' ' ||
            coalesce(jsonb_path_query_array(sections, '$[*].heading')::text, '')
        )
    ) STORED;
CREATE INDEX idx_papers_fulltext_sections_tsv
    ON papers_fulltext USING gin (sections_tsv);
```

## Reciprocal Rank Fusion (RRF)

For each candidate document `d`, RRF aggregates ranks from multiple
retrievers as

```
score(d) = sum over retrievers r of 1 / (k + rank_r(d))
```

with `k = 60` per the original Cormack et al. recipe. `rank_r(d)` is
1-indexed (best = 1). Documents missing from a retriever contribute 0
for that retriever. Higher score is better.

For our tool, "documents" are `(bibcode, section_index)` pairs.

## Query encoding for nomic-embed-text-v1.5

Per the section pipeline module docstring and the model card, queries are
prefixed with `"search_query: "` while documents use `"search_document: "`.
The pipeline already exposes `NOMIC_DOC_PREFIX = "search_document: "` and a
private `_load_model` that imports `sentence_transformers` lazily. We will
reuse `_load_model`. We define a sibling `NOMIC_QUERY_PREFIX` locally (or
inline the literal) to avoid coupling the consumers.

## Non-goals / things NOT to add

- No paid-API SDK imports (openai / anthropic / cohere / voyageai). Per
  `feedback_no_paid_apis`, the encoder must be local open-weight only.
- No reranker call here — keep RRF as the single fusion step and let the
  agent layer decide whether a downstream rerank is worth it.
- We don't need to expose `section_index` in the response — the spec asks
  for `bibcode, section_heading, snippet, score, canonical_url` only.

## HNSW iterative scan settings

Acceptance criterion 4 mandates `SET LOCAL hnsw.iterative_scan = 'relaxed'`
and `SET LOCAL hnsw.ef_search = 100` *inside the same transaction* as the
SELECT. `SET LOCAL` rolls back on transaction end so we explicitly issue a
`BEGIN ... COMMIT` block (or rely on the implicit transaction the cursor
opens — psycopg auto-commits only when `autocommit=True`, which the pool
does NOT enable for pooled connections). We will wrap dense retrieval in a
manual transaction to be safe.

## Filter contract

Per the spec, `filters` accepts at minimum:

| key | type | source column |
| --- | --- | --- |
| discipline | str | `papers.discipline` |
| year_min | int | `papers.year` |
| year_max | int | `papers.year` |
| bibcode_prefix | str | `bibcode LIKE prefix || '%'` |

These are independent of the existing `_FILTERS_SCHEMA` (which is
search-tool-specific). We define a slimmer `_SECTION_FILTERS_SCHEMA`.

## Test strategy

CPU-only, no live DB, no model:

1. **RRF math** — call `_rrf_fuse` with two synthetic ranked lists and
   verify the standard RRF aggregation against a hand-computed expectation.
2. **Snippet truncation** — feed an oversized text into the snippet builder
   helper and assert `len(snippet) <= 500`.
3. **Filters apply** — mock `conn.cursor()` and assert the produced SQL
   contains the expected filter clauses and parameters when filters are
   present (and does NOT when they're absent).
4. **Tool registration** — call the registered `list_tools()` (via
   `startup_self_test` plumbing) and assert `section_retrieval` appears
   with a valid input schema.
5. **Schema of returned object** — call `_dispatch_tool(conn,
   "section_retrieval", ...)` with a mock that returns one synthetic dense
   row and one BM25 row, assert each result item has exactly the required
   keys.
