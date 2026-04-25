# Plan — `section_retrieval` MCP tool

## Goal

Add a new MCP tool `section_retrieval(query, k, filters)` that retrieves
section-grain results by fusing dense HNSW search over `section_embeddings`
with BM25 over `papers_fulltext.sections_tsv` via Reciprocal Rank Fusion
(k=60). Return per-section snippets with canonical URL.

## Edits

### `src/scix/mcp_server.py`

1. Add `section_retrieval` to `EXPECTED_TOOLS` (after `find_replications`),
   bringing the tuple length from 15 to 16.
2. Add `Tool(name="section_retrieval", ...)` to the `list_tools()` body
   right after the `find_replications` Tool. Schema:
   - `query: str` (required)
   - `k: int` (default 10)
   - `filters: object` with discipline, year_min, year_max, bibcode_prefix.
3. Route `name == "section_retrieval"` in `_dispatch_consolidated` to a
   new `_handle_section_retrieval(conn, args)`.
4. Implement `_handle_section_retrieval`:
   1. Validate / coerce `query`, `k`, `filters` (year coercion via
      `_coerce_year`).
   2. Encode query: lazy-load nomic via
      `scix.embeddings.section_pipeline._load_model`, prefix
      `"search_query: "`, encode with `truncate_dim=1024` and
      `normalize_embeddings=True`. Wrap the `sentence_transformers`
      `ImportError` and return a structured error.
   3. Run dense path inside an explicit transaction with
      `SET LOCAL hnsw.iterative_scan = 'relaxed'` and
      `SET LOCAL hnsw.ef_search = 100`. SELECT
      `(bibcode, section_index, section_heading, embedding <=> %s::halfvec)`
      from `section_embeddings` joined to `papers` for filter columns,
      ORDER BY distance, LIMIT a fanout (10*k or 50, whichever is greater).
   4. Run BM25 path: SELECT bibcode + each section JSON entry's heading +
      text by indexing into `papers_fulltext.sections`. Use
      `papers_fulltext.sections_tsv @@ plainto_tsquery('english', %s)` and
      `ts_rank(sections_tsv, plainto_tsquery(...))` for ordering. Apply
      filters via JOIN to `papers`.
      - For each matching paper, ts_rank ranks the *paper*, not the
        *section*. We unnest the JSONB array to find sections whose text
        matches `plainto_tsquery` and produce one (bibcode, section_index)
        row per match. Limit to fanout.
   5. Fuse via `_rrf_fuse([dense_keys, bm25_keys], k_rrf=60)` to produce a
      ranked list of `(bibcode, section_index)` keys with fused scores.
   6. For the top `k` keys, build response items: bibcode, section_heading,
      snippet (truncate text to 500), score, canonical_url (resolved via
      arXiv lookup; None if not derivable).
5. Implement `_rrf_fuse(ranked_lists, k_rrf=60)`: pure helper, takes a
   list of ranked iterables (best first) and returns a list of
   `(key, score)` sorted by descending score.
6. Implement `_truncate_snippet(text, max_chars=500)`: pure helper.
7. Update `_smoke_call_new_tools` to also call `section_retrieval` if the
   new tool reports it is reachable; tolerate failures the same way the
   existing claim_blame/find_replications calls do.

### `tests/test_mcp_section_retrieval.py` (new)

CPU-only, no real DB. Tests:

- `test_rrf_fuse_basic` — two ranked lists with overlap; assert correct
  RRF score ordering.
- `test_rrf_fuse_disjoint` — no overlap; both items appear with their
  individual contributions.
- `test_rrf_fuse_k_constant` — verify the `k` constant is 60 by default
  (call without override and compare against expected 1/(60+1) value).
- `test_truncate_snippet_under_limit` — text shorter than 500 returns
  unchanged.
- `test_truncate_snippet_over_limit` — text longer than 500 returns ≤500.
- `test_section_retrieval_registered` — call `startup_self_test` (with
  `_init_model_impl` patched) and assert `section_retrieval` appears in
  the tool names with a schema declaring required `query`.
- `test_section_retrieval_filters_apply` — patch `_load_model`,
  `encode_batch`, mock `conn.cursor` execute calls, dispatch with
  filters, and assert the SQL (or params) carry the filter values.
- `test_section_retrieval_returns_required_keys` — patch the encoder and
  return synthetic rows for both dense and BM25 paths; assert each
  returned item has exactly `{bibcode, section_heading, snippet, score,
  canonical_url}`.
- `test_section_retrieval_no_paid_api_imports` — open the file and grep
  for forbidden import strings.

### `tests/test_mcp_smoke.py`

- Bump `assert len(EXPECTED_TOOLS) == 15` → 16.
- Bump tool_count assertions (15 → 16) where they appear.
- Add a `test_section_retrieval` smoke test (mock `_load_model` and
  cursor execute to return empty).

## Risks / open issues

1. **arxiv_id lookup cost.** Calling `papers.identifier` lookup per
   result is N round-trips. We will batch via a single `WHERE bibcode =
   ANY(%s)` query at the end.
2. **BM25 row dedup.** A single paper may match multiple sections; we
   need to dedup `(bibcode, section_index)` pairs before fusion.
3. **Empty results.** If both dense and BM25 are empty (e.g. no
   matching documents), return `{"results": [], "total": 0}` —
   matches the convention of other tools.
4. **Section heading & text source.** Both dense and BM25 paths must
   read the heading and text consistently. We will join to
   `papers_fulltext` once at the end to fetch the JSONB section by
   index, rather than carrying text through fusion.

## Acceptance check

After implementation:

```
python -m pytest tests/test_mcp_section_retrieval.py tests/test_mcp_smoke.py -v
```

Both must exit 0.

```
awk '/async def list_tools/,/@server.call_tool/' src/scix/mcp_server.py \
    | grep -oE 'name="[a-z_]+"' | sort -u | wc -l
```

Must print 17.
