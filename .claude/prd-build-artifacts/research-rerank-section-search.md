# Research — rerank-section-search (M5)

## Current `search_within_paper` (src/scix/search.py:2779-2888)

- Signature: `search_within_paper(conn, bibcode, query) -> SearchResult`
- Implementation: single SQL pass that uses `ts_headline` over `papers.body` and
  returns one paper dict with `headline` (single concatenated fragment string).
- ADR-006 guard branch: when `papers_fulltext.source` is LaTeX-derived, the
  `headline` is forced through `apply_snippet_budget_if_needed` and
  `canonical_url` is attached.
- Errors paths: paper-not-found → empty result, paper-with-no-body → empty +
  `metadata.has_body=False`.
- All callers use positional (`conn, bibcode, query`) — `mcp_server._handle_read_paper`
  at src/scix/mcp_server.py:2141 and `tests/test_mcp_paper_tools.py::test_search_within_paper_dispatches`.
  New signature must keep these 3 positional args first.

## CrossEncoderReranker (src/scix/search.py:900-983)

- `CrossEncoderReranker(model_name='cross-encoder/ms-marco-MiniLM-L-12-v2')`
- Lazy-loads weights inside `_load()`; first `__call__` triggers load.
- API: `__call__(query: str, papers: list[dict], top_n: int | None = None) -> list[dict]`
  - Uses `(p.get('title') or '') + '. ' + (p.get('abstract_snippet') or '')` for
    the document text.  For section-level rerank we'll build paper-shaped dicts
    where `abstract_snippet` is the section text.
- Already supports MiniLM (default) and BGE-large.

## section_parser.parse_sections (src/scix/section_parser.py)

- `parse_sections(body: str) -> list[tuple[str, int, int, str]]` — returns
  `(section_name, start_char, end_char, text)`.
- Falls back to `[("full", 0, len(body), body)]` when no headers found, or
  `[("full", 0, 0, "")]` for empty bodies.
- Recognizes IMRaD-style headers (introduction/methods/observations/results/...).

## SCIX_RERANK_DEFAULT_MODEL wiring (src/scix/mcp_server.py:580-639)

- `_RERANK_MODEL_ALIASES = {'minilm': 'cross-encoder/ms-marco-MiniLM-L-12-v2', 'bge-large': 'BAAI/bge-reranker-large'}`
- `_resolve_default_reranker_model()` returns the model string or `None`
  (default 'off' or unknown values).
- `_get_default_reranker()` constructs/caches a `CrossEncoderReranker`.
- The 'off' default means: even if `use_rerank=True` is passed, no model
  is loaded unless an operator flips the env. This is the M4-FAIL gate.

## Existing tests we must not regress

- `tests/test_mcp_search.py` — 11 tests around the MCP `search` tool's
  rerank wiring; all mocked (no real models).
- `tests/test_mcp_paper_tools.py::TestSearchWithinPaper` — 3 tests using
  mock cursors. The dispatch test calls `search.search_within_paper(conn,
  bibcode, query)` positionally.
- `tests/test_body_adr006_guard.py::TestSearchWithinPaperADR006Guard`
  — exercises ADR-006 guard. Need to preserve guard behaviour and keep
  existing assertion shape (`paper["headline"]` is the snippet for top-1).

## Plan-relevant design constraints

1. **Backward compat (AC1, AC2)**: keep `headline` populated from the top-1
   reranked section so existing code continues to work; add `sections` list.
2. **No paid APIs**: rerank uses local sentence-transformers (MiniLM).
3. **Default = off** until env flipped — but signature default `use_rerank=True`
   is what the PRD spec asks for. Implementation: when called with
   `use_rerank=True`, attempt to construct/use the default reranker via the
   same env lookup path; if the env says 'off' (default), simply skip the
   rerank step and use BM25 ts_rank ordering.
4. **Top-K candidates from per-section ts_rank**: parse_sections → for each
   section compute `ts_rank` via SQL (one query per paper, fetching body+title
   then doing per-section ranking in Python — cheap because we have the body
   already), pick top-K=20.
5. **ts_headline still used for snippet**: per-section snippet via Python
   slicing OR a second SQL pass with `ts_headline` over each section's text.
   Simpler: in Python build a snippet by finding the first match window of
   ~200 chars around any query token; this avoids a second SQL round-trip per
   section. We will use `ts_headline` once on the whole body to get the
   backward-compat headline blob for top-1 (preserving existing behaviour),
   and Python-side snippets for the 3 reranked sections.

## File targets

- src/scix/search.py:2779 — modify `search_within_paper` only.
- tests/test_search_within_paper_rerank.py — new unit + eval tests.
- tests/fixtures/within_paper_rerank_gold_20.jsonl — 20 synthetic items.
- results/within_paper_rerank_eval.md — eval summary (gitignored, force-add).
