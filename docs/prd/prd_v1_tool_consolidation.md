# PRD: V1 Tool Consolidation — 17 → 10 MCP tools

## Status (2026-04-26)

**Ready for `/prd-build`.** All upstream dependencies have landed: chunk
embeddings (`chunk_search`), section embeddings (`section_retrieval`), claim
extraction (`read_paper_claims` / `find_claims`), and the per-type NER quality
profile (dbl.3 close). The tool-surface eval that motivates the design has
also shipped (`docs/eval/tool_surface_eval.md`).

This PRD defines the v1 MCP tool surface that ships in the SciX Deep Search
v1 release.

## Goal

Collapse the current **17 tools (+1 optional `chunk_search`) = 18** registered
MCP tools to **exactly 10 top-level tools**, while preserving 100% of the
underlying capability via consolidated dispatchers. Backward compatibility is
preserved through `_DEPRECATED_ALIASES`.

The 10-tool v1 surface:

| # | Tool | Consolidates | Dispatch parameter |
|---|---|---|---|
| 1 | `search` | `search`, `concept_search`, `section_retrieval`, `chunk_search`, `find_claims` | `grain ∈ {paper, concept, section, chunk, claim}` |
| 2 | `paper` | `get_paper`, `read_paper`, `read_paper_claims` | `action ∈ {metadata, read, claims}` |
| 3 | `citation` | `citation_traverse`, `citation_similarity` | `mode ∈ {traverse, similarity}` |
| 4 | `entity` | `entity`, `entity_context` | `action ∈ {lookup, context, papers}` |
| 5 | `claim_blame` | (top-level, no change) | — |
| 6 | `find_replications` | (top-level, no change) | — |
| 7 | `graph_context` | (top-level, no change) | — |
| 8 | `find_gaps` | (top-level, no change) | — |
| 9 | `temporal_evolution` | (top-level, no change) | — |
| 10 | `facet_counts` | (top-level, no change) | — |

## Why this exact shape

The proposed 8-tool surface evaluated in `docs/eval/tool_surface_eval.md`
folded `claim_blame` and `find_replications` into `paper(action=...)`. The
eval (90 runs / 30 queries / Sonnet via OAuth subagent) showed:

| intent | v0 (18 tools) | v1 (8 tools) |
|---|---:|---:|
| `paper_blame` | 100% | **33.3%** |
| `paper_replications` | 100% | **66.7%** |
| `search_section` | 66.7% | 100% |
| `graph_context` | 66.7% | 100% |
| (all others) | 100% | 100% |

Aggregate v1 tool_accuracy was 96.7% (vs 94.4% for v0) — the consolidation
helps in 17/18 intent clusters and only hurts the two cross-paper provenance
clusters. **Cause:** `claim_blame` and `find_replications` are not
operations on a single paper keyed by bibcode — they are graph queries
across many papers. Folding them into `paper(action=...)` mismatches the
agent's intent representation and produces selection misses.

**Decision:** keep the v1 consolidation pattern, promote `claim_blame` and
`find_replications` back to top-level. Net: 8 + 2 = 10 tools. This is below
the 15-tool premortem ceiling (`CLAUDE.md §Tool Count Concern`) with
headroom for a future tool to land without breaking the budget.

The post-promotion design is referred to as **v1′** (or just **v1** for the
shipping release; the 8-tool variant is retired).

## Non-Goals

- Adding new capabilities. Every v1 tool surfaces existing handlers.
- Changing the underlying retrieval / graph / claim-extraction
  implementations. The consolidation is purely a surface re-shape.
- Removing the `chunk_search` Qdrant requirement — it is folded into
  `search(grain=chunk)` which is gated on `_qdrant_enabled()`. If Qdrant is
  not configured, calls with `grain=chunk` return a structured "backend not
  available" error rather than 404. (See WU-1 acceptance.)
- Changing semantics of the 6 unchanged top-level tools.
- Removing the `_DEPRECATED_ALIASES` machinery. All 17 current tool names
  remain callable as deprecated aliases per the existing pattern.

## Deliverables / Work Units

Each WU is independently committable, typically <300 LOC, and has explicit
acceptance criteria for `/prd-build`'s scoring loop.

---

### WU-1. `search` dispatcher with `grain` enum

**Files:** `src/scix/mcp_server.py` (search tool registration + dispatcher),
`tests/test_search_grain.py` (new).

**Schema:**
```jsonschema
{
  "name": "search",
  "inputSchema": {
    "type": "object",
    "properties": {
      "query":  {"type": "string"},
      "grain":  {"type": "string", "enum": ["paper","concept","section","chunk","claim"], "default": "paper"},
      "mode":   {"type": "string", "enum": ["hybrid","semantic","keyword"], "default": "hybrid"},
      "limit":  {"type": "integer", "default": 10},
      "year_min":      {"type": "integer"},
      "year_max":      {"type": "integer"},
      "arxiv_class":   {"type": "string"},
      "doctype":       {"type": "string"},
      "include_retracted": {"type": "boolean", "default": false}
    },
    "required": ["query"]
  }
}
```

**Routing:**
- `grain=paper` → existing whole-paper hybrid search handler
- `grain=concept` → existing `concept_search` handler
- `grain=section` → existing `section_retrieval` handler
- `grain=chunk` → existing `chunk_search` handler (gated on `_qdrant_enabled()`)
- `grain=claim` → existing `find_claims` handler

**Acceptance:**
- `search(query="dark energy", grain="paper")` returns identical results to
  current `search(query="dark energy")` (byte-for-byte after stripping
  per-query timestamps).
- Each of the 5 grains is exercised in `tests/test_search_grain.py` with a
  result-shape assertion.
- When Qdrant is disabled, `search(grain="chunk")` returns a JSON envelope
  `{"error": "qdrant_unavailable", "use_instead": "search(grain=section)"}`
  with HTTP 200 (structured error, not exception).

---

### WU-2. `paper` dispatcher with `action` enum

**Files:** `src/scix/mcp_server.py` (new tool registration + dispatcher),
`tests/test_paper_dispatcher.py` (new).

**Schema:**
```jsonschema
{
  "name": "paper",
  "inputSchema": {
    "type": "object",
    "properties": {
      "bibcode": {"type": "string"},
      "action":  {"type": "string", "enum": ["metadata","read","claims"], "default": "metadata"},
      "section":  {"type": "string"},
      "include_entities": {"type": "boolean", "default": false}
    },
    "required": ["bibcode"]
  }
}
```

**Routing:**
- `action=metadata` → existing `get_paper` handler
- `action=read` → existing `read_paper` handler (forwards `section`)
- `action=claims` → existing `read_paper_claims` handler

**Explicitly excluded** (per eval evidence): `blame` and `replications`
actions. These remain as top-level tools `claim_blame` and
`find_replications`.

**Acceptance:**
- All 3 actions return identical payload shape to the current standalone
  tool of the same name (snapshot test in
  `tests/test_paper_dispatcher.py`).
- `paper(bibcode=X, action="invalid")` returns a structured error
  identifying valid actions.

---

### WU-3. `citation` dispatcher with `mode` enum

**Files:** `src/scix/mcp_server.py`, `tests/test_citation_dispatcher.py` (new).

**Schema:**
```jsonschema
{
  "name": "citation",
  "inputSchema": {
    "type": "object",
    "properties": {
      "bibcode":   {"type": "string"},
      "mode":      {"type": "string", "enum": ["traverse","similarity"], "default": "traverse"},
      "direction": {"type": "string", "enum": ["forward","backward","both"], "default": "both"},
      "depth":     {"type": "integer", "default": 1},
      "method":    {"type": "string", "enum": ["co_citation","bibliographic_coupling"], "default": "co_citation"},
      "limit":     {"type": "integer", "default": 20}
    },
    "required": ["bibcode"]
  }
}
```

**Routing:**
- `mode=traverse` → existing `citation_traverse` handler (uses `direction`, `depth`)
- `mode=similarity` → existing `citation_similarity` handler (uses `method`)

**Acceptance:**
- Both modes return identical payloads to standalone tools (snapshot test).
- Irrelevant params for the chosen mode are silently ignored (e.g.
  `method` when `mode=traverse`).

---

### WU-4. `entity` dispatcher with `action` enum

**Files:** `src/scix/mcp_server.py`, `tests/test_entity_dispatcher.py` (new).

**Schema:**
```jsonschema
{
  "name": "entity",
  "inputSchema": {
    "type": "object",
    "properties": {
      "name":   {"type": "string"},
      "qid":    {"type": "string"},
      "action": {"type": "string", "enum": ["lookup","context","papers"], "default": "lookup"},
      "type":   {"type": "string"},
      "limit":  {"type": "integer", "default": 20}
    }
  }
}
```

**Routing:**
- `action=lookup` → existing `entity` handler (resolves name/QID → canonical row)
- `action=context` → existing `entity_context` handler
- `action=papers` → SQL `SELECT bibcode, title, year FROM document_entities
  JOIN papers USING (bibcode) WHERE entity_id = $1 LIMIT $2` against the
  resolved entity. **New tiny handler** (≤30 LOC); not previously exposed.

**Acceptance:**
- `lookup` and `context` actions byte-equivalent to current standalones.
- `papers` action returns at least 10 rows for a high-coverage instrument
  entity (e.g. `name="JWST"`).

---

### WU-5. `EXPECTED_TOOLS` rewrite + registration cleanup

**Files:** `src/scix/mcp_server.py` (lines 913–947 region),
`tests/test_startup_self_test.py`.

**Changes:**
- `EXPECTED_TOOLS` becomes the 10-tuple from §Goal.
- Remove standalone registration of `concept_search`, `get_paper`,
  `read_paper`, `read_paper_claims`, `find_claims`, `section_retrieval`,
  `citation_traverse`, `citation_similarity`, `entity_context`. (Their
  handlers stay; only the `@mcp.tool`-style registration goes.)
- `_OPTIONAL_TOOLS` becomes empty tuple — `chunk_search` is no longer a
  separate optional tool. `_qdrant_enabled()` now only gates the
  `grain=chunk` branch inside the `search` dispatcher.
- Startup self-test asserts exactly 10 tools registered.

**Depends on:** WU-1, WU-2, WU-3, WU-4 (the dispatchers must exist before
their predecessors are removed from the registration list).

**Acceptance:**
- `python -c "from scix.mcp_server import startup_self_test; print(startup_self_test()['tool_count'])"` prints `10`.
- All existing tests that depended on old tool names continue to pass via
  WU-6 deprecation aliases.

---

### WU-6. Deprecation alias map extension

**Files:** `src/scix/mcp_server.py` (`_DEPRECATED_ALIASES` dict + 
`_transform_deprecated_args`).

**New entries:**
```python
_DEPRECATED_ALIASES.update({
    # search consolidation
    "concept_search":     "search",   # transform: grain=concept
    "section_retrieval":  "search",   # transform: grain=section
    "chunk_search":       "search",   # transform: grain=chunk
    "find_claims":        "search",   # transform: grain=claim
    # paper consolidation
    "get_paper":          "paper",    # transform: action=metadata
    "read_paper":         "paper",    # transform: action=read
    "read_paper_claims":  "paper",    # transform: action=claims
    # citation consolidation
    "citation_traverse":   "citation", # transform: mode=traverse
    "citation_similarity": "citation", # transform: mode=similarity
    # entity consolidation
    "entity_context":     "entity",   # transform: action=context
})
```

`_transform_deprecated_args` gets corresponding branches that inject the
right `grain`/`action`/`mode` into the args dict before dispatch. Existing
17 deprecation entries (`semantic_search`, `keyword_search`, `citation_graph`,
`citation_chain`, etc.) chain through: e.g., `semantic_search` → `search` is
unchanged, but anything that currently maps to `get_paper` should be updated
to map to `paper` with `action=metadata` so the deprecation envelope's
`use_instead` field surfaces the v1 name.

**Acceptance:**
- `tests/test_deprecated_aliases.py` extended: every old name returns
  `{"deprecated": true, "use_instead": "<v1_name>"}` in the envelope and
  the underlying payload byte-equivalent to the v1 call.
- New: assert `_DEPRECATED_ALIASES` covers every name in the pre-v1
  `EXPECTED_TOOLS` list (regression guard).

---

### WU-7. Tool-surface eval — add `v1_prime` variant + re-run

**Files:** `src/scix/eval/tool_surface/stubs.py` (add `_v1_prime_tools()`),
`scripts/run_tool_surface_eval.py` (or however the runner is invoked),
`docs/eval/tool_surface_eval_v1prime.md` (new writeup).

**Changes:**
- Add `_v1_prime_tools()` returning the 10-tool list (v1's 8 tools minus
  `paper.action` enum values `blame|replications`, plus standalone
  `claim_blame` and `find_replications` tools — copy schemas from v0).
- Register variant `v1_prime` in `_VARIANTS`.
- Re-run the same 90-run / 30-query eval against `v1_prime`.

**Acceptance:**
- `paper_blame` cluster ≥ 90% on `v1_prime` (vs 33.3% on `v1`).
- `paper_replications` cluster ≥ 90% on `v1_prime` (vs 66.7% on `v1`).
- Aggregate `tool_accuracy` ≥ `v1`'s 96.7%.
- All other clusters unchanged within ±5pt.
- Writeup includes the v0/v1/v1′ side-by-side table and the recommendation
  to ship v1′ as the v1 release surface.

---

### WU-8. Documentation update

**Files:** `CLAUDE.md` (§Tool Count Concern), `docs/mcp_tool_audit_2026-04.md`,
`docs/prd/scix_deep_search_v1.md` (if it references tool count).

**Changes:**
- Update tool count: 13 → 10.
- Document the v0 → v1 mapping table.
- Cross-link the eval evidence (`docs/eval/tool_surface_eval.md` + the new
  `tool_surface_eval_v1prime.md`).
- Add an ADR-style note in `mcp_tool_audit_2026-04.md` explaining the
  `claim_blame` / `find_replications` carve-out.

**Depends on:** WU-7 (eval results inform the writeup).

**Acceptance:**
- `grep -c "13 MCP tools" CLAUDE.md` returns 0.
- The phrase "10 tools" or "10 top-level tools" appears in
  `CLAUDE.md §Tool Count Concern`.

---

## Execution order for `/prd-build`

```
WU-1, WU-2, WU-3, WU-4    (parallel — independent dispatchers)
        ↓
      WU-5                (sequential — registration cleanup)
        ↓
      WU-6                (sequential — alias map needs final names)
        ↓
   WU-7, WU-8             (parallel — eval and docs are independent)
```

Estimate: 4–6 hours of agent wall-clock with 4-way parallelism on WU-1..WU-4.

## Risks / Open questions

1. **Qdrant-disabled environments.** When `QDRANT_URL` is unset,
   `search(grain=chunk)` must fail gracefully. WU-1 specifies a structured
   error envelope; agents must verify their planning copes with the empty
   case. Mitigation: ensure `tool_surface_eval` `v1_prime` variant runs
   include a Qdrant-disabled regression case.

2. **Schema drift in deprecated callers.** The CLAUDE.md statement that "21
   legacy names still work as aliases" was current as of commit
   `7fe258d` (the 28→13 consolidation). After v1 lands the count grows —
   ~28 legacy aliases. WU-6 acceptance must include a regression test that
   every pre-v1 name routes correctly. If any internal SciX caller (scripts/
   subagents) still uses an old name, the deprecation warnings will start
   firing in logs — that's expected, not a failure.

3. **`entity(action=papers)` is a new code path.** WU-4 introduces ~30 LOC
   of SQL. It is the only WU that adds capability rather than re-shaping
   existing surface. Out-of-scope alternative: drop the `papers` action and
   defer to `search(query="<entity_name>", grain=paper)`. Recommendation:
   keep — the v1 stubs already advertise it, and the underlying join is
   one cheap index lookup.

4. **Eval cost.** WU-7 re-runs 90 Sonnet calls via OAuth subagent. Per
   `feedback_no_paid_apis.md` and `feedback_claude_judge_via_oauth.md` this
   is the right channel. Confirm the harness still uses subagents, not the
   `anthropic` SDK.

## Out of scope (not part of v1, may follow later)

- Wiring `ner_quality_profile.precision_estimate` into MCP responses (bead
  `scix_experiments-bqva` / dbl.3.1) — separate PRD/bead.
- Phase 2 body NER (dbl.3.2).
- The `paper(action=read)` `section` parameter calling out to the
  full-text parser — already works; not changed by this PRD.
- Renaming `claim_blame` / `find_replications` to a unified `provenance`
  tool. Possible future v2 consolidation, but the eval shows the current
  names are the agent-selectable sweet spot.
