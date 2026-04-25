# SciX MCP Tool Audit & Consolidation Plan

## Metadata

| Field | Value |
|---|---|
| Date | 2026-04-25 |
| Source file | `src/scix/mcp_server.py` (2032 lines, branch `prd-build/section-embeddings-mcp-consolidation`) |
| Authoritative registry | `list_tools()` handler, lines 700-1152; `EXPECTED_TOOLS` tuple at line 522 |
| **Currently registered tool count (verified)** | **13** |
| **Target tool count** | **12** |
| Net change | -1 (citation_chain folded into citation_graph) |
| Headroom under <=15 cap | 3 |

The PRD's "current 22 MCP tools" claim is incorrect. The codebase has had a
consolidation pass that landed 28 -> 13 (per the docstring at line 8). Fact-check:

```
$ awk '/async def list_tools/,/@server.call_tool/' src/scix/mcp_server.py \
    | grep -cE '^[[:space:]]+name="'
13
```

The wider dispatchable surface (~40 paths) includes 21 entries in
`_DEPRECATED_ALIASES` and 7 legacy in-process handlers (`add_to_working_set`,
`get_working_set`, `get_session_summary`, `clear_working_set`,
`get_citation_context`, `get_author_papers`, `health_check`,
`entity_profile`). None of those are advertised by `list_tools()`, so the
agent-visible tool count is 13.

## Defending the target count of 12

Section 4.4 of the planned ADASS paper depends on the MCP being lean enough
for an agent to pick the right tool quickly. The consolidation premortem
flagged "tool count concern" as a top risk. Twelve tools fits the
LLM-as-router heuristic of "a small enough menu that every option is in the
prompt and clearly differentiated":

- 5 by capability axis (search, paper access, citation graph, entity system,
  structure) -> matches the SKILL.md mental model.
- Each tool's required-args set is unique among siblings, so no two tools
  compete for the same agent intent.
- Headroom of 3 absorbs near-term additions (e.g. an author-graph tool, a
  body-NER probe) before the next audit.

Going below 12 by collapsing further (e.g. merging `entity_context` into
`entity` or `read_paper` into `get_paper`) would force polymorphic
parameters where the *type* of the input flips the *shape* of the output.
That is exactly the failure mode that hurts agent tool-use accuracy.

## Per-tool audit

| # | Tool | Signature summary | Classification | Rationale |
|---|---|---|---|---|
| 1 | `search` | `query: str` (req); `mode: hybrid\|semantic\|keyword = hybrid`; `filters: {year_min, year_max, arxiv_class, doctype, first_author, entity_types[<=100], entity_ids[<=100]}`; `limit: int = 10`; `disambiguate: bool = true` | KEEP | Top-of-funnel tool; already absorbed `semantic_search`, `keyword_search`, `get_author_papers` as deprecated aliases. Hybrid + INDUS + body-BM25 fusion is unique to this entry point. |
| 2 | `concept_search` | `query: str` (req — UAT label or URI); `include_subtopics: bool = true`; `limit: int = 20` | KEEP | UAT-only retrieval. Distinct from `search` because the input is a curated taxonomy term, not natural language; merging would erode the "I have a concept ID" signal. |
| 3 | `get_paper` | `bibcode: str` (req); `include_entities: bool = false` | KEEP | Cheap (<5s timeout) metadata-only lookup. Already absorbed `document_context`, `get_openalex_topics`. Output shape (paper struct) differs from `read_paper`; merging would force the caller to branch on body-vs-metadata. |
| 4 | `read_paper` | `bibcode: str` (req); `section: str = full` (full/introduction/methods/results/discussion/conclusions); `search_query: str?`; `char_offset: int = 0`; `limit: int = 5000` | KEEP | Already absorbed `read_paper_section` and `search_within_paper`. The body-text contract (offset + limit + section) is materially different from `get_paper`'s metadata contract — see hypothesis 1 below. |
| 5 | `citation_graph` | `bibcode: str` (req); `direction: forward\|backward\|both = forward`; `include_context: bool = false`; `limit: int = 20` | CONSOLIDATE-INTO `citation_traverse` | Merges with `citation_chain` via a `mode` enum: `neighbors` (this tool's behavior, default) and `path` (current `citation_chain`). |
| 6 | `citation_similarity` | `bibcode: str` (req); `method: co_citation\|coupling = co_citation`; `min_overlap: int = 2`; `limit: int = 20` | KEEP | Already merges `co_citation_analysis` + `bibliographic_coupling` (verified at `_handle_citation_similarity`, line 1684). Both methods take identical inputs and return identical shapes — clean enum dispatch. |
| 7 | `citation_chain` | `source_bibcode: str` (req); `target_bibcode: str` (req); `max_depth: int = 5` (clamped 1..5) | CONSOLIDATE-INTO `citation_traverse` | Two-endpoint shortest path. Folds into `citation_traverse(mode=path)`; the conditional `target_bibcode` requirement is enforceable via JSON-Schema `oneOf`. |
| 8 | `entity` | `action: search\|resolve` (req); `query: str` (req); `entity_type: methods\|datasets\|instruments\|materials` (required when `action=search`); `discipline: str?`; `fuzzy: bool = false`; `limit: int = 20`; `min_confidence_tier: 1\|2\|3?`; `sources: [str]?` | KEEP | Already merges `entity_search` + `resolve_entity`. The `action` enum keeps the input space cohesive (free text in, candidates out). |
| 9 | `entity_context` | `entity_id: int` (req) | KEEP | Strict-typed lookup by integer id. Merging into `entity` would force `query` to be `str \| int` — a polymorphic parameter that hurts agent tool-use accuracy more than it saves a tool slot. See hypothesis 4. |
| 10 | `graph_context` | `bibcode: str` (req); `include_community: bool = false`; `resolution: coarse\|medium\|fine = coarse`; `signal: citation\|semantic\|taxonomic = semantic`; `limit: int = 20` | KEEP | Already merges `get_paper_metrics` + `explore_community`. The three-signal community pivot (citation vs semantic vs taxonomic) is the differentiator from `citation_graph`. |
| 11 | `find_gaps` | (no required args); `resolution: coarse\|medium\|fine = coarse`; `limit: int = 20`; `clear_first: bool = false` | KEEP | Reads from implicit session state (papers seen via `get_paper`). The only tool that does cross-community recommendation; not duplicated anywhere. |
| 12 | `temporal_evolution` | `bibcode_or_query: str` (req); `year_start: int?`; `year_end: int?` | KEEP | Year-axis aggregation; output shape (per-year buckets with anchor papers + dominant communities) is distinct from `facet_counts`. The polymorphic `bibcode_or_query` is acceptable here because the routing on bibcode-shape vs query-shape is mechanical. |
| 13 | `facet_counts` | `field: year\|doctype\|arxiv_class\|database\|bibgroup\|property` (req); `filters: {...}`; `limit: int = 50` | KEEP | Single-field distribution with the same filter syntax as `search`. Distinct from `temporal_evolution` because there is no anchor query/bibcode. |

### Final tool roster (12)

1. search
2. concept_search
3. get_paper
4. read_paper
5. **citation_traverse** (merges citation_graph + citation_chain)
6. citation_similarity
7. entity
8. entity_context
9. graph_context
10. find_gaps
11. temporal_evolution
12. facet_counts

## PRD consolidation hypotheses — explicit verdicts

### H1. `get_paper + read_paper -> single read_paper with depth param`
**Verdict: REJECT.**

`get_paper` returns a single paper struct (title, abstract, authors, citation
counts, optional entity links). `read_paper` returns a body chunk plus offset
metadata (or a list of highlighted passages when `search_query` is set).
A single tool would need `depth ∈ {metadata, body, body_search}` and three
mutually exclusive output shapes, which is exactly the polymorphic-output
anti-pattern that hurts agent tool-use accuracy. Their timeouts (5s vs 10s)
also diverge for principled reasons — body reads can hit the 47%-coverage
body GIN. Keep separate.

### H2. `citation_similarity already covers co-citation + bibliographic_coupling?`
**Verdict: ACCEPT (already done — ratify).**

Confirmed at `mcp_server.py:1684` (`_handle_citation_similarity`). The
`method ∈ {co_citation, coupling}` enum dispatches to
`search.co_citation_analysis` or `search.bibliographic_coupling`. Both old
names route here via `_DEPRECATED_ALIASES`. Documented in the migration table.

### H3. `citation_graph + citation_chain -> single citation_traverse`
**Verdict: ACCEPT (modify to a `mode` enum).**

Both tools walk the citation graph. They differ in:

| | citation_graph | citation_chain |
|---|---|---|
| required | `bibcode` | `source_bibcode`, `target_bibcode` |
| axis | direction (forward/backward/both) | depth (1..5) |
| output | neighbor list | ordered path |

The merged `citation_traverse` has:

```
mode: neighbors | path  (default: neighbors)
bibcode: str           (required when mode=neighbors)
source_bibcode: str    (required when mode=path)
target_bibcode: str    (required when mode=path)
direction: forward|backward|both = forward   (mode=neighbors only)
include_context: bool = false                (mode=neighbors only)
max_depth: int = 5  (clamped 1..5)           (mode=path only)
limit: int = 20                              (mode=neighbors only)
```

JSON-Schema `oneOf` enforces the conditional requirement. The output is one
of two clearly-named shapes (`{neighbors: [...]}` vs `{path: [...]}`),
selected by `mode` — the caller knows which to expect. This is the only
hypothesis that clears the "shapes are compatible AND the agent's mental
model treats them as the same decision" bar.

### H4. `entity + entity_context — should they merge?`
**Verdict: REJECT.**

`entity` takes free text (`query: str`) and returns candidates with
confidence/aliases. `entity_context` takes an integer (`entity_id: int`)
and returns the full entity record with relationships. Folding into one
tool requires `query: str | int` — polymorphic input — and either a
polymorphic output (candidate list when input is text, full record when
input is int) or a third action `context` whose meaning depends on input
type. The current split is two-tools-by-input-type, which is the cleanest
agent-facing contract.

### H5. `find_similar_by_examples — keep / drop?`
**Verdict: REJECT (drop from plan; not present in code).**

`grep -r find_similar_by_examples src/scix/` returns no matches. The PRD
references this as a hypothetical tool. There is no demand signal in the
session logs (this audit did not query `query_log`, but the tool does not
exist to be invoked). Out of scope for this audit.

## Migration table

Columns: `old_tool` (the deprecated name an agent might still try),
`new_tool` (where it routes today / will route after this audit lands),
`parameter mapping`, `deprecation note` (the 1-2 sentence message to
surface in the response so agents migrate).

### Already-implemented alias migrations (ratified by this audit)

| old_tool | new_tool | parameter mapping | deprecation note |
|---|---|---|---|
| semantic_search | search | `mode = "semantic"` | Use `search` with `mode="semantic"`; hybrid mode is usually a better default. |
| keyword_search | search | `terms -> query`; `mode = "keyword"` | Use `search` with `mode="keyword"` and pass terms as `query`. |
| get_citations | citation_graph | `direction = "forward"` | Use `citation_graph(direction="forward")` to fetch citing papers. |
| get_references | citation_graph | `direction = "backward"` | Use `citation_graph(direction="backward")` to fetch references. |
| co_citation_analysis | citation_similarity | `method = "co_citation"` | Use `citation_similarity(method="co_citation")`. |
| bibliographic_coupling | citation_similarity | `method = "coupling"` | Use `citation_similarity(method="coupling")`. |
| entity_search | entity | `entity_name -> query`; `action = "search"` | Use `entity(action="search")` and pass the entity name as `query`. |
| resolve_entity | entity | `action = "resolve"` | Use `entity(action="resolve")` to map a free-text mention to canonical entity records. |
| entity_profile | get_paper | dedicated handler preserves the legacy raw-extractions schema | Use `get_paper(include_entities=true)` for the modern profile shape. |
| document_context | get_paper | `include_entities = true` | Use `get_paper(include_entities=true)` to fetch metadata plus linked entities in one call. |
| get_openalex_topics | get_paper | `include_entities = true` | Use `get_paper(include_entities=true)` — OpenAlex topic links are emitted alongside other entity links. |
| get_paper_metrics | graph_context | `include_community = false` | Use `graph_context(include_community=false)` for influence and authority scores. |
| explore_community | graph_context | `include_community = true` | Use `graph_context(include_community=true)` to get sibling papers in the same community. |
| get_author_papers | search | (legacy passthrough) | Use `search` with `filters.first_author=<name>`; the dedicated handler still works but is no longer advertised. |
| add_to_working_set | get_paper | (legacy passthrough) | Working-set tracking is now implicit in `get_paper`. Call `get_paper` for each bibcode you care about; `find_gaps` reads the same state. |
| get_working_set | find_gaps | (legacy passthrough) | The implicit working set is consumed by `find_gaps`; there is no longer a separate read tool. |
| get_session_summary | find_gaps | (legacy passthrough) | Session summary metadata is folded into `find_gaps` output. |
| clear_working_set | find_gaps | (legacy passthrough) | Pass `clear_first=true` to `find_gaps` to reset the focused set before searching. |
| get_citation_context | citation_graph | (legacy passthrough) | Use `citation_graph(include_context=true)` to receive citation context sentences inline. |
| read_paper_section | read_paper | passthrough | Use `read_paper(section=<name>)`. |
| search_within_paper | read_paper | `query -> search_query` | Use `read_paper(search_query=<terms>)` to search inside a single paper's body. |

### New migration introduced by this audit

| old_tool | new_tool | parameter mapping | deprecation note |
|---|---|---|---|
| citation_graph | citation_traverse | `mode = "neighbors"`; all other args passthrough (`bibcode`, `direction`, `include_context`, `limit`) | Use `citation_traverse(mode="neighbors", bibcode=...)`; default mode and arguments are unchanged. |
| citation_chain | citation_traverse | `mode = "path"`; all other args passthrough (`source_bibcode`, `target_bibcode`, `max_depth`) | Use `citation_traverse(mode="path", source_bibcode=..., target_bibcode=...)` to find the shortest citation path. |

## Implementation notes (for the consolidation work unit that follows)

This audit is documentation-only. To realize the new state:

1. Add `citation_traverse` Tool registration in `list_tools()` with the
   `mode`-conditional schema described under H3. Implement
   `_handle_citation_traverse` that branches on `mode` and calls the existing
   `search.citation_chain` / neighbor functions.
2. Add `citation_graph` and `citation_chain` to `_DEPRECATED_ALIASES` mapping
   to `citation_traverse`. Extend `_transform_deprecated_args` with the two
   `mode` injections.
3. Update `EXPECTED_TOOLS` from 13 entries to 12 (drop `citation_graph` and
   `citation_chain`, add `citation_traverse`). Update `startup_self_test`'s
   `tool_count != 13` check to 12.
4. Update `/home/ds/.claude/skills/scix-mcp/SKILL.md` tool overview (the
   "Tool Overview (13 tools)" header and the citation-graph row).
5. Update the module docstring at `mcp_server.py:1` from "13 consolidated
   tools" to "12 consolidated tools".

These steps are not part of the tool-audit work unit; they belong to the
follow-on consolidation unit.

## Audit follow-up (2026-04-25)

The original audit above was written against an older base of 13
agent-facing tools. Between when the audit landed and the consolidation
work unit ran, the integration branch
`prd-build/section-embeddings-mcp-consolidation` had grown the active
tool set to 17 entries via three additions and one feature-flag-gated
optional tool. This appendix records the follow-up decisions that
shipped with the consolidation unit.

### Tools the original audit missed

| Tool | Source | Verdict (2026-04-25) | Reason |
|---|---|---|---|
| `claim_blame` | PRD MH-4 — Deep Search v1 | KEEP | Provenance-specific (chronologically earliest origin walk over citation contexts). No semantic overlap with another tool; merging would force a polymorphic output shape. |
| `find_replications` | PRD MH-4 — Deep Search v1 | KEEP | Replication / refutation relation inference over forward citations. Disjoint mental model from `citation_traverse(mode="graph", direction="forward")` — agents picking the right tool benefit from the explicit semantic. |
| `find_similar_by_examples` | qdrant_tools probe (gated by `_qdrant_enabled()`) | RETIRE | The Qdrant backend is currently DOWN per the project's `mcp_deployment_state` record (no NAS-Qdrant migration has landed yet). The tool was opt-in only and not in active use; retiring it removes a "phantom" surface that confused agents reading `EXPECTED_TOOLS` against `_OPTIONAL_TOOLS`. |
| `section_retrieval` | PRD section-embeddings-mcp-consolidation (sibling unit) | KEEP | Section-grain hybrid retrieval (HNSW over halfvec + BM25 over `papers_fulltext.sections_tsv`, RRF-fused). Distinct from `search` (paper-level) and `read_paper` (single-paper body access). |

### Final active tool roster (15)

1. search
2. concept_search
3. get_paper
4. read_paper
5. section_retrieval
6. **citation_traverse** (merges citation_graph + citation_chain, `mode` enum)
7. citation_similarity
8. entity
9. entity_context
10. graph_context
11. find_gaps
12. temporal_evolution
13. facet_counts
14. claim_blame
15. find_replications

`find_similar_by_examples` is hard-removed: not registered in `list_tools()`, not in `_DEPRECATED_ALIASES`, dispatch returns
``{"error": "tool_removed", "removed_in": "2026-04-25"}``. The
private `_handle_find_similar_by_examples` function and the
`scix.qdrant_tools` module are retained on the off chance the NAS-Qdrant
migration revives the surface; nothing in the active path references them.

### Strategy used for the citation merge

**Shim**, not remove. `citation_graph` and `citation_chain` are added to
`_DEPRECATED_ALIASES` and routed through `_transform_deprecated_args`
which injects `mode="graph"` or `mode="chain"` before forwarding to
`citation_traverse`. Existing callers (including the
`deep_search_investigator` persona allowlist) continue to work and
receive a `deprecated: true` envelope so agents migrate organically.
Pre-existing aliases that targeted `citation_graph` (`get_citations`,
`get_references`, `get_citation_context`) were remapped to
`citation_traverse` with the appropriate `mode` injection to avoid
double-hop deprecation.

### Acceptance verification

```
$ awk '/async def list_tools/,/@server.call_tool/' src/scix/mcp_server.py \
    | grep -oE 'name="[a-z_]+"' | sort -u | wc -l
15
```

The acceptance criterion of <=15 active registrations is met exactly.
Headroom for future additions before the next audit is 0; the next
single-tool addition pushes the surface to 16 and should be paired with
either another consolidation or a re-justification of the cap.
