# Plan — MCP Tool Audit

> Scratch plan for the tool-audit work unit. Final deliverable: `../mcp_tool_audit.md`.

## Structure of the deliverable

1. **Metadata block** at top:
   - Date
   - True tool count (13, fact-checked via grep on `list_tools()` body)
   - Target tool count (12 — see decision below)
   - Net change (-1)
   - Source file with line refs
2. **Per-tool table** — one row per active tool: name, signature summary,
   classification (KEEP / DEPRECATE / CONSOLIDATE-INTO), one-sentence rationale.
3. **Section per PRD consolidation hypothesis** — explicit verdict (accept /
   reject / modify) with citation to the implementation that justifies it.
4. **Migration table** — old_tool, new_tool, parameter mapping, deprecation note.
   Covers both already-implemented aliases (so SKILL.md updates can reuse the
   text) and the new proposed change.
5. **Defense of target count** — short paragraph tying the proposal to ADASS
   §4.4 ("agent picks the right tool quickly") and the consolidation premortem.

## Decision criteria for each consolidation hypothesis

For each hypothesis the audit answers three questions:

1. Are the input shapes compatible? (If required args differ, merging forces
   polymorphic params that hurt agent legibility.)
2. Are the output shapes compatible? (Different shapes force the caller to
   branch in post-processing — no real consolidation gain.)
3. Does the SKILL.md table already model them as one decision point? (If yes,
   merge. If they sit in different rows of the agent's mental model, keep
   separate.)

## Verdicts (will be written into the deliverable)

| Hypothesis | Verdict | Why |
|---|---|---|
| get_paper + read_paper -> read_paper(depth) | reject | Different shapes (struct vs body chunk), different timeouts; `depth` would be a polymorphic flag that swaps return type. |
| citation_similarity covers co-citation + coupling | accept (already done) | `_handle_citation_similarity` dispatches on `method` enum. Document as ratified. |
| citation_graph + citation_chain -> citation_traverse | accept-with-modification | Merge with a `mode` enum {neighbors, path}. `mode=path` requires `target_bibcode`; `mode=neighbors` ignores it. Saves one tool slot, preserves type safety via conditional schema. |
| entity + entity_context merge | reject | `entity_context` requires int `entity_id`; folding it into `entity` overloads `query` as polymorphic. Two tools is clearer for agents. |
| find_similar_by_examples | reject | Does not exist in code; no demand signal; out of scope. |

## Target tool count: 12

Rationale: only `citation_graph + citation_chain -> citation_traverse` survives
the rejection filters above. Net change: 13 -> 12. Headroom of 3 tools below
the <=15 ceiling leaves room for two future additions before another audit is
needed.

## Migration table contents

The deliverable's migration table covers BOTH:

(a) **Already-merged aliases** (history) — semantic_search, keyword_search,
    get_citations, get_references, co_citation_analysis, bibliographic_coupling,
    entity_search, resolve_entity, document_context, get_openalex_topics,
    get_paper_metrics, explore_community, get_author_papers, add_to_working_set,
    get_working_set, get_session_summary, clear_working_set, get_citation_context,
    read_paper_section, search_within_paper, entity_profile.

(b) **Newly proposed** — citation_chain -> citation_traverse(mode=path).

Each row has:
- old_tool
- new_tool
- parameter mapping
- deprecation note text (1-2 sentences telling agents what to do instead)

## Acceptance-criterion mapping

| AC | How the deliverable satisfies it |
|---|---|
| 1. Lists every registered tool | Per-tool table covers all 13 names from `list_tools()`. |
| 2. Records signature + classification + rationale | One row per tool with all three columns. |
| 3. Final tool set <=15, counted at top | Metadata block states "Target: 12" explicitly. |
| 4. Each PRD hypothesis has explicit verdict | "Verdict on PRD consolidation hypotheses" section. |
| 5. Migration table | Dedicated section near end. |
| 6. True count as fact at top | Metadata block: "Currently registered: 13 (verified)". |
| 7. No code outside docs/ modified | Deliverable is .md under `docs/prd/artifacts/`; no source edits. |

## Note on artifact path

The work unit's PHASE 5 referenced `.claude/prd-build-artifacts/...` for the
research/plan/test scratch notes. That path is permission-blocked by the agent
policy in this environment. Scratch notes are kept under
`docs/prd/artifacts/tool-audit/` instead so they remain version-controlled and
co-located with the deliverable.
