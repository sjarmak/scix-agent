# Plan: consolidate-mcp-tools

## Phase 1: Session changes (session.py)

1. Add `focused_papers: set[str]` to `_SessionData`
2. Add `track_seen(bibcodes)` method - adds to seen set
3. Add `track_focused(bibcode)` method - adds to focused set
4. Add `get_focused_papers()` method
5. Add `clear_focused()` method

## Phase 2: \_DEPRECATED_ALIASES dict

Add mapping of old names to new names at module level.

## Phase 3: list_tools() - replace all 25+ tools with exactly 13

Define the 13 new tool schemas.

## Phase 4: \_dispatch_tool() - implement new routing

1. Check \_DEPRECATED_ALIASES first - if old name, resolve to new name, set deprecated flag
2. Implement each of the 13 new tool handlers
3. Auto-track bibcodes in results (seen set)
4. Auto-track get_paper bibcodes (focused set)
5. Keep health_check as internal (not in list_tools, handled in dispatch for deprecated alias)

## Phase 5: Update TOOL_TIMEOUTS

## Phase 6: Tests

1. Test search dispatches for all 3 modes
2. Test citation_graph forward/backward
3. Test citation_similarity both methods
4. Test entity search/resolve actions
5. Test get_paper with/without include_entities
6. Test deprecated aliases return deprecated:true
7. Test find_gaps reads from focused set
8. Test list_tools returns exactly 13 tools

## Risk mitigation

- Keep old dispatch logic available through aliases
- All new tools delegate to same search.py functions
- Auto-tracking is additive (doesn't break existing behavior)
