---
name: scix-mcp-tool-surface
description: Maintainer guide to the SciX MCP tool surface — the 15-tool visible cap and its three enforcement layers, deprecated-alias routing via _ALIAS_TRANSFORMS, the published contract (contract/scix_mcp_v1.json) and the gen_mcp_contract.py regen workflow, the closed error_code catalog, and the add/consolidate-a-tool checklist. Load when adding, renaming, merging, hiding, or retiring an MCP tool; when the contract conformance test or the import-time cap guard fails; when touching mcp_server.py, mcp_tool_specs.py, mcp_handlers/, mcp_errors.py, or mcp_contract.py; or when deciding whether a new capability deserves a new tool. NOT for using the tools to do literature research — use the scix-mcp skill (query-side). NOT for search/RRF internals — use scix-retrieval-architecture. NOT for query_log analysis — use scix-db-safety-and-telemetry. NOT for CI mechanics — use scix-build-test-ci.
---

# SciX MCP Tool Surface — Maintainer Guide

The MCP server (`python -m scix.mcp_server`, stdio) is the agent-facing
deliverable of this project: other agents and projects depend on its tool
names, schemas, and error shapes. This skill is how you change that surface
without breaking consumers or tripping the three guardrails that police it.

Provenance: verified against branch `bd/0yp5-external-copy-accuracy-audit` at
commit `452ab86` (2026-07-07). Not `main`, but every file this skill cites is
byte-identical between this HEAD and `origin/main` (verified with
`git diff HEAD..origin/main -- <mcp files>` → empty). All MCP-surface files
are clean in the working tree; the uncommitted s7cy embedding fix does not
touch them.

**Gating (PROVISIONAL pending Stephanie, discovery Q5):** treat any change to
the visible tool surface — add, remove, merge, rename, hide/unhide, cap
change — as HALT-branch-ready: prepare the branch, run the gates, stop for
sign-off. The cap itself is ADR-pinned ("change only via ADR", CLAUDE.md).
See sibling `scix-change-control`.

## Vocabulary (once)

| Term                 | Meaning here                                                                                                                                                          |
| -------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Registered tool**  | Has a handler in the dispatch registry and (usually) a spec in `_TOOL_SPECS`. 19 names in `EXPECTED_TOOLS` + the Qdrant-optional `chunk_search`.                      |
| **Visible tool**     | Advertised by `tools/list` to agents. Visible = registered − hidden. The cap governs THIS number.                                                                     |
| **Hidden tool**      | Registered and callable, but filtered out of `tools/list` (`_HIDDEN_TOOLS`, env `SCIX_HIDDEN_TOOLS`). Used for tools whose backing tables are empty.                  |
| **Deprecated alias** | An old tool name that still dispatches (rewritten to a consolidated target) but returns `deprecated: true` metadata. 26 entries in `_ALIAS_TRANSFORMS`.               |
| **Contract**         | `contract/scix_mcp_v1.json` — the committed, versioned, environment-independent description of the visible surface (names + inputSchemas + error catalog + envelope). |
| **Envelope**         | The response shape. Success: tool-specific JSON, no uniform wrapper. Error: always `{"error": <human msg>, "error_code": <catalog member>}`.                          |

## File map

| File                                                               | Role                                                                                                                                                                                                                                                                                      |
| ------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `src/scix/mcp_server.py`                                           | Server wiring: `EXPECTED_TOOLS`, cap guard, `_HIDDEN_TOOLS`, `TOOL_TIMEOUTS`, `_ALIAS_TRANSFORMS`, dispatch (`call_tool` → `_dispatch_tool` → registry), `startup_self_test`. Its 800-line cap is relaxed to 1327 by PL ruling (see module docstring, bead 2qx3) — do not "fix" its size. |
| `src/scix/mcp_tool_specs.py`                                       | Pure data: `_TOOL_SPECS` (name + description + `inputSchema` per tool), `_FILTERS_SCHEMA`, `_CHUNK_SEARCH_SPEC`. The contract is built from these unchanged.                                                                                                                              |
| `src/scix/mcp_handlers/`                                           | The handlers, split by domain: `search.py`, `paper.py`, `citation.py`, `entity.py`, `claim.py`, `sections.py`, `synthesis.py`, `_common.py` (bead pebe).                                                                                                                                  |
| `src/scix/mcp_runtime.py`                                          | Pure stateless helpers, re-exported into `mcp_server` for the historical patch surface.                                                                                                                                                                                                   |
| `src/scix/mcp_errors.py`                                           | `ErrorCode` constants + `CATALOG` frozenset — the closed error-code catalog (single source of truth).                                                                                                                                                                                     |
| `src/scix/mcp_contract.py`                                         | `build_contract()` / `write_published_contract()` / `CONTRACT_VERSION` ("1").                                                                                                                                                                                                             |
| `scripts/gen_mcp_contract.py`                                      | Regenerates the committed artifact.                                                                                                                                                                                                                                                       |
| `contract/scix_mcp_v1.json`                                        | The committed artifact CI diffs against.                                                                                                                                                                                                                                                  |
| `tests/test_mcp_contract_conformance.py`                           | Cap + schema + catalog + drift gate.                                                                                                                                                                                                                                                      |
| `tests/test_mcp_error_envelopes.py`                                | Pins each emitted `error_code` at its emit site.                                                                                                                                                                                                                                          |
| `tests/test_mcp_dispatch_alias_transform.py`                       | Parametrized over the alias table; routing + args + deprecation envelope.                                                                                                                                                                                                                 |
| `docs/mcp_tool_audit_2026-04.md`, `docs/mcp_tool_audit_2026-06.md` | The audits. 06 supersedes 04's _count_ sections; 04's telemetry/error-catalog sections remain authoritative.                                                                                                                                                                              |
| `docs/prd/prd_v1_tool_consolidation.md`                            | The consolidation PRD CLAUDE.md tells you to read before adding a tool.                                                                                                                                                                                                                   |

## The 15-tool cap

`VISIBLE_TOOL_CAP = 15` (`src/scix/mcp_server.py`, near line 625). The cap is
premortem-driven: agent tool-selection accuracy degrades as the surface
grows, independent of whether the tools overlap. It is ADR-pinned; raising it
requires an ADR (audit 2026-06 §6 Option C was exactly that path, and it was
rejected in favor of consolidation).

History that bought the guards: the surface silently drifted 15 → 17 over
two months (four new tools landed after the cap was last declared "met";
`lit_review` was never even recorded in the 2026-04 audit table). Bead `xjqi`
caught it; audit `gzjq` (2026-06-15) ranked the paths back; Stephanie chose
Option A (merge real overlaps, bead `9afa`): `entity_context` →
`entity(action='profile')` and `cited_by_intent` + `find_replications` →
`forward_citations(annotate='intent'|'relation')`. Surface back at exactly 15.

### Arithmetic of the current surface (2026-07-07)

- `EXPECTED_TOOLS` = **19** registered names.
- `_DEFAULT_HIDDEN_TOOLS_STR` = `chunk_search,section_retrieval,read_paper_claims,find_claims,claim_search` — 4 of those are in `EXPECTED_TOOLS` (`chunk_search` is `_OPTIONAL_TOOLS`, registered only when Qdrant is enabled, and hidden by default anyway).
- Visible = 19 − 4 = **15**, exactly at cap. **There is no headroom: the next tool addition must merge or hide something, or ship an ADR.**

The 15 visible tools (from `contract/scix_mcp_v1.json`, registry order):
`search`, `lit_review`, `concept_search`, `get_paper`, `read_paper`,
`citation_traverse`, `citation_similarity`, `entity`, `graph_context`,
`find_gaps`, `temporal_evolution`, `facet_counts`, `claim_blame`,
`forward_citations`, `synthesize_findings`.

The default-hidden tools and why (all registered + tested; only `tools/list`
visibility is gated; restore with `SCIX_HIDDEN_TOOLS=` — empty string shows
all):

| Tool                               | Hidden because (2026-07-07)                                              |
| ---------------------------------- | ------------------------------------------------------------------------ |
| `chunk_search`                     | Qdrant-optional; `scix_chunks_v1` collection not populated               |
| `section_retrieval`                | `section_embeddings` table not populated                                 |
| `read_paper_claims`, `find_claims` | `paper_claims` table empty (migration 062 exists, no extraction run)     |
| `claim_search`                     | `extractions` has 0 rows for `negative_result`/`quant_claim` (bead c996) |

### Triple enforcement

| #   | Layer                   | Where                                                                                                                                | What trips it                                                                                                                                                                                                                                                                            |
| --- | ----------------------- | ------------------------------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | **Import-time assert**  | `mcp_server.py` ~line 631: `if len(_DEFAULT_VISIBLE_TOOLS) > VISIBLE_TOOL_CAP: raise RuntimeError(...)`                              | Adding a 16th default-visible name to `EXPECTED_TOOLS`. The server will not even import — every test file that imports `scix.mcp_server` fails collection. Evaluates the DEFAULT hidden set, not the live `SCIX_HIDDEN_TOOLS`, so an operator unhiding tools for testing cannot trip it. |
| 2   | **Conformance test**    | `tests/test_mcp_contract_conformance.py::test_visible_surface_within_cap` (+ `test_visible_surface_matches_default` catches renames) | Contract tool count > 15, or duplicates, or names diverging from `default_visible_tool_names()`.                                                                                                                                                                                         |
| 3   | **Contract drift gate** | `::test_published_contract_matches_live`                                                                                             | ANY surface change (name, description, schema, error catalog, envelope) without regenerating `contract/scix_mcp_v1.json`.                                                                                                                                                                |

A fourth, runtime guard: `startup_self_test` (runs inside `create_server`)
asserts the live `tools/list` equals the expected set exactly — missing,
extra, or duplicate tools abort server startup. With `SCIX_TEST_DSN` set it
also smoke-calls `claim_blame` / `find_replications` / `section_retrieval`.

All conformance tests are DB-less and model-less (they run in the CI marker
subset `-m "not integration and not network"`; CI installs
`.[dev,qdrant,graph,mcp]`).

## Deprecated-alias routing — `_ALIAS_TRANSFORMS`

Single source of truth: the `_ALIAS_TRANSFORMS` dict in `mcp_server.py`
(~line 1042), **26 entries**, one `_AliasTransform` per legacy name. It
replaced a 24-branch if-chain plus a parallel `_DEPRECATED_ALIASES` map that
had to be hand-synced and drifted. Do not reintroduce a second table.

`_AliasTransform` fields:

| Field         | Meaning                                                                                                                                                                                                                                                              |
| ------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `target`      | Consolidated tool name actually dispatched.                                                                                                                                                                                                                          |
| `set_args`    | Keys force-set on the args (e.g. `{"mode": "semantic"}`); they override caller values.                                                                                                                                                                               |
| `arg_fn`      | Optional in-place key rename. Only three aliases need one: `keyword_search` (`terms`→`query`), `entity_search` (`entity_name`→`query`), `search_within_paper` (`query`→`search_query`).                                                                              |
| `use_instead` | Modern name advertised to agents; defaults to `target`. Diverges only for the seven self-targeting passthroughs (e.g. `get_citation_context` dispatches to its own handler but advertises `citation_traverse`; the session tools advertise `get_paper`/`find_gaps`). |

Call path: `call_tool(name, args)` → alias lookup → per-tool
`SET LOCAL statement_timeout` (resolved via the alias's `guidance`, so
deprecated calls get a sensible timeout) → `_dispatch_tool` rewrites
`(name, args)` via `_transform_deprecated_args` → `_dispatch_consolidated`
looks up `_handler_registry()` → result wrapped by `_wrap_deprecated` with
`{"deprecated": true, "use_instead": <name>, "original_tool": <old name>}`.

Notes that save you a wrong turn:

- Aliases are **not** in `tools/list` and do not count against the cap; they
  are dispatch-only.
- `find_similar_by_examples` is not an alias — it is hard-removed; its
  registry entry returns a structured `tool_removed` error.
- Aliases still get real traffic: as of the 2026-06-15 telemetry window,
  `keyword_search` had 620 logged calls (likely an eval harness pinned to the
  pre-consolidation name). Retiring an alias outright is a breaking change
  for such consumers — treat it like a tool removal (HALT-branch-ready,
  PROVISIONAL pending Stephanie, Q5).
- Unknown names (not registered, not aliased) return
  `{"error": "Unknown tool: ...", "error_code": "unknown_tool"}` — they do
  not raise.

Tests: `tests/test_mcp_dispatch_alias_transform.py` is parametrized over the
table itself, so a new alias entry is covered automatically for routing,
forced args, caller-args immutability, and the deprecation envelope. The
9afa merges were additionally verified byte-for-byte against the old
handlers (modulo timing/deprecation metadata) in
`tests/test_mcp_tool_consolidation.py`.

## The closed error_code catalog

`src/scix/mcp_errors.py` defines `ErrorCode` (24 string constants as of
2026-07-07, grouped: input validation, query guards, routing,
backend-unavailable, operation failures) and derives `CATALOG` from it.
Convention (beads ir2h/x5jg): `error_code` at the response root,
machine-branchable; `error` for humans. `call_tool`'s last-resort wrapper
tags escaped exceptions `internal_error`.

Three tests police it — know which one fired:

| Failure                                                                                                                            | Test                                                                                         | It means                                                                                                                                               |
| ---------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| A dict literal `{"error": ...}` with no `error_code` sibling anywhere in `mcp_server.py`, `mcp_tool_specs.py`, or `mcp_handlers/*` | `test_mcp_contract_conformance.py::test_every_error_return_carries_an_error_code` (AST scan) | You emitted a bare error envelope. Add a catalog member. (The 2026-06 deep audit found 30/77 error returns bare — this scan is why that cannot recur.) |
| An emitted code not in `CATALOG`                                                                                                   | `test_mcp_error_envelopes.py::test_every_emitted_error_code_is_in_the_closed_catalog`        | You invented a string instead of adding an `ErrorCode` constant.                                                                                       |
| `contract["error_codes"] != sorted(CATALOG)`                                                                                       | conformance `test_error_codes_are_the_closed_catalog` + the drift gate                       | You added a code but did not regen the contract.                                                                                                       |

Adding a code: add the constant to `ErrorCode` (CATALOG derives
automatically) → emit it → add an assertion in
`test_mcp_error_envelopes.py` → regen the contract → per the
`test_mcp_error_envelopes.py` docstring, also add a row in
`docs/mcp_tool_audit_2026-04.md` (its error-catalog section is still the
documented public contract). Never remove or rename an existing code without
a `CONTRACT_VERSION` bump — consumers branch on these strings.

## Contract + regen workflow

`build_contract()` is deliberately **deterministic and
environment-independent**: it clears `_HIDDEN_TOOLS` to enumerate the full
registered surface, forces `_qdrant_enabled` off, then selects
`EXPECTED_TOOLS − _DEFAULT_HIDDEN_TOOLS`. The same artifact is produced on a
laptop, in CI, and on prod regardless of `SCIX_HIDDEN_TOOLS` / `QDRANT_URL`.
It is DB-less and model-less (`create_server(_run_self_test=False,
_preload_model=False)`).

After any intentional surface change:

```bash
.venv/bin/python scripts/gen_mcp_contract.py   # prints: wrote .../contract/scix_mcp_v1.json
git diff contract/                              # REVIEW the diff — this IS the change review
```

Light command (no DB, no model download; needs the `mcp` extra installed).
Review rules for the diff:

- **Additive / compatible** (new tool within cap, widened enum, new optional
  param, clarified description): update `scix_mcp_v1.json` in place.
- **Breaking** (rename, removal, required-param change, error-code removal):
  bump `CONTRACT_VERSION` in `src/scix/mcp_contract.py` → a NEW
  `contract/scix_mcp_v2.json` is published alongside v1 (the path derives
  from the version). As of 2026-07-07 only v1 exists; no breaking change has
  shipped.

Committing a regenerated contract you cannot explain line-by-line defeats
the gate. If the diff surprises you, the surface changed by accident — find
the cause before regenerating.

## Checklist — add or consolidate a tool

Read first: `docs/prd/prd_v1_tool_consolidation.md` +
`docs/mcp_tool_audit_2026-06.md` (CLAUDE.md requires this). Then:

1. **Budget against the cap.** Surface is AT 15. Options, in preference
   order (audit §6): merge a real overlap (Option A precedent), hide a
   cold/unbacked tool (temporary only — hiding a working tool silently
   breaks agents that depend on it; `claim_blame` is the cautionary
   example), or write an ADR to raise the cap. A new tool with empty backing
   tables ships default-hidden (add it to `_DEFAULT_HIDDEN_TOOLS_STR`) — that
   is the established pattern and costs no cap headroom.
2. **Gate it.** Tool-surface change = HALT-branch-ready (PROVISIONAL pending
   Stephanie, Q5). Prepare everything below on a branch; do not merge
   without sign-off.
3. **Spec**: add the entry to `_TOOL_SPECS` in `src/scix/mcp_tool_specs.py`
   (name, trigger-rich description, JSON-schema `inputSchema` with
   `type: "object"` and `properties`). Follow the §5 refinement conventions:
   default `limit` 20, text param named `query`, bibcode param named
   `bibcode`. Reuse `_FILTERS_SCHEMA` where filters apply.
4. **Handler**: implement in the matching `src/scix/mcp_handlers/<domain>.py`
   with signature `(conn, args) -> str` (JSON string). Every error return is
   `{"error": ..., "error_code": <ErrorCode member>}` — the AST scan checks
   the literal shape.
5. **Wire**: add the name to `EXPECTED_TOOLS` and to the dict in
   `_build_handler_registry()` (both in `mcp_server.py`). If default-hidden,
   also to `_DEFAULT_HIDDEN_TOOLS_STR`.
6. **Timeout**: add a `TOOL_TIMEOUTS` entry with a `SCIX_TIMEOUT_*` env
   override, sized to the slowest legitimate query (existing entries: 3–30s).
7. **Consolidating?** Add an `_ALIAS_TRANSFORMS` entry per retired name
   (target + forced args + `arg_fn` if keys rename + `use_instead` if the
   dispatch target differs from the advertised modern tool). Never delete the
   old name outright. Verify old-name parity against the new path (the 9afa
   byte-for-byte precedent, `test_mcp_tool_consolidation.py`).
8. **Regen the contract** (section above) and review the diff.
9. **Tests ship with the change**: handler unit tests + error-envelope
   assertions; the conformance and alias-transform suites pick up the new
   entries automatically. Run the CI-equivalent subset:
   `make check-ci` (or minimally
   `.venv/bin/pytest tests/test_mcp_contract_conformance.py tests/test_mcp_dispatch_alias_transform.py tests/test_mcp_error_envelopes.py -q`
   — DB-less; on this host wrap anything heavier in `scix-batch`, see
   sibling `scix-memory-and-batch-discipline`).
10. **Record it**: update the audit doc's tool table (the `lit_review`
    silent-drift lesson: a tool that isn't in the audit table is invisible
    to the next auditor), and realign in-repo consumers — the
    `deep_search_investigator` agent allow-list and the query-side
    `scix-mcp` skill were the two touched by 9afa.

## Telemetry pointer (one fact, one home)

Guard-blocked and structured-error responses log `query_log.success = TRUE`
(handlers return error JSON without raising). Analyze failures with
`WHERE success = FALSE OR error_msg IS NOT NULL`. Full treatment in sibling
`scix-db-safety-and-telemetry`.

## Known-stale bits in the query-side `scix-mcp` skill (recorded 2026-07-07, DO NOT edit it here)

The existing `.claude/skills/scix-mcp/SKILL.md` is user/query-facing and
complements this skill. Its tool table was realigned to the 15-tool surface
under bead 9afa, but these parts contradict committed reality — fix only
under an explicit bead (PROVISIONAL pending Stephanie, Q4):

1. "Ranking Model" + workflow text claim **4 RRF signals including
   `text-embedding-3-large` (OpenAI)** — that lane is removed (commit
   8b9cc90, bead 7gb4); the live stack is ~3 signals with the INDUS dense
   lane served from Qdrant (ADR-013).
2. "Connection Config" advertises a **trycloudflare URL + bearer token** —
   the public deployment was decommissioned 2026-06-12 (intentional, not an
   outage); local stdio is the only supported transport.
3. "rate limit is 60/min per token" — property of the decommissioned public
   deployment.
4. "INDUS embeddings: 32M papers (complete coverage)" — dense-lane coverage
   currently has a ~83K-paper gap from the s7cy incident (remediation
   in-flight, uncommitted; PROVISIONAL pending Stephanie, Q2 — see sibling
   `scix-embedding-pipeline`).
5. "Entity dictionary: ~90K entities" — disagrees with the current entity
   graph scale (~57.7M paper-entity links, 13 types; see sibling
   `scix-entity-ner-system`).

## Provenance and maintenance

Authored 2026-07-07 against branch `bd/0yp5-external-copy-accuracy-audit`,
HEAD `452ab86` (all cited MCP files identical to `origin/main` at `e59d89d`).
Volatile facts (tool list, hidden set, alias count, telemetry numbers) are
date-stamped above. Re-verify (all read-only):

```bash
git -C . branch --show-current && git rev-parse --short HEAD
python3 -c "import json; c=json.load(open('contract/scix_mcp_v1.json')); print(len(c['tools']), [t['name'] for t in c['tools']], len(c['error_codes']))"   # expect 15 tools, 24 codes
grep -n "VISIBLE_TOOL_CAP = " src/scix/mcp_server.py                                    # expect 15
grep -c '": _AliasTransform(' src/scix/mcp_server.py                                    # expect 26
grep -n "_DEFAULT_HIDDEN_TOOLS_STR = " -A2 src/scix/mcp_server.py                       # hidden set
grep -n 'CONTRACT_VERSION = ' src/scix/mcp_contract.py                                  # expect "1"; ls contract/
.venv/bin/pytest tests/test_mcp_contract_conformance.py --collect-only -q | tail -3     # suite still collects
```
