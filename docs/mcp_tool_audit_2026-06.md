# MCP Tool Audit + Consolidation Proposal — 2026-06

**Bead:** `scix_experiments-gzjq` (Stephanie 2026-06-15, Slack: "Audit our tools
to determine whether they need refinement and consolidation").
**Feeds:** `scix_experiments-xjqi` (cap-reconcile decision — 17 visible vs the
ADR-pinned ≤ 15). **Pairs with:** `scix_experiments-x2dp` (contract conformance
suite).
**Scope (analysis only):** refresh the live surface audit, reconcile against
`docs/mcp_tool_audit_2026-04.md`, and present a ranked path to ≤ 15. This doc
does **not** change any contract, hide/remove any tool, or decide the cap — the
cap is ADR-pinned and Stephanie's call (xjqi).

Source of truth: `src/scix/mcp_server.py::EXPECTED_TOOLS` (line 1281) minus
`_HIDDEN_TOOLS` (line 240), plus `_OPTIONAL_TOOLS` when `QDRANT_URL` is set.

---

## 1. Live surface — confirmed count

`EXPECTED_TOOLS` registers **21** tools. `_HIDDEN_TOOLS` defaults to
`{chunk_search, section_retrieval, read_paper_claims, find_claims, claim_search}`
— 4 of those are in `EXPECTED_TOOLS` (`chunk_search` is `_OPTIONAL_TOOLS`, only
registered when Qdrant is enabled). Net agent-visible surface:

> **21 registered − 4 default-hidden = 17 agent-visible.**

This confirms the `xjqi` finding exactly: **17 visible vs the ≤ 15 cap — over by 2.**

### 1a. The 17 visible tools

| # | Tool | Required input | One-line purpose | Prod calls (since 2026-04-14) |
|---|---|---|---|---|
| 1 | `search` | `query` | Hybrid/semantic/keyword paper search (`mode`); RRF over INDUS + body BM25. | 753 |
| 2 | `lit_review` | `query` | Composite: seed search + citation expansion + community decomposition + facets, populates working set. | 29 |
| 3 | `concept_search` | `query` | Controlled-vocabulary concept lookup (10 vocabs) with lexical fallback. | 38 |
| 4 | `get_paper` | `bibcode` | Single-paper metadata + optional linked entities. | 686 |
| 5 | `read_paper` | `bibcode` | Read/search inside one paper's full-text body (`search_query` toggles). | 1891 |
| 6 | `citation_traverse` | per-mode | Citation graph: neighborhood walk (`graph`) or shortest path (`chain`). | 103 |
| 7 | `citation_similarity` | `bibcode` | Structural similarity via shared citations (`co_citation` / `coupling`). | 41 |
| 8 | `entity` | `action` | Named-entity lookup: `resolve` (text→id), `papers` (id→papers), `search` (legacy extractions). | 45 |
| 9 | `entity_context` | `entity_id` | Full profile of one entity by numeric id. | 17 |
| 10 | `graph_context` | `bibcode` | PageRank/HITS + community membership (3 signals) for a paper. | 61 |
| 11 | `find_gaps` | _(session)_ | Cross-community gap detection over the working set. | 14 |
| 12 | `temporal_evolution` | _(query/bibcodes)_ | Publications- or citations-per-year trend with anchor papers. | 38 |
| 13 | `facet_counts` | `field` | Single-field distribution (year/doctype/arxiv_class/…) with filters. | 10 |
| 14 | `claim_blame` | `claim_text` | Trace a claim to its earliest non-retracted origin via reverse references. | **0** |
| 15 | `find_replications` | `target_bibcode` | Forward citations annotated with replication relation + hedge flag. | 6 |
| 16 | `cited_by_intent` | `target_bibcode` | Forward citations filtered by citation intent (method/background/result_comparison). | 21 |
| 17 | `synthesize_findings` | _(working set)_ | Bin a working set into a section outline (mechanical aggregation). | 7 |

### 1b. The 5 default-hidden tools (registered, not advertised)

| Tool | Why hidden | Restore |
|---|---|---|
| `chunk_search` | `_OPTIONAL_TOOLS`; only registered when `QDRANT_URL` set + `scix_chunks_v1` reachable. | set `QDRANT_URL` |
| `section_retrieval` | `section_embeddings` table not populated. | `SCIX_HIDDEN_TOOLS=` |
| `read_paper_claims` | `paper_claims` table empty (no extraction run; migration 062 exists). | `SCIX_HIDDEN_TOOLS=` |
| `find_claims` | same — `paper_claims` empty. | `SCIX_HIDDEN_TOOLS=` |
| `claim_search` | `extractions` has 0 rows for `negative_result` / `quant_claim` (bead c996). | `SCIX_HIDDEN_TOOLS=` |

All five are registered + tested unconditionally; only `tools/list` visibility
is gated. None has any prod call (consistent with being hidden).

---

## 2. Reconciliation against `docs/mcp_tool_audit_2026-04.md`

The 2026-04 audit's classification table enumerated **12 core** tools (rows 1–13
with row 7 folded into `citation_traverse`) + 1 optional Qdrant tool
(`find_similar_by_examples`), and its Summary claimed "**Met at 15 visible**"
after adding `cited_by_intent` and the `entity(action='papers')` action.

Since then the visible surface drifted **15 → 17**. Net deltas:

| Change | Tool | Bead / PRD | Net |
|---|---|---|---|
| Added | `lit_review` | `nn03` (composite review session) | +1 — **never recorded in the 04 classification table** (silent drift) |
| Added | `claim_blame` | PRD MH-4 (Deep Search v1) | +1 |
| Added | `find_replications` | PRD MH-4 (Deep Search v1) | +1 |
| Added | `synthesize_findings` | `cfh9` (terminal synthesis) | +1 |
| Retired | `find_similar_by_examples` | 2026-04-25; dead handler removed in `5aab3af` (2026-06) | −1 (was optional, not counted in 15) |

So the over-cap state is **structural, not accidental**: four genuinely new
capabilities (one review composite, two Deep-Search provenance tools, one
synthesis tool) landed after the cap was last declared met. The 04 doc itself
flagged this would happen — "subsequent additions … push the registered count
higher; the visible-by-default surface is bounded by `_HIDDEN_TOOLS`" — but the
new tools were **not** hidden, because their backing data *is* populated
(`citation_contexts`, `paper_metrics`), unlike the claims/section tools.

**Documentation debt:** `lit_review` is on the live surface but absent from the
04 classification table; the 04 table still lists "13 core + 1 optional" in its
header while enumerating 12. The 04 doc's telemetry + error-envelope sections
(query_log conventions, closed `error_code` catalog) remain **accurate** and are
not superseded by this doc.

---

## 3. Telemetry (read-only, guard-inclusive)

`query_log`: **4,401 rows, 2026-04-14 → 2026-06-15.** Predicate used:
`WHERE success = FALSE OR error_msg IS NOT NULL` (per CLAUDE.md telemetry note —
does **not** drop guard-blocked rows).

Observations that bear on consolidation:

- **Error/block rate is near-zero across the board** — `search` 1.3 % (10 rows,
  the unscoped-broad guard firing), `facet_counts` 10 % (1 of 10). No tool shows
  a systemic error pattern. The closed error-envelope catalog (bead `ir2h`) is
  holding.
- **Usage is heavily skewed.** `read_paper` (1891) + `search` (753) + `get_paper`
  (686) = **75 %** of all logged calls. The Deep-Search / synthesis cluster is
  cold: `cited_by_intent` 21, `synthesize_findings` 7, `find_replications` 6,
  **`claim_blame` 0 (never invoked in prod)**.
- **Deprecated aliases are still hot.** `keyword_search` **620**, `semantic_search`
  4, `citation_graph` 6, `citation_chain` 5, `get_author_papers` 5. The 620
  `keyword_search` calls almost certainly come from an eval harness pinned to the
  pre-consolidation name — worth confirming, but the alias layer is doing its job
  (0 errors on all alias rows).

**Caveat (do not over-read low counts):** `query_log` reflects whoever exercised
the local MCP since April — largely eval/dev traffic, not a representative agent
population. `claim_blame=0` means "not yet exercised in prod," **not** "no value"
— it is the core tool of the `deep_search_investigator` subagent and PRD MH-4.
Low usage is an argument for *visibility discipline*, not deletion.

---

## 4. Overlap / redundancy analysis

Grouped by the agent's mental model. Verdict per cluster: **merge / keep / hide.**

### 4a. Paper-finding entry points — `search` · `concept_search` · `lit_review`
- `search`: natural-language → ranked papers (the workhorse, 753 calls).
- `concept_search`: controlled-vocabulary label/URI → concepts + tagged papers.
  Distinct **input contract** (taxonomy term, not NL).
- `lit_review`: composite that *calls* hybrid search then expands. Distinct
  **output shape** (a seeded session, not a flat list) and side effect.

**Verdict: keep all three.** Different input contracts and output shapes; merging
would overload one tool with three result schemas. This matches the 04
recommendation. `lit_review` is the weakest-justified (29 calls, fully composes
existing tools) — a *defensible* drop candidate if the cap forces a cut, but it
encodes a multi-call workflow agents otherwise have to orchestrate by hand.

### 4b. Entity tools — `entity` · `entity_context` · `graph_context`
- `entity`: text↔id↔papers (3 actions), text/id input.
- `entity_context`: `entity_id` → full profile. **Same key space as
  `entity(action='papers')`**, different output (profile vs papers).
- `graph_context`: `bibcode` → graph metrics + community. Different key (bibcode,
  not entity_id), different domain (papers, not entities).

**Verdict: `entity_context` is the strongest merge candidate on the surface.** It
takes a numeric `entity_id` and returns a profile — a natural fourth action on
`entity` (`action='profile'`). The 04 doc kept it separate "because the input is
a numeric id, not text," but `entity(action='papers')` *already* accepts a bare
`entity_id`, so the input-shape argument no longer holds. Folding it removes one
tool with a near-mechanical change. `graph_context` stays separate (bibcode-keyed).

### 4c. Citation-graph cluster — `citation_traverse` · `citation_similarity` · `cited_by_intent` · `find_replications` · `claim_blame`
Five tools touch the citation graph. The two **forward-citation annotators** overlap:
- `cited_by_intent(target_bibcode, intent)`: forward citations filtered by
  `citation_contexts.intent` (method/background/result_comparison).
- `find_replications(target_bibcode, relation)`: forward citations annotated with
  an inferred replication relation + hedge flag.

Both answer "who cites X, and *why/how*," read overlapping rows
(`citation_contexts`), and key on `target_bibcode`. They differ only in the
annotation axis (intent classification vs replication-relation inference).

**Verdict: `cited_by_intent` + `find_replications` are a real merge candidate** —
one `forward_citations(target_bibcode, annotate='intent'|'relation', …)` tool, or
fold `cited_by_intent`'s intent filter into `citation_traverse(mode='graph',
direction='forward')` (which already annotates edges with intent per bead 79n).
`claim_blame` (reverse lineage) and `citation_similarity` (structural, not direct)
stay separate — different traversal direction / relation.

### 4d. Distribution tools — `temporal_evolution` · `facet_counts`
**Verdict: keep both.** Trend-over-time-anchored-to-a-topic vs flat single-field
distribution. Distinct, and the descriptions already cross-reference each other.

### 4e. `synthesize_findings`
**Verdict: keep.** No overlap — it is the only terminal-aggregation tool (bins a
working set into a section outline). Low usage (7) but unique capability and the
documented end of the lit-review → synthesis arc.

---

## 5. Refinement findings (independent of the cap)

These are contract-quality issues worth a follow-up bead regardless of the cap
decision; none requires a contract change to *fix the count*.

1. **`limit` ranges are inconsistent and partly undocumented.** Defaults vary
   (10 / 20 / 25 / 50) and documented caps differ (`1..200` on `cited_by_intent`,
   `claim_search`; `1..500` on `read_paper_claims`, `find_claims`; uncapped/implicit
   elsewhere). A single documented convention (e.g. default 20, cap 200 unless a
   tool justifies otherwise) would reduce agent guesswork.
2. **Query-param naming drift.** `temporal_evolution` uses `bibcode_or_query`
   while every other text entry point uses `query`; the forward-citation tools use
   `target_bibcode` while `citation_traverse` uses `bibcode` / `source_bibcode`.
   Consistent naming aids tool-selection accuracy (the same reason the cap exists).
3. **`citation_traverse` per-mode required params can't be expressed in JSON
   Schema.** Already mitigated by handler-side `missing_required_params` (bead
   `zjt9`) — noted as accepted debt, not a new finding.
4. **Documentation drift** (see §2): `lit_review` missing from the 04 table; the
   04 header count is stale. This doc supersedes the *count* sections of the 04
   audit; the telemetry + error-catalog sections of 04 remain authoritative.
5. **Error envelope: healthy.** Closed `error_code` catalog (bead `ir2h`) verified
   present; telemetry shows no off-catalog error leakage. No action.

---

## 6. Ranked path to ≤ 15 — **options for xjqi, not a decision**

The cap (premortem tool-count concern, ADR-pinned) is over by 2. Three coherent
ways to land it; each is reversible and contract-test-guarded. **The choice
between them — and whether to instead raise the cap — is `xjqi` / Stephanie's.**

### Option A — Merge two genuine overlaps (recommended path to 15; lowest risk)
Drops 2 by consolidating real redundancy, not by hiding capability:
1. **Fold `entity_context` → `entity(action='profile')`** (§4b). Near-mechanical;
   `entity` already accepts `entity_id`. −1.
2. **Merge `cited_by_intent` + `find_replications`** into one forward-citation
   annotator (§4c), *or* fold `cited_by_intent` into `citation_traverse`'s forward
   mode. −1.
→ **15 visible.** No capability lost; two fewer top-level names for the agent to
disambiguate. Cost: a deprecation alias for each merged name + conformance-suite
updates (x2dp). This is the option that best serves the *reason* the cap exists
(agent tool-selection accuracy).

### Option B — Hide the coldest tools (fastest; zero contract change)
Add `claim_blame` (0 calls) + `synthesize_findings` (7) — or `claim_blame` +
`lit_review` — to `_HIDDEN_TOOLS`, restorable via `SCIX_HIDDEN_TOOLS=`.
→ **15 visible**, no code change beyond the env default.
**Caveat:** this hides *working, populated-backing* tools purely for the count.
`claim_blame` is the Deep-Search subagent's core tool; hiding it from the default
surface would silently break that workflow for any agent not setting the env var.
**Not recommended** as a standing state — acceptable only as a temporary measure
while Option A lands.

### Option C — Raise the cap to 17 (requires an ADR; xjqi's call)
Argue the surface is 17 *coherent, non-redundant* tools and the 15 figure
predates the Deep-Search (MH-4) + synthesis (cfh9) capabilities. **Evidence for:**
§4 finds only two true overlaps; the rest are distinct contracts. **Evidence
against:** the premortem cap is about *agent selection accuracy*, which degrades
with surface size independent of redundancy; §3 shows 4 of the 17 tools are
near-cold, so the marginal capability of the 16th/17th tool is currently low. A
cap change must go through an ADR per CLAUDE.md ("change only via ADR").

### Recommendation (advisory only)
**Option A**, with Option B's `claim_blame` hide as an *interim* step if the
surface must be ≤ 15 before the merges land. Reasoning: A removes the two real
redundancies §4 identifies, preserves every capability, and directly serves the
selection-accuracy rationale behind the cap — whereas B trades that rationale for
speed and C needs an ADR. **Decision deferred to xjqi / Stephanie.**

### Status — Option A implemented (bead `scix_experiments-9afa`, 2026-06-15)
Stephanie chose Option A ("go with A, and do the refinement"). Implemented on
branch `scix_experiments-lq32/qdrant-missing-bibcode-keyfix` (branch-ready, not
yet published):
- `entity_context` → `entity(action='profile')` (A1); `cited_by_intent` +
  `find_replications` → `forward_citations(bibcode, annotate='intent'|'relation')`
  (A2). Agent-visible surface back at **15** (`EXPECTED_TOOLS` 21→19 minus 4
  default-hidden).
- Non-breaking: `entity_context`, `cited_by_intent`, `find_replications` retained
  as deprecated aliases (verified byte-for-byte parity modulo timing/dep-meta).
- §5 refinement: `DEFAULT_RESULT_LIMIT=20` convention applied (claims-tool default
  drift 50/25→20; 500 cap kept as a documented per-paper-bulk exception);
  `temporal_evolution.bibcode_or_query` → `query` (synonym retained);
  `forward_citations` anchors on `bibcode` (consistent with `citation_traverse`).
- Guard: import-time assert `len(default-visible) ≤ VISIBLE_TOOL_CAP (15)` in
  `mcp_server.py`; conformance expectations updated (pairs with x2dp). In-repo
  consumers realigned (`deep_search_investigator` agent allow-list, `scix-mcp`
  SKILL.md).

---

## 7. Acceptance checklist (this audit)

- [x] Enumerated the agent-visible set from `mcp_server.py` + `_HIDDEN_TOOLS`;
      confirmed **17 visible** (21 registered − 4 hidden).
- [x] Per-tool purpose, required input, and observed prod usage (query_log,
      guard-inclusive predicate).
- [x] Reconciled against `docs/mcp_tool_audit_2026-04.md`; identified the
      15 → 17 drift drivers and the `lit_review` documentation gap.
- [x] Overlap analysis with merge/keep/hide verdict per cluster.
- [x] Refinement findings (limit ranges, param naming, doc drift).
- [x] Ranked path to ≤ 15 presented as **options for xjqi** — cap not decided here.
- [x] No contract changed, no tool hidden/removed, no ADR written.
