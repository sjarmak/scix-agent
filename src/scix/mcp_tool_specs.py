"""Static MCP tool specifications — pure JSON-schema data, no runtime deps on handlers.

Extracted from :mod:`scix.mcp_server` (bead scix_experiments-pebe) so the tool
surface (the ~1.2k lines of inline ``inputSchema`` JSON) lives apart from the
server wiring. This module is pure data: it imports only ``typing`` and the
synthesize enums the descriptions are derived from, and is imported back into
``scix.mcp_server`` for registration. The published contract
(``contract/scix_mcp_v1.json``) is built from these specs unchanged.
"""

from __future__ import annotations

from typing import Any

from scix.synthesize import (
    DEFAULT_SECTIONS as _SYNTH_DEFAULT_SECTIONS,
)
from scix.synthesize import (
    SIGNAL_USED_VALUES as _SYNTH_SIGNAL_USED_VALUES,
)

# Bead qxcp: derive the signal_used enum prose from the single source of
# truth in synthesize.py so the tool description never drifts when a new
# signal value is added. Hand-typed before; one-place change now.
_SIGNAL_USED_DESCRIPTION: str = (
    "signal_used (" + " | ".join(f"'{v}'" for v in _SYNTH_SIGNAL_USED_VALUES) + ")"
)

MAX_ENTITY_FILTER_ITEMS = 100

_FILTERS_SCHEMA = {
    "type": "object",
    "properties": {
        "year_min": {"type": "integer"},
        "year_max": {"type": "integer"},
        "arxiv_class": {"type": "string"},
        "doctype": {"type": "string"},
        "first_author": {"type": "string"},
        "entity_types": {
            "type": "array",
            "items": {"type": "string"},
            "maxItems": MAX_ENTITY_FILTER_ITEMS,
            "description": (
                "Restrict results to papers linked to at least one entity of "
                "these types (e.g. 'instrument', 'mission', 'dataset'). "
                f"Empty list disables the filter. Max {MAX_ENTITY_FILTER_ITEMS} items."
            ),
        },
        "entity_ids": {
            "type": "array",
            "items": {"type": "integer"},
            "maxItems": MAX_ENTITY_FILTER_ITEMS,
            "description": (
                "Restrict results to papers linked to at least one of these "
                "specific entity ids (from the entities table). "
                f"Empty list disables the filter. Max {MAX_ENTITY_FILTER_ITEMS} items."
            ),
        },
    },
}

# A `discipline` filter was previously advertised here, but the `papers`
# table has no `discipline` column, so any caller passing it crashed at
# SQL execute() time. Re-adding it is tracked by scix_experiments-dbl.10
# (facet_counts: add 'discipline' as first-class facet field).

_SECTION_FILTERS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "year_min": {"type": "integer"},
        "year_max": {"type": "integer"},
        "bibcode_prefix": {
            "type": "string",
            "description": (
                "Restrict to bibcodes that start with this prefix "
                "(e.g. '2024ApJ' for ApJ papers from 2024)."
            ),
        },
    },
}

_TOOL_SPECS: tuple[dict[str, Any], ...] = (
    # --- M1: Unified search ---
    dict(
        name="search",
        description=(
            "Search the scientific literature corpus (32M papers: NASA ADS "
            "astro/planetary/earth/helio/biophysics + the arxiv mirror across "
            "cs.*, stat.*, physics.*, etc.) for papers matching a natural-language "
            "query. Returns ranked papers with titles, abstracts, authors, years, "
            "and citation counts. Defaults to hybrid mode, the best general-purpose "
            "search. For broad multi-keyword queries, pass filters.arxiv_class "
            "(e.g. 'cs.SE') or a filters.year_min — unscoped queries run a full-text "
            "scan over all 32M papers and may hit the statement timeout. "
            "Unscoped + broad queries (no filters AND >=3 tokens or >=30 chars) are "
            "blocked up-front with a structured {error_code: 'unscoped_broad_query', "
            "error: '<human msg>', hint, suggestions, ...} response — branch on "
            "error_code, read the hint, and retry with filters, or pass "
            "bypass_unscoped_guard=true to run the unscoped query anyway. "
            "Use concept_search instead when the query is a "
            "formal astronomy taxonomy term (e.g., 'Exoplanets'). Use entity with "
            "action='search' when looking up a named method, dataset, or instrument. "
            "Optional filters.entity_types / filters.entity_ids restrict results to "
            "papers linked to entities of given types (e.g. ['instrument']) or to "
            "specific entity ids resolved via the entity tool — useful for queries "
            "like 'papers about JWST' once JWST's entity id is known."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural language query or search terms",
                },
                "mode": {
                    "type": "string",
                    "enum": ["hybrid", "semantic", "keyword"],
                    "default": "hybrid",
                    "description": "Search mode: hybrid (default), semantic, or keyword",
                },
                "filters": _FILTERS_SCHEMA,
                "limit": {"type": "integer", "default": 10},
                "disambiguate": {
                    "type": "boolean",
                    "default": True,
                    "description": (
                        "If true (default), check the query for ambiguous entity "
                        "mentions (e.g., 'Hubble' = mission or person). If ambiguity "
                        "is detected, the server returns disambiguation candidates "
                        "instead of search results. Set false to run the search directly."
                    ),
                },
                "use_rerank": {
                    "type": "boolean",
                    "default": True,
                    "description": (
                        "If true (default) AND the SCIX_RERANK_DEFAULT_MODEL env var "
                        "is set to a model alias other than 'off' AND limit <= 20, "
                        "rerank the fused candidate set with a cross-encoder. The "
                        "production default for SCIX_RERANK_DEFAULT_MODEL is 'off' "
                        "because the M1 ablation showed both candidate rerankers "
                        "regress nDCG@10 on this corpus; this flag exists so an "
                        "operator can flip on a future domain-tuned reranker without "
                        "code changes. Setting use_rerank=false bypasses reranking "
                        "even when a model is configured."
                    ),
                },
                "bypass_unscoped_guard": {
                    "type": "boolean",
                    "default": False,
                    "description": (
                        "If true, skip the unscoped-broad-query guard and run the "
                        "search even when no filters are set and the query is broad "
                        "(>=3 tokens or >=30 chars). The default is false: such "
                        "queries are blocked up-front with a structured "
                        "{error_code: 'unscoped_broad_query', error: '<human msg>', ...} "
                        "response so the agent can retry with scoping filters instead "
                        "of waiting for the statement timeout. Use only when you "
                        "genuinely need an unscoped scan (tests, power-user exploration)."
                    ),
                },
                "community_expand": {
                    "type": "boolean",
                    "default": False,
                    "description": (
                        "When true, run a community-expansion lane (entity "
                        "co-occurrence retrieval) instead of standard hybrid "
                        "retrieval. The seed entity is taken from "
                        "filters.entity_ids (when it has exactly one entry); "
                        "otherwise the server resolves the query string to a "
                        "single unambiguous entity. If neither resolves to a "
                        "unique entity, the tool returns a structured "
                        "{error_code: 'community_expand_no_seed', ...} response. "
                        "Super-hub seeds (>50k linked papers) are blocked with a "
                        "{error_code: 'seed_too_broad', ...} envelope. Off by "
                        "default — gated behind this flag until per-prop eval "
                        "lands. See bead scix_experiments-xz4.1.40."
                    ),
                },
            },
            "required": ["query"],
        },
    ),
    # --- lit_review (composite — bead nn03) ---
    dict(
        name="lit_review",
        description=(
            "Open a literature-review session in one call. Composes "
            "hybrid_search for seed papers, citation expansion (refs + "
            "forward citations on the top-K seeds), community decomposition "
            "(semantic-medium clusters with labels), top-venue and "
            "year-distribution facets over the resulting working set, and "
            "full abstracts on the highest-ranked seeds. Side effect: the "
            "working set is populated in session state so follow-up tools "
            "(find_gaps, etc.) operate on it without re-listing bibcodes. "
            "Use this as the FIRST call when the user asks for a literature "
            "review or topic survey; use plain search instead when you only "
            "need a flat ranked list."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Free-text research question or topic.",
                },
                "year_min": {
                    "type": "integer",
                    "description": "Earliest publication year to include in "
                    "seed retrieval (inclusive).",
                },
                "year_max": {
                    "type": "integer",
                    "description": "Latest publication year to include (inclusive). "
                    "Filters seeds AND citation-expanded papers.",
                },
                "top_seeds": {
                    "type": "integer",
                    "default": 20,
                    "description": "Hybrid-search seed count (1..100).",
                },
                "expansion_seeds": {
                    "type": "integer",
                    "default": 5,
                    "description": "How many top seeds to expand via "
                    "refs+citations (0..top_seeds).",
                },
                "expand_per_seed": {
                    "type": "integer",
                    "default": 20,
                    "description": "Per-seed reference + citation fetch limit (0..50).",
                },
                "sample_abstracts": {
                    "type": "integer",
                    "default": 5,
                    "description": "Number of seeds to attach full abstracts "
                    "to in the response (0..top_seeds).",
                },
                "discipline": {
                    "type": "string",
                    "description": "Optional discipline hint (currently "
                    "informational, surfaced in metadata).",
                },
            },
            "required": ["query"],
        },
    ),
    # --- concept_search (multi-vocabulary router, dbl.7) ---
    dict(
        name="concept_search",
        description=(
            "Look up concepts across controlled vocabularies and return ranked "
            "candidates tagged with their source vocabulary. Searched by default: "
            "UAT (astronomy), OpenAlex Topics, ACM CCS (computing), MSC "
            "(mathematics), PhySH (physics), GCMD (earth science), MeSH "
            "(biomedical), NCBI Taxonomy (organisms), ChEBI (chemistry), Gene "
            "Ontology (biology). Pass vocabulary=['mesh'] (or a single string) "
            "to restrict search. Accepts a concept label (case-insensitive), an "
            "alternate label, or a URI (concept_id or external_uri). Returns "
            "concept hits in metadata.concepts; for UAT hits, also returns "
            "papers tagged with the best concept (and its descendants by "
            "default). Free-text fallback: when no controlled-vocabulary "
            "concept resolves, the tool falls through to lexical (BM25) search "
            "on the corpus and returns those papers with "
            "metadata.fallback='lexical_search' and metadata.concept_found=false. "
            "Disable via fallback=false for strict vocabulary-only behavior. "
            "For best results on short, taxonomy-shaped queries (e.g. "
            "'Exoplanets', 'Black Holes'), use this tool. For free-form natural "
            "language, search is usually a better first call."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Concept label, alternate label, or URI",
                },
                "vocabulary": {
                    "description": (
                        "Restrict search to one or more vocabularies. Allowed: "
                        "'uat', 'openalex', 'acm_ccs', 'msc', 'physh', 'gcmd', "
                        "'mesh', 'ncbi_tax', 'chebi', 'gene_ontology'. "
                        "Omit to search all."
                    ),
                    "anyOf": [
                        {"type": "string"},
                        {"type": "array", "items": {"type": "string"}},
                        {"type": "null"},
                    ],
                },
                "include_subtopics": {
                    "type": "boolean",
                    "default": True,
                    "description": (
                        "When the best hit is a UAT concept, include papers "
                        "tagged with descendants in the hierarchy."
                    ),
                },
                "fallback": {
                    "type": "boolean",
                    "default": True,
                    "description": (
                        "When true (default), free-text queries that resolve "
                        "to no vocabulary concept fall through to lexical "
                        "(BM25) search and return those papers with "
                        "metadata.fallback='lexical_search'. Set false for "
                        "strict vocabulary-only behavior (legacy)."
                    ),
                },
                "limit": {"type": "integer", "default": 20},
            },
            "required": ["query"],
        },
    ),
    # --- M6: get_paper absorbs document_context ---
    dict(
        name="get_paper",
        description=(
            "Fetch full metadata for a single paper by its ADS bibcode: title, "
            "abstract, authors, affiliations, year, keywords, and citation counts. "
            "Optionally include linked entities (methods, datasets, instruments) "
            "detected in the paper, each annotated with precision_estimate "
            "(0..1) and precision_band (high/medium/low/noisy) from the dbl.3 "
            "NER quality profile. Use search or concept_search instead when you "
            "do not yet know the bibcode. Use read_paper to access the full body text."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "bibcode": {
                    "type": "string",
                    "description": "ADS bibcode (e.g., '2024ApJ...962L..15J')",
                },
                "include_entities": {
                    "type": "boolean",
                    "default": False,
                    "description": "Include linked entities from document_context view",
                },
                "min_precision": {
                    "type": "number",
                    "description": (
                        "Optional precision_estimate floor (0..1). Drops linked "
                        "entities below the threshold; only applies when "
                        "include_entities=true. Default surfaces all entities with "
                        "precision metadata attached."
                    ),
                },
            },
            "required": ["bibcode"],
        },
    ),
    # --- S1: read_paper (combines read_paper_section + search_within_paper) ---
    dict(
        name="read_paper",
        description=(
            "Read or search inside one paper's full-text body. Without search_query, "
            "returns a paginated chunk of a named section (introduction, methods, "
            "results, discussion, conclusions, or full). With search_query, returns "
            "highlighted passages matching terms inside that paper. Use get_paper "
            "instead when you only need metadata and abstract, not body text."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "bibcode": {
                    "type": "string",
                    "description": "ADS bibcode of the paper",
                },
                "section": {
                    "type": "string",
                    "default": "full",
                    "description": (
                        "Section name: 'full', 'introduction', 'methods', "
                        "'results', 'discussion', 'conclusions'"
                    ),
                },
                "role": {
                    "type": "string",
                    "enum": [
                        "background",
                        "method",
                        "result",
                        "conclusion",
                        "other",
                    ],
                    "description": (
                        "Optional canonical role. When provided, selects "
                        "the first parsed section whose name maps to this "
                        "role (e.g. 'method' picks Methods/Observations/"
                        "Data). Takes precedence over 'section'."
                    ),
                },
                "search_query": {
                    "type": "string",
                    "description": "If provided, search within the paper "
                    "body instead of reading",
                },
                "char_offset": {
                    "type": "integer",
                    "default": 0,
                    "description": "Character offset for pagination (read mode)",
                },
                "limit": {
                    "type": "integer",
                    "default": 5000,
                    "description": "Max characters to return (read mode)",
                },
            },
            "required": ["bibcode"],
        },
    ),
    # --- citation_traverse (merges citation_graph + citation_chain, 2026-04-25) ---
    dict(
        name="citation_traverse",
        description=(
            "Traverse the citation graph. mode='graph' walks the neighborhood of a "
            "single paper (citing or cited papers); mode='chain' traces the shortest "
            "citation path between a source and a target paper. "
            "REQUIRED PARAMS DIFFER BY MODE (JSON Schema can't express this): "
            "mode='graph' (default) requires ONE of {bibcode, bibcodes=[...], "
            "a focused session working set} — multi-paper neighborhoods are "
            "returned under by_bibcode; supports direction "
            "(forward=citing, backward=references, both=all) and include_context. "
            "mode='chain' requires BOTH source_bibcode AND target_bibcode and "
            "accepts max_depth (1..5); working-set scoping does not apply. "
            "Calls missing the per-mode required params return a structured "
            "{error_code:'missing_required_params', mode, required, got} "
            "payload before any DB access. "
            "Use citation_similarity instead when you want papers related via "
            "shared citation patterns rather than direct citation links."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "mode": {
                    "type": "string",
                    "enum": ["graph", "chain"],
                    "default": "graph",
                    "description": (
                        "graph: walk neighbors of a single bibcode. "
                        "chain: trace shortest path between source and target."
                    ),
                },
                "bibcode": {
                    "type": "string",
                    "description": (
                        "ADS bibcode (used when mode='graph' and bibcodes is not provided)"
                    ),
                },
                "bibcodes": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Optional list of bibcodes for working-set "
                        "expansion (mode='graph' only). When provided, "
                        "the neighborhood walk runs per source bibcode "
                        "and results are returned under by_bibcode. "
                        "When omitted, falls through to the session's "
                        "focused papers (papers inspected via get_paper)."
                    ),
                },
                "direction": {
                    "type": "string",
                    "enum": ["forward", "backward", "both"],
                    "default": "forward",
                    "description": (
                        "forward=citing papers, backward=references, both=all "
                        "(mode='graph' only)"
                    ),
                },
                "include_context": {
                    "type": "boolean",
                    "default": False,
                    "description": ("Include citation context text (mode='graph' only, slower)"),
                },
                "source_bibcode": {
                    "type": "string",
                    "description": ("Starting paper of the chain (required when mode='chain')"),
                },
                "target_bibcode": {
                    "type": "string",
                    "description": ("Destination paper of the chain (required when mode='chain')"),
                },
                "max_depth": {
                    "type": "integer",
                    "default": 5,
                    "description": ("Maximum number of hops 1..5 (mode='chain' only)"),
                },
                "limit": {
                    "type": "integer",
                    "default": 20,
                    "description": "Max neighbors to return (mode='graph' only)",
                },
            },
        },
    ),
    # --- M3: citation_similarity (merges co_citation + coupling) ---
    dict(
        name="citation_similarity",
        description=(
            "Find papers related to a seed paper through shared citation patterns. "
            "method='co_citation' returns papers often cited together with the seed "
            "(peer works in the same discussion). method='coupling' returns papers "
            "that share many references with the seed (papers built on similar "
            "foundations). Use citation_traverse(mode='graph') instead when you "
            "want direct citing or cited papers rather than structurally similar ones."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "bibcode": {"type": "string", "description": "ADS bibcode"},
                "method": {
                    "type": "string",
                    "enum": ["co_citation", "coupling"],
                    "default": "co_citation",
                    "description": "co_citation or coupling",
                },
                "min_overlap": {
                    "type": "integer",
                    "default": 2,
                    "description": "Minimum shared citing/referenced papers",
                },
                "limit": {"type": "integer", "default": 20},
            },
            "required": ["bibcode"],
        },
    ),
    # --- M4: entity (merges entity_search + resolve_entity) ---
    #
    # NOTE (scix_experiments-mh14, 2026-04-27): the entity tool
    # advertises three distinct workflows behind the ``action`` enum.
    # ``negative_result`` and ``quant_claim`` used to be smuggled in
    # under ``entity_type`` for action='search', but those are
    # claim/finding extractions, not entities — they share no
    # parameter shape with real entity types. They were removed from
    # the schema; surfacing them is tracked under the follow-up bead
    # for a dedicated claim_search tool. See
    # ``docs/mcp_tool_audit_2026-04.md`` for the rationale.
    dict(
        name="entity",
        description=(
            "Look up named scientific entities across 13 vocabularies "
            "(method, dataset, instrument, material, gene, software, "
            "mission, organism, target, observable, chemical, "
            "location, taxon — ~9M entities total). The tool exposes "
            "three distinct actions; pick by what you have on input "
            "and what you want back. "
            "action='resolve' (text→entity_id): you have a free-text "
            "mention ('p53', 'transformer', 'JWST') and want canonical "
            "entity records to disambiguate it. "
            "action='papers' (entity_id→papers): you already have an "
            "entity_id (from a prior resolve, or document_entities "
            "joins) and want the papers tagged with it; pass "
            "entity_id directly, or query='<text>' to auto-resolve "
            "first. "
            "action='search' (legacy entity-extractions search): a "
            "narrower path that scans the older extractions table "
            "for one of four containment-payload entity types "
            "(methods/datasets/instruments/materials). "
            "action='profile' (entity_id->full profile): you have an "
            "entity_id and want its canonical name, type, discipline, "
            "external ids, aliases, related entities, and mention "
            "count (this is the former entity_context tool, folded in)."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["search", "resolve", "papers", "profile"],
                    "description": (
                        "resolve: text mention -> canonical entity records "
                        "(cross-discipline disambiguation). "
                        "papers: entity_id (or auto-resolved query) -> "
                        "papers tagged with that entity via "
                        "document_entities. "
                        "search: legacy narrow scan of the extractions "
                        "table for one of the four containment-payload "
                        "entity types (methods/datasets/instruments/"
                        "materials). "
                        "profile: entity_id -> full entity profile "
                        "(name, type, discipline, external ids, aliases, "
                        "related entities, paper count) — formerly the "
                        "entity_context tool."
                    ),
                },
                "entity_id": {
                    "type": "integer",
                    "description": (
                        "Entity id (from resolve). Used by action='papers' "
                        "and required by action='profile'."
                    ),
                },
                "entity_type": {
                    "type": "string",
                    "enum": [
                        "methods",
                        "datasets",
                        "instruments",
                        "materials",
                    ],
                    "description": (
                        "Entity type (required for action=search). "
                        "Searches the in-line containment payload on "
                        "extractions rows. NOTE: 'negative_result' "
                        "and 'quant_claim' were removed in 2026-04 "
                        "(bead scix_experiments-mh14) — those are "
                        "claim/finding extractions, not entities. A "
                        "dedicated tool for surfacing them is tracked "
                        "as a follow-up; see docs/mcp_tool_audit_2026"
                        "-04.md."
                    ),
                },
                "query": {
                    "type": "string",
                    "description": "Entity name to search for or resolve",
                },
                "discipline": {
                    "type": "string",
                    "description": "Discipline for ranking boost (resolve only)",
                },
                "fuzzy": {
                    "type": "boolean",
                    "default": False,
                    "description": "Enable fuzzy matching (resolve only)",
                },
                "limit": {"type": "integer", "default": 20},
                "min_confidence_tier": {
                    "type": "integer",
                    "enum": [1, 2, 3],
                    "description": (
                        "Only return extractions whose confidence_tier is "
                        ">= this value (1=low, 2=medium, 3=high). Omit to "
                        "include all tiers. Applies to action='search' only."
                    ),
                },
                "sources": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Filter extractions to these provenance sources "
                        "only (e.g. ['metadata', 'ner', 'llm']). Omit to "
                        "include all sources. Applies to action='search' "
                        "only."
                    ),
                },
            },
            "required": ["action"],
        },
    ),
    # entity_context folded into entity(action='profile') (bead 9afa);
    # the old name remains callable via _ALIAS_TRANSFORMS.
    # --- S2: graph_context (merges get_paper_metrics + explore_community) ---
    dict(
        name="graph_context",
        description=(
            "Get citation-graph analytics for a paper: influence scores, authority "
            "and hub metrics, and the communities it belongs to under all three "
            "signals (citation, semantic, taxonomic) at coarse/medium/fine "
            "resolution. The response includes a top-level `communities` block "
            "with a sub-block per signal, each carrying `community_id`, `label`, "
            "and `top_keywords` where available. Optionally also returns sibling "
            "papers in the same community ranked by influence; the `signal` "
            "parameter selects which community signal to explore for siblings. "
            "Default `resolution='medium'` matches `find_gaps` so back-to-back "
            "calls land on the same partition. "
            "Use citation_traverse(mode='graph') instead when you want direct "
            "citing or cited papers rather than computed scores and community membership."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "bibcode": {
                    "type": "string",
                    "description": "ADS bibcode",
                },
                "include_community": {
                    "type": "boolean",
                    "default": False,
                    "description": "Include sibling papers from same community",
                },
                "resolution": {
                    "type": "string",
                    "enum": ["coarse", "medium", "fine"],
                    "default": "medium",
                    "description": (
                        "Community resolution level. Default 'medium' "
                        "matches find_gaps for cross-tool consistency."
                    ),
                },
                "signal": {
                    "type": "string",
                    "enum": ["citation", "semantic", "taxonomic"],
                    "default": "semantic",
                    "description": (
                        "Community signal to explore for sibling papers: "
                        "'citation' (co-citation-derived Leiden), 'semantic' "
                        "(INDUS-embedding k-means), or 'taxonomic' (arXiv-class). "
                        "Invalid values return a structured error."
                    ),
                },
                "limit": {"type": "integer", "default": 20},
            },
            "required": ["bibcode"],
        },
    ),
    # --- M5: find_gaps (reads from implicit session state) ---
    dict(
        name="find_gaps",
        description=(
            "Surface papers in communities you have not yet explored that still "
            "cite papers you already inspected via get_paper. Helps catch adjacent "
            "literature you might be missing during a research session. Two ways to "
            "seed the working set: (a) call get_paper on one or more papers first "
            "(implicit session state); (b) pass query='<topic>' to auto-seed via "
            "concept_search in a single call. Use citation_traverse(mode='graph') "
            "instead when you want direct citations of a single paper rather than "
            "cross-community gap detection. The 'signal' parameter picks which "
            "community partition to traverse: 'semantic' (default, INDUS k-means, "
            "full 32M-paper coverage) or 'citation' (currently offline — Leiden "
            "Phase B has not completed, so this path returns empty)."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "Optional. When the working set is empty, auto-seeds it "
                        "via concept_search(query) so the gap analysis runs in a "
                        "single call. Ignored when prior get_paper calls have "
                        "already populated the working set."
                    ),
                },
                "signal": {
                    "type": "string",
                    "enum": ["semantic", "citation"],
                    "default": "semantic",
                    "description": (
                        "Which community partition to traverse. "
                        "'semantic' covers all 32M papers today; "
                        "'citation' is offline until Leiden Phase B lands."
                    ),
                },
                "resolution": {
                    "type": "string",
                    "enum": ["coarse", "medium", "fine"],
                    "default": "medium",
                    "description": (
                        "Community resolution. 'medium' (default, "
                        "k=200) gives sharp CS-readable labels. "
                        "'coarse' (k=20) lumps many disciplines into "
                        "mega-buckets; 'fine' (k=2000) has crisper "
                        "partitions but TF-IDF labels get noisy as "
                        "buckets shrink (driven by outlier terms)."
                    ),
                },
                "limit": {"type": "integer", "default": 20},
                "clear_first": {
                    "type": "boolean",
                    "default": False,
                    "description": "Reset the focused set before searching",
                },
            },
        },
    ),
    # --- temporal_evolution ---
    dict(
        name="temporal_evolution",
        description=(
            "Show how activity around a topic or paper evolves over time. Given a "
            "single bibcode, returns citations-per-year for that paper. Given "
            "bibcodes=[...] (or with the working set populated via prior "
            "get_paper calls), returns aggregate citations-per-year across the "
            "set. Given search terms, returns publications-per-year plus per-year "
            "'buckets' with top anchor papers (ranked by PageRank) and dominant "
            "communities, so a single call yields a usable temporal narrative "
            "instead of raw counts. Useful for tracking rising or fading topics "
            "and paper impact trajectories. Use facet_counts instead when you "
            "want a single distribution by year without a topic or bibcode anchor."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "A bibcode (citation trends) or search terms "
                        "(pub volume). Optional when bibcodes=[...] or "
                        "the session has focused papers. (The legacy "
                        "param name 'bibcode_or_query' is still accepted "
                        "for backward compatibility.)"
                    ),
                },
                "bibcodes": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Optional list of bibcodes for working-set "
                        "aggregate citations-per-year. When omitted, "
                        "falls through to the session's focused papers."
                    ),
                },
                "year_start": {"type": "integer"},
                "year_end": {"type": "integer"},
            },
        },
    ),
    # --- facet_counts (unchanged) ---
    dict(
        name="facet_counts",
        description=(
            "Return a distribution of paper counts grouped by a single metadata "
            "field: year, doctype, arxiv_class, database, bibgroup, or property. "
            "Accepts the same filters as search to scope the distribution to a "
            "subset. Pass bibcodes=[...] (or rely on the session's focused "
            "papers) to scope the distribution to a working set — useful for "
            "characterizing the year/doctype profile of a curated paper set. "
            "Useful for dataset overviews and filter discovery. Use "
            "temporal_evolution instead when you need year-over-year trends tied "
            "to a specific query or bibcode rather than a flat distribution."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "field": {
                    "type": "string",
                    "enum": [
                        "year",
                        "doctype",
                        "arxiv_class",
                        "database",
                        "bibgroup",
                        "property",
                    ],
                },
                "filters": _FILTERS_SCHEMA,
                "bibcodes": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Optional list of bibcodes to scope the facet "
                        "distribution. When omitted, falls through to "
                        "the session's focused papers; when neither is "
                        "set, runs over the full corpus."
                    ),
                },
                "limit": {"type": "integer", "default": 50},
            },
            "required": ["field"],
        },
    ),
    # --- PRD MH-4: claim_blame ---
    dict(
        name="claim_blame",
        description=(
            "Trace a claim back to its chronologically earliest non-retracted "
            "origin paper by walking reverse references over all citation "
            "contexts. Returns the origin bibcode, a Hop chain that surfaces "
            "intent and intent_weight at every step, a confidence score in "
            "[0,1], and a list of retraction warnings for any paper in the "
            "lineage with a retraction event. Ranking is "
            "(chronological_priority, intent_weight, semantic_match) with "
            "weights {result_comparison: 1.0, method: 0.6, background: 0.3}. "
            "Use citation_chain instead when you already know both endpoints "
            "and just want a path; use find_replications to enumerate forward "
            "citations to a paper."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "claim_text": {
                    "type": "string",
                    "description": (
                        "Natural-language claim to trace, e.g. "
                        "'local H0 measurement is 73 km/s/Mpc'."
                    ),
                },
                "scope": {
                    "type": "object",
                    "description": (
                        "Optional ResearchScope filters (year_window, "
                        "community_ids, methodology_class, instruments, "
                        "exclude_authors, exclude_funders, min_venue_tier, "
                        "leiden_resolution). All keys optional."
                    ),
                },
                "candidate_limit": {
                    "type": "integer",
                    "default": 20,
                    "description": "Max claim-asserting candidates to seed.",
                },
                "lineage_limit": {
                    "type": "integer",
                    "default": 10,
                    "description": "Max hops to walk per candidate.",
                },
            },
            "required": ["claim_text"],
        },
    ),
    # --- forward_citations (bead 9afa): merges find_replications + cited_by_intent ---
    dict(
        name="forward_citations",
        description=(
            "Enumerate forward citations to a paper (papers that cite it), "
            "annotated by one of two axes selected with the 'annotate' param. "
            "annotate='intent' (default) surfaces the citation_contexts.intent "
            "classification (method / background / result_comparison) so you can "
            "ask 'which papers used X as their method?' or 'which papers compared "
            "their results to X?' — each result carries the citing bibcode, the "
            "intent label, and a 400-char excerpt; optional 'intent' filters to one "
            "class. annotate='relation' surfaces an inferred replication relation "
            "(replicates / refutes / qualifies / partial / unknown) plus a "
            "hedge_present flag, derived from a documented heuristic substitute for "
            "NegBERT; optional 'relation' filters, and 'scope' applies ResearchScope "
            "filters (year_window applies to the citing paper's year). Coverage of "
            "the intent axis is partial (~825K contexts across 30K source / 250K "
            "cited papers); uncovered papers return empty cleanly. Use "
            "citation_traverse(mode='graph', direction='forward') instead when you "
            "want raw forward citations with no annotation; use claim_blame for the "
            "earliest origin of a claim rather than its forward citations."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "bibcode": {
                    "type": "string",
                    "description": (
                        "ADS bibcode whose forward citations (incoming "
                        "citations) to enumerate. 'target_bibcode' is "
                        "accepted as a synonym for backward compatibility."
                    ),
                },
                "annotate": {
                    "type": "string",
                    "enum": ["intent", "relation"],
                    "default": "intent",
                    "description": (
                        "Annotation axis. 'intent' = citation-intent "
                        "classification + excerpt (former cited_by_intent); "
                        "'relation' = inferred replication relation + hedge "
                        "flag (former find_replications)."
                    ),
                },
                "intent": {
                    "type": "string",
                    "enum": ["method", "background", "result_comparison"],
                    "description": (
                        "(annotate='intent' only) Filter to one citation "
                        "intent: 'method' = papers that used this work's "
                        "method; 'background' = papers citing it as "
                        "background; 'result_comparison' = papers comparing "
                        "results to it. Omit for any-intent."
                    ),
                },
                "relation": {
                    "type": "string",
                    "enum": ["replicates", "refutes", "qualifies", "partial"],
                    "description": (
                        "(annotate='relation' only) Filter on "
                        "relation_inferred. Omit to return all relations "
                        "including unknown."
                    ),
                },
                "scope": {
                    "type": "object",
                    "description": (
                        "(annotate='relation' only) Optional ResearchScope "
                        "filters; year_window applies to the citing paper's year."
                    ),
                },
                "limit": {
                    "type": "integer",
                    "default": 20,
                    "description": "Max papers to return (1..200).",
                },
            },
            "required": ["bibcode"],
        },
    ),
    # --- Claim/finding extraction surface (bead c996) ---
    #
    # NOTE (scix_experiments-c996, 2026-04-27): split out from the
    # entity tool's ``entity_type`` enum under bead mh14, where
    # ``negative_result`` and ``quant_claim`` were misfiled as
    # entities. They are claim/finding extraction kinds with their
    # own payload shapes (``spans[].evidence_span`` for
    # negative_result; ``claims[].{quantity, value, uncertainty,
    # unit}`` for quant_claim). This tool is the dedicated home.
    # Default-hidden in deployments where the extractions table has
    # no rows of these types — see ``_HIDDEN_TOOLS``.
    dict(
        name="claim_search",
        description=(
            "Search the extractions table for claim/finding rows. "
            "Two action kinds: action='negative_result' surfaces "
            "spans of explicit negative findings (no detection, "
            "no significant excess, upper limits) — each row "
            "carries a top-level ``evidence_span`` with the "
            "matching sentence plus the full ``spans`` list; "
            "action='quant_claim' surfaces measured-quantity "
            "claims — the first claim is promoted to a top-level "
            "``payload`` of {quantity, value, uncertainty, unit} "
            "and the full ``claims`` list is preserved. The "
            "optional ``query`` parameter narrows the result "
            "set: for negative_result it is a free-text needle "
            "matched against ``evidence_span`` / ``match_text``; "
            "for quant_claim it is an exact match on the "
            "canonical ``quantity`` value (e.g. 'H0'). Use the "
            "entity tool for named-entity lookups; this tool is "
            "for paper-level findings."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["negative_result", "quant_claim"],
                    "description": (
                        "Which extraction kind to search. "
                        "negative_result = explicit negative "
                        "findings; quant_claim = measured "
                        "quantities."
                    ),
                },
                "query": {
                    "type": "string",
                    "description": (
                        "Optional filter. negative_result: "
                        "free-text needle matched against "
                        "evidence_span / match_text. "
                        "quant_claim: exact match on the "
                        "canonical 'quantity' value."
                    ),
                },
                "limit": {
                    "type": "integer",
                    "default": 20,
                    "description": "Max papers to return (1..200).",
                },
            },
            "required": ["action"],
        },
    ),
    # --- Terminal synthesis (bead cfh9) ---
    dict(
        name="synthesize_findings",
        description=(
            "Bin a working set of papers into a section outline ready "
            "for the agent to write up. Pure mechanical aggregation: "
            "each paper is assigned to a section by override first, "
            "then by modal citation_contexts.intent (method->methods, "
            "background->background, result_comparison->results), then "
            "by weighted community fall-through driven by each "
            "community's share of the working set: share>=0.15 "
            "('core') -> 'background'; share>=0.05 ('supporting') -> "
            "'methods' (or 'background' overflow when methods is "
            "absent); share<0.05 ('peripheral') -> left unattributed "
            "and eligible for the Tier-3 citation-count fallback "
            "that fills any still-empty section from the unattributed "
            "pool by within-set citation_count desc. Each "
            "cited_papers entry exposes the signals that produced "
            "its assignment so the agent can audit and re-bucket: "
            + _SIGNAL_USED_DESCRIPTION
            + ", section_assigned, and a signals payload "
            "with intent_counts (Counter of intent->n_rows), "
            "intent_total_rows, community_id, community_share "
            "(fraction of working set in the same community), "
            "share_tier ('core' | 'supporting' | 'peripheral' | null), "
            "is_modal_community, modal_community_id. Each entry also "
            "carries alternative_sections — the other sections the "
            "paper could have landed in given the available signals "
            "— which an agent can feed straight back into "
            "section_overrides to re-bucket. Each section also "
            "emits a structured 'theme' object — "
            "{communities:[{community_id,label,top_keywords,"
            "top_arxiv_classes,paper_count_in_section}, ... top 3 "
            "by count], top_papers_by_citation:[{bibcode,title,"
            "citation_count}, ... top 3]} — meant as raw signals "
            "for the agent to compose its own theme description; "
            "do NOT treat communities[].label as prose, it is a "
            "metadata tag. The legacy 'theme_summary' string "
            "(concatenated community labels) is preserved alongside "
            "for backwards compat but is deprecated in favor of "
            "'theme'. Returns unattributed_bibcodes for papers "
            "with no signal, and a coverage block reporting how "
            "many bibcodes had each kind of signal. Each "
            "cited_papers row also carries 'first_author' "
            "(populated from papers.first_author) so the agent "
            "can attribute citations without parsing the "
            "abstract. Two opt-in grounding flags expand the "
            "payload: include_full_abstracts=true adds an "
            "'abstract_full' field with the untruncated abstract "
            "alongside 'abstract_snippet'; "
            "include_citation_contexts=true adds a "
            "'citation_excerpts' field (up to 3 rows from "
            "citation_contexts) on intent_modal-bucketed papers "
            "so the agent sees at least one actual citing "
            "sentence per intent-attributed paper. Falls through "
            "to the session's focused papers when "
            "working_set_bibcodes is omitted (mirrors find_gaps). "
            "Use after lit_review or a retrieval+characterization "
            "loop when you want a scaffold to write the synthesis."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "working_set_bibcodes": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Bibcodes to synthesise. Omit to fall "
                        "through to session-focused papers (capped "
                        "at 200)."
                    ),
                },
                "sections": {
                    "type": "array",
                    "items": {"type": "string"},
                    "default": list(_SYNTH_DEFAULT_SECTIONS),
                    "description": (
                        "Section names to emit, in order. Default: "
                        "background, methods, results, "
                        "open_questions. Custom names that don't "
                        "appear in the intent map only receive "
                        "papers via the community fall-through."
                    ),
                },
                "max_papers_per_section": {
                    "type": "integer",
                    "default": 8,
                    "description": (
                        "Cap on per-section cited_papers length. "
                        "Sorted by year desc, bibcode asc for "
                        "deterministic output."
                    ),
                },
                "section_overrides": {
                    "type": "object",
                    "additionalProperties": {"type": "string"},
                    "maxProperties": 200,
                    "description": (
                        "Optional {bibcode: section_name} mapping "
                        "that pins specific papers to specific "
                        "sections, overriding the intent / community "
                        "rules. The override target must appear in "
                        "the requested sections list — out-of-set "
                        "targets are ignored and the paper falls "
                        "back to the normal rules. Use this to "
                        "re-bucket papers based on the signals "
                        "exposed in a previous synthesize_findings "
                        "call."
                    ),
                },
                "include_full_abstracts": {
                    "type": "boolean",
                    "default": False,
                    "description": (
                        "When true, every cited_papers row also "
                        "carries an 'abstract_full' field with the "
                        "untruncated papers.abstract text alongside "
                        "the existing 'abstract_snippet'. Use when "
                        "you need full abstracts for grounded "
                        "synthesis without round-tripping through "
                        "read_paper. Default false (preserves the "
                        "default wire format)."
                    ),
                },
                "include_citation_contexts": {
                    "type": "boolean",
                    "default": False,
                    "description": (
                        "When true, papers attributed via "
                        "signal_used='intent_modal' carry a "
                        "'citation_excerpts' field with up to 3 "
                        "{context_text, intent, citing_bibcode} "
                        "rows from citation_contexts so the agent "
                        "can see at least one actual citing "
                        "sentence per intent-bucketed paper. "
                        "Papers attributed via community / "
                        "override / citation-count fallback do "
                        "NOT receive excerpts (their bucket "
                        "assignment did not come from a "
                        "citation_contexts row). Default false."
                    ),
                },
            },
        },
    ),
    # --- PRD section-embeddings-mcp-consolidation: section_retrieval ---
    dict(
        name="section_retrieval",
        description=(
            "Retrieve passages at section granularity. Encodes the query "
            "with a local nomic-embed-text-v1.5 model and runs a hybrid "
            "search that fuses (a) dense HNSW search over "
            "section_embeddings (halfvec(1024)) and (b) BM25 over the "
            "papers_fulltext.sections tsvector via Reciprocal Rank Fusion "
            "(k=60). Returns a ranked list of section hits each carrying "
            "bibcode, section_heading, a snippet (<=500 chars of the "
            "section text), the fused score, and a canonical_url for "
            "attribution. Use search instead when you need paper-level "
            "results rather than section-level passages; use read_paper "
            "to read a specific section once you have the bibcode."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural-language passage query.",
                },
                "k": {
                    "type": "integer",
                    "default": 10,
                    "description": "Max number of section hits to return.",
                },
                "filters": _SECTION_FILTERS_SCHEMA,
            },
            "required": ["query"],
        },
    ),
    # --- PRD nanopub-claim-extraction: read_paper_claims ---
    dict(
        name="read_paper_claims",
        description=(
            "Return claims extracted from a single paper, ordered by "
            "their position in the source text "
            "(section_index, paragraph_index, char_span_start). Each "
            "claim carries its provenance contract (bibcode + section/"
            "paragraph/char span), the verbatim claim_text, a "
            "claim_type tag (factual / methodological / comparative / "
            "speculative / cited_from_other), an optional "
            "subject-predicate-object decomposition, and an optional "
            "extractor confidence. Use find_claims instead when you "
            "want to search across all papers by claim text."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "bibcode": {
                    "type": "string",
                    "description": "ADS bibcode of the paper.",
                },
                "claim_type": {
                    "type": "string",
                    "enum": [
                        "factual",
                        "methodological",
                        "comparative",
                        "speculative",
                        "cited_from_other",
                    ],
                    "description": ("Optional filter on claim_type. Omit to return all types."),
                },
                "limit": {
                    "type": "integer",
                    "default": 20,
                    "description": (
                        "Max claims to return. Default 20 (shared "
                        "convention); cap 500 — a single paper can carry "
                        "many claims, so bulk per-paper reads justify the "
                        "higher ceiling (audit §5 exception)."
                    ),
                },
            },
            "required": ["bibcode"],
        },
    ),
    # --- PRD nanopub-claim-extraction: find_claims ---
    dict(
        name="find_claims",
        description=(
            "Search across all extracted paper_claims by natural-"
            "language query, ranked by ts_rank against the GIN "
            "to_tsvector('english', claim_text) index. The query is "
            "treated as a phrase via plainto_tsquery (no operators "
            "required). Optional entity_id filters to claims linked "
            "to that entity as either subject or object. Optional "
            "claim_type narrows by category. Use read_paper_claims "
            "instead when you already know the bibcode and want all "
            "claims from one paper."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "Natural-language query over claim text. "
                        "Treated as a phrase by plainto_tsquery."
                    ),
                },
                "claim_type": {
                    "type": "string",
                    "enum": [
                        "factual",
                        "methodological",
                        "comparative",
                        "speculative",
                        "cited_from_other",
                    ],
                    "description": ("Optional filter on claim_type. Omit to return all types."),
                },
                "entity_id": {
                    "type": "integer",
                    "description": (
                        "Optional entity id; restricts results to "
                        "claims where linked_entity_subject_id OR "
                        "linked_entity_object_id matches."
                    ),
                },
                "limit": {
                    "type": "integer",
                    "default": 20,
                    "description": (
                        "Max claims to return. Default 20 (shared "
                        "convention); cap 500 (audit §5 exception — "
                        "bulk claim search)."
                    ),
                },
            },
            "required": ["query"],
        },
    ),
)

_CHUNK_SEARCH_SPEC: dict[str, Any] = dict(
    name="chunk_search",
    description=(
        "Chunk-grain semantic search over the scix_chunks_v1 "
        "Qdrant collection. Encodes the query with INDUS "
        "(768-dim, mean-pooled) and runs an ANN query against "
        "section-aware sliding-window chunks of paper bodies. "
        "Best for method/dataset queries and other narrow, "
        "passage-level questions where the relevant text is a "
        "few sentences inside a section. Use search "
        "(abstract-level) when broader paper-level relevance "
        "is enough; use section_retrieval when section-level "
        "(rather than chunk-level) granularity is preferred. "
        "Filters cover year window, arxiv_class, "
        "community_id_med (medium-resolution Leiden "
        "community), section_heading (canonical norm), and "
        "bibcode (restrict to specific papers)."
    ),
    inputSchema={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": (
                    "Natural-language passage query. "
                    "Encoded with INDUS (mean pooling, "
                    "768-dim) before the ANN call."
                ),
            },
            "filters": {
                "type": "object",
                "description": (
                    "Optional payload filters; all keys " "AND-combined. Omit any to disable."
                ),
                "properties": {
                    "year_min": {
                        "type": "integer",
                        "description": "Earliest paper year (inclusive).",
                    },
                    "year_max": {
                        "type": "integer",
                        "description": "Latest paper year (inclusive).",
                    },
                    "arxiv_class": {
                        "description": (
                            "arXiv class filter; accepts a "
                            "string or a list of strings "
                            "(e.g. 'astro-ph.EP' or "
                            "['astro-ph.EP', 'astro-ph.GA'])."
                        ),
                        "oneOf": [
                            {"type": "string"},
                            {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                        ],
                    },
                    "community_id_med": {
                        "description": (
                            "Medium-resolution Leiden "
                            "community id; integer or list "
                            "of integers."
                        ),
                        "oneOf": [
                            {"type": "integer"},
                            {
                                "type": "array",
                                "items": {"type": "integer"},
                            },
                        ],
                    },
                    "section_heading": {
                        "description": (
                            "Canonical normalized section "
                            "heading(s) (e.g. 'methods', "
                            "'results'). Pass-through to the "
                            "section_heading_norm payload "
                            "filter."
                        ),
                        "oneOf": [
                            {"type": "string"},
                            {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                        ],
                    },
                    "bibcode": {
                        "description": ("Restrict to specific paper bibcode(s); string or list."),
                        "oneOf": [
                            {"type": "string"},
                            {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                        ],
                    },
                },
            },
            "limit": {
                "type": "integer",
                "default": 20,
                "description": ("Max chunk hits to return; clamped to [1, 100]."),
            },
        },
        "required": ["query"],
    },
)
