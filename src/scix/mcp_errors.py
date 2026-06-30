"""Closed catalog of MCP error codes (beads scix_experiments-ir2h / -x5jg).

Every error envelope returned by the MCP tool surface carries a stable,
machine-readable ``error_code`` alongside the human-readable ``error`` message,
so consumers can branch on failure modes without parsing prose. This module is
the single source of truth for the closed set of codes; the conformance test
asserts every ``error_code`` emitted by the server is a member of
:data:`CATALOG`.

The envelope convention (bead ``scix_experiments-x5jg``): ``error_code`` lives
at the response root and is machine-readable; ``error`` is for humans.
"""

from __future__ import annotations


class ErrorCode:
    """String constants for the closed MCP error-code catalog.

    Values are plain strings so they serialize directly via ``json.dumps``.
    Members are grouped by failure class; :data:`CATALOG` is the authoritative
    enumeration derived from these constants.
    """

    # --- Input validation (generic, used across all tools) ---
    MISSING_REQUIRED_PARAMS = "missing_required_params"
    INVALID_PARAM_TYPE = "invalid_param_type"
    INVALID_PARAM_VALUE = "invalid_param_value"

    # --- Input validation (pre-existing per-parameter codes) ---
    INVALID_MODE = "invalid_mode"
    INVALID_DIRECTION = "invalid_direction"
    INVALID_METHOD = "invalid_method"
    INVALID_ACTION = "invalid_action"
    INVALID_LIMIT = "invalid_limit"
    INVALID_FILTERS = "invalid_filters"
    INVALID_SCOPE = "invalid_scope"
    ENTITY_LEGACY_EXTRACTION_TYPE = "entity_legacy_extraction_type"

    # --- Query guards ---
    UNSCOPED_BROAD_QUERY = "unscoped_broad_query"
    COMMUNITY_EXPAND_NO_SEED = "community_expand_no_seed"
    SEED_TOO_BROAD = "seed_too_broad"

    # --- Routing ---
    UNKNOWN_TOOL = "unknown_tool"
    TOOL_REMOVED = "tool_removed"

    # --- Backend / dependency unavailable ---
    DEPENDENCY_MISSING = "dependency_missing"
    VECTOR_INDEX_UNAVAILABLE = "vector_index_unavailable"
    QDRANT_DISABLED = "qdrant_disabled"

    # --- Operation failures ---
    ENCODE_FAILED = "encode_failed"
    QDRANT_FAILED = "qdrant_failed"
    DENSE_RETRIEVE_FAILED = "dense_retrieve_failed"
    BM25_RETRIEVE_FAILED = "bm25_retrieve_failed"
    INTERNAL_ERROR = "internal_error"


CATALOG: frozenset[str] = frozenset(
    value
    for name, value in vars(ErrorCode).items()
    if not name.startswith("_") and isinstance(value, str)
)
