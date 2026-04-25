"""ResearchScope — JSON dict threaded through every Deep Search tool call.

Implements MH-5 of the SciX Deep Search v1 PRD
(``docs/prd/scix_deep_search_v1.md``). Provides a single source of truth for
the scope shape plus deterministic helpers that derive parameterised SQL
``WHERE`` clauses from a scope.

Design notes
------------

* **Frozen dataclass.** Per ``CLAUDE.md`` and ``rules/python/coding-style.md``,
  immutable data structures are preferred. ``ResearchScope`` is a
  ``@dataclass(frozen=True)`` with all-optional fields.

* **psycopg-style ``%s`` parameters.** All clauses use positional ``%s``
  placeholders matching the rest of the codebase (see ``src/scix/db.py``).

* **table_aliases.** Callers map logical table names to the SQL alias used in
  their query. Supported keys (all optional — only the keys needed by the
  active scope fields are required):

    - ``papers`` → alias for the ``papers`` table (default ``p``)
    - ``citation_edges`` → alias for ``citation_edges`` (default ``ce``)
    - ``citation_contexts`` → alias for ``citation_contexts`` (default ``cc``)
    - ``paper_metrics`` → alias for ``paper_metrics`` (default ``pm``);
      required when ``community_ids`` or ``leiden_resolution`` is set, since
      the Leiden community columns live on ``paper_metrics`` (per migration
      ``006_graph_metrics.sql``).

* **exclude_authors / exclude_funders.** We use ``NOT EXISTS`` subqueries
  against forward-looking join tables ``papers_authors(paper_id, author)`` and
  ``papers_funders(paper_id, funder)``. Rationale: ``papers.authors`` is a
  ``TEXT[]`` column (per migration ``001_initial_schema.sql``), but a clean
  agent-facing exclusion list is best modelled by a normalised join table —
  the same shape SH-1 (``venues`` lookup) follows. Until those tables ship,
  the generated subqueries return zero rows and the exclusion is a no-op
  (which is the safe failure mode: NOT EXISTS over an empty table is always
  TRUE, so no papers are excluded). Documented here rather than enforced via
  schema check because callers may pre-create the tables independently.

* **min_venue_tier.** References ``papers.venue_tier``, which is a
  forward-looking column (PRD SH-1). Until SH-1 ships, the column is absent
  and the clause will fail at execution — callers must therefore not pass
  ``min_venue_tier`` until SH-1 lands. Documented but not silently dropped;
  see ``scope_to_sql_clauses`` raising ``KeyError`` if the alias is missing.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any, Literal

LeidenResolution = Literal["coarse", "medium", "fine"]
"""Allowed values for ``ResearchScope.leiden_resolution``."""

_VALID_LEIDEN_RESOLUTIONS: frozenset[str] = frozenset({"coarse", "medium", "fine"})

_DEFAULT_ALIASES: dict[str, str] = {
    "papers": "p",
    "citation_edges": "ce",
    "citation_contexts": "cc",
    "paper_metrics": "pm",
}


@dataclass(frozen=True)
class ResearchScope:
    """Optional filters threaded through every Deep Search tool call.

    Every field is optional; an empty ``ResearchScope()`` denotes "no
    constraints". See module docstring for the SQL-derivation contract.

    Attributes:
        community_ids: Restrict to papers in any of these Leiden community
            ids (resolved against ``leiden_resolution``).
        year_window: Inclusive ``(start, end)`` filter on ``papers.year``.
        methodology_class: Restrict to papers with this methodology class
            (forward-looking column ``papers.methodology_class``).
        instruments: Restrict to papers mentioning any of these instruments
            (forward-looking column ``papers.instruments TEXT[]``).
        exclude_authors: Drop papers authored by any of these names.
        exclude_funders: Drop papers funded by any of these grant agencies.
        min_venue_tier: Keep only papers whose ``venue_tier`` is ``<=`` this
            value (lower tier = better venue per PRD SH-1; tier 1 = top,
            tier 4 = predatory). No-op until SH-1 ships the column.
        leiden_resolution: Which Leiden resolution to resolve
            ``community_ids`` against. Required when ``community_ids`` is
            set (defaults to ``'medium'``).
    """

    community_ids: list[int] | None = None
    year_window: tuple[int, int] | None = None
    methodology_class: str | None = None
    instruments: list[str] | None = None
    exclude_authors: list[str] | None = None
    exclude_funders: list[str] | None = None
    min_venue_tier: int | None = None
    leiden_resolution: LeidenResolution | None = None

    def __post_init__(self) -> None:
        if (
            self.leiden_resolution is not None
            and self.leiden_resolution not in _VALID_LEIDEN_RESOLUTIONS
        ):
            raise ValueError(
                f"leiden_resolution must be one of "
                f"{sorted(_VALID_LEIDEN_RESOLUTIONS)}; got "
                f"{self.leiden_resolution!r}"
            )
        if self.year_window is not None:
            start, end = self.year_window
            if start > end:
                raise ValueError(
                    f"year_window start ({start}) must be <= end ({end})"
                )


def _alias(table_aliases: dict[str, str], logical: str) -> str:
    """Resolve ``logical`` to the caller's SQL alias, falling back to the
    canonical default. Raises ``KeyError`` only if neither is provided.
    """
    if logical in table_aliases:
        return table_aliases[logical]
    if logical in _DEFAULT_ALIASES:
        return _DEFAULT_ALIASES[logical]
    raise KeyError(
        f"No alias provided for logical table {logical!r} and no default known"
    )


def scope_to_sql_clauses(
    scope: ResearchScope,
    table_aliases: dict[str, str],
) -> tuple[str, list[Any]]:
    """Derive a parameterised ``WHERE`` clause fragment from ``scope``.

    Args:
        scope: The active ResearchScope.
        table_aliases: Mapping of logical table names (``'papers'``,
            ``'citation_edges'``, ``'citation_contexts'``, ``'paper_metrics'``)
            to the SQL alias the caller is using in their query. Missing
            keys fall back to the canonical defaults defined in this module.

    Returns:
        A tuple ``(clause, params)``. ``clause`` is a SQL fragment with
        ``%s`` placeholders (psycopg style); it is empty when ``scope`` has
        no active fields. ``params`` is the matching ordered parameter list,
        suitable to extend an existing ``cur.execute(sql, params)`` call.

        The returned clause does NOT include a leading ``WHERE`` or
        ``AND`` — the caller is expected to compose it into their query
        with the appropriate connector. Multiple sub-clauses are joined
        with ``AND``.
    """
    parts: list[str] = []
    params: list[Any] = []

    if scope.year_window is not None:
        papers = _alias(table_aliases, "papers")
        start, end = scope.year_window
        parts.append(f"{papers}.year >= %s AND {papers}.year <= %s")
        params.extend([start, end])

    if scope.community_ids is not None:
        # Default to 'medium' when leiden_resolution unset, per Phase B
        # default documented in MEMORY.md (semantic medium is the working
        # signal). Required because we have to pick a column.
        resolution = scope.leiden_resolution or "medium"
        pm = _alias(table_aliases, "paper_metrics")
        column = f"community_id_{resolution}"
        parts.append(f"{pm}.{column} = ANY(%s)")
        params.append(list(scope.community_ids))

    if scope.methodology_class is not None:
        papers = _alias(table_aliases, "papers")
        parts.append(f"{papers}.methodology_class = %s")
        params.append(scope.methodology_class)

    if scope.instruments is not None:
        papers = _alias(table_aliases, "papers")
        # papers.instruments is a forward-looking TEXT[] column; use
        # array-overlap to keep papers mentioning ANY of the supplied
        # instruments.
        parts.append(f"{papers}.instruments && %s::text[]")
        params.append(list(scope.instruments))

    if scope.exclude_authors is not None:
        papers = _alias(table_aliases, "papers")
        # NOT EXISTS subquery against forward-looking papers_authors join
        # table. See module docstring for the rationale and the no-op
        # behaviour while the table is absent.
        parts.append(
            f"NOT EXISTS ("
            f"SELECT 1 FROM papers_authors pa "
            f"WHERE pa.paper_id = {papers}.id "
            f"AND pa.author = ANY(%s)"
            f")"
        )
        params.append(list(scope.exclude_authors))

    if scope.exclude_funders is not None:
        papers = _alias(table_aliases, "papers")
        # NOT EXISTS subquery against forward-looking papers_funders join
        # table — same shape as exclude_authors.
        parts.append(
            f"NOT EXISTS ("
            f"SELECT 1 FROM papers_funders pf "
            f"WHERE pf.paper_id = {papers}.id "
            f"AND pf.funder = ANY(%s)"
            f")"
        )
        params.append(list(scope.exclude_funders))

    if scope.min_venue_tier is not None:
        papers = _alias(table_aliases, "papers")
        # Per PRD SH-1: lower tier number = better venue. ``min_venue_tier``
        # is the worst tier we'll keep, so we filter ``venue_tier <= N``.
        # No-op until SH-1 ships the column; will raise at execution if the
        # column is missing.
        parts.append(f"{papers}.venue_tier <= %s")
        params.append(scope.min_venue_tier)

    clause = " AND ".join(parts)
    return clause, params


def scope_from_dict(d: dict[str, Any]) -> ResearchScope:
    """Construct a :class:`ResearchScope` from a JSON-RPC-friendly dict.

    Tolerant of missing keys (treated as ``None``) and of JSON's preference
    for ``list`` over ``tuple`` (year_window is normalised back to a tuple).
    Unknown keys raise ``ValueError`` so client/server schema drift is
    caught loudly rather than silently dropped.
    """
    if not isinstance(d, dict):
        raise TypeError(f"scope_from_dict requires a dict, got {type(d).__name__}")

    known = {f.name for f in fields(ResearchScope)}
    unknown = set(d) - known
    if unknown:
        raise ValueError(
            f"Unknown ResearchScope keys: {sorted(unknown)}. "
            f"Allowed keys: {sorted(known)}"
        )

    year_window = d.get("year_window")
    if year_window is not None:
        if not (isinstance(year_window, (list, tuple)) and len(year_window) == 2):
            raise ValueError(
                f"year_window must be a 2-element list/tuple, got {year_window!r}"
            )
        year_window = (int(year_window[0]), int(year_window[1]))

    community_ids = d.get("community_ids")
    if community_ids is not None:
        community_ids = [int(x) for x in community_ids]

    instruments = d.get("instruments")
    if instruments is not None:
        instruments = [str(x) for x in instruments]

    exclude_authors = d.get("exclude_authors")
    if exclude_authors is not None:
        exclude_authors = [str(x) for x in exclude_authors]

    exclude_funders = d.get("exclude_funders")
    if exclude_funders is not None:
        exclude_funders = [str(x) for x in exclude_funders]

    min_venue_tier = d.get("min_venue_tier")
    if min_venue_tier is not None:
        min_venue_tier = int(min_venue_tier)

    methodology_class = d.get("methodology_class")
    if methodology_class is not None:
        methodology_class = str(methodology_class)

    leiden_resolution = d.get("leiden_resolution")
    # Pass through as-is; __post_init__ validates against the allowed set.

    return ResearchScope(
        community_ids=community_ids,
        year_window=year_window,
        methodology_class=methodology_class,
        instruments=instruments,
        exclude_authors=exclude_authors,
        exclude_funders=exclude_funders,
        min_venue_tier=min_venue_tier,
        leiden_resolution=leiden_resolution,
    )


def scope_to_dict(s: ResearchScope) -> dict[str, Any]:
    """Serialise a :class:`ResearchScope` to a JSON-RPC-friendly dict.

    ``year_window`` is emitted as a 2-element list (JSON has no tuple
    type). ``None`` fields are preserved so the wire shape is stable and
    self-documenting; clients that prefer compact payloads can strip them.
    """
    out: dict[str, Any] = asdict(s)
    if s.year_window is not None:
        out["year_window"] = [s.year_window[0], s.year_window[1]]
    return out


__all__ = [
    "LeidenResolution",
    "ResearchScope",
    "scope_from_dict",
    "scope_to_dict",
    "scope_to_sql_clauses",
]
