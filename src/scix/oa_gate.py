"""OA/preprint gate — single source of truth for body-AI pipelines.

Publisher-agreement risk surface: applying AI tools (NER, embeddings,
citation-context extraction) to non-OA body text is constrained by Wiley /
Springer / etc. TDM clauses. The gate identifies papers that are either
explicitly open-access (``'OPENACCESS' = ANY(papers.property)``) or arxiv
preprints (``array_length(papers.arxiv_class, 1) > 0``); these are safe
to feed into AI pipelines without per-publisher policy review.

The canonical predicate lives in
``migrations/068_papers_is_oa_or_preprint.sql`` as the IMMUTABLE SQL
function ``papers_is_oa_or_preprint(p papers)`` plus a partial expression
index ``idx_papers_is_oa ON papers ((papers_is_oa_or_preprint(papers)))
WHERE body IS NOT NULL``. Body-AI scripts call the function in their
WHERE clauses (``... AND papers_is_oa_or_preprint(p)``) so the planner
can hit the index.

This module is the Python mirror so unit tests can pin predicate
semantics without a live DB, and so any future caller that needs to
classify in-process data can use the same logic.

Keep the SQL function and :func:`is_oa_or_preprint` in lockstep — if you
change one, change the other and re-run ``tests/test_oa_gate.py``.
"""

from __future__ import annotations

#: Canonical SQL function name. Scripts call this in their WHERE clauses;
#: tests pin the migration file against this constant.
SQL_FUNCTION_NAME: str = "papers_is_oa_or_preprint"


def is_oa_or_preprint(
    property_: list[str] | None,
    arxiv_class: list[str] | None,
) -> bool:
    """Return True if the paper is OA or a preprint per the gate predicate.

    Mirrors the SQL function ``papers_is_oa_or_preprint(p papers)``:

        COALESCE('OPENACCESS' = ANY(p.property), FALSE)
        OR COALESCE(array_length(p.arxiv_class, 1) > 0, FALSE)

    NULL inputs are treated as FALSE (matching the SQL ``COALESCE``
    wrap), so a paper with both fields NULL is considered closed-access.
    """
    if property_ is not None and not isinstance(property_, list):
        raise TypeError(f"property_ must be list[str] or None, got {type(property_).__name__}")
    if arxiv_class is not None and not isinstance(arxiv_class, list):
        raise TypeError(f"arxiv_class must be list[str] or None, got {type(arxiv_class).__name__}")

    has_openaccess = property_ is not None and "OPENACCESS" in property_
    has_arxiv_class = arxiv_class is not None and len(arxiv_class) > 0
    return has_openaccess or has_arxiv_class
