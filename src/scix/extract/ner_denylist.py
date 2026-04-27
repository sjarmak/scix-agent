"""Denylist of noisy NER canonical-name + entity-type pairs.

Per bead ``scix_experiments-eq95``: a small mechanical filter applied at
query time keeps the most obviously-spurious entity matches from
surfacing to agents. This is a precision lift on the bottom-tier types
(``dataset``, ``method``, ``software``, ``gene``, ``organism``) without
re-running the model or rewriting any document_entities rows.

Verified against prod 2026-04-26 — these three entries alone produce
163K spurious mentions:

    canonical_name      entity_type    mentions
    'experimental data' 'dataset'      80,540
    'data'              'dataset'      55,423
    'method'            'method'       26,976

The remainder of the seed list comes from the bead's quality-assessment
notes plus a same-day cross-check against ``document_entities``. Ratios
are dominated by gliner-sourced rows (gliner's wide-net recall produces
generic-word matches its dictionary-based predecessors didn't).

To extend the denylist:

1. Add a ``(canonical_name_lower, entity_type)`` tuple to ``_DENYLIST``.
2. The match is exact + case-insensitive on canonical_name and exact on
   entity_type. ``protein`` under ``gene`` and ``protein`` under
   ``chemical`` are independent entries — list both if both are noisy.
3. Run ``tests/test_ner_denylist.py`` to verify the new entry is
   covered.
4. Document the discovery context in this docstring.

Design notes:

- Filter is *applied at the agent surface*, not at extraction time. The
  document_entities rows are still written; only the entity tool and
  ``get_paper(include_entities=True)`` hide them. A future insert-time
  filter (eq95's secondary scope) is a separate, more invasive change
  that requires a backfill DELETE on existing rows.
- ``(canonical_name, entity_type)`` granularity means the denylist
  doesn't accidentally hide a *different* entity that happens to share
  a canonical name across types — e.g. ``'protein'`` is denylisted as a
  ``gene`` (the noisy interpretation) but a hypothetical
  ``'protein dynamics'`` ``method`` entry would still surface.
"""
from __future__ import annotations

#: ``(canonical_name_lower, entity_type)`` pairs that should never
#: surface from any agent-facing entity query.
#:
#: Generic-word false positives observed in production (2026-04-26):
_DENYLIST: frozenset[tuple[str, str]] = frozenset(
    {
        # MUST-include per the bead acceptance criteria — 163K combined mentions.
        ("experimental data", "dataset"),
        ("data", "dataset"),
        ("method", "method"),
        # Type-confusion: 'protein' is not a gene; 'dna' is a molecule.
        ("protein", "gene"),
        ("proteins", "gene"),
        ("dna", "gene"),
        ("genes", "gene"),
        # 'cells' tagged as organism is anatomy, not a species.
        ("cells", "organism"),
        # Generic software words (heuristic-noisy).
        ("code", "software"),
        ("library", "software"),
        ("program", "software"),
        # Generic dataset words.
        ("training samples", "dataset"),
        ("kinematic data", "dataset"),
        ("library", "dataset"),
    }
)


def is_denylisted(canonical_name: str | None, entity_type: str | None) -> bool:
    """Return True if this (canonical_name, entity_type) is on the denylist.

    Both arguments are normalized internally — callers don't need to
    pre-lowercase canonical_name. ``None`` or empty inputs return False
    (the denylist is exclusion-by-match; missing inputs are not noise).
    """
    if not canonical_name or not entity_type:
        return False
    return (canonical_name.casefold().strip(), entity_type) in _DENYLIST


def filter_denylisted_rows(
    rows: list[dict],
    *,
    name_key: str = "canonical_name",
    type_key: str = "entity_type",
) -> list[dict]:
    """Drop rows whose (name_key, type_key) values are on the denylist.

    Convenience wrapper for in-memory filtering of result lists. Returns
    a new list; does not mutate the input.
    """
    return [
        r
        for r in rows
        if not is_denylisted(r.get(name_key), r.get(type_key))
    ]
