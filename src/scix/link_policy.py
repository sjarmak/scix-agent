"""Pure classifier for entity link policy.

Determines what tier of linking each entity is eligible for, based on
provenance (source, scheme) and ambiguity classification.  The five
policy values map to ``entities.link_policy``::

    'open'             — eligible for tier-1 keyword matching
    'context_required' — needs context validation (tier-2+)
    'llm_only'         — requires LLM judgment (tier-3+)
    'banned'           — no automatic linking

This module does NOT touch the database.  The script
``scripts/set_link_policy.py`` is the DB-touching wrapper.
"""

from __future__ import annotations

from typing import Any, Literal

LinkPolicy = Literal["open", "context_required", "llm_only", "banned"]

# ---------------------------------------------------------------------------
# Rule constants
# ---------------------------------------------------------------------------

# GCMD "providers" scheme entities are data-center codes (e.g.
# DOC/NOAA/NESDIS, CA/NFLD/FRA/SOIL), not scientific concepts.
# Their short-name aliases collide with common English words.
GCMD_PROVIDER_SCHEME: str = "providers"

# GCMD hierarchical entities use " > " to encode parent context
# (e.g. "SEA ICE > SALINITY").  Matching on the leaf term alone
# without verifying the parent domain produces cross-domain FPs.
GCMD_HIERARCHY_MARKER: str = " > "

# SPASE observables (Frequency, Entropy, ModeAmplitude) are
# heliophysics-specific.  Keyword matching without domain context
# (arxiv_class check) produces FPs in other fields.
SPASE_SOURCE: str = "spase"


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------


def determine_link_policy(
    source: str,
    canonical_name: str,
    ambiguity_class: str | None,
    properties: dict[str, Any],
) -> LinkPolicy:
    """Return the link policy for an entity.

    Rule priority (first match wins):

    1. ``ambiguity_class == 'banned'`` → ``'banned'``
    2. GCMD providers scheme → ``'banned'``
    3. GCMD hierarchical (canonical contains ``" > "``) → ``'context_required'``
    4. SPASE source → ``'context_required'``
    5. Everything else → ``'open'``

    Args:
        source: Entity source ontology (e.g. ``'gcmd'``, ``'spase'``).
        canonical_name: The entity's canonical label.
        ambiguity_class: Current ``entities.ambiguity_class`` value, or
            ``None`` if not yet classified.
        properties: The entity's ``properties`` JSONB dict.

    Returns:
        One of ``'open'``, ``'context_required'``, ``'banned'``.
        (``'llm_only'`` is reserved in :data:`LinkPolicy` but no rule
        currently produces it.)
    """
    # Rule 1: propagate Zipf / short-name bans.
    if ambiguity_class == "banned":
        return "banned"

    # Rule 2: GCMD data-center codes are never valid keyword matches.
    if source == "gcmd" and properties.get("gcmd_scheme") == GCMD_PROVIDER_SCHEME:
        return "banned"

    # Rule 3: GCMD hierarchical entities need parent-context validation.
    if source == "gcmd" and GCMD_HIERARCHY_MARKER in canonical_name:
        return "context_required"

    # Rule 4: SPASE entities need domain filtering.
    if source == SPASE_SOURCE:
        return "context_required"

    # Rule 5: default — eligible for tier-1.
    return "open"
