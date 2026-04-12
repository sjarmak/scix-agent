"""Pure classifier for entity name ambiguity.

Populates the four buckets of ``entities.ambiguity_class``:

* ``banned`` — name matches the top-20K English words at Zipf >= 3.0,
  or is <= 2 characters long. The classifier considers both the
  canonical name and every alias; if any one of them is banned, the
  whole entity is banned.
* ``homograph`` — name collides with another entity's canonical or alias.
* ``domain_safe`` — canonical is >= 6 characters, the name is unique in
  the graph, and the entity has a single ontology of origin.
* ``unique`` — none of the above.

This module does NOT touch the database. The script
``scripts/classify_entity_ambiguity.py`` is the DB-touching wrapper: it
computes ``source_count`` and ``collision_count`` in SQL/Python and
passes them in here.
"""

from __future__ import annotations

from typing import Literal

from wordfreq import zipf_frequency

AmbiguityClass = Literal["banned", "homograph", "domain_safe", "unique"]

# "Top 20K English words at Zipf >= 3.0" — wordfreq's top-of-distribution
# cutoff. Any name whose lowercased form has zipf_frequency >= 3.0 is
# considered a common English word and banned from auto-linking.
ZIPF_BANNED_THRESHOLD: float = 3.0

# Names of 2 characters or less are banned outright — too short to
# disambiguate (e.g., "a", "IR", "UV").
SHORT_NAME_MAX_LEN: int = 2

# domain_safe requires a canonical name at least this long.
DOMAIN_SAFE_MIN_LEN: int = 6


def is_banned_name(name: str) -> bool:
    """Return True if ``name`` is either too short or a common English word.

    A name is banned if:

    * Its trimmed length is <= 2 characters, OR
    * ``zipf_frequency(name.lower(), "en") >= 3.0`` (top-20K English words).
    """
    if name is None:
        return False
    stripped = name.strip()
    if len(stripped) <= SHORT_NAME_MAX_LEN:
        return True
    if zipf_frequency(stripped.lower(), "en") >= ZIPF_BANNED_THRESHOLD:
        return True
    return False


def classify(
    canonical_name: str,
    aliases: list[str],
    source_count: int,
    collision_count: int,
) -> AmbiguityClass:
    """Classify an entity's ambiguity bucket.

    Args:
        canonical_name: The entity's canonical label.
        aliases: Every alias string for this entity.
        source_count: Distinct source ontologies across the collision
            group, INCLUDING this entity's own source.
        collision_count: Number of OTHER entities that share the
            canonical name or any alias with this entity (case-insensitive).

    Returns:
        One of ``"banned"``, ``"homograph"``, ``"domain_safe"``, ``"unique"``.

    Precedence (highest to lowest): banned > homograph > domain_safe > unique.
    """
    # 1. banned — short or common English word (canonical or any alias).
    if is_banned_name(canonical_name):
        return "banned"
    for alias in aliases:
        if is_banned_name(alias):
            return "banned"

    # 2. homograph — collides with another entity's canonical or alias.
    if collision_count > 0:
        return "homograph"

    # 3. domain_safe — long, unique, single source of origin.
    canonical_stripped = canonical_name.strip()
    if (
        len(canonical_stripped) >= DOMAIN_SAFE_MIN_LEN
        and collision_count == 0
        and source_count == 1
    ):
        return "domain_safe"

    # 4. unique — everything else.
    return "unique"
