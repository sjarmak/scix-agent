"""Generic loader and types for the cross-discipline concept substrate.

Shared infrastructure for migration ``056_concepts_vocabularies.sql``: the
``vocabularies``, ``concepts``, and ``concept_relationships`` tables.

The pattern parallels ``scix.uat`` (UAT keeps its own legacy tables) but
generalizes: every vocabulary loader produces ``Concept`` /
``ConceptRelationship`` records and a ``Vocabulary`` metadata row, then
hands them to :func:`load_vocabulary`. COPY + staging tables make the load
fast and idempotent.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field

import psycopg

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Vocabulary:
    """Metadata for a single source vocabulary (license + provenance)."""

    vocabulary: str
    name: str
    license: str
    source_url: str
    description: str | None = None
    license_url: str | None = None
    homepage_url: str | None = None
    version: str | None = None
    properties: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class Concept:
    """A single concept within a vocabulary."""

    vocabulary: str
    concept_id: str
    preferred_label: str
    alternate_labels: tuple[str, ...] = ()
    definition: str | None = None
    external_uri: str | None = None
    level: int | None = None
    properties: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class ConceptRelationship:
    """A parent → child relationship within one vocabulary."""

    vocabulary: str
    parent_id: str
    child_id: str
    relationship: str = "broader"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _pg_text_array(values: tuple[str, ...] | list[str]) -> str:
    """Format a sequence of strings as a PostgreSQL text array literal."""
    if not values:
        return "{}"
    escaped: list[str] = []
    for v in values:
        v = v.replace("\\", "\\\\").replace('"', '\\"')
        escaped.append(f'"{v}"')
    return "{" + ",".join(escaped) + "}"


# ---------------------------------------------------------------------------
# Vocabulary metadata
# ---------------------------------------------------------------------------


_VOCAB_UPSERT_SQL = """
    INSERT INTO vocabularies
        (vocabulary, name, description, license, license_url, homepage_url,
         source_url, version, record_count, properties)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb)
    ON CONFLICT (vocabulary) DO UPDATE SET
        name         = EXCLUDED.name,
        description  = EXCLUDED.description,
        license      = EXCLUDED.license,
        license_url  = EXCLUDED.license_url,
        homepage_url = EXCLUDED.homepage_url,
        source_url   = EXCLUDED.source_url,
        version      = EXCLUDED.version,
        record_count = EXCLUDED.record_count,
        properties   = EXCLUDED.properties,
        ingested_at  = now()
"""


def upsert_vocabulary(
    conn: psycopg.Connection,
    vocab: Vocabulary,
    record_count: int,
) -> None:
    """Insert or refresh a row in ``vocabularies``."""
    with conn.cursor() as cur:
        cur.execute(
            _VOCAB_UPSERT_SQL,
            (
                vocab.vocabulary,
                vocab.name,
                vocab.description,
                vocab.license,
                vocab.license_url,
                vocab.homepage_url,
                vocab.source_url,
                vocab.version,
                record_count,
                json.dumps(vocab.properties),
            ),
        )


# ---------------------------------------------------------------------------
# Concepts (staging COPY + merge)
# ---------------------------------------------------------------------------


_CONCEPT_STAGING_DROP = "DROP TABLE IF EXISTS _concept_staging"
_CONCEPT_STAGING_CREATE = "CREATE TEMP TABLE _concept_staging (LIKE concepts INCLUDING DEFAULTS)"

_CONCEPT_MERGE_SQL = """
    WITH inserted AS (
        INSERT INTO concepts
            (vocabulary, concept_id, preferred_label, alternate_labels,
             definition, external_uri, level, properties)
        SELECT vocabulary, concept_id, preferred_label, alternate_labels,
               definition, external_uri, level, properties
        FROM _concept_staging
        ON CONFLICT (vocabulary, concept_id) DO UPDATE SET
            preferred_label  = EXCLUDED.preferred_label,
            alternate_labels = EXCLUDED.alternate_labels,
            definition       = EXCLUDED.definition,
            external_uri     = EXCLUDED.external_uri,
            level            = EXCLUDED.level,
            properties       = EXCLUDED.properties
        RETURNING 1
    )
    SELECT COUNT(*) FROM inserted
"""


def load_concepts(conn: psycopg.Connection, concepts: list[Concept]) -> int:
    """Load ``Concept`` records via TEMP-staging + COPY + upsert."""
    if not concepts:
        return 0

    t0 = time.monotonic()
    with conn.cursor() as cur:
        cur.execute(_CONCEPT_STAGING_DROP)
        cur.execute(_CONCEPT_STAGING_CREATE)
        with cur.copy(
            "COPY _concept_staging "
            "(vocabulary, concept_id, preferred_label, alternate_labels, "
            "definition, external_uri, level, properties) FROM STDIN"
        ) as copy:
            for c in concepts:
                copy.write_row(
                    (
                        c.vocabulary,
                        c.concept_id,
                        c.preferred_label,
                        _pg_text_array(c.alternate_labels),
                        c.definition,
                        c.external_uri,
                        c.level,
                        json.dumps(c.properties),
                    )
                )
        cur.execute(_CONCEPT_MERGE_SQL)
        upserted = cur.fetchone()[0]

    elapsed = time.monotonic() - t0
    logger.info("concepts: %d upserted in %.1fs", upserted, elapsed)
    return upserted


# ---------------------------------------------------------------------------
# Relationships (staging COPY + merge, FK-safe)
# ---------------------------------------------------------------------------


_REL_STAGING_DROP = "DROP TABLE IF EXISTS _concept_rel_staging"
_REL_STAGING_CREATE = (
    "CREATE TEMP TABLE _concept_rel_staging "
    "(vocabulary TEXT, parent_id TEXT, child_id TEXT, relationship TEXT)"
)

_REL_FK_DROP_COUNT_SQL = """
    SELECT COUNT(*) FROM (
        SELECT DISTINCT vocabulary, parent_id, child_id, relationship
        FROM _concept_rel_staging s
        WHERE NOT EXISTS (
            SELECT 1 FROM concepts c
            WHERE c.vocabulary = s.vocabulary AND c.concept_id = s.parent_id)
           OR NOT EXISTS (
            SELECT 1 FROM concepts c
            WHERE c.vocabulary = s.vocabulary AND c.concept_id = s.child_id)
    ) AS dropped
"""

_REL_MERGE_SQL = """
    WITH inserted AS (
        INSERT INTO concept_relationships
            (vocabulary, parent_id, child_id, relationship)
        SELECT DISTINCT s.vocabulary, s.parent_id, s.child_id, s.relationship
        FROM _concept_rel_staging s
        WHERE EXISTS (
            SELECT 1 FROM concepts c
            WHERE c.vocabulary = s.vocabulary AND c.concept_id = s.parent_id)
          AND EXISTS (
            SELECT 1 FROM concepts c
            WHERE c.vocabulary = s.vocabulary AND c.concept_id = s.child_id)
        ON CONFLICT (vocabulary, parent_id, child_id, relationship) DO NOTHING
        RETURNING 1
    )
    SELECT COUNT(*) FROM inserted
"""

_REL_PRESENT_COUNT_SQL = """
    SELECT COUNT(*) FROM concept_relationships r
    WHERE EXISTS (
        SELECT 1 FROM _concept_rel_staging s
        WHERE s.vocabulary   = r.vocabulary
          AND s.parent_id    = r.parent_id
          AND s.child_id     = r.child_id
          AND s.relationship = r.relationship
    )
"""


def load_relationships(
    conn: psycopg.Connection,
    rels: list[ConceptRelationship],
) -> int:
    """Load relationships via TEMP-staging + COPY + FK-safe insert."""
    if not rels:
        return 0

    t0 = time.monotonic()
    with conn.cursor() as cur:
        cur.execute(_REL_STAGING_DROP)
        cur.execute(_REL_STAGING_CREATE)
        with cur.copy(
            "COPY _concept_rel_staging "
            "(vocabulary, parent_id, child_id, relationship) FROM STDIN"
        ) as copy:
            for r in rels:
                copy.write_row((r.vocabulary, r.parent_id, r.child_id, r.relationship))

        cur.execute(_REL_FK_DROP_COUNT_SQL)
        fk_dropped = cur.fetchone()[0]
        if fk_dropped:
            logger.warning(
                "Skipping %d relationships with parent_id or child_id "
                "missing from concepts (FK preflight)",
                fk_dropped,
            )

        cur.execute(_REL_MERGE_SQL)
        newly_inserted = cur.fetchone()[0]

        cur.execute(_REL_PRESENT_COUNT_SQL)
        present = cur.fetchone()[0]

    elapsed = time.monotonic() - t0
    logger.info(
        "concept_relationships: %d new, %d present, %d FK-dropped in %.1fs",
        newly_inserted,
        present,
        fk_dropped,
        elapsed,
    )
    return present


# ---------------------------------------------------------------------------
# Convenience: full per-vocabulary pipeline
# ---------------------------------------------------------------------------


def load_vocabulary(
    conn: psycopg.Connection,
    vocab: Vocabulary,
    concepts: list[Concept],
    rels: list[ConceptRelationship],
) -> tuple[int, int]:
    """Idempotent end-to-end load of one vocabulary into the substrate.

    Order matters: vocabularies row first (FK target), then concepts (FK
    target for relationships), then relationships. Caller commits.
    """
    upsert_vocabulary(conn, vocab, record_count=len(concepts))
    n_concepts = load_concepts(conn, concepts)
    n_rels = load_relationships(conn, rels)
    return n_concepts, n_rels
