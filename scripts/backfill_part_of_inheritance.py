#!/usr/bin/env python3
"""Backfill flagship-instrument document_entities + part_of inheritance.

Problem
-------

The flagship-seed instruments (``NIRSpec``, ``ACIS``, ``MIRI``, ...) have
:class:`entity_relationships` ``part_of`` edges to their parent missions
(``James Webb Space Telescope``, ``Chandra X-ray Observatory``, ...) but have
**zero rows** in :class:`document_entities`. Because the
``SpecificEntityBackend`` in ``scripts/eval_entity_value_props.py`` resolves a
gold entity name to ``entities.id`` and then looks up papers via
``document_entities``, queries like ``spec-004`` (NIRSpec) and ``spec-005``
(ACIS) return zero results. The eval scores them 0.

This script closes the gap by:

1. Iterating over ``part_of`` edges from ``flagship_seed`` (instrument →
   mission).
2. For each instrument, building a precision-first surface-form list — the
   canonical name plus *disambiguating* aliases (≥10 chars OR multi-token via
   whitespace/slash/hyphen) — and finding papers whose ``papers.tsv`` matches
   any of those surface forms via :func:`phraseto_tsquery`. The precision-first
   filter is the same disambiguation rule the existing tier-2 Aho-Corasick
   linker uses (see :data:`scix.aho_corasick.DISAMBIGUATOR_MIN_CHARS`); short
   bare acronyms like ``OM``, ``IRS``, ``ACS``, ``COS``, ``HRC`` are deliberately
   skipped because the lemmatizer mangles them ('ACIS' → 'aci') and they
   collide with chemistry/engineering jargon.
3. Inserting :class:`document_entities` rows for the instrument with
   ``link_type='abstract_match'``, ``tier=2``, ``tier_version=1``,
   ``match_method='part_of_backfill_tsv'``, and ``evidence={'method':
   'part_of_backfill_tsv', 'matched_surfaces': [...]}``.
4. Mirroring the inverse — for every (paper, instrument) row produced, ensure
   the parent mission has a row too. Uses ``link_type='inherited'``,
   ``match_method='part_of_inheritance'`` and
   ``evidence={'method':'part_of_inheritance', 'via_instrument_id':...,
   'via_instrument_name':...}``. The distinct ``link_type`` keeps these rows
   from colliding with the existing tier-2 ``abstract_match`` rows on the
   primary key ``(bibcode, entity_id, link_type, tier)``.

Idempotency
-----------

Every insert uses ``ON CONFLICT (bibcode, entity_id, link_type, tier) DO
NOTHING``; re-running is a no-op.

Production safety
-----------------

Mirrors the guard pattern in ``scripts/seed_flagship_instruments.py`` and
``scripts/populate_papers_fulltext.py``:

* Refuses to write to a production DSN unless ``--allow-prod`` is passed.
* When ``--allow-prod`` is set, refuses to run unless launched inside a
  systemd-managed scope (``INVOCATION_ID`` env var set by
  ``systemd-run --scope``, which ``scix-batch`` uses).

Usage
-----

::

    .venv/bin/python scripts/backfill_part_of_inheritance.py --db scix_test
    .venv/bin/python scripts/backfill_part_of_inheritance.py --dry-run
    scix-batch --allow-prod python scripts/backfill_part_of_inheritance.py --allow-prod -v
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pathlib
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Optional

import psycopg

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from scix.db import DEFAULT_DSN, get_connection, is_production_dsn, redact_dsn  # noqa: E402

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------

# Re-use the same threshold the tier-2 Aho-Corasick linker uses for "this
# alias is specific enough to fire without further disambiguation".
DISAMBIGUATOR_MIN_CHARS: int = 10

# A single CamelCase / mixed-case token of >= this length is also treated as
# specific because the postgres english lemmatizer preserves it as a unique
# lexeme (e.g. 'NIRSpec', 'NIRCam', 'NIRISS', 'NICMOS', 'LSSTCam'). Bare
# acronyms shorter than this (ACS=3, ACIS=4, OM=2) are too noisy to backfill
# from solely.
SAFE_COMPOUND_MIN_CHARS: int = 6

# Sources of part_of edges we trust to build inheritance from. Both are
# operator-curated; we deliberately skip the bulk SsODNet + GCMD edges
# because their part_of relationships are minor-body / Earth-science
# taxonomies, not flagship instrument hierarchies.
TRUSTED_PART_OF_SOURCES: tuple[str, ...] = ("flagship_seed", "curated_flagship_v1")

# document_entities knobs.
INSTRUMENT_LINK_TYPE: str = "abstract_match"
INSTRUMENT_MATCH_METHOD: str = "part_of_backfill_tsv"
PARENT_LINK_TYPE: str = "inherited"
PARENT_MATCH_METHOD: str = "part_of_inheritance"
LINK_TIER: int = 2
LINK_TIER_VERSION: int = 1
INSTRUMENT_CONFIDENCE: float = 0.85
PARENT_CONFIDENCE: float = 0.75


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class ProdGuardError(SystemExit):
    """Raised when production safety guards refuse to proceed."""


# ---------------------------------------------------------------------------
# DTOs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PartOfEdge:
    """One ``part_of`` instrument → mission edge."""

    instrument_id: int
    instrument_name: str
    mission_id: int
    mission_name: str


@dataclass(frozen=True)
class BackfillStats:
    """Per-run aggregate counters."""

    edges_processed: int
    instruments_with_surfaces: int
    instruments_skipped_no_surfaces: int
    instrument_rows_inserted: int
    parent_rows_inserted: int


# ---------------------------------------------------------------------------
# Surface-form selection
# ---------------------------------------------------------------------------


def is_specific_surface(surface: str) -> bool:
    """Return True if ``surface`` is precise enough to backfill from.

    A surface is "specific" if any of the following hold:

    * length ≥ :data:`DISAMBIGUATOR_MIN_CHARS` (10 chars), OR
    * contains whitespace (multi-word phrase like ``Chandra ACIS``), OR
    * contains a slash (mission-prefixed alias like ``Chandra/ACIS``), OR
    * length ≥ :data:`SAFE_COMPOUND_MIN_CHARS` (6) AND contains at least
      one lowercase letter (CamelCase tokens like ``NIRSpec``, ``NIRCam``,
      ``NIRISS``, ``NICMOS``, ``LSSTCam`` survive the tsvector lemmatizer
      as a unique lexeme).

    Bare uppercase acronyms ``OM``, ``ACS``, ``COS``, ``HRC``, ``IRS``,
    ``MDI``, ``BOSS``, ``EPIC``, ``ACIS``, ``RGS``, ``MIPS``, ``IRAC``,
    ``FGS`` are intentionally rejected: the english lemmatizer mangles
    them and they collide with unrelated jargon. Their disambiguating
    long-form aliases (``Chandra ACIS``, ``Chandra/ACIS``, ``Advanced CCD
    Imaging Spectrometer``) are accepted instead.
    """
    s = surface.strip()
    if not s:
        return False
    if len(s) >= DISAMBIGUATOR_MIN_CHARS:
        return True
    if any(ch.isspace() for ch in s) or "/" in s or "-" in s:
        return True
    if len(s) >= SAFE_COMPOUND_MIN_CHARS and any(ch.islower() for ch in s):
        return True
    return False


# ---------------------------------------------------------------------------
# SQL
# ---------------------------------------------------------------------------


_FETCH_PART_OF_EDGES_SQL = """
    SELECT er.subject_entity_id,
           e1.canonical_name,
           er.object_entity_id,
           e2.canonical_name
      FROM entity_relationships er
      JOIN entities e1 ON e1.id = er.subject_entity_id
      JOIN entities e2 ON e2.id = er.object_entity_id
     WHERE er.predicate = 'part_of'
       AND er.source = ANY(%(sources)s)
       AND e1.entity_type = 'instrument'
     ORDER BY er.subject_entity_id
"""

_FETCH_SURFACES_SQL = """
    SELECT canonical_name AS surface
      FROM entities
     WHERE id = %(entity_id)s
     UNION
    SELECT alias AS surface
      FROM entity_aliases
     WHERE entity_id = %(entity_id)s
"""

_INSERT_INSTRUMENT_DOC_ENTITIES_SQL = """
    WITH matched AS (
        SELECT DISTINCT p.bibcode
          FROM papers p
         WHERE p.tsv @@ %(tsquery)s::tsquery
    )
    INSERT INTO document_entities (
        bibcode, entity_id, link_type, tier, tier_version,
        confidence, match_method, evidence
    )
    SELECT m.bibcode,
           %(entity_id)s,
           %(link_type)s,
           %(tier)s::smallint,
           %(tier_version)s,
           %(confidence)s::real,
           %(match_method)s,
           %(evidence)s::jsonb
      FROM matched m
    ON CONFLICT (bibcode, entity_id, link_type, tier) DO NOTHING
"""

_INSERT_PARENT_DOC_ENTITIES_SQL = """
    INSERT INTO document_entities (
        bibcode, entity_id, link_type, tier, tier_version,
        confidence, match_method, evidence
    )
    SELECT de.bibcode,
           %(parent_id)s,
           %(link_type)s,
           %(tier)s::smallint,
           %(tier_version)s,
           %(confidence)s::real,
           %(match_method)s,
           %(evidence)s::jsonb
      FROM document_entities de
     WHERE de.entity_id = %(instrument_id)s
       AND de.link_type = %(instrument_link_type)s
       AND de.tier = %(tier)s::smallint
    ON CONFLICT (bibcode, entity_id, link_type, tier) DO NOTHING
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def fetch_part_of_edges(
    conn: psycopg.Connection,
    *,
    sources: tuple[str, ...] = TRUSTED_PART_OF_SOURCES,
) -> list[PartOfEdge]:
    """Pull all instrument ``part_of`` mission edges from trusted sources."""
    with conn.cursor() as cur:
        cur.execute(_FETCH_PART_OF_EDGES_SQL, {"sources": list(sources)})
        rows = cur.fetchall()
    return [
        PartOfEdge(
            instrument_id=int(instr_id),
            instrument_name=str(instr_name),
            mission_id=int(mission_id),
            mission_name=str(mission_name),
        )
        for instr_id, instr_name, mission_id, mission_name in rows
    ]


def fetch_surfaces(conn: psycopg.Connection, entity_id: int) -> list[str]:
    """Return canonical_name + every alias for an entity (deduplicated)."""
    with conn.cursor() as cur:
        cur.execute(_FETCH_SURFACES_SQL, {"entity_id": entity_id})
        rows = cur.fetchall()
    seen: set[str] = set()
    out: list[str] = []
    for (surface,) in rows:
        s = (surface or "").strip()
        if s and s not in seen:
            seen.add(s)
            out.append(s)
    return out


def build_tsquery_or(surfaces: list[str]) -> Optional[str]:
    """Render a tsquery string OR-ing every surface as a phrase query.

    Each surface is wrapped with :func:`phraseto_tsquery` and the
    sub-queries are joined with ``||``. Returns ``None`` when no
    surfaces remain.

    We let postgres do the lexeme conversion server-side via
    ``phraseto_tsquery('english', %s)::text`` rather than building the
    tsquery client-side; that keeps lemma/stopword behaviour aligned
    with the GIN index.
    """
    if not surfaces:
        return None
    return " || ".join(f"phraseto_tsquery('english', {_quote(s)})" for s in surfaces)


def _quote(s: str) -> str:
    """Return a single-quoted SQL literal for ``s`` with quote-doubling."""
    escaped = s.replace("'", "''")
    return f"'{escaped}'"


def insert_instrument_doc_entities(
    conn: psycopg.Connection,
    *,
    instrument_id: int,
    surfaces: list[str],
) -> int:
    """Insert ``document_entities`` rows for one instrument. Returns count."""
    if not surfaces:
        return 0
    tsquery_expr = build_tsquery_or(surfaces)
    if tsquery_expr is None:
        return 0

    # We need a single tsquery expression as a parameter. The cleanest path
    # is to materialize it via SELECT, then bind the resulting text. Inline
    # composition is safe here because every surface flows through
    # _quote() which doubles single-quotes.
    with conn.cursor() as cur:
        cur.execute(f"SELECT ({tsquery_expr})::text")
        row = cur.fetchone()
        if row is None or row[0] is None:
            return 0
        tsquery_text = str(row[0])

    evidence = json.dumps(
        {
            "method": INSTRUMENT_MATCH_METHOD,
            "matched_surfaces": surfaces,
        }
    )
    with conn.cursor() as cur:
        cur.execute(
            _INSERT_INSTRUMENT_DOC_ENTITIES_SQL,
            {
                "tsquery": tsquery_text,
                "entity_id": instrument_id,
                "link_type": INSTRUMENT_LINK_TYPE,
                "tier": LINK_TIER,
                "tier_version": LINK_TIER_VERSION,
                "confidence": INSTRUMENT_CONFIDENCE,
                "match_method": INSTRUMENT_MATCH_METHOD,
                "evidence": evidence,
            },
        )
        return cur.rowcount or 0


def insert_parent_doc_entities(
    conn: psycopg.Connection,
    *,
    instrument_id: int,
    instrument_name: str,
    parent_id: int,
) -> int:
    """Mirror instrument doc_entities to the parent mission. Returns count.

    Re-reads ``document_entities`` for the instrument (which includes
    rows we just inserted in the same transaction) and INSERTs a parallel
    row for the parent under ``link_type='inherited'``. The distinct
    ``link_type`` avoids colliding with existing tier-2 abstract_match
    rows on the parent mission's primary key.
    """
    evidence = json.dumps(
        {
            "method": PARENT_MATCH_METHOD,
            "via_instrument_id": instrument_id,
            "via_instrument_name": instrument_name,
        }
    )
    with conn.cursor() as cur:
        cur.execute(
            _INSERT_PARENT_DOC_ENTITIES_SQL,
            {
                "parent_id": parent_id,
                "instrument_id": instrument_id,
                "instrument_link_type": INSTRUMENT_LINK_TYPE,
                "link_type": PARENT_LINK_TYPE,
                "tier": LINK_TIER,
                "tier_version": LINK_TIER_VERSION,
                "confidence": PARENT_CONFIDENCE,
                "match_method": PARENT_MATCH_METHOD,
                "evidence": evidence,
            },
        )
        return cur.rowcount or 0


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def run_backfill(
    conn: psycopg.Connection,
    *,
    dry_run: bool = False,
    sources: tuple[str, ...] = TRUSTED_PART_OF_SOURCES,
) -> BackfillStats:
    """Execute the backfill end-to-end.

    On ``dry_run=True`` rolls back at the end; the returned stats reflect
    what *would have been* inserted.
    """
    edges = fetch_part_of_edges(conn, sources=sources)
    logger.info("found %d part_of edges from sources %s", len(edges), sources)

    instruments_with_surfaces = 0
    instruments_skipped = 0
    instrument_rows_inserted = 0
    parent_rows_inserted = 0

    for edge in edges:
        all_surfaces = fetch_surfaces(conn, edge.instrument_id)
        specific_surfaces = [s for s in all_surfaces if is_specific_surface(s)]
        if not specific_surfaces:
            instruments_skipped += 1
            logger.warning(
                "skipping instrument %r (id=%d): no specific surfaces "
                "(all surfaces: %s)",
                edge.instrument_name,
                edge.instrument_id,
                all_surfaces,
            )
            continue
        instruments_with_surfaces += 1

        instr_inserted = insert_instrument_doc_entities(
            conn,
            instrument_id=edge.instrument_id,
            surfaces=specific_surfaces,
        )
        instrument_rows_inserted += instr_inserted

        parent_inserted = insert_parent_doc_entities(
            conn,
            instrument_id=edge.instrument_id,
            instrument_name=edge.instrument_name,
            parent_id=edge.mission_id,
        )
        parent_rows_inserted += parent_inserted

        logger.info(
            "%s -> %s: %d instrument rows + %d inherited parent rows "
            "(surfaces: %s)",
            edge.instrument_name,
            edge.mission_name,
            instr_inserted,
            parent_inserted,
            specific_surfaces,
        )

    if dry_run:
        conn.rollback()
        logger.info("DRY RUN — rolled back all inserts")
    else:
        conn.commit()

    return BackfillStats(
        edges_processed=len(edges),
        instruments_with_surfaces=instruments_with_surfaces,
        instruments_skipped_no_surfaces=instruments_skipped,
        instrument_rows_inserted=instrument_rows_inserted,
        parent_rows_inserted=parent_rows_inserted,
    )


# ---------------------------------------------------------------------------
# Production-safety guard
# ---------------------------------------------------------------------------


def enforce_prod_guard(
    *,
    dsn: str,
    allow_prod: bool,
    env: Mapping[str, str],
) -> None:
    """Refuse to run against prod unless ``--allow-prod`` AND systemd scope.

    Mirrors ``scripts/populate_papers_fulltext.py::_enforce_prod_guard``.
    Raises :class:`ProdGuardError` (a SystemExit) on policy violation.
    """
    if is_production_dsn(dsn) and not allow_prod:
        msg = (
            f"refusing to write to production DSN {redact_dsn(dsn)} — "
            "pass --allow-prod to override"
        )
        logger.error(msg)
        raise ProdGuardError(2)

    if allow_prod and not env.get("INVOCATION_ID"):
        msg = (
            "refusing to run --allow-prod outside a systemd scope. "
            "Invoke via: scix-batch --allow-prod python "
            "scripts/backfill_part_of_inheritance.py --allow-prod"
        )
        logger.error(msg)
        raise ProdGuardError(2)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _resolve_dsn(db_arg: Optional[str]) -> str:
    if db_arg:
        if "=" in db_arg or "://" in db_arg:
            return db_arg
        return f"dbname={db_arg}"
    return os.environ.get("SCIX_DSN") or DEFAULT_DSN


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Backfill flagship-instrument document_entities and propagate "
            "part_of inheritance to their parent missions."
        ),
    )
    parser.add_argument(
        "--db",
        type=str,
        default=None,
        help="Database name or full DSN (default: SCIX_DSN env or scix).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Run the backfill but roll back without committing.",
    )
    parser.add_argument(
        "--allow-prod",
        action="store_true",
        default=False,
        help="Required to run against the production database.",
    )
    parser.add_argument(
        "--sources",
        nargs="+",
        default=list(TRUSTED_PART_OF_SOURCES),
        help=(
            "entity_relationships.source values to read part_of edges from. "
            f"Default: {list(TRUSTED_PART_OF_SOURCES)}."
        ),
    )
    parser.add_argument("--verbose", "-v", action="store_true", default=False)
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    dsn = _resolve_dsn(args.db)
    try:
        enforce_prod_guard(dsn=dsn, allow_prod=args.allow_prod, env=os.environ)
    except ProdGuardError as exc:
        return int(exc.code) if isinstance(exc.code, int) else 2

    logger.info(
        "backfill_part_of_inheritance: dsn=%s dry_run=%s sources=%s",
        redact_dsn(dsn),
        args.dry_run,
        args.sources,
    )

    conn = get_connection(dsn)
    try:
        stats = run_backfill(
            conn,
            dry_run=args.dry_run,
            sources=tuple(args.sources),
        )
    finally:
        conn.close()

    logger.info(
        "backfill_part_of_inheritance: edges=%d instruments_linked=%d "
        "instruments_skipped=%d instrument_rows=%d parent_rows=%d",
        stats.edges_processed,
        stats.instruments_with_surfaces,
        stats.instruments_skipped_no_surfaces,
        stats.instrument_rows_inserted,
        stats.parent_rows_inserted,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
