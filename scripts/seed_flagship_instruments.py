#!/usr/bin/env python3
"""Seed ~30 flagship instrument entities that are missing from ``entities``.

Problem: the curated flagship seed (``scripts/curate_flagship_entities.py``)
covers flagship missions and telescopes (JWST, HST, Chandra, ALMA, ...) but
does **not** enumerate sub-instruments (JWST/NIRSpec, HST/ACS, Chandra/ACIS,
ALMA Band-6, ...). Downstream MCP agents ask for specific instruments by name
and get empty result sets because the instrument row simply does not exist.

This script seeds a curated list of flagship *instruments* (plus a few
observatories / missions when a parent is missing) under
``source='flagship_seed'`` and writes ``part_of`` relationships linking each
sub-instrument to its parent mission/observatory.

Idempotent: every insert uses ``ON CONFLICT DO NOTHING``; re-running is a
no-op. Adding a new instrument to ``CURATED_INSTRUMENTS`` is additive.

Usage::

    .venv/bin/python scripts/seed_flagship_instruments.py --db scix_test
    .venv/bin/python scripts/seed_flagship_instruments.py --dry-run
    .venv/bin/python scripts/seed_flagship_instruments.py --allow-prod -v
"""

from __future__ import annotations

import argparse
import logging
import os
import pathlib
import sys
from dataclasses import dataclass
from typing import Optional

import psycopg

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from scix.db import DEFAULT_DSN, get_connection, is_production_dsn, redact_dsn  # noqa: E402

logger = logging.getLogger(__name__)

SEED_SOURCE = "flagship_seed"
ALIAS_SOURCE = "flagship_seed"
PART_OF_PREDICATE = "part_of"


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ParentEntity:
    """Parent mission/observatory to ensure exists before linking children.

    ``source_preference`` lists acceptable existing ``source`` values to
    match against first (e.g., ``curated_flagship_v1``). If no match is
    found, we create the parent under ``SEED_SOURCE``.
    """

    canonical_name: str
    entity_type: str  # 'mission' | 'observatory'
    discipline: str
    aliases: tuple[str, ...] = ()
    source_preference: tuple[str, ...] = ("curated_flagship_v1", SEED_SOURCE)


@dataclass(frozen=True)
class Instrument:
    """One curated sub-instrument and its parent linkage."""

    canonical_name: str
    entity_type: str  # typically 'instrument'
    discipline: str
    aliases: tuple[str, ...]
    parent: Optional[ParentEntity] = None


# ---------------------------------------------------------------------------
# Parents (shared across multiple children)
# ---------------------------------------------------------------------------

_JWST = ParentEntity(
    "James Webb Space Telescope",
    "mission",
    "astrophysics",
    aliases=("JWST", "James Webb Space Telescope"),
)
_HST = ParentEntity(
    "Hubble Space Telescope",
    "mission",
    "astrophysics",
    aliases=("HST", "Hubble Space Telescope"),
)
_CHANDRA = ParentEntity(
    "Chandra X-ray Observatory",
    "mission",
    "astrophysics",
    aliases=("Chandra", "Chandra X-ray Observatory"),
)
_XMM = ParentEntity(
    "XMM-Newton",
    "mission",
    "astrophysics",
    aliases=("XMM-Newton", "XMM Newton"),
)
_SPITZER = ParentEntity(
    "Spitzer Space Telescope",
    "mission",
    "astrophysics",
    aliases=("Spitzer", "Spitzer Space Telescope"),
)
_ALMA = ParentEntity(
    "Atacama Large Millimeter Array",
    "observatory",
    "astrophysics",
    aliases=("ALMA", "Atacama Large Millimeter Array"),
)
_RUBIN = ParentEntity(
    "Vera C. Rubin Observatory",
    "observatory",
    "astrophysics",
    aliases=("Rubin Observatory", "Vera C. Rubin Observatory", "LSST Observatory"),
)
_SDSS = ParentEntity(
    "Sloan Digital Sky Survey",
    "observatory",
    "astrophysics",
    aliases=("SDSS", "Sloan Digital Sky Survey"),
)


# ---------------------------------------------------------------------------
# Curated list (≥25 instruments + standalone missions)
# ---------------------------------------------------------------------------


CURATED_INSTRUMENTS: tuple[Instrument, ...] = (
    # JWST instruments ------------------------------------------------------
    Instrument(
        "NIRSpec",
        "instrument",
        "astrophysics",
        aliases=("NIRSpec", "JWST NIRSpec", "JWST/NIRSpec", "Near-Infrared Spectrograph"),
        parent=_JWST,
    ),
    Instrument(
        "NIRCam",
        "instrument",
        "astrophysics",
        aliases=("NIRCam", "JWST NIRCam", "JWST/NIRCam", "Near-Infrared Camera"),
        parent=_JWST,
    ),
    Instrument(
        "MIRI",
        "instrument",
        "astrophysics",
        aliases=("MIRI", "JWST MIRI", "JWST/MIRI", "Mid-Infrared Instrument"),
        parent=_JWST,
    ),
    Instrument(
        "NIRISS",
        "instrument",
        "astrophysics",
        aliases=(
            "NIRISS",
            "JWST NIRISS",
            "JWST/NIRISS",
            "Near-Infrared Imager and Slitless Spectrograph",
        ),
        parent=_JWST,
    ),
    Instrument(
        "FGS",
        "instrument",
        "astrophysics",
        aliases=("FGS", "JWST FGS", "JWST/FGS", "Fine Guidance Sensor"),
        parent=_JWST,
    ),
    # HST instruments -------------------------------------------------------
    Instrument(
        "STIS",
        "instrument",
        "astrophysics",
        aliases=(
            "STIS",
            "HST STIS",
            "HST/STIS",
            "Space Telescope Imaging Spectrograph",
        ),
        parent=_HST,
    ),
    Instrument(
        "ACS",
        "instrument",
        "astrophysics",
        aliases=("ACS", "HST ACS", "HST/ACS", "Advanced Camera for Surveys"),
        parent=_HST,
    ),
    Instrument(
        "WFC3",
        "instrument",
        "astrophysics",
        aliases=("WFC3", "HST WFC3", "HST/WFC3", "Wide Field Camera 3"),
        parent=_HST,
    ),
    Instrument(
        "COS",
        "instrument",
        "astrophysics",
        aliases=("COS", "HST COS", "HST/COS", "Cosmic Origins Spectrograph"),
        parent=_HST,
    ),
    Instrument(
        "NICMOS",
        "instrument",
        "astrophysics",
        aliases=(
            "NICMOS",
            "HST NICMOS",
            "HST/NICMOS",
            "Near Infrared Camera and Multi-Object Spectrometer",
        ),
        parent=_HST,
    ),
    # Chandra instruments --------------------------------------------------
    Instrument(
        "ACIS",
        "instrument",
        "astrophysics",
        aliases=(
            "ACIS",
            "Chandra ACIS",
            "Chandra/ACIS",
            "Advanced CCD Imaging Spectrometer",
        ),
        parent=_CHANDRA,
    ),
    Instrument(
        "HRC",
        "instrument",
        "astrophysics",
        aliases=("HRC", "Chandra HRC", "Chandra/HRC", "High Resolution Camera"),
        parent=_CHANDRA,
    ),
    # XMM-Newton instruments -----------------------------------------------
    Instrument(
        "EPIC",
        "instrument",
        "astrophysics",
        aliases=(
            "EPIC",
            "XMM EPIC",
            "XMM-Newton EPIC",
            "European Photon Imaging Camera",
        ),
        parent=_XMM,
    ),
    Instrument(
        "RGS",
        "instrument",
        "astrophysics",
        aliases=(
            "RGS",
            "XMM RGS",
            "XMM-Newton RGS",
            "Reflection Grating Spectrometer",
        ),
        parent=_XMM,
    ),
    Instrument(
        "OM",
        "instrument",
        "astrophysics",
        aliases=("XMM OM", "XMM-Newton OM", "Optical Monitor"),
        parent=_XMM,
    ),
    # Spitzer instruments --------------------------------------------------
    Instrument(
        "IRAC",
        "instrument",
        "astrophysics",
        aliases=(
            "IRAC",
            "Spitzer IRAC",
            "Spitzer/IRAC",
            "Infrared Array Camera",
        ),
        parent=_SPITZER,
    ),
    Instrument(
        "IRS",
        "instrument",
        "astrophysics",
        aliases=("IRS", "Spitzer IRS", "Spitzer/IRS", "Infrared Spectrograph"),
        parent=_SPITZER,
    ),
    Instrument(
        "MIPS",
        "instrument",
        "astrophysics",
        aliases=(
            "MIPS",
            "Spitzer MIPS",
            "Spitzer/MIPS",
            "Multiband Imaging Photometer for Spitzer",
        ),
        parent=_SPITZER,
    ),
    # Standalone missions (no parent) --------------------------------------
    Instrument(
        "Transiting Exoplanet Survey Satellite",
        "mission",
        "astrophysics",
        aliases=("TESS", "Transiting Exoplanet Survey Satellite"),
        parent=None,
    ),
    Instrument(
        "Kepler Space Telescope",
        "mission",
        "astrophysics",
        aliases=("Kepler", "Kepler Space Telescope", "Kepler mission"),
        parent=None,
    ),
    Instrument(
        "Gaia Space Observatory",
        "mission",
        "astrophysics",
        aliases=("Gaia", "Gaia Space Observatory", "Gaia satellite"),
        parent=None,
    ),
    Instrument(
        "Fermi Gamma-ray Space Telescope",
        "mission",
        "astrophysics",
        aliases=("Fermi", "Fermi Gamma-ray Space Telescope", "Fermi-LAT", "Fermi GBM"),
        parent=None,
    ),
    Instrument(
        "Neil Gehrels Swift Observatory",
        "mission",
        "astrophysics",
        aliases=(
            "SWIFT",
            "Swift satellite",
            "Neil Gehrels Swift Observatory",
            "Swift Gamma-Ray Burst Mission",
        ),
        parent=None,
    ),
    Instrument(
        "Neutron star Interior Composition Explorer",
        "mission",
        "astrophysics",
        aliases=("NICER", "Neutron star Interior Composition Explorer"),
        parent=None,
    ),
    # Ground-based / surveys -----------------------------------------------
    Instrument(
        "LSSTCam",
        "instrument",
        "astrophysics",
        aliases=(
            "LSSTCam",
            "LSST Camera",
            "Rubin LSSTCam",
            "Rubin/LSSTCam",
            "LSST",
        ),
        parent=_RUBIN,
    ),
    Instrument(
        "Dark Energy Spectroscopic Instrument",
        "instrument",
        "astrophysics",
        aliases=("DESI", "Dark Energy Spectroscopic Instrument"),
        parent=None,
    ),
    Instrument(
        "BOSS",
        "instrument",
        "astrophysics",
        aliases=("BOSS", "SDSS BOSS", "SDSS/BOSS", "Baryon Oscillation Spectroscopic Survey"),
        parent=_SDSS,
    ),
    Instrument(
        "APOGEE",
        "instrument",
        "astrophysics",
        aliases=(
            "APOGEE",
            "SDSS APOGEE",
            "SDSS/APOGEE",
            "Apache Point Observatory Galactic Evolution Experiment",
        ),
        parent=_SDSS,
    ),
    # ALMA bands (sub-instruments of ALMA) ---------------------------------
    Instrument(
        "ALMA Band 3",
        "instrument",
        "astrophysics",
        aliases=("ALMA Band 3", "ALMA Band-3", "ALMA/Band 3"),
        parent=_ALMA,
    ),
    Instrument(
        "ALMA Band 6",
        "instrument",
        "astrophysics",
        aliases=("ALMA Band 6", "ALMA Band-6", "ALMA/Band 6"),
        parent=_ALMA,
    ),
    Instrument(
        "ALMA Band 7",
        "instrument",
        "astrophysics",
        aliases=("ALMA Band 7", "ALMA Band-7", "ALMA/Band 7"),
        parent=_ALMA,
    ),
)


# ---------------------------------------------------------------------------
# SQL
# ---------------------------------------------------------------------------


_INSERT_ENTITY_SQL = """
    INSERT INTO entities (canonical_name, entity_type, discipline, source, properties)
    VALUES (%(canonical_name)s, %(entity_type)s, %(discipline)s, %(source)s, '{}'::jsonb)
    ON CONFLICT (canonical_name, entity_type, source) DO NOTHING
    RETURNING id
"""

_LOOKUP_ENTITY_BY_KEY_SQL = """
    SELECT id FROM entities
     WHERE canonical_name = %(canonical_name)s
       AND entity_type = %(entity_type)s
       AND source = %(source)s
"""

_LOOKUP_ENTITY_ANY_SOURCE_SQL = """
    SELECT id, source FROM entities
     WHERE canonical_name = %(canonical_name)s
       AND entity_type = %(entity_type)s
     ORDER BY CASE source WHEN %(preferred)s THEN 0 ELSE 1 END, id
     LIMIT 1
"""

_INSERT_ALIAS_SQL = """
    INSERT INTO entity_aliases (entity_id, alias, alias_source)
    VALUES (%(entity_id)s, %(alias)s, %(alias_source)s)
    ON CONFLICT (entity_id, alias) DO NOTHING
    RETURNING entity_id
"""

_INSERT_RELATIONSHIP_SQL = """
    INSERT INTO entity_relationships
        (subject_entity_id, predicate, object_entity_id, source, confidence)
    VALUES (%(subject)s, %(predicate)s, %(object)s, %(source)s, 1.0)
    ON CONFLICT (subject_entity_id, predicate, object_entity_id) DO NOTHING
    RETURNING id
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SeedStats:
    """Summary of a seed run."""

    entities_created: int
    entities_existing: int
    aliases_created: int
    aliases_existing: int
    relationships_created: int
    relationships_existing: int


def _upsert_entity(
    conn: psycopg.Connection,
    *,
    canonical_name: str,
    entity_type: str,
    discipline: str,
    source: str,
) -> tuple[int, bool]:
    """Insert-or-lookup an entity by (canonical, type, source).

    Returns ``(entity_id, created_new)``.
    """
    params = {
        "canonical_name": canonical_name,
        "entity_type": entity_type,
        "discipline": discipline,
        "source": source,
    }
    with conn.cursor() as cur:
        cur.execute(_INSERT_ENTITY_SQL, params)
        row = cur.fetchone()
        if row is not None:
            return int(row[0]), True
        cur.execute(_LOOKUP_ENTITY_BY_KEY_SQL, params)
        row = cur.fetchone()
        if row is None:
            raise RuntimeError(
                f"failed to upsert or lookup entity ({canonical_name!r}, "
                f"{entity_type!r}, {source!r})"
            )
        return int(row[0]), False


def _insert_alias(conn: psycopg.Connection, *, entity_id: int, alias: str) -> bool:
    """Insert one alias; returns True if a new row was created."""
    with conn.cursor() as cur:
        cur.execute(
            _INSERT_ALIAS_SQL,
            {
                "entity_id": entity_id,
                "alias": alias,
                "alias_source": ALIAS_SOURCE,
            },
        )
        return cur.fetchone() is not None


def _ensure_parent(
    conn: psycopg.Connection, parent: ParentEntity
) -> tuple[int, bool, int]:
    """Find or create a parent entity. Uses ``source_preference`` order.

    Returns ``(entity_id, created_new, new_alias_count)``.

    If a row with the parent's canonical_name+entity_type already exists
    under any preferred source, we reuse that id and layer aliases on it.
    Otherwise we create a new row under ``SEED_SOURCE``.
    """
    preferred = parent.source_preference[0] if parent.source_preference else SEED_SOURCE
    with conn.cursor() as cur:
        cur.execute(
            _LOOKUP_ENTITY_ANY_SOURCE_SQL,
            {
                "canonical_name": parent.canonical_name,
                "entity_type": parent.entity_type,
                "preferred": preferred,
            },
        )
        row = cur.fetchone()

    if row is not None:
        entity_id = int(row[0])
        created = False
    else:
        entity_id, created = _upsert_entity(
            conn,
            canonical_name=parent.canonical_name,
            entity_type=parent.entity_type,
            discipline=parent.discipline,
            source=SEED_SOURCE,
        )

    new_aliases = 0
    for alias in parent.aliases:
        if _insert_alias(conn, entity_id=entity_id, alias=alias):
            new_aliases += 1
    return entity_id, created, new_aliases


def _insert_relationship(
    conn: psycopg.Connection,
    *,
    subject_id: int,
    predicate: str,
    object_id: int,
) -> bool:
    """Insert a relationship edge; returns True if a new row was created."""
    with conn.cursor() as cur:
        cur.execute(
            _INSERT_RELATIONSHIP_SQL,
            {
                "subject": subject_id,
                "predicate": predicate,
                "object": object_id,
                "source": SEED_SOURCE,
            },
        )
        return cur.fetchone() is not None


def seed(conn: psycopg.Connection, *, dry_run: bool = False) -> SeedStats:
    """Seed all ``CURATED_INSTRUMENTS`` and their parent ``part_of`` edges.

    When ``dry_run`` is True, all inserts are rolled back at the end and
    the returned counts reflect what *would have been* changed. The caller
    is responsible for closing the connection.
    """
    entities_created = 0
    entities_existing = 0
    aliases_created = 0
    aliases_total = 0
    relationships_created = 0
    relationships_total = 0

    # Track parents we've already ensured this run to avoid redundant work.
    parent_ids: dict[tuple[str, str], int] = {}

    for inst in CURATED_INSTRUMENTS:
        parent_id: Optional[int] = None
        if inst.parent is not None:
            key = (inst.parent.canonical_name, inst.parent.entity_type)
            if key in parent_ids:
                parent_id = parent_ids[key]
            else:
                parent_id, parent_created, parent_new_aliases = _ensure_parent(
                    conn, inst.parent
                )
                parent_ids[key] = parent_id
                if parent_created:
                    entities_created += 1
                else:
                    entities_existing += 1
                aliases_created += parent_new_aliases
                aliases_total += len(inst.parent.aliases)

        entity_id, created = _upsert_entity(
            conn,
            canonical_name=inst.canonical_name,
            entity_type=inst.entity_type,
            discipline=inst.discipline,
            source=SEED_SOURCE,
        )
        if created:
            entities_created += 1
        else:
            entities_existing += 1

        inst_new_aliases = 0
        for alias in inst.aliases:
            if _insert_alias(conn, entity_id=entity_id, alias=alias):
                inst_new_aliases += 1
        aliases_created += inst_new_aliases
        aliases_total += len(inst.aliases)

        edge_created = False
        if parent_id is not None:
            edge_created = _insert_relationship(
                conn,
                subject_id=entity_id,
                predicate=PART_OF_PREDICATE,
                object_id=parent_id,
            )
            relationships_total += 1
            if edge_created:
                relationships_created += 1

        logger.info(
            "%-6s entity_id=%-7d aliases=%d/%d %s %s",
            "NEW" if created else "exists",
            entity_id,
            inst_new_aliases,
            len(inst.aliases),
            "part_of->" + str(parent_id) if parent_id is not None else "(no parent)",
            inst.canonical_name,
        )

    if dry_run:
        conn.rollback()
        logger.info("DRY RUN — rolled back all inserts")
    else:
        conn.commit()

    return SeedStats(
        entities_created=entities_created,
        entities_existing=entities_existing,
        aliases_created=aliases_created,
        aliases_existing=aliases_total - aliases_created,
        relationships_created=relationships_created,
        relationships_existing=relationships_total - relationships_created,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _resolve_dsn(db_arg: Optional[str]) -> str:
    """Resolve DSN from --db argument, SCIX_DSN env, or the package default."""
    if db_arg:
        # Accept either a bare dbname (e.g., ``scix_test``) or a full DSN.
        if "=" in db_arg or "://" in db_arg:
            return db_arg
        return f"dbname={db_arg}"
    return os.environ.get("SCIX_DSN") or DEFAULT_DSN


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Seed flagship instrument entities + part_of relationships.",
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
        help="Print planned inserts and roll back without committing.",
    )
    parser.add_argument(
        "--allow-prod",
        action="store_true",
        default=False,
        help="Required to run against the production database.",
    )
    parser.add_argument("--verbose", "-v", action="store_true", default=False)
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    dsn = _resolve_dsn(args.db)
    if is_production_dsn(dsn) and not args.allow_prod:
        logger.error(
            "refusing to run against production DSN %s — pass --allow-prod to override",
            redact_dsn(dsn),
        )
        return 2

    logger.info(
        "seed_flagship_instruments: dsn=%s dry_run=%s",
        redact_dsn(dsn),
        args.dry_run,
    )

    conn = get_connection(dsn)
    try:
        stats = seed(conn, dry_run=args.dry_run)
    finally:
        conn.close()

    logger.info(
        "seed_flagship_instruments: %d new entities (%d already existed), "
        "%d new aliases (%d already existed), "
        "%d new part_of edges (%d already existed)",
        stats.entities_created,
        stats.entities_existing,
        stats.aliases_created,
        stats.aliases_existing,
        stats.relationships_created,
        stats.relationships_existing,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
