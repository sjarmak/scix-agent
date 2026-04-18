#!/usr/bin/env python3
"""Seed curated flagship telescopes / missions / instruments.

Problem: ADS-derived entities (GCMD, AAS verified facilities, etc.) have
inconsistent canonical names — "NASA/GSFC/SED/ASD/JWST", "European Space
Agency (ESA) 6.5m James Webb Space Telescope (JWST) Satellite Mission",
etc. — and several are classified ``banned`` or ``homograph`` without
the long-form aliases needed for co-presence firing. As a result,
MCP queries for "JWST" or "Hubble" return the tiny subset of papers
that mention the mangled canonical verbatim.

This script seeds a curated set of ~25 flagship instruments/missions
under ``source='curated_flagship_v1'``. Each is classified
``domain_safe`` and given science-unambiguous aliases (acronym + full
name). The 25K per-entity cap in Tier 2 remains the safety net for
runaway matches.

Idempotent: uses ``ON CONFLICT DO NOTHING`` at every step. Re-running
is a no-op; adding a new flagship is additive.

Run after the entity table is populated; re-run Tier 2 after to pick
up new matches.

Usage::

    .venv/bin/python scripts/curate_flagship_entities.py --allow-prod -v
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

CURATION_SOURCE = "curated_flagship_v1"
ALIAS_SOURCE = "curated_flagship_v1"


@dataclass(frozen=True)
class Flagship:
    """One curated flagship entity and its aliases."""

    canonical_name: str
    entity_type: str  # 'mission' | 'instrument' | 'telescope'
    discipline: str  # 'astrophysics' | ... — coarse
    aliases: tuple[str, ...]
    ambiguity_class: str = "domain_safe"


FLAGSHIPS: tuple[Flagship, ...] = (
    # Space telescopes / flagship observatories
    Flagship(
        "James Webb Space Telescope",
        "mission",
        "astrophysics",
        ("JWST", "James Webb Space Telescope"),
    ),
    Flagship(
        "Hubble Space Telescope",
        "mission",
        "astrophysics",
        ("HST", "Hubble Space Telescope"),
    ),
    Flagship(
        "Kepler Space Telescope",
        "mission",
        "astrophysics",
        ("Kepler Space Telescope", "Kepler mission"),
    ),
    Flagship(
        "Transiting Exoplanet Survey Satellite",
        "mission",
        "astrophysics",
        ("TESS", "Transiting Exoplanet Survey Satellite"),
    ),
    Flagship(
        "Chandra X-ray Observatory",
        "mission",
        "astrophysics",
        ("Chandra X-ray Observatory", "Chandra Observatory"),
    ),
    Flagship(
        "Spitzer Space Telescope",
        "mission",
        "astrophysics",
        ("Spitzer Space Telescope",),
    ),
    Flagship(
        "XMM-Newton",
        "mission",
        "astrophysics",
        ("XMM-Newton", "XMM Newton"),
    ),
    Flagship(
        "Gaia Space Observatory",
        "mission",
        "astrophysics",
        ("Gaia Space Observatory", "Gaia satellite", "Gaia mission"),
    ),
    Flagship(
        "Planck Space Observatory",
        "mission",
        "astrophysics",
        ("Planck satellite", "Planck mission", "Planck Space Observatory"),
    ),
    Flagship(
        "Fermi Gamma-ray Space Telescope",
        "mission",
        "astrophysics",
        ("Fermi Gamma-ray Space Telescope", "Fermi-LAT", "Fermi GBM"),
    ),
    Flagship(
        "Neil Gehrels Swift Observatory",
        "mission",
        "astrophysics",
        ("Swift satellite", "Swift Gamma-Ray Burst Mission", "Neil Gehrels Swift Observatory"),
    ),
    Flagship(
        "Herschel Space Observatory",
        "mission",
        "astrophysics",
        ("Herschel Space Observatory", "Herschel satellite"),
    ),
    Flagship(
        "Wide-field Infrared Survey Explorer",
        "mission",
        "astrophysics",
        ("WISE", "NEOWISE", "Wide-field Infrared Survey Explorer"),
    ),
    Flagship(
        "Wilkinson Microwave Anisotropy Probe",
        "mission",
        "astrophysics",
        ("WMAP", "Wilkinson Microwave Anisotropy Probe"),
    ),
    Flagship(
        "Rossi X-ray Timing Explorer",
        "mission",
        "astrophysics",
        ("RXTE", "Rossi X-ray Timing Explorer"),
    ),
    Flagship(
        "Nuclear Spectroscopic Telescope Array",
        "mission",
        "astrophysics",
        ("NuSTAR", "Nuclear Spectroscopic Telescope Array"),
    ),
    Flagship(
        "INTErnational Gamma-Ray Astrophysics Laboratory",
        "mission",
        "astrophysics",
        ("INTEGRAL satellite", "INTErnational Gamma-Ray Astrophysics Laboratory"),
    ),
    Flagship(
        "Compton Gamma Ray Observatory",
        "mission",
        "astrophysics",
        ("CGRO", "Compton Gamma Ray Observatory"),
    ),
    Flagship(
        "Galaxy Evolution Explorer",
        "mission",
        "astrophysics",
        ("GALEX", "Galaxy Evolution Explorer"),
    ),
    # Solar / heliophysics
    Flagship(
        "Solar Dynamics Observatory",
        "mission",
        "heliophysics",
        ("SDO", "Solar Dynamics Observatory"),
    ),
    Flagship(
        "Solar and Heliospheric Observatory",
        "mission",
        "heliophysics",
        ("SOHO", "Solar and Heliospheric Observatory"),
    ),
    # Ground-based flagship instruments
    Flagship(
        "Atacama Large Millimeter Array",
        "instrument",
        "astrophysics",
        ("ALMA", "Atacama Large Millimeter Array"),
    ),
    Flagship(
        "Very Large Telescope",
        "instrument",
        "astrophysics",
        ("VLT", "Very Large Telescope"),
    ),
    Flagship(
        "Very Large Array",
        "instrument",
        "astrophysics",
        ("VLA", "Very Large Array"),
    ),
    Flagship(
        "Keck Observatory",
        "instrument",
        "astrophysics",
        ("Keck Observatory", "W. M. Keck Observatory"),
    ),
    # Gravitational wave
    Flagship(
        "Laser Interferometer Gravitational-Wave Observatory",
        "instrument",
        "astrophysics",
        ("LIGO", "Laser Interferometer Gravitational-Wave Observatory"),
    ),
    Flagship(
        "Virgo Gravitational-Wave Interferometer",
        "instrument",
        "astrophysics",
        ("Virgo Gravitational-Wave Interferometer", "Virgo interferometer"),
    ),
)


@dataclass(frozen=True)
class CurationStats:
    entities_created: int
    entities_existing: int
    aliases_created: int
    aliases_existing: int


_INSERT_ENTITY_SQL = """
    INSERT INTO entities (
        canonical_name, entity_type, discipline, source,
        ambiguity_class, link_policy, properties
    )
    VALUES (
        %(canonical_name)s, %(entity_type)s, %(discipline)s, %(source)s,
        %(ambiguity_class)s::entity_ambiguity_class, NULL, '{}'::jsonb
    )
    ON CONFLICT (canonical_name, entity_type, source) DO NOTHING
    RETURNING id
"""

_LOOKUP_ENTITY_SQL = """
    SELECT id FROM entities
     WHERE canonical_name = %(canonical_name)s
       AND entity_type = %(entity_type)s
       AND source = %(source)s
"""

_INSERT_ALIAS_SQL = """
    INSERT INTO entity_aliases (entity_id, alias, alias_source)
    VALUES (%(entity_id)s, %(alias)s, %(alias_source)s)
    ON CONFLICT (entity_id, alias) DO NOTHING
    RETURNING entity_id
"""


def upsert_flagship(conn: psycopg.Connection, flagship: Flagship) -> tuple[int, bool, int]:
    """Upsert one flagship entity plus its aliases.

    Returns
    -------
    (entity_id, created_entity, aliases_created)
    """
    params = {
        "canonical_name": flagship.canonical_name,
        "entity_type": flagship.entity_type,
        "discipline": flagship.discipline,
        "source": CURATION_SOURCE,
        "ambiguity_class": flagship.ambiguity_class,
    }

    with conn.cursor() as cur:
        cur.execute(_INSERT_ENTITY_SQL, params)
        row = cur.fetchone()
        if row is not None:
            entity_id = int(row[0])
            created = True
        else:
            cur.execute(_LOOKUP_ENTITY_SQL, params)
            row = cur.fetchone()
            assert row is not None, f"failed to upsert or lookup {flagship.canonical_name}"
            entity_id = int(row[0])
            created = False

        aliases_created = 0
        for alias in flagship.aliases:
            cur.execute(
                _INSERT_ALIAS_SQL,
                {
                    "entity_id": entity_id,
                    "alias": alias,
                    "alias_source": ALIAS_SOURCE,
                },
            )
            if cur.fetchone() is not None:
                aliases_created += 1

    return entity_id, created, aliases_created


def run_curation(conn: psycopg.Connection, *, dry_run: bool = False) -> CurationStats:
    """Upsert every :data:`FLAGSHIPS` entity and its aliases."""
    entities_created = 0
    entities_existing = 0
    aliases_created = 0
    total_aliases = sum(len(f.aliases) for f in FLAGSHIPS)

    for flagship in FLAGSHIPS:
        entity_id, created, new_aliases = upsert_flagship(conn, flagship)
        if created:
            entities_created += 1
        else:
            entities_existing += 1
        aliases_created += new_aliases
        logger.info(
            "%-6s entity_id=%-7d aliases=%d/%d  %s",
            "NEW" if created else "exists",
            entity_id,
            new_aliases,
            len(flagship.aliases),
            flagship.canonical_name,
        )

    if dry_run:
        conn.rollback()
        logger.info("DRY RUN — rolled back all inserts")
    else:
        conn.commit()

    return CurationStats(
        entities_created=entities_created,
        entities_existing=entities_existing,
        aliases_created=aliases_created,
        aliases_existing=total_aliases - aliases_created,
    )


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--db-url", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true", default=False)
    parser.add_argument("--allow-prod", action="store_true", default=False)
    parser.add_argument("--verbose", "-v", action="store_true", default=False)
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    dsn = args.db_url or os.environ.get("SCIX_TEST_DSN") or DEFAULT_DSN
    if is_production_dsn(dsn) and not args.allow_prod:
        logger.error(
            "refusing to run against production DSN %s — pass --allow-prod to override",
            redact_dsn(dsn),
        )
        return 2

    conn = get_connection(dsn)
    try:
        stats = run_curation(conn, dry_run=args.dry_run)
    finally:
        conn.close()

    print(
        f"curate_flagship_entities: {stats.entities_created} new entities, "
        f"{stats.entities_existing} already existed, "
        f"{stats.aliases_created} new aliases "
        f"({stats.aliases_existing} already existed)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
