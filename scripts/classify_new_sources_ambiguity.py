#!/usr/bin/env python3
"""Populate ``entities.ambiguity_class`` for selected sources.

Targeted equivalent of :mod:`scripts/classify_entity_ambiguity.py`. The
global classifier reclassifies *every* row, which means streaming 21M
entities through Python — too slow to be useful when only a handful of
new sources need backfill.

This variant:

* Loads the top-20K English words via :mod:`wordfreq` (same banned
  threshold as :mod:`scix.ambiguity`).
* Pulls collision counts from the ``entities`` table with one SQL
  aggregation — ``COUNT(*) - 1 OVER (PARTITION BY lower(canonical_name))``
  computed via a pre-aggregated subquery — joined with the alias graph.
* Iterates only entities in ``--source`` (default: cran, bioconductor,
  pypi) and bulk-updates them in batches.

Safety: same DSN / ``--allow-prod`` gates as the global classifier.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections import defaultdict
from pathlib import Path

import psycopg

# Repo's src/ on path for sibling imports.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from scix.ambiguity import classify  # noqa: E402

logger = logging.getLogger("classify_new_sources_ambiguity")

_PRODUCTION_DB_NAMES = {"scix"}
DEFAULT_SOURCES = ("cran", "bioconductor", "pypi")


def _is_production_dsn(dsn: str) -> bool:
    for token in dsn.split():
        if "=" in token:
            key, _, value = token.partition("=")
            if key.strip() == "dbname" and value.strip() in _PRODUCTION_DB_NAMES:
                return True
    return False


def _default_dsn() -> str:
    return os.environ.get("SCIX_TEST_DSN") or "dbname=scix_test"


def fetch_target_entities(
    conn: psycopg.Connection,
    sources: tuple[str, ...],
) -> dict[int, tuple[str, str]]:
    """``{entity_id: (canonical_name, source)}`` for the targeted sources."""
    placeholders = ",".join(["%s"] * len(sources))
    sql = (
        "SELECT id, canonical_name, source "
        "  FROM entities "
        f" WHERE source IN ({placeholders})"
    )
    out: dict[int, tuple[str, str]] = {}
    with conn.cursor() as cur:
        cur.execute(sql, sources)
        for entity_id, canonical, source in cur.fetchall():
            out[int(entity_id)] = (canonical or "", source or "")
    return out


def fetch_aliases_for(
    conn: psycopg.Connection,
    entity_ids: list[int],
) -> dict[int, list[str]]:
    """``{entity_id: [alias, ...]}`` for the given entity ids."""
    out: dict[int, list[str]] = defaultdict(list)
    if not entity_ids:
        return out
    with conn.cursor() as cur:
        cur.execute(
            "SELECT entity_id, alias FROM entity_aliases WHERE entity_id = ANY(%s)",
            (entity_ids,),
        )
        for entity_id, alias in cur.fetchall():
            if alias:
                out[int(entity_id)].append(alias)
    return out


def fetch_collision_groups(
    conn: psycopg.Connection,
    candidate_names: set[str],
) -> dict[str, set[str]]:
    """Return ``{lower_name: {source, ...}}`` for every name with >1 source.

    Only groups whose lowered ``canonical_name`` matches one of the
    ``candidate_names`` (lowercased) are returned — this lets us collide-
    check the new source set without scanning the full corpus.
    """
    if not candidate_names:
        return {}

    placeholder_chunks = list(candidate_names)
    out: dict[str, set[str]] = defaultdict(set)

    # Process in chunks of 5000 to keep query plans cheap.
    chunk = 5000
    with conn.cursor() as cur:
        for i in range(0, len(placeholder_chunks), chunk):
            batch = placeholder_chunks[i : i + chunk]
            cur.execute(
                """
                SELECT lower(canonical_name) AS lname, source
                  FROM entities
                 WHERE lower(canonical_name) = ANY(%s)
                """,
                (batch,),
            )
            for lname, source in cur.fetchall():
                if lname and source:
                    out[str(lname)].add(str(source))

    # Also grab alias collisions: any alias matching a candidate name
    # whose entity has a different canonical_name → collision group.
    with conn.cursor() as cur:
        for i in range(0, len(placeholder_chunks), chunk):
            batch = placeholder_chunks[i : i + chunk]
            cur.execute(
                """
                SELECT lower(ea.alias) AS lname, e.source
                  FROM entity_aliases ea
                  JOIN entities e ON ea.entity_id = e.id
                 WHERE lower(ea.alias) = ANY(%s)
                """,
                (batch,),
            )
            for lname, source in cur.fetchall():
                if lname and source:
                    out[str(lname)].add(str(source))

    return out


def apply_classifications(
    conn: psycopg.Connection,
    sources: tuple[str, ...],
    *,
    batch_size: int = 1000,
) -> dict[str, int]:
    """Classify every entity in ``sources`` and update ambiguity_class."""
    logger.info("loading target entities for sources=%s", list(sources))
    entities = fetch_target_entities(conn, sources)
    logger.info("loaded %d entities", len(entities))

    logger.info("loading aliases for target entities")
    aliases = fetch_aliases_for(conn, list(entities.keys()))
    logger.info(
        "loaded %d alias rows across %d entities",
        sum(len(v) for v in aliases.values()),
        len(aliases),
    )

    candidate_lower: set[str] = set()
    for canonical, _src in entities.values():
        if canonical:
            candidate_lower.add(canonical.strip().lower())
    for alist in aliases.values():
        for a in alist:
            if a:
                candidate_lower.add(a.strip().lower())

    logger.info(
        "fetching collision groups for %d candidate names",
        len(candidate_lower),
    )
    collisions = fetch_collision_groups(conn, candidate_lower)
    logger.info("collision groups built: %d names", len(collisions))

    counts: dict[str, int] = {"banned": 0, "homograph": 0, "domain_safe": 0, "unique": 0}
    pending: list[tuple[str, int]] = []

    for entity_id, (canonical, source) in entities.items():
        alias_list = aliases.get(entity_id, [])
        names: set[str] = set()
        if canonical:
            names.add(canonical.strip().lower())
        for a in alias_list:
            if a:
                names.add(a.strip().lower())

        all_sources: set[str] = {source} if source else set()
        collision_count = 0
        for n in names:
            sources_at_name = collisions.get(n, set())
            other_sources = sources_at_name - {source}
            if other_sources:
                collision_count += len(other_sources)
                all_sources.update(sources_at_name)

        result = classify(
            canonical_name=canonical,
            aliases=alias_list,
            source_count=max(1, len(all_sources)),
            collision_count=collision_count,
        )
        counts[result] += 1
        pending.append((result, entity_id))

        if len(pending) >= batch_size:
            with conn.cursor() as cur:
                cur.executemany(
                    "UPDATE entities "
                    "SET ambiguity_class = %s::entity_ambiguity_class "
                    "WHERE id = %s",
                    pending,
                )
            pending.clear()

    if pending:
        with conn.cursor() as cur:
            cur.executemany(
                "UPDATE entities "
                "SET ambiguity_class = %s::entity_ambiguity_class "
                "WHERE id = %s",
                pending,
            )
    conn.commit()

    logger.info("classified counts: %s", counts)
    return counts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dsn",
        default=_default_dsn(),
        help="PostgreSQL DSN (default: SCIX_TEST_DSN or dbname=scix_test)",
    )
    parser.add_argument(
        "--source",
        action="append",
        default=None,
        help="Restrict to this source (repeatable). Default: cran, bioconductor, pypi.",
    )
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument(
        "--allow-prod",
        action="store_true",
        help="Allow running against the production database.",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
    )

    sources = tuple(args.source) if args.source else DEFAULT_SOURCES

    if _is_production_dsn(args.dsn) and not args.allow_prod:
        logger.error(
            "refusing to run against production DSN %r — pass --allow-prod to override",
            args.dsn,
        )
        return 2

    logger.info("connecting to %s", args.dsn)
    with psycopg.connect(args.dsn) as conn:
        counts = apply_classifications(conn, sources, batch_size=args.batch_size)

    print("\nambiguity_class counts (target sources only):")
    for k, v in counts.items():
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
