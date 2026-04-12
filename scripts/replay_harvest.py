#!/usr/bin/env python3
"""Replay an entity harvest snapshot back into the staging tables.

A "snapshot" is a gzipped JSON-Lines file whose records have the shape:

    {
      "entity": {
        "canonical_name": "...",
        "entity_type": "...",
        "source": "...",
        "discipline": "...",          # optional
        "source_version": "...",      # optional
        "ambiguity_class": "...",     # optional
        "link_policy": "...",         # optional
        "properties": { ... }          # optional
      },
      "aliases": [
        {"alias": "...", "alias_source": "..."}
      ],
      "identifiers": [
        {"id_scheme": "...", "external_id": "...", "is_primary": false}
      ]
    }

Usage:

    SCIX_TEST_DSN=dbname=scix_test \
        python scripts/replay_harvest.py \
        --source VizieR --snapshot 2026-04-10

Snapshots live under ``data/entities/snapshots/{source}/{YYYY-MM-DD}.jsonl.gz``.

The script inserts one row per entity into ``entities_staging`` with a fresh
``staging_run_id`` tied to a newly-created ``harvest_runs`` row (status
``'replayed'``), plus rows into the aliases and identifiers staging tables.
The resulting ``run_id`` is printed to stdout so downstream tooling can
call :func:`scix.harvest_promotion.promote_harvest` on it.
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import os
import pathlib
import sys
from typing import Iterable, Iterator, Optional

import psycopg

from scix.db import DEFAULT_DSN, get_connection

logger = logging.getLogger(__name__)


SNAPSHOTS_ROOT = pathlib.Path("data/entities/snapshots")


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------


def snapshot_path(source: str, snapshot: str, root: pathlib.Path = SNAPSHOTS_ROOT) -> pathlib.Path:
    """Return the on-disk path for a given source / date snapshot."""
    return root / source / f"{snapshot}.jsonl.gz"


def iter_snapshot(path: pathlib.Path) -> Iterator[dict]:
    """Yield parsed records from a gzipped JSON-Lines snapshot."""
    if not path.exists():
        raise FileNotFoundError(f"Snapshot not found: {path}")
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: invalid JSON: {exc.msg}") from exc


def write_snapshot(records: Iterable[dict], path: pathlib.Path) -> int:
    """Write records to a gzipped JSON-Lines snapshot. Returns count written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, sort_keys=True) + "\n")
            n += 1
    return n


# ---------------------------------------------------------------------------
# Replay
# ---------------------------------------------------------------------------


def _json_or_empty(value: Optional[dict]) -> str:
    return json.dumps(value or {})


def replay_snapshot(
    source: str,
    snapshot: str,
    *,
    dsn: Optional[str] = None,
    snapshots_root: pathlib.Path = SNAPSHOTS_ROOT,
) -> int:
    """Load a snapshot into the staging tables and return the new run_id."""
    path = snapshot_path(source, snapshot, root=snapshots_root)
    records = list(iter_snapshot(path))

    conn = get_connection(dsn or DEFAULT_DSN, autocommit=False)
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO harvest_runs (source, status, config)
                VALUES (%s, 'replayed', %s::jsonb)
                RETURNING id
                """,
                (
                    source,
                    json.dumps(
                        {
                            "mode": "replay",
                            "snapshot": snapshot,
                            "path": str(path),
                        }
                    ),
                ),
            )
            run_id = cur.fetchone()[0]

            for rec in records:
                entity = rec.get("entity") or {}
                cur.execute(
                    """
                    INSERT INTO entities_staging (
                        staging_run_id, canonical_name, entity_type,
                        discipline, source, source_version,
                        ambiguity_class, link_policy, properties
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb)
                    RETURNING id
                    """,
                    (
                        run_id,
                        entity.get("canonical_name"),
                        entity.get("entity_type"),
                        entity.get("discipline"),
                        entity.get("source", source),
                        entity.get("source_version"),
                        entity.get("ambiguity_class"),
                        entity.get("link_policy"),
                        _json_or_empty(entity.get("properties")),
                    ),
                )
                staging_entity_id = cur.fetchone()[0]

                for alias in rec.get("aliases") or []:
                    cur.execute(
                        """
                        INSERT INTO entity_aliases_staging (
                            staging_run_id, staging_entity_id, canonical_name,
                            entity_type, source, alias, alias_source
                        )
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                        """,
                        (
                            run_id,
                            staging_entity_id,
                            entity.get("canonical_name"),
                            entity.get("entity_type"),
                            entity.get("source", source),
                            alias.get("alias"),
                            alias.get("alias_source"),
                        ),
                    )

                for ident in rec.get("identifiers") or []:
                    cur.execute(
                        """
                        INSERT INTO entity_identifiers_staging (
                            staging_run_id, staging_entity_id, canonical_name,
                            entity_type, source, id_scheme, external_id, is_primary
                        )
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                        """,
                        (
                            run_id,
                            staging_entity_id,
                            entity.get("canonical_name"),
                            entity.get("entity_type"),
                            entity.get("source", source),
                            ident.get("id_scheme"),
                            ident.get("external_id"),
                            bool(ident.get("is_primary", False)),
                        ),
                    )

            cur.execute(
                "UPDATE harvest_runs SET records_fetched = %s WHERE id = %s",
                (len(records), run_id),
            )
        conn.commit()
        return int(run_id)
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def dump_staging_to_snapshot(
    run_id: int,
    output_path: pathlib.Path,
    *,
    dsn: Optional[str] = None,
) -> int:
    """Write all staging rows for a run back out to a snapshot file.

    Used for round-trip validation in tests.
    """
    conn = get_connection(dsn or DEFAULT_DSN, autocommit=False)
    records: list[dict] = []
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, canonical_name, entity_type, discipline, source,
                       source_version, ambiguity_class, link_policy, properties
                  FROM entities_staging
                 WHERE staging_run_id = %s
                 ORDER BY id
                """,
                (run_id,),
            )
            entity_rows = cur.fetchall()

            for row in entity_rows:
                (
                    ent_id,
                    canonical_name,
                    entity_type,
                    discipline,
                    source,
                    source_version,
                    ambiguity_class,
                    link_policy,
                    properties,
                ) = row

                cur.execute(
                    """
                    SELECT alias, alias_source
                      FROM entity_aliases_staging
                     WHERE staging_run_id = %s AND staging_entity_id = %s
                     ORDER BY id
                    """,
                    (run_id, ent_id),
                )
                aliases = [{"alias": a, "alias_source": s} for (a, s) in cur.fetchall()]

                cur.execute(
                    """
                    SELECT id_scheme, external_id, is_primary
                      FROM entity_identifiers_staging
                     WHERE staging_run_id = %s AND staging_entity_id = %s
                     ORDER BY id
                    """,
                    (run_id, ent_id),
                )
                identifiers = [
                    {
                        "id_scheme": scheme,
                        "external_id": ext_id,
                        "is_primary": bool(prim),
                    }
                    for (scheme, ext_id, prim) in cur.fetchall()
                ]

                records.append(
                    {
                        "entity": {
                            "canonical_name": canonical_name,
                            "entity_type": entity_type,
                            "discipline": discipline,
                            "source": source,
                            "source_version": source_version,
                            "ambiguity_class": ambiguity_class,
                            "link_policy": link_policy,
                            "properties": properties or {},
                        },
                        "aliases": aliases,
                        "identifiers": identifiers,
                    }
                )
        conn.commit()
    finally:
        conn.close()

    return write_snapshot(records, output_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="Harvest source name")
    parser.add_argument("--snapshot", required=True, help="Snapshot date YYYY-MM-DD")
    parser.add_argument("--dsn", default=None, help="Database DSN (defaults to SCIX_DSN env var)")
    parser.add_argument(
        "--snapshots-root",
        default=str(SNAPSHOTS_ROOT),
        help="Root directory for snapshots",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = _parse_args(argv)
    root = pathlib.Path(args.snapshots_root)
    run_id = replay_snapshot(args.source, args.snapshot, dsn=args.dsn, snapshots_root=root)
    print(run_id)
    return 0


if __name__ == "__main__":
    sys.exit(main())
