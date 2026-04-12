#!/usr/bin/env python3
"""Populate ``entities.ambiguity_class`` across the entity graph.

Runs a two-pass classification pipeline against a scix database and
writes a 50-example-per-class spot-check audit report to
``build-artifacts/ambiguity_audit.md``.

Usage::

    SCIX_TEST_DSN=dbname=scix_test \
      python scripts/classify_entity_ambiguity.py

Safety: the default DSN is ``SCIX_TEST_DSN`` or ``dbname=scix_test``.
The script refuses to run against production unless ``--allow-prod``
is explicitly passed.
"""

from __future__ import annotations

import argparse
import logging
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

import psycopg

# Ensure repo's src/ is importable when invoked as a script.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from scix.ambiguity import AmbiguityClass, classify  # noqa: E402

logger = logging.getLogger("classify_entity_ambiguity")

_PRODUCTION_DB_NAMES = {"scix"}

_AUDIT_CLASSES: tuple[AmbiguityClass, ...] = (
    "banned",
    "homograph",
    "domain_safe",
    "unique",
)


def _is_production_dsn(dsn: str) -> bool:
    for token in dsn.split():
        if "=" in token:
            key, _, value = token.partition("=")
            if key.strip() == "dbname" and value.strip() in _PRODUCTION_DB_NAMES:
                return True
    return False


def _default_dsn() -> str:
    return os.environ.get("SCIX_TEST_DSN") or "dbname=scix_test"


def _fetch_entities(conn: psycopg.Connection) -> dict[int, tuple[str, str]]:
    """Return ``{entity_id: (canonical_name, source)}``."""
    out: dict[int, tuple[str, str]] = {}
    with conn.cursor() as cur:
        cur.execute("SELECT id, canonical_name, source FROM entities")
        for row in cur:
            entity_id, canonical, source = row
            out[int(entity_id)] = (canonical or "", source or "")
    return out


def _fetch_aliases(conn: psycopg.Connection) -> dict[int, list[str]]:
    """Return ``{entity_id: [alias, ...]}``."""
    out: dict[int, list[str]] = defaultdict(list)
    with conn.cursor() as cur:
        cur.execute("SELECT entity_id, alias FROM entity_aliases")
        for row in cur:
            entity_id, alias = row
            if alias:
                out[int(entity_id)].append(alias)
    return out


def _build_name_index(
    entities: dict[int, tuple[str, str]],
    aliases: dict[int, list[str]],
) -> dict[str, set[int]]:
    """Map every lowercased canonical/alias name to the set of entity ids."""
    index: dict[str, set[int]] = defaultdict(set)
    for entity_id, (canonical, _src) in entities.items():
        if canonical:
            index[canonical.strip().lower()].add(entity_id)
    for entity_id, alias_list in aliases.items():
        for alias in alias_list:
            if alias:
                index[alias.strip().lower()].add(entity_id)
    return index


def _names_for_entity(canonical: str, alias_list: list[str]) -> set[str]:
    names: set[str] = set()
    if canonical:
        names.add(canonical.strip().lower())
    for alias in alias_list:
        if alias:
            names.add(alias.strip().lower())
    return names


def _apply_updates(
    conn: psycopg.Connection,
    batch: list[tuple[str, int]],
) -> None:
    """Bulk-apply a batch of (ambiguity_class, entity_id) updates."""
    if not batch:
        return
    with conn.cursor() as cur:
        cur.executemany(
            "UPDATE entities " "SET ambiguity_class = %s::entity_ambiguity_class " "WHERE id = %s",
            batch,
        )


def classify_all(
    conn: psycopg.Connection,
    batch_size: int = 1000,
    progress_every: int = 10_000,
) -> dict[AmbiguityClass, int]:
    """Classify every row in ``entities`` and populate ``ambiguity_class``.

    Returns a count dict keyed by class.
    """
    logger.info("pass 1/2 — loading entities + aliases")
    entities = _fetch_entities(conn)
    aliases = _fetch_aliases(conn)
    logger.info(
        "loaded %d entities, %d alias rows",
        len(entities),
        sum(len(v) for v in aliases.values()),
    )

    name_index = _build_name_index(entities, aliases)
    logger.info("built name index with %d unique names", len(name_index))

    logger.info("pass 2/2 — classifying + updating")
    counts: dict[AmbiguityClass, int] = {c: 0 for c in _AUDIT_CLASSES}
    pending: list[tuple[str, int]] = []
    processed = 0

    for entity_id, (canonical, source) in entities.items():
        alias_list = aliases.get(entity_id, [])
        names = _names_for_entity(canonical, alias_list)

        collision_ids: set[int] = set()
        for name in names:
            collision_ids.update(name_index.get(name, set()))
        collision_ids.discard(entity_id)
        collision_count = len(collision_ids)

        sources: set[str] = {source} if source else set()
        for other_id in collision_ids:
            other = entities.get(other_id)
            if other and other[1]:
                sources.add(other[1])
        source_count = max(1, len(sources))

        result = classify(
            canonical_name=canonical,
            aliases=alias_list,
            source_count=source_count,
            collision_count=collision_count,
        )
        counts[result] += 1
        pending.append((result, entity_id))
        processed += 1

        if len(pending) >= batch_size:
            _apply_updates(conn, pending)
            pending.clear()

        if processed % progress_every == 0:
            logger.info("  classified %d / %d", processed, len(entities))

    _apply_updates(conn, pending)
    conn.commit()

    logger.info("done — %s", counts)
    return counts


def _fetch_class_summary(
    conn: psycopg.Connection,
) -> list[tuple[str | None, int]]:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT ambiguity_class::text, count(*) "
            "FROM entities "
            "GROUP BY ambiguity_class "
            "ORDER BY ambiguity_class"
        )
        return [(row[0], int(row[1])) for row in cur.fetchall()]


def _sample_for_class(
    conn: psycopg.Connection,
    cls: AmbiguityClass,
    sample_size: int,
) -> list[tuple[int, str, str, str]]:
    """Fetch a random sample for a class. Uses Python-side shuffling so
    the sample is reproducible on small fixtures (ORDER BY random() can
    return fewer than requested on tiny tables — not an issue, but this
    keeps the audit stable)."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT id, canonical_name, entity_type, source "
            "FROM entities "
            "WHERE ambiguity_class = %s::entity_ambiguity_class",
            (cls,),
        )
        rows = [(int(r[0]), r[1] or "", r[2] or "", r[3] or "") for r in cur.fetchall()]
    rnd = random.Random(42)
    rnd.shuffle(rows)
    return rows[:sample_size]


def _escape_md(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ").strip()


def write_audit_report(
    conn: psycopg.Connection,
    out_path: Path,
    sample_size: int = 50,
) -> None:
    """Write a 50-per-class spot-check audit report as markdown."""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    summary = _fetch_class_summary(conn)

    lines: list[str] = []
    lines.append("# Entity Ambiguity Classification — Spot-Check Audit")
    lines.append("")
    lines.append(
        "Generated by `scripts/classify_entity_ambiguity.py`. Up to "
        f"{sample_size} random examples per class."
    )
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append("| ambiguity_class | count |")
    lines.append("| --- | --- |")
    for cls, count in summary:
        lines.append(f"| {cls or '(null)'} | {count} |")
    lines.append("")

    for cls in _AUDIT_CLASSES:
        rows = _sample_for_class(conn, cls, sample_size)
        lines.append(f"## {cls}")
        lines.append("")
        lines.append(f"Showing {len(rows)} of up to {sample_size} random examples.")
        lines.append("")
        lines.append("| id | canonical_name | entity_type | source |")
        lines.append("| --- | --- | --- | --- |")
        for eid, canonical, etype, source in rows:
            lines.append(
                f"| {eid} | {_escape_md(canonical)} | "
                f"{_escape_md(etype)} | {_escape_md(source)} |"
            )
        lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("audit report written to %s", out_path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dsn",
        default=_default_dsn(),
        help="PostgreSQL DSN (default: SCIX_TEST_DSN or dbname=scix_test)",
    )
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument(
        "--audit-out",
        type=Path,
        default=Path("build-artifacts/ambiguity_audit.md"),
    )
    parser.add_argument(
        "--allow-prod",
        action="store_true",
        help="Allow running against the production database (dangerous).",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
    )

    if _is_production_dsn(args.dsn) and not args.allow_prod:
        logger.error(
            "refusing to run against production DSN %r — pass --allow-prod " "to override",
            args.dsn,
        )
        return 2

    logger.info("connecting to %s", args.dsn)
    with psycopg.connect(args.dsn) as conn:
        classify_all(conn, batch_size=args.batch_size)

        summary = _fetch_class_summary(conn)
        print("\nambiguity_class counts:")
        for cls, count in summary:
            print(f"  {cls or '(null)'}: {count}")

        audit_path = args.audit_out
        if not audit_path.is_absolute():
            audit_path = _REPO_ROOT / audit_path
        write_audit_report(conn, audit_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
