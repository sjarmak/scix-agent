#!/usr/bin/env python3
"""Populate ``entities.link_policy`` across the entity graph.

Applies provenance-based and ambiguity-based rules to determine which
entities are eligible for tier-1 keyword matching (``'open'``), which
require context validation (``'context_required'``), and which are
banned from automatic linking (``'banned'``).

Usage::

    SCIX_TEST_DSN=dbname=scix_test python scripts/set_link_policy.py
    python scripts/set_link_policy.py --allow-prod

Safety: the default DSN is ``SCIX_TEST_DSN`` or ``dbname=scix_test``.
The script refuses to run against production unless ``--allow-prod``
is explicitly passed.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from collections import Counter
from pathlib import Path

import psycopg

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from scix.db import is_production_dsn, redact_dsn  # noqa: E402
from scix.link_policy import LinkPolicy, determine_link_policy  # noqa: E402

logger = logging.getLogger("set_link_policy")


def _default_dsn() -> str:
    return os.environ.get("SCIX_TEST_DSN") or "dbname=scix_test"


def _fetch_entities(
    conn: psycopg.Connection,
) -> list[tuple[int, str, str, str | None, dict]]:
    """Return ``[(id, source, canonical_name, ambiguity_class, properties)]``."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT id, COALESCE(source, ''), canonical_name, "
            "       ambiguity_class::text, COALESCE(properties, '{}'::jsonb) "
            "FROM entities"
        )
        return [(row[0], row[1], row[2] or "", row[3], row[4]) for row in cur]


def _apply_updates(
    conn: psycopg.Connection,
    batch: list[tuple[str, int]],
) -> None:
    """Bulk-apply ``(link_policy, entity_id)`` updates."""
    if not batch:
        return
    with conn.cursor() as cur:
        cur.executemany(
            "UPDATE entities SET link_policy = %s::entity_link_policy WHERE id = %s",
            batch,
        )


def set_all_link_policies(
    conn: psycopg.Connection,
    batch_size: int = 1000,
    progress_every: int = 10_000,
) -> Counter[LinkPolicy]:
    """Classify every entity and populate ``link_policy``.

    Returns a count of entities per policy value.
    """
    logger.info("loading entities")
    entities = _fetch_entities(conn)
    logger.info("loaded %d entities", len(entities))

    counts: Counter[LinkPolicy] = Counter()
    pending: list[tuple[str, int]] = []
    total = len(entities)

    for i, (entity_id, source, canonical, ambiguity, props) in enumerate(entities, 1):
        policy = determine_link_policy(
            source=source,
            canonical_name=canonical,
            ambiguity_class=ambiguity,
            properties=props,
        )
        counts[policy] += 1
        pending.append((policy, entity_id))

        if len(pending) >= batch_size:
            _apply_updates(conn, pending)
            pending = []

        if i % progress_every == 0:
            logger.info("  classified %d / %d", i, total)

    _apply_updates(conn, pending)
    conn.commit()

    logger.info("done — %s", dict(counts))
    return counts


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dsn",
        default=_default_dsn(),
        help="PostgreSQL DSN (default: SCIX_TEST_DSN or dbname=scix_test)",
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

    effective_dsn = args.dsn or _default_dsn()
    if is_production_dsn(effective_dsn) and not args.allow_prod:
        logger.error(
            "refusing to run against production DSN %r — pass --allow-prod to override",
            redact_dsn(effective_dsn),
        )
        return 2

    logger.info("connecting to %s", redact_dsn(effective_dsn))
    with psycopg.connect(effective_dsn) as conn:
        counts = set_all_link_policies(conn, batch_size=args.batch_size)

        print("\nlink_policy counts:")
        for policy in ("open", "context_required", "llm_only", "banned"):
            print(f"  {policy}: {counts.get(policy, 0)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
