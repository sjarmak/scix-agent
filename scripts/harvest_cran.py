#!/usr/bin/env python3
"""Harvest the CRAN (Comprehensive R Archive Network) catalog into entity tables.

Downloads the full CRAN package catalog from https://crandb.r-pkg.org/-/all
and ingests it as ``entity_type='software'`` rows in two places:

* ``entity_dictionary`` (legacy compatibility, via :func:`scix.dictionary.bulk_load`).
* ``entities`` / ``entity_aliases`` / ``entity_identifiers`` (graph tables, via
  :mod:`scix.harvest_utils` helpers) — same shape as the bio.tools harvester.

The :data:`entities.properties` JSONB column carries ``homepage``, ``license``,
``description``, and ``author`` extracted from the latest DESCRIPTION entry.

The script writes a ``harvest_runs`` row (``source='cran'``) and refreshes
agent materialized views on completion (skip with ``--no-refresh-views``).

Safety: the default DSN is ``SCIX_TEST_DSN`` or ``dbname=scix_test``. Pass
``--allow-prod`` to write to production. ``--allow-prod`` requires the script
to run inside a systemd scope (use ``scix-batch``), checked via the
``INVOCATION_ID`` env var that systemd-run sets automatically.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from scix.db import get_connection
from scix.dictionary import bulk_load
from scix.harvest_utils import (
    HarvestRunLog,
    upsert_entity,
    upsert_entity_alias,
    upsert_entity_identifier,
)
from scix.http_client import ResilientClient

logger = logging.getLogger(__name__)

CRAN_ALL_URL = "https://crandb.r-pkg.org/-/all"
CRAN_DESC_URL = "https://crandb.r-pkg.org/-/desc"
SOURCE = "cran"
ID_SCHEME = "cran_package"

_PRODUCTION_DB_NAMES = {"scix"}

_client: ResilientClient | None = None


def _get_client() -> ResilientClient:
    """Return a shared ResilientClient instance."""
    global _client
    if _client is None:
        _client = ResilientClient(
            user_agent="scix-experiments/1.0",
            max_retries=3,
            backoff_base=2.0,
            rate_limit=5.0,
            timeout=300.0,
        )
    return _client


def _is_production_dsn(dsn: str) -> bool:
    for token in dsn.split():
        if "=" in token:
            key, _, value = token.partition("=")
            if key.strip() == "dbname" and value.strip() in _PRODUCTION_DB_NAMES:
                return True
    return False


def download_catalog(url: str = CRAN_ALL_URL) -> dict[str, Any]:
    """Download the CRAN package catalog as a dict keyed by package name."""
    client = _get_client()
    response = client.get(url)
    data = response.json()
    if not isinstance(data, dict):
        raise RuntimeError(f"Unexpected CRAN response type: {type(data).__name__}")
    logger.info("Downloaded CRAN catalog: %d packages", len(data))
    return data


def _normalize_url(url: str | None) -> str | None:
    """Strip CRAN's embedded newlines and trailing whitespace from URL fields."""
    if not url:
        return None
    cleaned = " ".join(url.split())
    if "," in cleaned:
        cleaned = cleaned.split(",", 1)[0].strip()
    return cleaned or None


def _normalize_text(text: str | None) -> str | None:
    """Collapse whitespace in DESCRIPTION free-text fields."""
    if not text:
        return None
    cleaned = " ".join(text.split())
    return cleaned or None


def parse_catalog(raw: dict[str, Any]) -> list[dict[str, Any]]:
    """Parse the CRAN /-/all response into entity records.

    Each returned dict is compatible with both :func:`scix.dictionary.bulk_load`
    (uses ``metadata``) and :func:`scix.harvest_utils.upsert_entity`
    (uses ``properties``). The two payloads share the same content.
    """
    entries: list[dict[str, Any]] = []
    skipped = 0

    for name, doc in raw.items():
        package_name = (doc.get("name") or name or "").strip()
        if not package_name:
            skipped += 1
            continue

        latest_version = (doc.get("latest") or "").strip() or None
        versions = doc.get("versions") or {}
        if latest_version and isinstance(versions, dict) and latest_version in versions:
            desc = versions[latest_version]
        else:
            desc = doc

        title = _normalize_text(desc.get("Title")) or _normalize_text(doc.get("title"))
        description = _normalize_text(desc.get("Description"))
        license_ = _normalize_text(desc.get("License"))
        url = _normalize_url(desc.get("URL"))
        bug_reports = _normalize_url(desc.get("BugReports"))
        author = _normalize_text(desc.get("Author")) or _normalize_text(desc.get("Authors@R"))

        properties: dict[str, Any] = {}
        if title:
            properties["title"] = title
        if description:
            properties["description"] = description
        if license_:
            properties["license"] = license_
        if url:
            properties["homepage"] = url
        if bug_reports:
            properties["bug_reports"] = bug_reports
        if author:
            properties["author"] = author
        if latest_version:
            properties["version"] = latest_version

        aliases: list[str] = []
        lower = package_name.lower()
        if lower != package_name:
            aliases.append(lower)

        entries.append(
            {
                "canonical_name": package_name,
                "entity_type": "software",
                "source": SOURCE,
                "external_id": package_name,
                "aliases": aliases,
                "properties": properties,
                "metadata": properties,
            }
        )

    if skipped:
        logger.warning("Skipped %d CRAN entries missing a package name", skipped)

    logger.info("Parsed %d CRAN packages into entity records", len(entries))
    return entries


def _write_entity_graph(
    conn: Any,
    entries: list[dict[str, Any]],
    harvest_run_id: int,
    discipline: str | None = None,
) -> int:
    """Upsert each entry into entities/entity_aliases/entity_identifiers."""
    count = 0
    for entry in entries:
        entity_id = upsert_entity(
            conn,
            canonical_name=entry["canonical_name"],
            entity_type=entry["entity_type"],
            source=entry["source"],
            discipline=discipline,
            harvest_run_id=harvest_run_id,
            properties=entry.get("properties", {}),
        )

        external_id = entry.get("external_id")
        if external_id:
            upsert_entity_identifier(
                conn,
                entity_id=entity_id,
                id_scheme=ID_SCHEME,
                external_id=external_id,
                is_primary=True,
            )

        for alias in entry.get("aliases", []):
            upsert_entity_alias(
                conn,
                entity_id=entity_id,
                alias=alias,
                alias_source=SOURCE,
            )

        count += 1

    conn.commit()
    return count


def run_harvest(
    dsn: str | None = None,
    *,
    dry_run: bool = False,
    catalog_url: str = CRAN_ALL_URL,
    refresh_views: bool = True,
) -> int:
    """Run the full CRAN harvest pipeline.

    Returns the number of entities upserted into ``entities``. In dry-run
    mode, returns the number of parsed entries without touching the DB.
    """
    t0 = time.monotonic()
    raw = download_catalog(catalog_url)
    entries = parse_catalog(raw)

    if dry_run:
        logger.info("Dry run — would load %d entries", len(entries))
        return len(entries)

    conn = get_connection(dsn)
    run_log = HarvestRunLog(conn, SOURCE)
    try:
        run_log.start(config={"catalog_url": catalog_url})
        bulk_load(conn, entries)
        graph_count = _write_entity_graph(conn, entries, run_log.run_id)
        run_log.complete(
            records_fetched=len(entries),
            records_upserted=graph_count,
            counts={"software": graph_count},
            refresh_views=refresh_views,
        )
    except Exception as exc:
        try:
            run_log.fail(str(exc))
        except Exception:
            logger.warning("Failed to mark harvest run as failed")
        raise
    finally:
        conn.close()

    elapsed = time.monotonic() - t0
    logger.info("CRAN harvest complete: %d entities upserted in %.1fs", graph_count, elapsed)
    return graph_count


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Harvest CRAN package catalog into the entities table",
    )
    parser.add_argument(
        "--dsn",
        default=os.environ.get("SCIX_TEST_DSN") or "dbname=scix_test",
        help="PostgreSQL DSN (default: SCIX_TEST_DSN or dbname=scix_test)",
    )
    parser.add_argument(
        "--allow-prod",
        action="store_true",
        help="Allow writes to the production database (requires systemd scope).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and report counts without DB writes.",
    )
    parser.add_argument(
        "--catalog-url",
        default=CRAN_ALL_URL,
        help=f"CRAN catalog URL (default: {CRAN_ALL_URL})",
    )
    parser.add_argument(
        "--no-refresh-views",
        action="store_true",
        help="Skip the post-harvest agent-views refresh (run separately later).",
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
            "refusing to run against production DSN %r — pass --allow-prod to override",
            args.dsn,
        )
        return 2
    if args.allow_prod and _is_production_dsn(args.dsn) and not os.environ.get("INVOCATION_ID"):
        logger.error(
            "--allow-prod requires a systemd scope. Invoke via: "
            "scix-batch python %s <args...>",
            Path(sys.argv[0]).name,
        )
        return 2

    count = run_harvest(
        dsn=args.dsn,
        dry_run=args.dry_run,
        catalog_url=args.catalog_url,
        refresh_views=not args.no_refresh_views,
    )
    if args.dry_run:
        print(f"Dry run: {count} CRAN packages would be loaded")
    else:
        print(f"Loaded {count} CRAN packages into entities")
    return 0


if __name__ == "__main__":
    sys.exit(main())
