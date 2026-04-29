#!/usr/bin/env python3
"""Harvest the Bioconductor software catalog into entity tables.

Bioconductor publishes one ``packages.json`` per release containing the
DESCRIPTION fields for every software package, plus ``biocViews`` topic tags.
This harvester:

* Downloads ``https://bioconductor.org/packages/json/<release>/bioc/packages.json``.
* Parses each package into an entity record (``entity_type='software'``,
  ``source='bioconductor'``, ``discipline='life_sciences'``).
* Writes to ``entity_dictionary`` (legacy) and to
  ``entities`` / ``entity_aliases`` / ``entity_identifiers`` (graph tables).

The :data:`entities.properties` JSONB column carries ``homepage``, ``license``,
``description``, ``biocViews`` (topic tags), and the release version.

Safety: the default DSN is ``SCIX_TEST_DSN`` or ``dbname=scix_test``. Pass
``--allow-prod`` to write to production (requires running inside a systemd
scope, i.e. ``scix-batch``).
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

DEFAULT_RELEASE = "3.20"
SOURCE = "bioconductor"
ID_SCHEME = "bioconductor_package"
DISCIPLINE = "life_sciences"

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
            timeout=120.0,
        )
    return _client


def _is_production_dsn(dsn: str) -> bool:
    for token in dsn.split():
        if "=" in token:
            key, _, value = token.partition("=")
            if key.strip() == "dbname" and value.strip() in _PRODUCTION_DB_NAMES:
                return True
    return False


def catalog_url(release: str = DEFAULT_RELEASE, repo: str = "bioc") -> str:
    """Return the packages.json URL for a given release + repository."""
    return f"https://bioconductor.org/packages/json/{release}/{repo}/packages.json"


def download_catalog(release: str = DEFAULT_RELEASE, repo: str = "bioc") -> dict[str, Any]:
    """Download the Bioconductor packages.json catalog."""
    url = catalog_url(release, repo)
    client = _get_client()
    response = client.get(url)
    data = response.json()
    if not isinstance(data, dict):
        raise RuntimeError(f"Unexpected Bioconductor response: {type(data).__name__}")
    logger.info("Downloaded Bioconductor %s/%s catalog: %d packages", release, repo, len(data))
    return data


def _normalize_text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, list):
        value = ", ".join(str(v) for v in value if v is not None)
    text = str(value).strip()
    if not text:
        return None
    return " ".join(text.split())


def _normalize_url(value: Any) -> str | None:
    text = _normalize_text(value)
    if not text:
        return None
    if "," in text:
        text = text.split(",", 1)[0].strip()
    return text or None


def parse_catalog(
    raw: dict[str, Any],
    *,
    release: str = DEFAULT_RELEASE,
) -> list[dict[str, Any]]:
    """Parse the Bioconductor catalog into entity records."""
    entries: list[dict[str, Any]] = []
    skipped = 0

    for name, doc in raw.items():
        package_name = (doc.get("Package") or name or "").strip()
        if not package_name:
            skipped += 1
            continue

        title = _normalize_text(doc.get("Title"))
        description = _normalize_text(doc.get("Description"))
        license_ = _normalize_text(doc.get("License"))
        url = _normalize_url(doc.get("URL"))
        version = _normalize_text(doc.get("Version"))
        biocviews_raw = doc.get("biocViews") or []
        if isinstance(biocviews_raw, str):
            biocviews = [v.strip() for v in biocviews_raw.split(",") if v.strip()]
        elif isinstance(biocviews_raw, list):
            biocviews = [str(v).strip() for v in biocviews_raw if str(v).strip()]
        else:
            biocviews = []

        author = _normalize_text(doc.get("Author"))
        maintainer = _normalize_text(doc.get("Maintainer"))
        git_url = _normalize_url(doc.get("git_url"))

        properties: dict[str, Any] = {"release": release}
        if title:
            properties["title"] = title
        if description:
            properties["description"] = description
        if license_:
            properties["license"] = license_
        if url:
            properties["homepage"] = url
        if git_url:
            properties["git_url"] = git_url
        if biocviews:
            properties["biocviews"] = biocviews
        if author:
            properties["author"] = author
        if maintainer:
            properties["maintainer"] = maintainer
        if version:
            properties["version"] = version

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
        logger.warning("Skipped %d Bioconductor entries missing a package name", skipped)

    logger.info("Parsed %d Bioconductor packages into entity records", len(entries))
    return entries


def _write_entity_graph(
    conn: Any,
    entries: list[dict[str, Any]],
    harvest_run_id: int,
) -> int:
    count = 0
    for entry in entries:
        entity_id = upsert_entity(
            conn,
            canonical_name=entry["canonical_name"],
            entity_type=entry["entity_type"],
            source=entry["source"],
            discipline=DISCIPLINE,
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
    release: str = DEFAULT_RELEASE,
    repo: str = "bioc",
    dry_run: bool = False,
    refresh_views: bool = True,
) -> int:
    """Run the full Bioconductor harvest pipeline."""
    t0 = time.monotonic()
    raw = download_catalog(release=release, repo=repo)
    entries = parse_catalog(raw, release=release)

    if dry_run:
        logger.info("Dry run — would load %d entries", len(entries))
        return len(entries)

    conn = get_connection(dsn)
    run_log = HarvestRunLog(conn, SOURCE)
    try:
        run_log.start(config={"release": release, "repo": repo})
        bulk_load(conn, entries, discipline=DISCIPLINE)
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
    logger.info(
        "Bioconductor harvest complete: %d entities upserted in %.1fs",
        graph_count,
        elapsed,
    )
    return graph_count


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Harvest the Bioconductor software catalog into entities",
    )
    parser.add_argument(
        "--dsn",
        default=os.environ.get("SCIX_TEST_DSN") or "dbname=scix_test",
        help="PostgreSQL DSN (default: SCIX_TEST_DSN or dbname=scix_test)",
    )
    parser.add_argument(
        "--release",
        default=DEFAULT_RELEASE,
        help=f"Bioconductor release version (default: {DEFAULT_RELEASE})",
    )
    parser.add_argument(
        "--repo",
        default="bioc",
        choices=["bioc", "data/annotation", "data/experiment", "workflows"],
        help="Bioconductor repository (default: bioc — software packages)",
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
        release=args.release,
        repo=args.repo,
        dry_run=args.dry_run,
        refresh_views=not args.no_refresh_views,
    )
    if args.dry_run:
        print(f"Dry run: {count} Bioconductor packages would be loaded")
    else:
        print(f"Loaded {count} Bioconductor packages into entities")
    return 0


if __name__ == "__main__":
    sys.exit(main())
