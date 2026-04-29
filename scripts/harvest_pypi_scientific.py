#!/usr/bin/env python3
"""Harvest the scientific subset of PyPI into entity tables.

PyPI has ~800K packages, so a blanket harvest would dwarf the entities
table. Instead, this harvester filters to two overlapping cohorts and
takes the union:

* **Mention cohort** (default) — package names that already appear as
  ``source='gliner'`` software entities. Each is checked against the PyPI
  Simple API to confirm it exists, then the per-package JSON is fetched
  for metadata (description, homepage, license, classifiers).
* **Classifier cohort** — packages on PyPI that carry a
  ``Topic :: Scientific/Engineering`` classifier (or any subtopic).
  Enabled with ``--include-simple-index``; explores up to
  ``--max-candidates`` extra names from the PyPI simple index.

Both cohorts are rate-limited and run through a thread pool. PyPI does
not enforce strict per-IP limits but advises users to be reasonable; see
https://warehouse.pypa.io/api-reference/json.html.

Safety: the default DSN is ``SCIX_TEST_DSN`` or ``dbname=scix_test``.
``--allow-prod`` writes to production and requires a systemd scope
(``scix-batch``) to coexist with the gascity OOM supervisor.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable

import psycopg

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

PYPI_SIMPLE_URL = "https://pypi.org/simple/"
PYPI_JSON_TEMPLATE = "https://pypi.org/pypi/{name}/json"
SOURCE = "pypi"
ID_SCHEME = "pypi_package"
SCIENTIFIC_TOPIC_PREFIX = "Topic :: Scientific/Engineering"
NAME_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]+$")

_PRODUCTION_DB_NAMES = {"scix"}

_client: ResilientClient | None = None
_client_lock = threading.Lock()


def _get_client() -> ResilientClient:
    """Return a shared ResilientClient instance."""
    global _client
    with _client_lock:
        if _client is None:
            _client = ResilientClient(
                user_agent="scix-experiments/1.0 (research; gibsonsteph42@gmail.com)",
                max_retries=3,
                backoff_base=2.0,
                rate_limit=20.0,
                timeout=60.0,
            )
        return _client


def _is_production_dsn(dsn: str) -> bool:
    for token in dsn.split():
        if "=" in token:
            key, _, value = token.partition("=")
            if key.strip() == "dbname" and value.strip() in _PRODUCTION_DB_NAMES:
                return True
    return False


def download_simple_index() -> set[str]:
    """Fetch every package name on PyPI via the JSON Simple API."""
    url = PYPI_SIMPLE_URL
    client = _get_client()
    response = client.get(
        url,
        headers={"Accept": "application/vnd.pypi.simple.v1+json"},
    )
    data = response.json()
    projects = data.get("projects", [])
    names = {str(p.get("name", "")).strip().lower() for p in projects if p.get("name")}
    logger.info("PyPI simple index: %d packages", len(names))
    return names


def load_gliner_candidates(
    conn: psycopg.Connection,
    *,
    min_mentions: int = 5,
    name_pattern: re.Pattern[str] = NAME_PATTERN,
    max_candidates: int | None = None,
    use_mention_count: bool = False,
) -> dict[str, int]:
    """Return ``{candidate_name_lower: mention_count}`` from existing data.

    Default: just enumerate gliner software names whose shape looks like a
    PyPI package (pattern ``[A-Za-z][A-Za-z0-9_.-]+``, length 3-40) — fast,
    uses only the ``entities`` table.

    With ``use_mention_count=True``, joins ``document_entities`` to weight
    candidates by paper mentions and require ``min_mentions`` distinct
    papers. That join is expensive (74M+ gliner doc rows, 14 GB table) and
    the helper index can be INVALID, so it is opt-in.
    """
    if use_mention_count:
        sql = """
            SELECT lower(e.canonical_name) AS name,
                   COUNT(DISTINCT de.bibcode) AS papers
              FROM entities e
              JOIN document_entities de ON de.entity_id = e.id
             WHERE e.source = 'gliner'
               AND e.entity_type = 'software'
               AND e.canonical_name ~ %s
               AND length(e.canonical_name) BETWEEN 3 AND 40
             GROUP BY lower(e.canonical_name)
            HAVING COUNT(DISTINCT de.bibcode) >= %s
             ORDER BY papers DESC
        """
        if max_candidates is not None:
            sql += "\n LIMIT %s"
            params: tuple = (name_pattern.pattern, min_mentions, max_candidates)
        else:
            params = (name_pattern.pattern, min_mentions)
    else:
        sql = """
            SELECT DISTINCT lower(canonical_name) AS name
              FROM entities
             WHERE source = 'gliner'
               AND entity_type = 'software'
               AND canonical_name ~ %s
               AND length(canonical_name) BETWEEN 3 AND 40
             ORDER BY name
        """
        if max_candidates is not None:
            sql += "\n LIMIT %s"
            params = (name_pattern.pattern, max_candidates)
        else:
            params = (name_pattern.pattern,)

    out: dict[str, int] = {}
    with conn.cursor() as cur:
        cur.execute(sql, params)
        for row in cur.fetchall():
            if use_mention_count:
                name, papers = row
                out[str(name)] = int(papers)
            else:
                (name,) = row
                out[str(name)] = 0
    logger.info(
        "Loaded %d gliner candidates (use_mention_count=%s, min_mentions=%d)",
        len(out),
        use_mention_count,
        min_mentions,
    )
    return out


def fetch_package_json(name: str) -> dict[str, Any] | None:
    """Fetch one package JSON document from PyPI; return None on 404."""
    client = _get_client()
    try:
        response = client.get(PYPI_JSON_TEMPLATE.format(name=name))
    except Exception as exc:  # noqa: BLE001 - resilient client wraps requests errors
        logger.debug("PyPI JSON fetch failed for %s: %s", name, exc)
        return None
    status = getattr(response, "status_code", 200)
    if status >= 400:
        return None
    try:
        return response.json()
    except Exception:
        return None


def _has_scientific_classifier(classifiers: Iterable[str]) -> bool:
    for c in classifiers:
        if isinstance(c, str) and c.startswith(SCIENTIFIC_TOPIC_PREFIX):
            return True
    return False


def _extract_topics(classifiers: Iterable[str]) -> list[str]:
    """Extract the leaf segments of every ``Topic :: ...`` classifier."""
    topics: list[str] = []
    for c in classifiers:
        if not isinstance(c, str) or not c.startswith("Topic ::"):
            continue
        parts = [p.strip() for p in c.split("::")]
        if len(parts) >= 2:
            topics.append(" / ".join(parts[1:]))
    return topics


def parse_pypi_doc(
    doc: dict[str, Any],
    *,
    fallback_name: str,
    mention_count: int = 0,
) -> dict[str, Any] | None:
    """Parse a single PyPI JSON document into an entity record.

    Returns ``None`` when the package fails the cohort filter — i.e. it
    has no Scientific/Engineering classifier and no recorded mentions.
    """
    info = doc.get("info") if isinstance(doc, dict) else None
    if not isinstance(info, dict):
        return None

    name = (info.get("name") or fallback_name or "").strip()
    if not name:
        return None

    classifiers = info.get("classifiers") or []
    if not isinstance(classifiers, list):
        classifiers = []

    is_scientific = _has_scientific_classifier(classifiers)
    if not is_scientific and mention_count <= 0:
        return None

    summary = (info.get("summary") or "").strip() or None
    description = summary
    license_ = (info.get("license") or "").strip() or None
    homepage = (info.get("home_page") or "").strip() or None
    project_urls = info.get("project_urls") or {}
    if not homepage and isinstance(project_urls, dict):
        for key in ("Homepage", "Home", "homepage", "Source", "Repository"):
            val = project_urls.get(key)
            if isinstance(val, str) and val.strip():
                homepage = val.strip()
                break

    version = (info.get("version") or "").strip() or None
    author = (info.get("author") or "").strip() or None
    topics = _extract_topics(classifiers)

    properties: dict[str, Any] = {}
    if summary:
        properties["title"] = summary
    if description:
        properties["description"] = description
    if license_:
        properties["license"] = license_
    if homepage:
        properties["homepage"] = homepage
    if version:
        properties["version"] = version
    if author:
        properties["author"] = author
    if topics:
        properties["topics"] = topics
    if is_scientific:
        properties["scientific_classifier"] = True
    if mention_count > 0:
        properties["paper_mentions"] = mention_count

    aliases: list[str] = []
    lower = name.lower()
    if lower != name:
        aliases.append(lower)

    return {
        "canonical_name": name,
        "entity_type": "software",
        "source": SOURCE,
        "external_id": name,
        "aliases": aliases,
        "properties": properties,
        "metadata": properties,
    }


def harvest_candidates(
    candidates: list[tuple[str, int]],
    *,
    workers: int = 16,
    progress_every: int = 500,
) -> list[dict[str, Any]]:
    """Fetch /pypi/<name>/json for each candidate, parsing in parallel."""
    out: list[dict[str, Any]] = []
    seen: set[str] = set()

    def worker(name: str, mentions: int) -> dict[str, Any] | None:
        doc = fetch_package_json(name)
        if doc is None:
            return None
        return parse_pypi_doc(doc, fallback_name=name, mention_count=mentions)

    processed = 0
    accepted = 0
    rejected = 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(worker, name, mentions): name
            for name, mentions in candidates
        }
        for fut in as_completed(futures):
            name = futures[fut]
            processed += 1
            try:
                record = fut.result()
            except Exception as exc:
                logger.debug("worker error on %s: %s", name, exc)
                record = None

            if record is None:
                rejected += 1
            else:
                key = record["canonical_name"].lower()
                if key in seen:
                    continue
                seen.add(key)
                out.append(record)
                accepted += 1

            if processed % progress_every == 0:
                logger.info(
                    "PyPI fetch progress: %d/%d (%d accepted, %d filtered)",
                    processed,
                    len(candidates),
                    accepted,
                    rejected,
                )

    logger.info(
        "PyPI fetch complete: %d candidates → %d accepted",
        len(candidates),
        accepted,
    )
    return out


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
            discipline=None,
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


def build_candidate_list(
    conn: psycopg.Connection,
    *,
    min_mentions: int,
    max_candidates: int | None,
    include_simple_index: bool,
    use_mention_count: bool = False,
) -> list[tuple[str, int]]:
    """Build the ordered candidate list of (name, mention_count) pairs."""
    gliner = load_gliner_candidates(
        conn,
        min_mentions=min_mentions,
        max_candidates=max_candidates,
        use_mention_count=use_mention_count,
    )

    pypi_names = download_simple_index()
    intersect = [(name, gliner[name]) for name in gliner if name in pypi_names]
    intersect.sort(key=lambda x: -x[1])
    logger.info("Mention-cohort candidates (in PyPI): %d", len(intersect))

    if not include_simple_index:
        return intersect

    mention_set = {n for n, _ in intersect}
    classifier_pool = sorted(pypi_names - mention_set)
    cap = max_candidates - len(intersect) if max_candidates else None
    if cap is not None and cap > 0:
        classifier_pool = classifier_pool[:cap]
    classifier_pairs = [(name, 0) for name in classifier_pool]
    logger.info(
        "Classifier-cohort candidates (need JSON to confirm): %d",
        len(classifier_pairs),
    )

    return intersect + classifier_pairs


def run_harvest(
    dsn: str | None = None,
    *,
    min_mentions: int = 5,
    max_candidates: int | None = 30000,
    workers: int = 16,
    include_simple_index: bool = False,
    dry_run: bool = False,
    refresh_views: bool = True,
    use_mention_count: bool = False,
) -> int:
    """Run the full PyPI scientific harvest pipeline."""
    t0 = time.monotonic()

    conn = get_connection(dsn)
    try:
        candidates = build_candidate_list(
            conn,
            min_mentions=min_mentions,
            max_candidates=max_candidates,
            include_simple_index=include_simple_index,
            use_mention_count=use_mention_count,
        )
    finally:
        conn.close()

    if not candidates:
        logger.warning("No PyPI candidates produced — nothing to harvest")
        return 0

    if dry_run:
        logger.info("Dry run — would fetch %d candidates", len(candidates))
        return len(candidates)

    entries = harvest_candidates(candidates, workers=workers)
    if not entries:
        logger.warning("No accepted PyPI packages — exiting without DB writes")
        return 0

    conn = get_connection(dsn)
    run_log = HarvestRunLog(conn, SOURCE)
    try:
        run_log.start(
            config={
                "min_mentions": min_mentions,
                "max_candidates": max_candidates,
                "include_simple_index": include_simple_index,
                "workers": workers,
            }
        )
        bulk_load(conn, entries)
        graph_count = _write_entity_graph(conn, entries, run_log.run_id)
        run_log.complete(
            records_fetched=len(candidates),
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
        "PyPI scientific harvest complete: %d entities upserted in %.1fs",
        graph_count,
        elapsed,
    )
    return graph_count


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Harvest the scientific subset of PyPI into entities",
    )
    parser.add_argument(
        "--dsn",
        default=os.environ.get("SCIX_TEST_DSN") or "dbname=scix_test",
        help="PostgreSQL DSN (default: SCIX_TEST_DSN or dbname=scix_test)",
    )
    parser.add_argument(
        "--min-mentions",
        type=int,
        default=5,
        help="Minimum distinct-paper count for the GLiNER mention cohort (default: 5)",
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=30000,
        help="Cap on number of candidates explored (default: 30000)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=16,
        help="Thread pool size for parallel /pypi/<name>/json fetches (default: 16)",
    )
    parser.add_argument(
        "--include-simple-index",
        action="store_true",
        help=(
            "In addition to the mention cohort, probe PyPI's simple index "
            "for Scientific/Engineering-classified packages."
        ),
    )
    parser.add_argument(
        "--use-mention-count",
        action="store_true",
        help=(
            "Weight gliner candidates by paper-mention count "
            "(joins document_entities — slow, requires healthy index)."
        ),
    )
    parser.add_argument(
        "--allow-prod",
        action="store_true",
        help="Allow writes to the production database (requires systemd scope).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build candidate list and report counts without DB writes.",
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
        min_mentions=args.min_mentions,
        max_candidates=args.max_candidates,
        workers=args.workers,
        include_simple_index=args.include_simple_index,
        dry_run=args.dry_run,
        refresh_views=not args.no_refresh_views,
        use_mention_count=args.use_mention_count,
    )
    if args.dry_run:
        print(f"Dry run: {count} PyPI candidates queued (not fetched)")
    else:
        print(f"Loaded {count} PyPI scientific packages into entities")
    return 0


if __name__ == "__main__":
    sys.exit(main())
