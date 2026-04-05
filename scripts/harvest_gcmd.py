#!/usr/bin/env python3
"""Harvest GCMD (Global Change Master Directory) keywords into entity_dictionary.

Downloads GCMD keyword scheme JSON files from two sources:
  1. GitHub (adiwg/gcmd-keywords) — instruments, platforms, sciencekeywords
  2. CMR KMS API — providers (data centers), projects

Parses 5 schemes into entity_dictionary entries:
  - Instruments/Sensors → entity_type='instrument'
  - Platforms/Sources → entity_type='instrument'
  - Earth Science Keywords (leaf nodes) → entity_type='observable'
  - Data Centers (providers) → entity_type='mission'
  - Projects → entity_type='mission'

All entries get source='gcmd', discipline='earth_science'.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, NamedTuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from scix.db import get_connection
from scix.dictionary import bulk_load

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# URLs
# ---------------------------------------------------------------------------

GITHUB_BASE = "https://raw.githubusercontent.com/adiwg/gcmd-keywords/master/resources/json"
KMS_BASE = "https://cmr.earthdata.nasa.gov/kms/concepts/concept_scheme"

# ---------------------------------------------------------------------------
# Scheme configuration
# ---------------------------------------------------------------------------


class SchemeConfig(NamedTuple):
    """Configuration for a single GCMD keyword scheme."""

    name: str
    entity_type: str
    source_kind: str  # 'github' or 'kms'
    url: str
    leaves_only: bool  # True for sciencekeywords — emit only leaf nodes


SCHEME_CONFIGS: dict[str, SchemeConfig] = {
    "instruments": SchemeConfig(
        name="instruments",
        entity_type="instrument",
        source_kind="github",
        url=f"{GITHUB_BASE}/instruments.json",
        leaves_only=False,
    ),
    "platforms": SchemeConfig(
        name="platforms",
        entity_type="instrument",
        source_kind="github",
        url=f"{GITHUB_BASE}/platforms.json",
        leaves_only=False,
    ),
    "sciencekeywords": SchemeConfig(
        name="sciencekeywords",
        entity_type="observable",
        source_kind="github",
        url=f"{GITHUB_BASE}/sciencekeywords.json",
        leaves_only=True,
    ),
    "providers": SchemeConfig(
        name="providers",
        entity_type="mission",
        source_kind="kms",
        url=f"{KMS_BASE}/providers?format=json",
        leaves_only=False,
    ),
    "projects": SchemeConfig(
        name="projects",
        entity_type="mission",
        source_kind="kms",
        url=f"{KMS_BASE}/projects?format=json",
        leaves_only=False,
    ),
}


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------


def _fetch_url(url: str, timeout: int = 60) -> bytes:
    """Download a URL with retry and exponential backoff.

    Args:
        url: URL to fetch.
        timeout: Request timeout in seconds.

    Returns:
        Raw response bytes.

    Raises:
        urllib.error.URLError: If the download fails after retries.
    """
    max_retries = 3
    for attempt in range(1, max_retries + 1):
        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "scix-experiments/1.0"},
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = resp.read()
            return data
        except (urllib.error.URLError, OSError) as exc:
            if attempt < max_retries:
                wait = 2**attempt
                logger.warning(
                    "Download attempt %d/%d failed (%s): %s — retrying in %ds",
                    attempt,
                    max_retries,
                    url,
                    exc,
                    wait,
                )
                time.sleep(wait)
            else:
                logger.error(
                    "Failed to download %s after %d attempts: %s",
                    url,
                    max_retries,
                    exc,
                )
                raise
    raise RuntimeError("_fetch_url: unexpected exit from retry loop")


def download_github_scheme(url: str) -> list[dict[str, Any]]:
    """Download a GCMD scheme JSON from the GitHub repository.

    Args:
        url: Raw GitHub URL for the scheme JSON.

    Returns:
        Parsed JSON (list of root nodes).
    """
    data = _fetch_url(url)
    result = json.loads(data)
    logger.info("Downloaded GitHub scheme from %s: %d root nodes", url, len(result))
    return result


def download_kms_scheme(base_url: str, page_size: int = 2000) -> list[dict[str, Any]]:
    """Download all pages of a GCMD KMS concept scheme.

    Args:
        base_url: KMS API URL with format=json parameter.
        page_size: Number of concepts per page.

    Returns:
        Combined list of all concept dicts.
    """
    all_concepts: list[dict[str, Any]] = []
    page_num = 1

    while True:
        sep = "&" if "?" in base_url else "?"
        url = f"{base_url}{sep}page_size={page_size}&page_num={page_num}"
        data = _fetch_url(url)
        payload = json.loads(data)

        concepts = payload.get("concepts", [])
        all_concepts.extend(concepts)

        total_hits = payload.get("hits", 0)
        logger.info(
            "KMS page %d: fetched %d concepts (%d/%d total)",
            page_num,
            len(concepts),
            len(all_concepts),
            total_hits,
        )

        if len(all_concepts) >= total_hits or not concepts:
            break
        page_num += 1

    return all_concepts


# ---------------------------------------------------------------------------
# Parsing: GitHub hierarchy schemes
# ---------------------------------------------------------------------------


def _walk_hierarchy(
    node: dict[str, Any],
    breadcrumb: list[str],
    leaves_only: bool,
    collected: list[tuple[str, str, list[str]]],
) -> None:
    """Recursively walk a GCMD hierarchy node, collecting entries.

    Each collected tuple is (label, uuid, breadcrumb_path_list).

    Args:
        node: Current node dict with uuid, label, children.
        breadcrumb: Ancestor labels leading to this node.
        leaves_only: If True, only collect nodes with no children.
        collected: Accumulator list for results.
    """
    label = node.get("label", "").strip()
    uuid = node.get("uuid", "").strip()
    children = node.get("children", [])

    if not label or not uuid:
        return

    current_path = [*breadcrumb, label]

    is_leaf = not children
    if not leaves_only or is_leaf:
        collected.append((label, uuid, current_path))

    for child in children:
        _walk_hierarchy(child, current_path, leaves_only, collected)


def parse_github_scheme(
    root_nodes: list[dict[str, Any]],
    config: SchemeConfig,
) -> list[dict[str, Any]]:
    """Parse a GitHub-hosted GCMD hierarchy into entity dictionary entries.

    Args:
        root_nodes: Parsed JSON root nodes (typically a single-element list).
        config: Scheme configuration.

    Returns:
        List of entity dictionary entry dicts.
    """
    # Collect all (label, uuid, path) tuples
    collected: list[tuple[str, str, list[str]]] = []
    for root in root_nodes:
        # Skip the root node itself (e.g. "Instruments", "Platforms", "EARTH SCIENCE")
        children = root.get("children", [])
        root_label = root.get("label", "").strip()
        for child in children:
            _walk_hierarchy(child, [root_label], config.leaves_only, collected)

    # Detect duplicate labels within this scheme
    label_counts: dict[str, int] = {}
    for label, _uuid, _path in collected:
        label_counts[label] = label_counts.get(label, 0) + 1

    # Build entries with disambiguation for duplicates
    entries: list[dict[str, Any]] = []
    for label, uuid, path in collected:
        hierarchy_str = " > ".join(path)

        if label_counts.get(label, 0) > 1 and len(path) >= 2:
            # Disambiguate: use "Parent > Name" as canonical_name
            canonical_name = f"{path[-2]} > {label}"
            aliases = [label]
        else:
            canonical_name = label
            aliases = []

        metadata: dict[str, Any] = {
            "gcmd_scheme": config.name,
            "gcmd_hierarchy": hierarchy_str,
            "uuid": uuid,
            "short_name": label,
        }

        entries.append(
            {
                "canonical_name": canonical_name,
                "entity_type": config.entity_type,
                "source": "gcmd",
                "external_id": uuid,
                "aliases": aliases,
                "metadata": metadata,
            }
        )

    logger.info(
        "Parsed %d entries from GCMD %s scheme (%d duplicates disambiguated)",
        len(entries),
        config.name,
        sum(1 for label, cnt in label_counts.items() if cnt > 1),
    )
    return entries


# ---------------------------------------------------------------------------
# Parsing: KMS API flat schemes
# ---------------------------------------------------------------------------


def parse_kms_scheme(
    concepts: list[dict[str, Any]],
    config: SchemeConfig,
) -> list[dict[str, Any]]:
    """Parse KMS API concepts into entity dictionary entries.

    Args:
        concepts: List of concept dicts from the KMS API.
        config: Scheme configuration.

    Returns:
        List of entity dictionary entry dicts.
    """
    entries: list[dict[str, Any]] = []
    skipped = 0

    for concept in concepts:
        uuid = concept.get("uuid", "").strip()
        pref_label = concept.get("prefLabel", "").strip()

        if not uuid or not pref_label:
            skipped += 1
            continue

        # For providers, the prefLabel often encodes hierarchy with slashes
        # e.g. "DOC/NOAA/NESDIS/STAR"
        parts = pref_label.split("/")
        short_name = parts[-1].strip() if parts else pref_label
        hierarchy_str = " > ".join(p.strip() for p in parts)

        # Build aliases: include short_name if different from full label
        aliases: list[str] = []
        if short_name != pref_label:
            aliases.append(short_name)

        # Extract definition text if available
        long_name = ""
        definitions = concept.get("definitions", [])
        if definitions and isinstance(definitions, list):
            first_def = definitions[0]
            if isinstance(first_def, dict):
                long_name = first_def.get("text", "").strip()

        metadata: dict[str, Any] = {
            "gcmd_scheme": config.name,
            "gcmd_hierarchy": hierarchy_str,
            "uuid": uuid,
            "short_name": short_name,
        }
        if long_name:
            metadata["long_name"] = long_name

        entries.append(
            {
                "canonical_name": pref_label,
                "entity_type": config.entity_type,
                "source": "gcmd",
                "external_id": uuid,
                "aliases": aliases,
                "metadata": metadata,
            }
        )

    if skipped:
        logger.warning("Skipped %d KMS concepts missing uuid or prefLabel", skipped)

    logger.info(
        "Parsed %d entries from GCMD %s scheme (KMS)",
        len(entries),
        config.name,
    )
    return entries


# ---------------------------------------------------------------------------
# Unified harvest
# ---------------------------------------------------------------------------


def harvest_scheme(config: SchemeConfig) -> list[dict[str, Any]]:
    """Download and parse a single GCMD scheme.

    Args:
        config: Scheme configuration.

    Returns:
        List of entity dictionary entry dicts.
    """
    if config.source_kind == "github":
        root_nodes = download_github_scheme(config.url)
        return parse_github_scheme(root_nodes, config)
    elif config.source_kind == "kms":
        concepts = download_kms_scheme(config.url)
        return parse_kms_scheme(concepts, config)
    else:
        raise ValueError(f"Unknown source_kind: {config.source_kind!r}")


def harvest_all(
    schemes: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Harvest one or more GCMD schemes.

    Args:
        schemes: List of scheme names to harvest. If None, harvests all.

    Returns:
        Combined list of entity dictionary entries.
    """
    target_schemes = schemes if schemes else list(SCHEME_CONFIGS.keys())
    all_entries: list[dict[str, Any]] = []

    for scheme_name in target_schemes:
        config = SCHEME_CONFIGS.get(scheme_name)
        if config is None:
            logger.error("Unknown scheme: %s", scheme_name)
            continue
        entries = harvest_scheme(config)
        all_entries.extend(entries)

    logger.info("Total GCMD entries harvested: %d", len(all_entries))
    return all_entries


def run_harvest(
    dsn: str | None = None,
    schemes: list[str] | None = None,
    dry_run: bool = False,
) -> int:
    """Run the full GCMD harvest pipeline.

    Downloads schemes, parses entries, and loads them into entity_dictionary.

    Args:
        dsn: Database connection string. Uses SCIX_DSN or default if None.
        schemes: List of scheme names to harvest. If None, harvests all.
        dry_run: If True, parse and report counts without DB writes.

    Returns:
        Number of entries loaded (or would be loaded in dry-run mode).
    """
    t0 = time.monotonic()

    entries = harvest_all(schemes=schemes)

    if dry_run:
        # Report counts by entity_type
        type_counts: dict[str, int] = {}
        for entry in entries:
            et = entry["entity_type"]
            type_counts[et] = type_counts.get(et, 0) + 1
        logger.info("Dry run — would load %d entries:", len(entries))
        for et, cnt in sorted(type_counts.items()):
            logger.info("  %s: %d", et, cnt)
        return len(entries)

    conn = get_connection(dsn)
    try:
        count = bulk_load(conn, entries, discipline="earth_science")
    finally:
        conn.close()

    elapsed = time.monotonic() - t0
    logger.info(
        "GCMD harvest complete: %d entries loaded in %.1fs",
        count,
        elapsed,
    )
    return count


def main() -> None:
    """Parse arguments and run the GCMD harvest pipeline."""
    parser = argparse.ArgumentParser(
        description="Harvest GCMD keyword schemes into entity_dictionary",
    )
    parser.add_argument(
        "--dsn",
        default=None,
        help="Database connection string (uses SCIX_DSN env var if not provided)",
    )
    parser.add_argument(
        "--scheme",
        choices=list(SCHEME_CONFIGS.keys()),
        default=None,
        help="Harvest a specific scheme only (default: all)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and report counts without writing to the database",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    schemes = [args.scheme] if args.scheme else None
    count = run_harvest(dsn=args.dsn, schemes=schemes, dry_run=args.dry_run)

    if args.dry_run:
        print(f"Dry run: {count} GCMD entries would be loaded")
    else:
        print(f"Loaded {count} GCMD entries into entity_dictionary")


if __name__ == "__main__":
    main()
