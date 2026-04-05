#!/usr/bin/env python3
"""Harvest SPASE (Space Physics Archive Search and Extract) vocabularies.

Downloads tab-delimited vocabulary files from the spase-group/spase-base-model
GitHub repository and extracts:

- MeasurementType + related quantity lists -> entity_type='observable'
- InstrumentType -> entity_type='instrument'
- ObservedRegion (Region hierarchy) -> entity_type='observable'

All entries use source='spase', discipline='heliophysics'.
CamelCase SPASE terms are split into space-separated aliases.

Usage:
    python scripts/harvest_spase.py --dry-run
    python scripts/harvest_spase.py --vocabulary measurement
    python scripts/harvest_spase.py --vocabulary instrument --verbose
    python scripts/harvest_spase.py --vocabulary region
    python scripts/harvest_spase.py  # harvest all vocabularies
"""

from __future__ import annotations

import argparse
import logging
import re
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

SPASE_VERSION = "2.7.1"
BASE_URL = (
    "https://raw.githubusercontent.com/spase-group/spase-base-model"
    f"/master/spase-base-{SPASE_VERSION}"
)
MEMBER_URL = f"{BASE_URL}/member.tab"
DICTIONARY_URL = f"{BASE_URL}/dictionary.tab"

SOURCE = "spase"
DISCIPLINE = "heliophysics"

# Lists whose members map to entity_type='observable' (measurement quantities)
MEASUREMENT_LISTS: frozenset[str] = frozenset(
    {
        "MeasurementType",
        "FieldQuantity",
        "ParticleQuantity",
        "WaveQuantity",
        "MixedQuantity",
    }
)

# Lists whose members map to entity_type='instrument'
INSTRUMENT_LISTS: frozenset[str] = frozenset({"InstrumentType"})

# Top-level region list and sub-region lists for ObservedRegion
REGION_TOP_LIST = "Region"
REGION_SUB_LISTS: frozenset[str] = frozenset(
    {
        "Earth",
        "Heliosphere",
        "Sun",
        "Magnetosphere",
        "Ionosphere",
        "NearSurface",
        "Jupiter",
        "Mars",
        "Mercury",
        "Neptune",
        "Saturn",
        "Uranus",
        "Venus",
    }
)


# ---------------------------------------------------------------------------
# Module-level client (lazy init)
# ---------------------------------------------------------------------------

_client: ResilientClient | None = None


def _get_client() -> ResilientClient:
    """Return a shared ResilientClient instance."""
    global _client
    if _client is None:
        _client = ResilientClient(
            user_agent="scix-experiments/1.0",
            max_retries=3,
            backoff_base=2.0,
            rate_limit=10.0,
        )
    return _client


def camel_case_split(term: str) -> str:
    """Split a PascalCase/camelCase term into space-separated words.

    Examples:
        MagneticField -> Magnetic Field
        ElectricField -> Electric Field
        EnergeticParticles -> Energetic Particles
        ACMagneticField -> AC Magnetic Field
        NearEarth -> Near Earth
        SPICE -> SPICE (all-caps unchanged)

    Args:
        term: PascalCase or camelCase term.

    Returns:
        Space-separated version of the term.
    """
    # Insert space before uppercase letter preceded by lowercase,
    # or before an uppercase letter followed by lowercase when preceded by uppercase
    result = re.sub(r"(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])", " ", term)
    return result


def _download_tab_file(url: str) -> str:
    """Download a tab-delimited file from GitHub using ResilientClient.

    Args:
        url: URL of the tab file.

    Returns:
        Raw text content of the file.
    """
    client = _get_client()
    response = client.get(url)
    data = response.text
    logger.info("Downloaded %s: %d bytes", url.split("/")[-1], len(data))
    return data


def parse_tab_file(content: str) -> list[dict[str, str]]:
    """Parse a tab-delimited file into a list of dicts keyed by header names.

    Parses by header name, not column index, for robustness.
    Lines starting with '#' have the '#' stripped from the first header name.

    Args:
        content: Raw text content of the tab file.

    Returns:
        List of dicts, one per data row, keyed by header names.
    """
    lines = content.strip().split("\n")
    if not lines:
        return []

    # Parse header — strip leading '#' if present
    header_line = lines[0]
    if header_line.startswith("#"):
        header_line = header_line[1:]

    headers = header_line.split("\t")
    rows: list[dict[str, str]] = []

    for line in lines[1:]:
        if not line.strip() or line.startswith("#"):
            continue
        fields = line.split("\t")
        row = {}
        for i, header in enumerate(headers):
            row[header] = fields[i] if i < len(fields) else ""
        rows.append(row)

    return rows


def _build_definition_map(dictionary_rows: list[dict[str, str]]) -> dict[str, str]:
    """Build a term -> definition lookup from dictionary.tab rows.

    Args:
        dictionary_rows: Parsed rows from dictionary.tab.

    Returns:
        Dict mapping term name to its definition string.
    """
    defs: dict[str, str] = {}
    for row in dictionary_rows:
        term = row.get("Term", "").strip()
        definition = row.get("Definition", "").strip()
        if term and definition:
            defs[term] = definition
    return defs


def _make_entry(
    canonical_name: str,
    entity_type: str,
    *,
    spase_list: str,
    definition: str = "",
) -> dict[str, Any]:
    """Create a single entity dictionary entry from a SPASE term.

    Args:
        canonical_name: The original SPASE term (PascalCase or dotted path).
        entity_type: 'observable' or 'instrument'.
        spase_list: The SPASE list this term belongs to.
        definition: Optional definition text from dictionary.tab.

    Returns:
        Dict compatible with bulk_load().
    """
    # Build alias from CamelCase split
    split_name = camel_case_split(canonical_name)
    aliases: list[str] = []
    if split_name != canonical_name:
        aliases.append(split_name)

    # For dotted paths like "Earth.Magnetosphere", also add space-separated form
    if "." in canonical_name:
        parts = canonical_name.split(".")
        space_form = " ".join(parts)
        if space_form not in aliases:
            aliases.append(space_form)
        # Also split each part's CamelCase
        split_parts = " ".join(camel_case_split(p) for p in parts)
        if split_parts not in aliases and split_parts != space_form:
            aliases.append(split_parts)

    metadata: dict[str, Any] = {"spase_list": spase_list}
    if definition:
        metadata["description"] = definition

    return {
        "canonical_name": canonical_name,
        "entity_type": entity_type,
        "source": SOURCE,
        "external_id": f"spase:{spase_list}:{canonical_name}",
        "aliases": aliases,
        "metadata": metadata,
    }


def parse_measurement_entries(
    member_rows: list[dict[str, str]],
    definitions: dict[str, str],
) -> list[dict[str, Any]]:
    """Extract measurement-type observable entries from member.tab.

    Combines MeasurementType, FieldQuantity, ParticleQuantity, WaveQuantity,
    and MixedQuantity lists to produce a comprehensive set of measurement
    observables.

    Args:
        member_rows: Parsed rows from member.tab.
        definitions: Term -> definition lookup from dictionary.tab.

    Returns:
        List of entity dictionary entry dicts with entity_type='observable'.
    """
    entries: list[dict[str, Any]] = []
    seen: set[str] = set()

    for row in member_rows:
        list_name = row.get("List", "").strip()
        item = row.get("Item", "").strip()

        if list_name not in MEASUREMENT_LISTS or not item:
            continue

        # Deduplicate across lists (some terms appear in multiple lists)
        key = f"{item}:{list_name}"
        if key in seen:
            continue
        seen.add(key)

        entries.append(
            _make_entry(
                item,
                "observable",
                spase_list=list_name,
                definition=definitions.get(item, ""),
            )
        )

    logger.info("Parsed %d measurement-type observable entries", len(entries))
    return entries


def parse_instrument_entries(
    member_rows: list[dict[str, str]],
    definitions: dict[str, str],
) -> list[dict[str, Any]]:
    """Extract instrument-type entries from member.tab.

    Args:
        member_rows: Parsed rows from member.tab.
        definitions: Term -> definition lookup from dictionary.tab.

    Returns:
        List of entity dictionary entry dicts with entity_type='instrument'.
    """
    entries: list[dict[str, Any]] = []

    for row in member_rows:
        list_name = row.get("List", "").strip()
        item = row.get("Item", "").strip()

        if list_name not in INSTRUMENT_LISTS or not item:
            continue

        entries.append(
            _make_entry(
                item,
                "instrument",
                spase_list=list_name,
                definition=definitions.get(item, ""),
            )
        )

    logger.info("Parsed %d instrument-type entries", len(entries))
    return entries


def parse_region_entries(
    member_rows: list[dict[str, str]],
    definitions: dict[str, str],
) -> list[dict[str, Any]]:
    """Extract observed-region observable entries from member.tab.

    Builds hierarchical region paths: top-level regions from the Region list,
    and dotted sub-regions (e.g., "Earth.Magnetosphere") from sub-region lists.

    Args:
        member_rows: Parsed rows from member.tab.
        definitions: Term -> definition lookup from dictionary.tab.

    Returns:
        List of entity dictionary entry dicts with entity_type='observable'.
    """
    entries: list[dict[str, Any]] = []
    seen: set[str] = set()

    for row in member_rows:
        list_name = row.get("List", "").strip()
        item = row.get("Item", "").strip()

        if not item:
            continue

        if list_name == REGION_TOP_LIST:
            # Top-level region: canonical name is just the region name
            canonical = item
            if canonical in seen:
                continue
            seen.add(canonical)

            entries.append(
                _make_entry(
                    canonical,
                    "observable",
                    spase_list="ObservedRegion",
                    definition=definitions.get(item, ""),
                )
            )

        elif list_name in REGION_SUB_LISTS:
            # Sub-region: canonical name is "Parent.Child"
            canonical = f"{list_name}.{item}"
            if canonical in seen:
                continue
            seen.add(canonical)

            entries.append(
                _make_entry(
                    canonical,
                    "observable",
                    spase_list="ObservedRegion",
                    definition=definitions.get(item, ""),
                )
            )

    logger.info("Parsed %d observed-region entries", len(entries))
    return entries


def download_and_parse(
    vocabulary: str = "all",
) -> list[dict[str, Any]]:
    """Download SPASE vocabulary files and parse selected entries.

    Args:
        vocabulary: Which vocabulary to harvest — 'measurement', 'instrument',
            'region', or 'all'.

    Returns:
        List of entity dictionary entry dicts.
    """
    member_text = _download_tab_file(MEMBER_URL)
    dictionary_text = _download_tab_file(DICTIONARY_URL)

    member_rows = parse_tab_file(member_text)
    dictionary_rows = parse_tab_file(dictionary_text)
    definitions = _build_definition_map(dictionary_rows)

    logger.info(
        "Parsed %d member rows, %d dictionary definitions",
        len(member_rows),
        len(definitions),
    )

    entries: list[dict[str, Any]] = []

    if vocabulary in ("all", "measurement"):
        entries.extend(parse_measurement_entries(member_rows, definitions))

    if vocabulary in ("all", "instrument"):
        entries.extend(parse_instrument_entries(member_rows, definitions))

    if vocabulary in ("all", "region"):
        entries.extend(parse_region_entries(member_rows, definitions))

    logger.info("Total SPASE entries: %d", len(entries))
    return entries


def _write_entity_graph(
    conn: Any,
    entries: list[dict[str, Any]],
    harvest_run_id: int,
) -> int:
    """Write entries to entities, entity_identifiers, and entity_aliases tables.

    Returns the number of entities upserted.
    """
    count = 0
    for entry in entries:
        metadata = entry.get("metadata", {})
        properties: dict[str, Any] = {
            "spase_list": metadata.get("spase_list", ""),
        }
        if "description" in metadata:
            properties["description"] = metadata["description"]

        entity_id = upsert_entity(
            conn,
            canonical_name=entry["canonical_name"],
            entity_type=entry["entity_type"],
            source=entry["source"],
            discipline=DISCIPLINE,
            harvest_run_id=harvest_run_id,
            properties=properties,
        )

        if entry.get("external_id"):
            upsert_entity_identifier(
                conn,
                entity_id=entity_id,
                id_scheme="spase_resource_id",
                external_id=entry["external_id"],
                is_primary=True,
            )

        for alias in entry.get("aliases", []):
            upsert_entity_alias(
                conn,
                entity_id=entity_id,
                alias=alias,
                alias_source="spase",
            )

        count += 1

    conn.commit()
    return count


def run_harvest(
    dsn: str | None = None,
    vocabulary: str = "all",
) -> int:
    """Run the full SPASE vocabulary harvest pipeline.

    Downloads vocabulary files, parses entries, and loads them into
    entity_dictionary and entity graph tables. Logs harvest run to harvest_runs.

    Args:
        dsn: Database connection string. Uses SCIX_DSN or default if None.
        vocabulary: Which vocabulary to harvest.

    Returns:
        Number of entries loaded.
    """
    t0 = time.monotonic()

    entries = download_and_parse(vocabulary=vocabulary)

    conn = get_connection(dsn)
    run_log = HarvestRunLog(conn, "spase")
    try:
        run_log.start(config={"vocabulary": vocabulary})

        # Backward-compatible: write to entity_dictionary
        dict_count = bulk_load(conn, entries, discipline=DISCIPLINE)

        # Write to entity graph tables
        graph_count = _write_entity_graph(conn, entries, run_log.run_id)

        # Build per-type counts for harvest_runs.counts
        type_counts: dict[str, int] = {}
        for entry in entries:
            et = entry["entity_type"]
            type_counts[et] = type_counts.get(et, 0) + 1

        run_log.complete(
            records_fetched=len(entries),
            records_upserted=graph_count,
            counts=type_counts,
        )
    except Exception as exc:
        try:
            run_log.fail(str(exc))
        except Exception:
            logger.warning("Failed to update harvest_run status to 'failed'")
        raise
    finally:
        conn.close()

    elapsed = time.monotonic() - t0
    logger.info(
        "SPASE harvest complete: %d entries loaded in %.1fs",
        dict_count,
        elapsed,
    )
    return dict_count


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and run the SPASE vocabulary harvest pipeline."""
    parser = argparse.ArgumentParser(
        description="Harvest SPASE heliophysics vocabularies into entity_dictionary",
    )
    parser.add_argument(
        "--dsn",
        default=None,
        help="Database connection string (uses SCIX_DSN env var if not provided)",
    )
    parser.add_argument(
        "--vocabulary",
        choices=["measurement", "instrument", "region", "all"],
        default="all",
        help="Which vocabulary to harvest (default: all)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Download and parse without loading into entity_dictionary",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.dry_run:
        entries = download_and_parse(vocabulary=args.vocabulary)

        # Print summary by entity_type
        by_type: dict[str, int] = {}
        for entry in entries:
            et = entry["entity_type"]
            by_type[et] = by_type.get(et, 0) + 1

        print(f"Dry run — parsed {len(entries)} SPASE entries:")
        for et, count in sorted(by_type.items()):
            print(f"  {et}: {count}")

        # Show first few entries per type
        for et in sorted(by_type):
            print(f"\nSample {et} entries:")
            samples = [e for e in entries if e["entity_type"] == et][:5]
            for s in samples:
                aliases_str = ", ".join(s["aliases"]) if s["aliases"] else "(none)"
                print(f"  {s['canonical_name']} -> aliases: [{aliases_str}]")

        return 0

    count = run_harvest(dsn=args.dsn, vocabulary=args.vocabulary)
    print(f"Loaded {count} SPASE entries into entity_dictionary")
    return 0


if __name__ == "__main__":
    sys.exit(main())
