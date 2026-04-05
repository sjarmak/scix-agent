#!/usr/bin/env python3
"""Compare entity availability from ADS metadata fields against NER/LLM extractions.

Samples astronomy papers, extracts entity-like data from metadata JSONB fields
(facility, data, keyword_norm, bibgroup), compares against the extractions table,
and outputs a JSON report showing what extraction adds beyond metadata.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import psycopg

# Add src/ to path for direct script execution
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from scix.db import DEFAULT_DSN, get_connection
from scix.normalize import normalize_entity

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Metadata field -> entity type mapping
# ---------------------------------------------------------------------------

METADATA_FIELD_TO_ENTITY_TYPE: dict[str, str] = {
    "facility": "instruments",
    "data": "datasets",
    "keyword_norm": "methods",
    "bibgroup": "instruments",
}


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TypeCoverage:
    """Per-type overlap statistics between metadata and extraction entities."""

    metadata_only_count: int
    extraction_only_count: int
    overlap_count: int

    def to_dict(self) -> dict[str, int]:
        return {
            "metadata_only_count": self.metadata_only_count,
            "extraction_only_count": self.extraction_only_count,
            "overlap_count": self.overlap_count,
        }


@dataclass(frozen=True)
class CoverageReport:
    """Full coverage analysis report."""

    sample_size: int
    per_type_coverage: dict[str, TypeCoverage]
    delta_summary: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_size": self.sample_size,
            "per_type_coverage": {k: v.to_dict() for k, v in self.per_type_coverage.items()},
            "delta_summary": self.delta_summary,
        }


# ---------------------------------------------------------------------------
# Metadata entity extraction
# ---------------------------------------------------------------------------


def extract_metadata_entities(
    raw_jsonb: dict[str, Any] | None,
    columns: dict[str, Any],
) -> dict[str, set[str]]:
    """Extract normalized entity sets from ADS metadata fields.

    Looks for facility, data, keyword_norm in raw JSONB and column values,
    and bibgroup from columns.

    Args:
        raw_jsonb: Parsed raw JSONB column (may be None).
        columns: Dict with column values for facility, data, keyword_norm, bibgroup.

    Returns:
        Dict mapping entity type -> set of normalized entity strings.
    """
    result: dict[str, set[str]] = {}

    for field, entity_type in METADATA_FIELD_TO_ENTITY_TYPE.items():
        raw_values: list[str] = []

        # Try column value first (for fields that have dedicated columns)
        col_val = columns.get(field)
        if isinstance(col_val, list):
            raw_values.extend(v for v in col_val if isinstance(v, str) and v.strip())

        # Also check raw JSONB for fields that might only be there
        if raw_jsonb and field in raw_jsonb:
            jsonb_val = raw_jsonb[field]
            if isinstance(jsonb_val, list):
                for v in jsonb_val:
                    if isinstance(v, str) and v.strip():
                        raw_values.append(v)

        normalized = {normalize_entity(v) for v in raw_values if v.strip()}
        # Remove empty strings that might result from normalization
        normalized.discard("")

        if entity_type not in result:
            result[entity_type] = set()
        result[entity_type] |= normalized

    return result


def extract_extraction_entities(
    payload: dict[str, Any] | None,
) -> dict[str, set[str]]:
    """Extract normalized entity sets from an extractions payload.

    Args:
        payload: The JSONB payload from the extractions table,
                 expected to contain {"entities": [{"name": ..., "type": ...}, ...]}.

    Returns:
        Dict mapping entity type -> set of normalized entity strings.
    """
    result: dict[str, set[str]] = {}
    if not payload:
        return result

    entities = payload.get("entities", [])
    if not isinstance(entities, list):
        return result

    for ent in entities:
        if not isinstance(ent, dict):
            continue
        name = ent.get("name", "")
        etype = ent.get("type", "")
        if not isinstance(name, str) or not name.strip():
            continue
        if not isinstance(etype, str) or not etype.strip():
            continue
        normalized_name = normalize_entity(name)
        if not normalized_name:
            continue
        etype_lower = etype.lower()
        if etype_lower not in result:
            result[etype_lower] = set()
        result[etype_lower].add(normalized_name)

    return result


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------


def compute_coverage(
    metadata_by_type: dict[str, set[str]],
    extraction_by_type: dict[str, set[str]],
) -> dict[str, TypeCoverage]:
    """Compute per-type overlap between metadata and extraction entity sets.

    Args:
        metadata_by_type: Entity type -> set of normalized metadata entities.
        extraction_by_type: Entity type -> set of normalized extraction entities.

    Returns:
        Dict mapping entity type -> TypeCoverage with overlap statistics.
    """
    all_types = set(metadata_by_type.keys()) | set(extraction_by_type.keys())
    result: dict[str, TypeCoverage] = {}

    for etype in sorted(all_types):
        meta_set = metadata_by_type.get(etype, set())
        extr_set = extraction_by_type.get(etype, set())
        overlap = meta_set & extr_set
        result[etype] = TypeCoverage(
            metadata_only_count=len(meta_set - extr_set),
            extraction_only_count=len(extr_set - meta_set),
            overlap_count=len(overlap),
        )

    return result


def build_delta_summary(per_type: dict[str, TypeCoverage]) -> dict[str, Any]:
    """Summarize what extraction adds beyond metadata.

    Args:
        per_type: Per-type coverage from compute_coverage().

    Returns:
        Summary dict with total counts and extraction lift ratio.
    """
    total_metadata_only = sum(tc.metadata_only_count for tc in per_type.values())
    total_extraction_only = sum(tc.extraction_only_count for tc in per_type.values())
    total_overlap = sum(tc.overlap_count for tc in per_type.values())
    total_metadata = total_metadata_only + total_overlap
    total_extraction = total_extraction_only + total_overlap

    extraction_lift = total_extraction_only / total_metadata if total_metadata > 0 else 0.0

    return {
        "total_metadata_entities": total_metadata,
        "total_extraction_entities": total_extraction,
        "total_overlap": total_overlap,
        "total_metadata_only": total_metadata_only,
        "total_extraction_only": total_extraction_only,
        "extraction_lift_ratio": round(extraction_lift, 4),
    }


# ---------------------------------------------------------------------------
# Database queries
# ---------------------------------------------------------------------------


def fetch_sample_papers(
    conn: psycopg.Connection,
    sample_size: int,
) -> list[dict[str, Any]]:
    """Fetch a sample of astronomy papers ordered by citation count.

    Args:
        conn: Database connection.
        sample_size: Number of papers to sample.

    Returns:
        List of dicts with bibcode, facility, data, keyword_norm, bibgroup, raw.
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT bibcode, facility, data, keyword_norm, bibgroup, raw
            FROM papers
            WHERE 'astronomy' = ANY(database)
            ORDER BY citation_count DESC NULLS LAST
            LIMIT %s
            """,
            (sample_size,),
        )
        rows = cur.fetchall()

    results = []
    for row in rows:
        bibcode, facility, data, keyword_norm, bibgroup, raw_val = row
        raw_parsed = None
        if raw_val is not None:
            if isinstance(raw_val, str):
                raw_parsed = json.loads(raw_val)
            elif isinstance(raw_val, dict):
                raw_parsed = raw_val

        results.append(
            {
                "bibcode": bibcode,
                "columns": {
                    "facility": facility,
                    "data": data,
                    "keyword_norm": keyword_norm,
                    "bibgroup": bibgroup,
                },
                "raw": raw_parsed,
            }
        )
    return results


def fetch_extractions_for_bibcodes(
    conn: psycopg.Connection,
    bibcodes: list[str],
) -> dict[str, list[dict[str, Any]]]:
    """Fetch all extractions for a set of bibcodes.

    Args:
        conn: Database connection.
        bibcodes: List of bibcodes to look up.

    Returns:
        Dict mapping bibcode -> list of extraction payloads.
    """
    if not bibcodes:
        return {}

    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT bibcode, payload
            FROM extractions
            WHERE bibcode = ANY(%s)
            """,
            (bibcodes,),
        )
        rows = cur.fetchall()

    result: dict[str, list[dict[str, Any]]] = {}
    for bibcode, payload in rows:
        parsed = payload if isinstance(payload, dict) else json.loads(payload)
        if bibcode not in result:
            result[bibcode] = []
        result[bibcode].append(parsed)

    return result


# ---------------------------------------------------------------------------
# Main analysis pipeline
# ---------------------------------------------------------------------------


def run_analysis(
    conn: psycopg.Connection,
    sample_size: int = 1000,
) -> CoverageReport:
    """Run the full metadata vs extraction coverage analysis.

    Args:
        conn: Database connection.
        sample_size: Number of papers to analyze.

    Returns:
        CoverageReport with per-type coverage and delta summary.
    """
    logger.info("Fetching %d astronomy papers...", sample_size)
    papers = fetch_sample_papers(conn, sample_size)
    actual_sample = len(papers)
    logger.info("Fetched %d papers", actual_sample)

    bibcodes = [p["bibcode"] for p in papers]
    logger.info("Fetching extractions for %d bibcodes...", len(bibcodes))
    extractions_map = fetch_extractions_for_bibcodes(conn, bibcodes)
    logger.info(
        "Found extractions for %d / %d papers",
        len(extractions_map),
        len(bibcodes),
    )

    # Accumulate entities across all papers
    all_metadata: dict[str, set[str]] = {}
    all_extraction: dict[str, set[str]] = {}

    for paper in papers:
        bib = paper["bibcode"]

        # Metadata entities
        meta_entities = extract_metadata_entities(paper["raw"], paper["columns"])
        for etype, eset in meta_entities.items():
            if etype not in all_metadata:
                all_metadata[etype] = set()
            all_metadata[etype] |= eset

        # Extraction entities
        payloads = extractions_map.get(bib, [])
        for payload in payloads:
            extr_entities = extract_extraction_entities(payload)
            for etype, eset in extr_entities.items():
                if etype not in all_extraction:
                    all_extraction[etype] = set()
                all_extraction[etype] |= eset

    per_type = compute_coverage(all_metadata, all_extraction)
    delta = build_delta_summary(per_type)

    return CoverageReport(
        sample_size=actual_sample,
        per_type_coverage=per_type,
        delta_summary=delta,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare metadata entity coverage against NER/LLM extractions"
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=1000,
        help="Number of astronomy papers to sample (default: 1000)",
    )
    parser.add_argument(
        "--dsn",
        default=None,
        help="PostgreSQL DSN (default: SCIX_DSN env var or 'dbname=scix')",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSON file path (default: stdout)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    conn = get_connection(args.dsn)
    try:
        report = run_analysis(conn, sample_size=args.sample_size)
    finally:
        conn.close()

    report_dict = report.to_dict()
    output_json = json.dumps(report_dict, indent=2)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(output_json + "\n")
        logger.info("Report written to %s", args.output)
    else:
        print(output_json)


if __name__ == "__main__":
    main()
