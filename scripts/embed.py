#!/usr/bin/env python3
"""CLI entry point for the SciX embedding pipeline."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Iterator, Sequence

# Add src/ to path for direct script execution
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from scix.embed import NIGHTLY_YEAR_LOOKBACK, default_year_floor, run_embedding_pipeline
from scix.ingest import open_jsonl


def _iter_bibcodes(paths: Sequence[Path]) -> Iterator[str]:
    """Yield validated bibcodes from ADS JSONL source files."""
    for path in paths:
        with open_jsonl(path) as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path} line {line_number}: invalid JSON") from exc
                if not isinstance(record, dict):
                    raise ValueError(f"{path} line {line_number}: expected a JSON object")
                bibcode = record.get("bibcode")
                if not isinstance(bibcode, str) or not bibcode.strip():
                    raise ValueError(f"{path} line {line_number}: missing or invalid bibcode")
                yield bibcode


def load_bibcodes(paths: Sequence[Path]) -> tuple[str, ...]:
    """Read unique bibcodes from JSONL files, preserving their source order."""
    return tuple(dict.fromkeys(_iter_bibcodes(paths)))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate INDUS embeddings for ADS papers and upsert to the Qdrant dense lane"
    )
    parser.add_argument(
        "--model",
        default="indus",
        help="Embedding model name (default: indus)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Papers per embedding batch (default: 32)",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Torch device: cpu, cuda, cuda:0, etc. (default: cpu)",
    )
    parser.add_argument(
        "--dsn",
        default=None,
        help="PostgreSQL DSN (default: SCIX_DSN env var or 'dbname=scix')",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max papers to embed (useful for testing; default: all)",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help=(
            "Scan the whole corpus instead of recent years. Use for an arbitrary "
            "backlog; bounded incremental runs can use --bibcodes-from-jsonl. "
            "Costs a full seq scan of papers (~530 s before the first row on a "
            "cold cache) — do not use for the nightly run."
        ),
    )
    parser.add_argument(
        "--year-floor",
        type=int,
        default=None,
        help=(
            "Only embed papers with year >= this value "
            f"(default: current year - {NIGHTLY_YEAR_LOOKBACK}). Ignored with --full."
        ),
    )
    parser.add_argument(
        "--bibcodes-from-jsonl",
        type=Path,
        action="append",
        default=None,
        help=(
            "Also embed unembedded bibcodes listed in this ADS JSONL file, even when "
            "their publication year is below --year-floor. May be repeated."
        ),
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )

    args = parser.parse_args()

    if args.full and args.year_floor is not None:
        parser.error("--full and --year-floor are mutually exclusive")
    year_floor = None if args.full else (args.year_floor or default_year_floor())
    bibcodes = load_bibcodes(args.bibcodes_from_jsonl or ())

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    total = run_embedding_pipeline(
        dsn=args.dsn,
        model_name=args.model,
        batch_size=args.batch_size,
        device=args.device,
        limit=args.limit,
        year_floor=year_floor,
        bibcodes=bibcodes,
    )
    logger = logging.getLogger(__name__)
    logger.info("Done. Embedded %d papers.", total)


if __name__ == "__main__":
    main()
