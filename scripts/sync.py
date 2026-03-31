#!/usr/bin/env python3
"""CLI for ADS data sync: harvest, incremental update, and gap detection."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from scix.sync import SyncClient, fill_gaps, harvest_years, incremental_sync


def parse_year_range(s: str) -> range:
    """Parse '2000-2020' or '2024' into a range."""
    if "-" in s:
        parts = s.split("-", 1)
        return range(int(parts[0]), int(parts[1]) + 1)
    return range(int(s), int(s) + 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Sync ADS metadata to JSONL files")
    parser.add_argument(
        "--data-dir", type=Path, default=Path("ads_metadata_by_year_picard"),
        help="Output directory for JSONL files",
    )
    parser.add_argument(
        "--batch-size", type=int, default=2000,
        help="Records per API request (default: 2000, ADS max)",
    )
    parser.add_argument(
        "--throttle", type=float, default=1.0,
        help="Seconds between API requests (default: 1.0)",
    )
    parser.add_argument("-v", "--verbose", action="store_true")

    sub = parser.add_subparsers(dest="command", required=True)

    # harvest subcommand
    p_harvest = sub.add_parser("harvest", help="Fetch all records for year range(s)")
    p_harvest.add_argument(
        "years", type=parse_year_range, nargs="+",
        help="Year range(s), e.g. '2000-2020' or '2024'",
    )

    # incremental subcommand
    p_incr = sub.add_parser("incremental", help="Fetch records entered since a date")
    p_incr.add_argument(
        "since", help="Start date (YYYY-MM-DD), e.g. '2026-03-01'",
    )

    # fill-gaps subcommand
    p_gaps = sub.add_parser("fill-gaps", help="Compare DB vs API counts per year")
    p_gaps.add_argument(
        "years", type=parse_year_range, nargs="+",
        help="Year range(s) to check",
    )
    p_gaps.add_argument("--dsn", default=None, help="PostgreSQL DSN")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    args.data_dir.mkdir(parents=True, exist_ok=True)
    client = SyncClient(batch_size=args.batch_size, throttle=args.throttle)

    if args.command == "harvest":
        all_years = range(0, 0)
        for yr in args.years:
            for year in yr:
                harvest_years(client, range(year, year + 1), args.data_dir)

    elif args.command == "incremental":
        path = incremental_sync(client, args.since, args.data_dir)
        print(f"Output: {path}")

    elif args.command == "fill-gaps":
        all_gaps = []
        for yr in args.years:
            gaps = fill_gaps(client, yr, args.data_dir, dsn=args.dsn)
            all_gaps.extend(gaps)
        if all_gaps:
            print(f"\n{len(all_gaps)} year(s) with gaps:")
            for year, db, api in all_gaps:
                print(f"  {year}: {db:,} in DB / {api:,} in ADS (missing {api - db:,})")
        else:
            print("No gaps found.")


if __name__ == "__main__":
    main()
