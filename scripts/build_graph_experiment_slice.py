#!/usr/bin/env python3
"""Build the graph-experiment slice snapshot.

Wrap with scix-batch for memory safety:

    scix-batch python scripts/build_graph_experiment_slice.py

Tracks bead scix_experiments-vdtd. Default slice is database='astronomy' +
1-hop reference neighborhood. Override via --slice flag once additional
slices are registered.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from scix.graph_experiment.loader import build_slice
from scix.graph_experiment.slice_config import SliceConfig

_SLICES = {
    "astronomy_1hop": SliceConfig.astronomy_1hop,
    "astronomy_recent_1hop": SliceConfig.astronomy_recent_1hop,
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--slice",
        choices=sorted(_SLICES),
        default="astronomy_1hop",
        help="Named slice to build",
    )
    parser.add_argument(
        "--no-snapshot",
        action="store_true",
        help="Skip writing the pickle snapshot (useful for sizing dry-runs)",
    )
    parser.add_argument(
        "--stats-out",
        type=Path,
        default=Path("results/graph_experiment_slice_stats.json"),
        help="Where to write build stats JSON",
    )
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    args = _parse_args()

    config = _SLICES[args.slice]()
    _, stats = build_slice(config, write_snapshot=not args.no_snapshot)

    args.stats_out.parent.mkdir(parents=True, exist_ok=True)
    args.stats_out.write_text(
        json.dumps(
            {
                "slice": stats.name,
                "snapshot_path": str(config.snapshot_path),
                "seed_count": stats.seed_count,
                "node_count": stats.node_count,
                "edge_count": stats.edge_count,
                "expansion_seconds": stats.expansion_seconds,
                "edge_fetch_seconds": stats.edge_fetch_seconds,
                "graph_build_seconds": stats.graph_build_seconds,
                "snapshot_bytes": stats.snapshot_bytes,
            },
            indent=2,
        )
        + "\n"
    )
    logging.info("stats written to %s", args.stats_out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
