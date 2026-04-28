"""Slice configuration: defines which subset of the corpus to materialize.

Two slices are registered for the vdtd spike:

  * ``astronomy_1hop`` — full astronomy database (~3M seeds) + 1-hop
    expansion. The originally-spec'd slice; on this host it expanded to
    14.1M nodes / 235M edges and triggered systemd-oomd at >30 GB RSS.
    Use only on hosts with ≥48 GB free RAM.
  * ``astronomy_recent_1hop`` — astronomy + ``year >= 2018`` seeds
    (~554K) + 1-hop expansion. Expected ~3-5M nodes, ~30-60M edges,
    well under 20 GB RSS. Default for benchmarking on this host.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class SliceConfig:
    """Definition of a corpus slice to load into an in-memory graph.

    The slice is built in two phases:
      1. Seed selection: papers matching ``seed_filter_sql`` (parameterised by
         ``seed_filter_params``).
      2. Neighborhood expansion: ``hop_depth`` rounds of edge expansion via
         ``citation_edges`` in both directions.
    """

    name: str
    seed_filter_sql: str
    seed_filter_params: tuple = field(default_factory=tuple)
    hop_depth: int = 1
    include_document_entities: bool = True
    snapshot_path: Path = field(default_factory=lambda: Path("data/graph_experiment/slice.pkl.gz"))

    @classmethod
    def astronomy_1hop(cls) -> "SliceConfig":
        return cls(
            name="astronomy_1hop",
            seed_filter_sql="'astronomy' = ANY(database)",
            seed_filter_params=(),
            hop_depth=1,
            include_document_entities=True,
            snapshot_path=Path("data/graph_experiment/astronomy_1hop.pkl.gz"),
        )

    @classmethod
    def astronomy_recent_1hop(cls) -> "SliceConfig":
        return cls(
            name="astronomy_recent_1hop",
            seed_filter_sql="'astronomy' = ANY(database) AND year >= 2018",
            seed_filter_params=(),
            hop_depth=1,
            include_document_entities=True,
            snapshot_path=Path(
                "data/graph_experiment/astronomy_recent_1hop.pkl.gz"
            ),
        )
