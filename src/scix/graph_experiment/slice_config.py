"""Slice configuration: defines which subset of the corpus to materialize.

Default slice for bead scix_experiments-vdtd is database='astronomy' + 1-hop
reference neighborhood — ~3M seed papers, ~15M total nodes after expansion,
~80M edges. Fits in ~5-8 GB RAM as an igraph.Graph.
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
