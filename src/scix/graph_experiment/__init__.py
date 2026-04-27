"""Graph-experiment spike: in-memory igraph slice + experimental MCP tools.

Tracks bead scix_experiments-vdtd. Goal is to test whether agents reach for
multi-hop graph queries when given primitive + freeform tools, or whether the
1-hop-heavy workload is intrinsic to the problem. Output of this spike feeds
the Apache AGE go/no-go decision.

Not registered in the production MCP server — runs as a separate process.
"""

from __future__ import annotations

from scix.graph_experiment.slice_config import SliceConfig
from scix.graph_experiment.trace import TraceLogger

__all__ = ["SliceConfig", "TraceLogger"]
