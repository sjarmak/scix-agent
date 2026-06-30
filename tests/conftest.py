"""Pytest configuration for the scix test suite.

Optional-dependency gating
--------------------------
CI installs the lightweight ``.[dev]`` dependency set (see
``.github/workflows/check.yml``). Several test modules import heavyweight or
domain-specific optional dependencies — torch, the viz/FastAPI stack, the Qdrant
client, igraph/leidenalg, etc. — that CI deliberately does not install (they are
model/GPU/service dependencies whose tests need the live corpus anyway). When
such a dependency is absent we skip *collecting* the whole module so import-time
``ModuleNotFoundError``s don't fail the run; full/prod environments that install
the relevant extra run these modules normally.

This is the whole-module case. Modules where only *some* tests need an optional
dependency guard those tests in-file with ``pytest.importorskip`` instead, so
their dependency-free tests still run in CI (e.g. ``test_project_embeddings_umap``,
``test_qdrant_tools``). Tracked in scix_experiments-o835.
"""

from __future__ import annotations

import importlib.util
import warnings

# Map an importable module name -> the test files that cannot be collected
# without it. Keep filenames bare (relative to this tests/ directory).
_OPTIONAL_DEP_MODULES: dict[str, list[str]] = {
    "fastapi": [
        "test_agent_trace_frontend.py",
        "test_mcp_trace_instrumentation.py",
        "test_sankey_frontend.py",
        "test_section_coverage_frontend.py",
        "test_streamgraph_frontend.py",
        "test_umap_frontend.py",
        "test_viz_demo_composite.py",
        "test_viz_demo_search.py",
        "test_viz_ego.py",
        "test_viz_resolution.py",
        "test_viz_server.py",
        # Needs fastapi (plus httpx/uvicorn, which arrive transitively via
        # qdrant-client, so fastapi is the determining absence).
        "test_trace_stream.py",
    ],
    "defusedxml": ["test_arxiv_s3.py"],
    "qdrant_client": [
        "test_backfill_qdrant_filter_fields.py",
        "test_qdrant_contract.py",
    ],
    "pgvector": ["test_qdrant_outbox_sync.py"],
    "igraph": ["test_graph_experiment.py"],
    "torch": ["test_somd_detect.py", "test_chunk_pass_embedder.py"],
    "leidenalg": ["test_recompute_citation_communities.py"],
}

collect_ignore: list[str] = []
for _dep, _modules in _OPTIONAL_DEP_MODULES.items():
    if importlib.util.find_spec(_dep) is None:
        collect_ignore.extend(_modules)
        warnings.warn(
            f"optional dependency {_dep!r} not installed — skipping collection of "
            f"{len(_modules)} test module(s): {', '.join(_modules)}",
            stacklevel=1,
        )
