"""Versioned, externally-pinnable MCP contract (bead scix_experiments-x2dp).

The MCP tool surface is a deliverable other projects depend on. This module
makes that contract *self-describing and self-enforcing*:

* :func:`build_contract` reads the live server and produces a deterministic,
  environment-independent description of the public surface — the default
  agent-visible tool names + JSON-schema ``inputSchema`` for each, the closed
  error-code catalog (:mod:`scix.mcp_errors`), and the response envelope shape.
* :func:`write_published_contract` serializes that to ``contract/scix_mcp_v1.json``
  (via ``scripts/gen_mcp_contract.py``).
* The conformance suite (``tests/test_mcp_contract_conformance.py``) asserts the
  committed artifact still matches the live server, so any drift fails CI until
  the artifact is regenerated and the change reviewed. A *breaking* change bumps
  :data:`CONTRACT_VERSION` (a new ``scix_mcp_v2.json``).

"Deterministic, environment-independent" matters: the published artifact is
built against the DEFAULT configuration (default hidden set, Qdrant off), not
the live ``SCIX_HIDDEN_TOOLS`` / ``QDRANT_URL`` of whatever host runs the
generator. So the same artifact is produced on a developer laptop, in CI, and
on prod.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from scix.mcp_errors import CATALOG

CONTRACT_VERSION = "1"


def _list_registered_tools() -> list[Any]:
    """Return every registered ``Tool`` (name + inputSchema), unfiltered.

    Builds a model-less server with the hidden-tool filter cleared so the full
    registered surface is visible, then invokes the ``tools/list`` handler
    synchronously. Callers select the public subset themselves (see
    :func:`build_contract`) — this returns whatever is registered regardless of
    the host's ``SCIX_HIDDEN_TOOLS``.
    """
    import asyncio

    from mcp.types import ListToolsRequest

    from scix import mcp_server

    # Clear the env-derived hidden set for the duration of the build so the
    # handler emits every registered tool; restore it afterwards. ``_qdrant_enabled``
    # is forced off so the optional, Qdrant-gated ``chunk_search`` tool does not
    # make the surface host-dependent (it is not part of the default-visible
    # contract anyway).
    saved_hidden = mcp_server._HIDDEN_TOOLS
    saved_qdrant = mcp_server._qdrant_enabled
    mcp_server._HIDDEN_TOOLS = frozenset()
    mcp_server._qdrant_enabled = lambda: False
    try:
        server = mcp_server.create_server(_run_self_test=False, _preload_model=False)
    finally:
        mcp_server._HIDDEN_TOOLS = saved_hidden
        mcp_server._qdrant_enabled = saved_qdrant

    handler = server.request_handlers[ListToolsRequest]
    loop = asyncio.new_event_loop()
    try:
        result = loop.run_until_complete(handler(ListToolsRequest(method="tools/list")))
    finally:
        loop.close()

    # Real handlers wrap the result in a ServerResult (`.root.tools`); raw
    # fixtures may expose `.tools` directly.
    if hasattr(result, "root") and hasattr(result.root, "tools"):
        return list(result.root.tools)
    if hasattr(result, "tools"):
        return list(result.tools)
    raise RuntimeError(f"unexpected tools/list result shape: {result!r}")


def default_visible_tool_names() -> list[str]:
    """The env-independent default agent-visible tool names, in registry order.

    ``EXPECTED_TOOLS`` minus the default hidden set — the public surface the
    cap (<= ``VISIBLE_TOOL_CAP``) governs and that the contract pins.
    """
    from scix import mcp_server

    return [t for t in mcp_server.EXPECTED_TOOLS if t not in mcp_server._DEFAULT_HIDDEN_TOOLS]


def build_contract() -> dict[str, Any]:
    """Build the deterministic public MCP contract from the live server."""
    by_name = {t.name: t.inputSchema for t in _list_registered_tools()}
    visible = default_visible_tool_names()
    missing = [n for n in visible if n not in by_name]
    if missing:
        raise RuntimeError(
            f"default-visible tools have no registered schema: {missing}; "
            "EXPECTED_TOOLS / list_tools are out of sync"
        )

    return {
        "contract_version": CONTRACT_VERSION,
        "envelope": {
            # No uniform success wrapper — each tool returns its own JSON shape.
            "success": "Tool-specific JSON object; there is no uniform success wrapper.",
            # Every structured error is uniform and machine-branchable.
            "error": {
                "required": ["error", "error_code"],
                "shape": {
                    "error": "<human-readable message>",
                    "error_code": "<member of error_codes>",
                },
            },
        },
        "error_codes": sorted(CATALOG),
        "tools": [{"name": name, "inputSchema": by_name[name]} for name in visible],
    }


def contract_path() -> Path:
    """Filesystem path of the committed artifact for the current version."""
    repo_root = Path(__file__).resolve().parents[2]
    return repo_root / "contract" / f"scix_mcp_v{CONTRACT_VERSION}.json"


def write_published_contract() -> Path:
    """Write the live contract to :func:`contract_path` and return the path.

    Canonical serialization — stable key order + trailing newline — so the
    committed artifact diffs cleanly on an intentional change.
    """
    path = contract_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(build_contract(), indent=2, sort_keys=True) + "\n")
    return path


def load_published_contract() -> dict[str, Any]:
    """Load the committed contract artifact."""
    return json.loads(contract_path().read_text())
