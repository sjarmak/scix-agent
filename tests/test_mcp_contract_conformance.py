"""MCP contract conformance suite (bead scix_experiments-x2dp).

Makes the externally-pinnable MCP contract self-enforcing. Two steps:

STEP 1 — surface conformance (would have caught both drifts the 2026-06-15
deep audit found: 17 > 15 visible, and 30/77 error returns missing an
``error_code``):

  * the agent-visible tool count stays within ``VISIBLE_TOOL_CAP`` (15);
  * every registered tool advertises a valid JSON-schema ``inputSchema``;
  * the response error envelope is uniform and every code is in the closed
    catalog (:mod:`scix.mcp_errors`; the source-scan in
    ``test_mcp_error_envelopes.py`` guards the emit sites).

STEP 2 — published-artifact drift gate:

  * ``contract/scix_mcp_v1.json`` matches the live server. Any divergence
    fails CI until the artifact is regenerated (``scripts/gen_mcp_contract.py``)
    and the change reviewed — a breaking change bumps ``CONTRACT_VERSION``.

The whole suite is DB-less and model-less: it only reads the static tool
surface, so it runs in the ``not integration and not network`` CI subset.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from scix import mcp_contract, mcp_server
from scix.mcp_errors import CATALOG
from scix.mcp_server import VISIBLE_TOOL_CAP


def _string_keys(node: ast.Dict) -> set[str]:
    return {k.value for k in node.keys if isinstance(k, ast.Constant) and isinstance(k.value, str)}


@pytest.fixture(scope="module")
def contract() -> dict:
    return mcp_contract.build_contract()


# ---------------------------------------------------------------------------
# STEP 1 — surface conformance
# ---------------------------------------------------------------------------


def test_visible_surface_within_cap(contract: dict) -> None:
    """The agent-visible tool count never exceeds the ADR-pinned cap."""
    names = [t["name"] for t in contract["tools"]]
    assert len(names) == len(set(names)), f"duplicate tool names: {names}"
    assert len(names) <= VISIBLE_TOOL_CAP, (
        f"visible tool surface ({len(names)}) exceeds the cap of "
        f"{VISIBLE_TOOL_CAP}: {sorted(names)}. Consolidate real overlaps or "
        "raise the cap via ADR — see docs/mcp_tool_audit_2026-06.md §6."
    )


def test_visible_surface_matches_default(contract: dict) -> None:
    """Contract tools are exactly the default-visible set (catches renames)."""
    names = [t["name"] for t in contract["tools"]]
    assert names == mcp_contract.default_visible_tool_names()


def test_every_tool_has_valid_input_schema(contract: dict) -> None:
    """Every published tool advertises a JSON-schema object inputSchema."""
    for tool in contract["tools"]:
        name = tool["name"]
        schema = tool["inputSchema"]
        assert isinstance(schema, dict), f"{name}: inputSchema is not a dict"
        assert (
            schema.get("type") == "object"
        ), f"{name}: inputSchema.type must be 'object', got {schema.get('type')!r}"
        assert isinstance(
            schema.get("properties"), dict
        ), f"{name}: inputSchema.properties missing or not a dict"


def test_error_codes_are_the_closed_catalog(contract: dict) -> None:
    """The published error-code list is exactly the closed catalog, non-empty."""
    assert contract["error_codes"], "error-code catalog is empty"
    assert contract["error_codes"] == sorted(CATALOG)


def test_error_envelope_shape_documented(contract: dict) -> None:
    """The error envelope pins both the human ``error`` and machine ``error_code``."""
    error_env = contract["envelope"]["error"]
    assert set(error_env["required"]) == {"error", "error_code"}
    assert set(error_env["shape"]) == {"error", "error_code"}


def test_every_error_return_carries_an_error_code() -> None:
    """Every error-envelope dict in the server source also carries ``error_code``.

    The deep audit found 30/77 error returns were bare ``{"error": "<str>"}``
    with no machine-readable ``error_code``, so a consumer branching on
    ``error_code`` silently missed a third of failure modes. This AST scan
    treats any dict literal with an ``"error"`` key as an error envelope and
    requires an ``"error_code"`` sibling — the structural complement to
    ``test_mcp_error_envelopes.py`` (which checks emitted codes are *in* the
    closed catalog, but not that a code is present at all).
    """
    tree = ast.parse(Path(mcp_server.__file__).read_text())
    offenders = sorted(
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Dict)
        and "error" in _string_keys(node)
        and "error_code" not in _string_keys(node)
    )
    assert not offenders, (
        "mcp_server has error-envelope dict literals with no error_code at "
        f"lines {offenders}. Every error return must carry an error_code from "
        "the closed catalog (src/scix/mcp_errors.py)."
    )


# ---------------------------------------------------------------------------
# STEP 2 — published-artifact drift gate
# ---------------------------------------------------------------------------


def test_contract_version_matches_artifact_name() -> None:
    """The committed artifact filename carries the declared contract version."""
    assert mcp_contract.contract_path().name == (f"scix_mcp_v{mcp_contract.CONTRACT_VERSION}.json")


def test_published_contract_matches_live(contract: dict) -> None:
    """The committed artifact must match the live server surface.

    On divergence, regenerate with ``python scripts/gen_mcp_contract.py`` and
    review the diff: an intentional, non-breaking change updates the artifact
    in place; a breaking change bumps ``CONTRACT_VERSION``. This is the gate
    that makes the contract self-enforcing.
    """
    try:
        published = mcp_contract.load_published_contract()
    except FileNotFoundError:  # pragma: no cover - first-run guard
        pytest.fail(
            f"{mcp_contract.contract_path()} is missing; generate it with "
            "`python scripts/gen_mcp_contract.py`."
        )
    assert published == contract, (
        "contract/scix_mcp_v1.json has drifted from the live MCP server. "
        "Regenerate with `python scripts/gen_mcp_contract.py` and review the "
        "diff (bump CONTRACT_VERSION for a breaking change)."
    )
