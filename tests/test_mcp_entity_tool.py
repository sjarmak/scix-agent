"""Integration tests for provenance filters on the MCP `entity` tool.

Exercises the new ``min_confidence_tier`` and ``sources`` arguments added by
the mcp-entity-provenance-filter work unit. These tests require a real
PostgreSQL instance with the ``extractions`` and ``papers`` tables in place
(migrations 001 and 017 applied). They seed a handful of extraction rows,
invoke ``_handle_entity`` directly (bypassing the MCP wire layer), and
assert that the returned paper set matches the expected provenance filter.

The tests skip cleanly when ``SCIX_TEST_DSN`` is not set or points at the
production database — see ``tests/helpers.get_test_dsn``.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

import psycopg
import pytest

from scix.mcp_server import _handle_entity
from tests.helpers import get_test_dsn

# Bibcode prefix used for all seeded rows so teardown can delete exactly the
# rows this fixture created, regardless of what else lives in scix_test. The
# extractions table has UNIQUE (bibcode, extraction_type, extraction_version)
# — migration 009 — so we use one bibcode per seed row to keep inserts
# independent and the seed data flat.
_TEST_BIBCODE_PREFIX = "TEST-ENTPROV-"

# The CHECK constraint on extractions.source restricts INSERTs to the set
# below (migration 017). Use allowed values so seeding actually succeeds —
# the filter contract is tested against these values. The PRD's example
# names (e.g. 'ads_metadata', 'ner_wiesp') are illustrative aliases for
# these same provenance lanes.
_SEED_ROWS: list[dict[str, Any]] = [
    # source='metadata', tier='high'
    {
        "bibcode": f"{_TEST_BIBCODE_PREFIX}0001",
        "extraction_type": "instruments",
        "extraction_version": "v1",
        "payload": {"instruments": ["HST"]},
        "source": "metadata",
        "confidence_tier": "high",
    },
    # source='metadata', tier='medium'
    {
        "bibcode": f"{_TEST_BIBCODE_PREFIX}0002",
        "extraction_type": "instruments",
        "extraction_version": "v1",
        "payload": {"instruments": ["HST"]},
        "source": "metadata",
        "confidence_tier": "medium",
    },
    # source='ner', tier='high'
    {
        "bibcode": f"{_TEST_BIBCODE_PREFIX}0003",
        "extraction_type": "instruments",
        "extraction_version": "v1",
        "payload": {"instruments": ["HST"]},
        "source": "ner",
        "confidence_tier": "high",
    },
    # source='ner', tier='low'
    {
        "bibcode": f"{_TEST_BIBCODE_PREFIX}0004",
        "extraction_type": "instruments",
        "extraction_version": "v1",
        "payload": {"instruments": ["HST"]},
        "source": "ner",
        "confidence_tier": "low",
    },
    # source='llm', tier='medium'
    {
        "bibcode": f"{_TEST_BIBCODE_PREFIX}0005",
        "extraction_type": "instruments",
        "extraction_version": "v1",
        "payload": {"instruments": ["HST"]},
        "source": "llm",
        "confidence_tier": "medium",
    },
    # source='llm', tier='low'
    {
        "bibcode": f"{_TEST_BIBCODE_PREFIX}0006",
        "extraction_type": "instruments",
        "extraction_version": "v1",
        "payload": {"instruments": ["HST"]},
        "source": "llm",
        "confidence_tier": "low",
    },
]


def _cleanup(conn: psycopg.Connection) -> None:
    """Remove any rows this fixture may have created on prior runs."""
    conn.rollback()
    with conn.cursor() as cur:
        cur.execute(
            "DELETE FROM extractions WHERE bibcode LIKE %s",
            (f"{_TEST_BIBCODE_PREFIX}%",),
        )
        cur.execute(
            "DELETE FROM papers WHERE bibcode LIKE %s",
            (f"{_TEST_BIBCODE_PREFIX}%",),
        )
    conn.commit()


@pytest.fixture()
def entity_conn() -> Iterator[psycopg.Connection]:
    """Yield a connection with 6 seeded extraction rows; clean up after."""
    dsn = get_test_dsn()
    if dsn is None:
        pytest.skip("SCIX_TEST_DSN not set or points at production")
    conn = psycopg.connect(dsn)
    conn.autocommit = False
    try:
        # Start from a known-clean state (prior run may have crashed before
        # teardown). Seed a test paper and the 6 extraction rows pointing
        # at it.
        _cleanup(conn)
        with conn.cursor() as cur:
            for row in _SEED_ROWS:
                cur.execute(
                    "INSERT INTO papers (bibcode, title) VALUES (%s, %s)",
                    (
                        row["bibcode"],
                        "Test Paper for Entity Provenance Filters",
                    ),
                )
                cur.execute(
                    "INSERT INTO extractions "
                    "(bibcode, extraction_type, extraction_version, "
                    " payload, source, confidence_tier) "
                    "VALUES (%s, %s, %s, %s::jsonb, %s, %s)",
                    (
                        row["bibcode"],
                        row["extraction_type"],
                        row["extraction_version"],
                        json.dumps(row["payload"]),
                        row["source"],
                        row["confidence_tier"],
                    ),
                )
        conn.commit()
        yield conn
    finally:
        try:
            _cleanup(conn)
        finally:
            conn.close()


def _invoke_search(
    conn: psycopg.Connection,
    *,
    min_confidence_tier: int | None = None,
    sources: list[str] | None = None,
) -> dict[str, Any]:
    args: dict[str, Any] = {
        "action": "search",
        "entity_type": "instruments",
        "query": "HST",
        "limit": 50,
    }
    if min_confidence_tier is not None:
        args["min_confidence_tier"] = min_confidence_tier
    if sources is not None:
        args["sources"] = sources
    return json.loads(_handle_entity(conn, args))


# ---------------------------------------------------------------------------
# Acceptance tests
# ---------------------------------------------------------------------------


def _ours(result: dict[str, Any]) -> list[dict[str, Any]]:
    """Return only rows belonging to this test fixture's bibcode namespace."""
    return [
        p
        for p in result["papers"]
        if p["bibcode"].startswith(_TEST_BIBCODE_PREFIX)
    ]


@pytest.mark.integration
def test_no_filter_returns_all_seeded_rows(
    entity_conn: psycopg.Connection,
) -> None:
    """AC4: both args omitted -> behavior matches pre-change entity tool."""
    result = _invoke_search(entity_conn)
    ours = _ours(result)
    assert len(ours) == len(_SEED_ROWS), (
        f"expected {len(_SEED_ROWS)} rows, got {len(ours)}: "
        f"{[p['bibcode'] for p in ours]}"
    )


@pytest.mark.integration
def test_sources_single_filters_to_subset(
    entity_conn: psycopg.Connection,
) -> None:
    """AC5 (part 1): sources=['metadata'] returns only metadata rows."""
    result = _invoke_search(entity_conn, sources=["metadata"])
    ours = _ours(result)
    expected_count = sum(1 for r in _SEED_ROWS if r["source"] == "metadata")
    assert len(ours) == expected_count == 2


@pytest.mark.integration
def test_sources_multiple_returns_union(
    entity_conn: psycopg.Connection,
) -> None:
    """AC5 (part 2): sources=['metadata','ner'] returns the union."""
    result = _invoke_search(entity_conn, sources=["metadata", "ner"])
    ours = _ours(result)
    expected_count = sum(
        1 for r in _SEED_ROWS if r["source"] in {"metadata", "ner"}
    )
    assert len(ours) == expected_count == 4


@pytest.mark.integration
def test_min_confidence_tier_filters_to_high_only(
    entity_conn: psycopg.Connection,
) -> None:
    """AC3: min_confidence_tier=3 returns only high-tier rows."""
    result = _invoke_search(entity_conn, min_confidence_tier=3)
    ours = _ours(result)
    expected_count = sum(
        1 for r in _SEED_ROWS if r["confidence_tier"] == "high"
    )
    assert len(ours) == expected_count == 2


@pytest.mark.integration
def test_min_confidence_tier_2_keeps_medium_and_high(
    entity_conn: psycopg.Connection,
) -> None:
    """AC3: min_confidence_tier=2 returns medium+high rows."""
    result = _invoke_search(entity_conn, min_confidence_tier=2)
    ours = _ours(result)
    expected_count = sum(
        1
        for r in _SEED_ROWS
        if r["confidence_tier"] in {"medium", "high"}
    )
    assert len(ours) == expected_count == 4


@pytest.mark.integration
def test_combined_filters_narrow_further(
    entity_conn: psycopg.Connection,
) -> None:
    """AC2+AC3: both filters together narrow further than either alone."""
    result = _invoke_search(
        entity_conn,
        sources=["metadata", "ner"],
        min_confidence_tier=3,  # high only
    )
    ours = _ours(result)
    # Expected: metadata/high + ner/high = 2 rows
    expected_count = sum(
        1
        for r in _SEED_ROWS
        if r["source"] in {"metadata", "ner"}
        and r["confidence_tier"] == "high"
    )
    assert len(ours) == expected_count == 2


# ---------------------------------------------------------------------------
# Schema assertions — these do NOT require a DB and must always run.
# ---------------------------------------------------------------------------


def _get_entity_tool_schema() -> dict[str, Any]:
    """Extract the entity tool's inputSchema from the live MCP server."""
    import asyncio

    from mcp.types import ListToolsRequest

    from scix.mcp_server import create_server

    server = create_server(_run_self_test=False)
    handler = server.request_handlers[ListToolsRequest]
    loop = asyncio.new_event_loop()
    try:
        result = loop.run_until_complete(
            handler(ListToolsRequest(method="tools/list"))
        )
    finally:
        loop.close()
    tools = result.root.tools if hasattr(result, "root") else result.tools
    entity_tool = next(t for t in tools if t.name == "entity")
    return entity_tool.inputSchema  # type: ignore[no-any-return]


def test_input_schema_exposes_min_confidence_tier() -> None:
    """AC8: schema documents min_confidence_tier with type + description."""
    try:
        schema = _get_entity_tool_schema()
    except (ImportError, AttributeError):
        pytest.skip("mcp SDK not installed or server API changed")
    props = schema["properties"]
    assert "min_confidence_tier" in props
    prop = props["min_confidence_tier"]
    assert prop["type"] == "integer"
    assert prop.get("description"), "description must be non-empty"
    assert prop.get("enum") == [1, 2, 3]


def test_input_schema_exposes_sources() -> None:
    """AC8: schema documents sources with type + description."""
    try:
        schema = _get_entity_tool_schema()
    except (ImportError, AttributeError):
        pytest.skip("mcp SDK not installed or server API changed")
    props = schema["properties"]
    assert "sources" in props
    prop = props["sources"]
    assert prop["type"] == "array"
    assert prop["items"]["type"] == "string"
    assert prop.get("description"), "description must be non-empty"


def test_input_schema_new_props_not_required() -> None:
    """Backward compat: new properties are NOT in the required list."""
    try:
        schema = _get_entity_tool_schema()
    except (ImportError, AttributeError):
        pytest.skip("mcp SDK not installed or server API changed")
    required = schema.get("required", [])
    assert "min_confidence_tier" not in required
    assert "sources" not in required
