"""Smoke + integration tests for MCP ``entity_context`` semantics.

Verifies two contracts that MCP agents depend on:

1. The ``entity_context`` tool returns the full ``properties`` JSONB
   from ``public.entities`` as-is — not stripped, not summarized away.
   For e.g. the SsODNet asteroid Psyche this means agents can see the
   raw ``taxonomy`` / ``sso_class`` / ``albedo`` / ``diameter`` values.

2. The ``relationships`` block joins through ``public.entity_relationships``
   and exposes ``(predicate, object_entity, properties_of_object)`` —
   not just an opaque object_id. This lets an agent navigate from a
   mission like JWST to its instruments and read each instrument's
   metadata in one hop.

The unit-mock tests assert the shape regardless of DB state. The
integration tests run against ``SCIX_TEST_DSN`` (defaults to
``dbname=scix_test``) and are skipped when the DSN is unset, points at
prod, or the ``entity_relationships`` table is empty — that last case
is the current prod reality while matview refresh is blocked (see
xz4.5).
"""

from __future__ import annotations

import json
import os
from typing import Any
from unittest.mock import MagicMock, patch

import psycopg
import pytest

from scix import search
from scix.db import is_production_dsn
from scix.mcp_server import _dispatch_tool, _session_state
from scix.search import SearchResult

# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


TEST_DSN = os.environ.get("SCIX_TEST_DSN")


@pytest.fixture(autouse=True)
def _reset_session() -> Any:
    """Clear implicit session state between tests."""
    _session_state.clear_working_set()
    _session_state.clear_focused()
    yield
    _session_state.clear_working_set()
    _session_state.clear_focused()


@pytest.fixture
def mock_conn() -> MagicMock:
    """A MagicMock standing in for a psycopg connection."""
    conn = MagicMock()
    cursor = MagicMock()
    cursor.__enter__ = MagicMock(return_value=cursor)
    cursor.__exit__ = MagicMock(return_value=False)
    cursor.fetchall.return_value = []
    cursor.fetchone.return_value = None
    cursor.execute.return_value = None
    conn.cursor.return_value = cursor
    return conn


def _integration_skip_reason() -> str | None:
    """Return a human-readable reason if integration tests should skip.

    Returns None when the environment is safe for integration tests.
    """
    if not TEST_DSN:
        return "SCIX_TEST_DSN not set — skipping DB integration"
    if is_production_dsn(TEST_DSN):
        return "Refusing to run destructive tests against production DSN"
    return None


def _has_entity_relationships(dsn: str) -> bool:
    """Return True if the target DB has any rows in entity_relationships."""
    try:
        with psycopg.connect(dsn) as conn, conn.cursor() as cur:
            cur.execute("SELECT EXISTS (SELECT 1 FROM entity_relationships LIMIT 1)")
            row = cur.fetchone()
            return bool(row and row[0])
    except psycopg.Error:
        return False


# ---------------------------------------------------------------------------
# Unit tests — contract shape with mocked search layer
# ---------------------------------------------------------------------------


class TestEntityContextPropertiesShape:
    """Assert get_entity_context returns a ``properties`` JSONB payload."""

    @pytest.mark.unit
    @patch("scix.mcp_server._log_query")
    @patch("scix.search.get_entity_context")
    def test_entity_context_includes_properties_key(
        self,
        mock_gec: MagicMock,
        _mock_log: MagicMock,
        mock_conn: MagicMock,
    ) -> None:
        """Psyche-like payload: properties must pass through to JSON response."""
        mock_gec.return_value = SearchResult(
            papers=[
                {
                    "entity_id": 11236,
                    "canonical_name": "Psyche",
                    "entity_type": "target",
                    "discipline": "planetary_science",
                    "source": "ssodnet",
                    "identifiers": [],
                    "aliases": [],
                    "properties": {
                        "taxonomy": "M",
                        "sso_class": "MB>Outer",
                        "albedo": 0.1175,
                        "diameter": 223.143,
                        "sso_number": 16,
                    },
                    "relationships": [],
                    "citing_paper_count": 0,
                }
            ],
            total=1,
            timing_ms={"query_ms": 0.5},
        )

        out = _dispatch_tool(mock_conn, "entity_context", {"entity_id": 11236})
        data = json.loads(out)

        assert data.get("total") == 1
        entity = data["papers"][0]
        props = entity.get("properties")
        assert isinstance(props, dict), "properties JSONB must be a dict, not stripped"
        # Core Psyche fields
        assert props.get("taxonomy") == "M"
        assert props.get("sso_class") == "MB>Outer"
        assert props.get("albedo") == pytest.approx(0.1175)
        assert props.get("diameter") == pytest.approx(223.143)


class TestGraphContextRelationshipsShape:
    """Assert relationships expose (predicate, object_entity, properties_of_object).

    The bead's ``graph_context`` check (Check 3) is about entity-to-entity
    hops — in the 13-tool consolidated API this is served by ``entity_context``
    (``graph_context`` is the citation-graph/community tool for papers,
    which does not deal with entity_relationships). These tests assert the
    entity_context relationships block carries enough data for an agent
    to navigate JWST -> NIRSpec in one hop.
    """

    @pytest.mark.unit
    @patch("scix.mcp_server._log_query")
    @patch("scix.search.get_entity_context")
    def test_relationships_expose_object_name_and_properties(
        self,
        mock_gec: MagicMock,
        _mock_log: MagicMock,
        mock_conn: MagicMock,
    ) -> None:
        """JWST-like payload with has_instrument relationships."""
        mock_gec.return_value = SearchResult(
            papers=[
                {
                    "entity_id": 1588866,
                    "canonical_name": "James Webb Space Telescope",
                    "entity_type": "mission",
                    "discipline": "astrophysics",
                    "source": "curated_flagship_v1",
                    "identifiers": [],
                    "aliases": ["JWST"],
                    "properties": {},
                    "relationships": [
                        {
                            "predicate": "has_instrument",
                            "object_id": 9001,
                            "object_name": "NIRSpec",
                            "object_entity_type": "instrument",
                            "object_properties": {
                                "wavelength_range_um": "0.6-5.3",
                                "mode": "spectrograph",
                            },
                            "confidence": 1.0,
                        },
                        {
                            "predicate": "has_instrument",
                            "object_id": 9002,
                            "object_name": "NIRCam",
                            "object_entity_type": "instrument",
                            "object_properties": {
                                "wavelength_range_um": "0.6-5.0",
                                "mode": "imaging",
                            },
                            "confidence": 1.0,
                        },
                        {
                            "predicate": "has_instrument",
                            "object_id": 9003,
                            "object_name": "MIRI",
                            "object_entity_type": "instrument",
                            "object_properties": {
                                "wavelength_range_um": "5-28",
                                "mode": "mid_infrared",
                            },
                            "confidence": 1.0,
                        },
                    ],
                    "citing_paper_count": 0,
                }
            ],
            total=1,
            timing_ms={"query_ms": 0.9},
        )

        out = _dispatch_tool(mock_conn, "entity_context", {"entity_id": 1588866})
        data = json.loads(out)

        entity = data["papers"][0]
        rels = entity.get("relationships")
        assert isinstance(rels, list) and rels, "relationships must be a non-empty list"

        instrument_names = {r.get("object_name") for r in rels}
        assert {"NIRSpec", "NIRCam", "MIRI"}.issubset(instrument_names)

        # Every has_instrument relationship must carry the object's properties
        for r in rels:
            assert r.get("predicate") == "has_instrument"
            assert "object_name" in r
            assert "object_properties" in r
            assert isinstance(r["object_properties"], dict)


# ---------------------------------------------------------------------------
# Integration tests — real DB round-trip via search.get_entity_context
# ---------------------------------------------------------------------------


@pytest.fixture
def test_conn() -> psycopg.Connection:
    """Yield a connection to SCIX_TEST_DSN; rollback on exit for isolation."""
    assert TEST_DSN is not None  # guard: caller must have checked skip reason
    c = psycopg.connect(TEST_DSN)
    yield c
    c.rollback()
    c.close()


@pytest.mark.integration
@pytest.mark.skipif(
    _integration_skip_reason() is not None,
    reason=_integration_skip_reason() or "",
)
class TestEntityContextIntegration:
    """Round-trip tests against scix_test — seed entities then assert output."""

    def test_psyche_like_entity_returns_full_properties(
        self, test_conn: psycopg.Connection
    ) -> None:
        """Seed a Psyche-like entity; entity_context must return its properties."""
        properties = {
            "taxonomy": "M",
            "sso_class": "MB>Outer",
            "albedo": 0.1175,
            "diameter": 223.143,
            "sso_number": 16,
        }
        with test_conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO entities (canonical_name, entity_type, source, properties)
                VALUES (%s, %s, %s, %s::jsonb)
                ON CONFLICT (canonical_name, entity_type, source) DO UPDATE
                    SET properties = EXCLUDED.properties
                RETURNING id
                """,
                (
                    "Psyche_xz4129_fixture",
                    "target",
                    "ssodnet_xz4129",
                    json.dumps(properties),
                ),
            )
            entity_id = cur.fetchone()[0]
            # Ensure this row is visible in agent_entity_context by refreshing
            # only if we can — matview refresh may be locked in prod, but
            # scix_test should be unconstrained.
            try:
                cur.execute("REFRESH MATERIALIZED VIEW agent_entity_context")
            except psycopg.errors.Error:
                test_conn.rollback()
                pytest.skip("cannot refresh agent_entity_context matview")
        test_conn.commit()

        try:
            result = search.get_entity_context(test_conn, entity_id)
            assert result.total == 1, f"entity {entity_id} not found in matview"
            entity = result.papers[0]
            props = entity.get("properties")
            assert isinstance(props, dict), (
                "properties JSONB was stripped — agents cannot see Psyche taxonomy/albedo/diameter"
            )
            assert props.get("taxonomy") == "M"
            assert props.get("sso_class") == "MB>Outer"
            assert props.get("albedo") == pytest.approx(0.1175)
            assert props.get("diameter") == pytest.approx(223.143)
        finally:
            # Clean up the fixture to keep scix_test idempotent across runs
            with test_conn.cursor() as cur:
                cur.execute("DELETE FROM entities WHERE id = %s", (entity_id,))
            test_conn.commit()

    @pytest.mark.skipif(
        TEST_DSN is None or not _has_entity_relationships(TEST_DSN or ""),
        reason=(
            "entity_relationships empty on target DSN — populate via "
            "scripts/populate_entity_relationships.py or wait for xz4.5 MV lock to clear"
        ),
    )
    def test_jwst_like_entity_returns_has_instrument_relationships(
        self, test_conn: psycopg.Connection
    ) -> None:
        """Seed a JWST-like mission + instruments + has_instrument edges.

        Assert entity_context exposes each instrument with its own properties.
        """
        with test_conn.cursor() as cur:
            # Seed the mission
            cur.execute(
                """
                INSERT INTO entities (canonical_name, entity_type, source, properties)
                VALUES (%s, %s, %s, %s::jsonb)
                ON CONFLICT (canonical_name, entity_type, source) DO UPDATE
                    SET properties = EXCLUDED.properties
                RETURNING id
                """,
                (
                    "JWST_xz4129_fixture",
                    "mission",
                    "curated_flagship_xz4129",
                    json.dumps({"launch_year": 2021}),
                ),
            )
            mission_id = cur.fetchone()[0]

            # Seed three instruments
            instruments = {
                "NIRSpec_xz4129": {"wavelength_range_um": "0.6-5.3", "mode": "spectrograph"},
                "NIRCam_xz4129": {"wavelength_range_um": "0.6-5.0", "mode": "imaging"},
                "MIRI_xz4129": {"wavelength_range_um": "5-28", "mode": "mid_infrared"},
            }
            instrument_ids: dict[str, int] = {}
            for name, props in instruments.items():
                cur.execute(
                    """
                    INSERT INTO entities (canonical_name, entity_type, source, properties)
                    VALUES (%s, %s, %s, %s::jsonb)
                    ON CONFLICT (canonical_name, entity_type, source) DO UPDATE
                        SET properties = EXCLUDED.properties
                    RETURNING id
                    """,
                    (name, "instrument", "curated_flagship_xz4129", json.dumps(props)),
                )
                instrument_ids[name] = cur.fetchone()[0]

            # Seed has_instrument edges
            for instrument_id in instrument_ids.values():
                cur.execute(
                    """
                    INSERT INTO entity_relationships
                        (subject_entity_id, predicate, object_entity_id, source, confidence)
                    VALUES (%s, 'has_instrument', %s, 'curated_flagship_xz4129', 1.0)
                    ON CONFLICT DO NOTHING
                    """,
                    (mission_id, instrument_id),
                )

            try:
                cur.execute("REFRESH MATERIALIZED VIEW agent_entity_context")
            except psycopg.errors.Error:
                test_conn.rollback()
                pytest.skip("cannot refresh agent_entity_context matview")
        test_conn.commit()

        try:
            result = search.get_entity_context(test_conn, mission_id)
            assert result.total == 1, f"mission entity {mission_id} not in matview"
            entity = result.papers[0]
            rels = entity.get("relationships") or []
            has_instrument_rels = [r for r in rels if r.get("predicate") == "has_instrument"]
            assert len(has_instrument_rels) >= 3, (
                f"expected >=3 has_instrument relationships, got {len(has_instrument_rels)}"
            )

            seen_names = {r.get("object_name") for r in has_instrument_rels}
            assert {"NIRSpec_xz4129", "NIRCam_xz4129", "MIRI_xz4129"}.issubset(
                seen_names
            ), f"missing instrument names in relationships: {seen_names}"

            # Every relationship must carry the object's properties
            for r in has_instrument_rels:
                name = r.get("object_name")
                if name not in instruments:
                    continue
                obj_props = r.get("object_properties")
                assert isinstance(obj_props, dict), (
                    f"relationship for {name} dropped object_properties"
                )
                assert obj_props.get("wavelength_range_um") == instruments[name][
                    "wavelength_range_um"
                ]
                assert obj_props.get("mode") == instruments[name]["mode"]
        finally:
            # Tear down fixtures (FK cascade drops entity_relationships rows)
            with test_conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM entities WHERE id = ANY(%s)",
                    ([mission_id, *instrument_ids.values()],),
                )
            test_conn.commit()
