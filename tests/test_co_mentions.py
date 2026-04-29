"""Tests for the ``co_mentions`` table and the entity_context co-mention surface.

Migration 063 introduces ``co_mentions`` (entity↔entity co-occurrence
within papers) and the ``populate_co_mentions.py`` rebuild script.
``entity_context`` (and the standalone ``get_top_co_mentions``) surface
the top-k partners as part of the entity card.

This module exercises three layers:

1. **Unit tests** (mocked cursor) assert ``_fetch_top_co_mentions``
   builds the expected payload shape and clamps the limit, even when
   the underlying table is missing.
2. **Integration tests** seed a tiny fixture in ``SCIX_TEST_DSN``
   (defaults to ``dbname=scix_test``), run
   ``scripts/populate_co_mentions.py`` end-to-end, and assert the
   resulting rows match the hand-computed expectations.
3. **MCP dispatch** test verifies the ``entity_context`` tool accepts
   the new ``co_mentions_limit`` argument and returns a co_mentions
   field in its JSON payload.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import psycopg
import pytest

from scix import search
from scix.db import is_production_dsn
from scix.mcp_server import _dispatch_tool
from scix.search import SearchResult

_REPO_ROOT = Path(__file__).resolve().parent.parent
TEST_DSN = os.environ.get("SCIX_TEST_DSN")


def _integration_skip_reason() -> str | None:
    if not TEST_DSN:
        return "SCIX_TEST_DSN not set — skipping DB integration"
    if is_production_dsn(TEST_DSN):
        return "Refusing to run destructive tests against production DSN"
    return None


# ---------------------------------------------------------------------------
# Unit tests — _fetch_top_co_mentions payload shape and clamping
# ---------------------------------------------------------------------------


class TestFetchTopCoMentionsUnit:
    """Verify the helper builds the right payload shape regardless of DB."""

    @pytest.mark.unit
    def test_returns_empty_list_when_limit_zero(self) -> None:
        cur = MagicMock()
        out = search._fetch_top_co_mentions(cur, entity_id=42, limit=0)
        assert out == []
        cur.execute.assert_not_called()

    @pytest.mark.unit
    def test_returns_empty_list_when_table_missing(self) -> None:
        """If migration 063 has not been applied, degrade gracefully."""
        cur = MagicMock()
        cur.execute.side_effect = psycopg.errors.UndefinedTable(
            'relation "co_mentions" does not exist'
        )
        out = search._fetch_top_co_mentions(cur, entity_id=42, limit=10)
        assert out == []
        cur.connection.rollback.assert_called_once()

    @pytest.mark.unit
    def test_payload_shape(self) -> None:
        """Each row is shaped as the documented partner record."""
        cur = MagicMock()
        cur.fetchall.return_value = [
            {
                "partner_id": 100,
                "partner_name": "Partner A",
                "partner_entity_type": "method",
                "partner_discipline": "astrophysics",
                "n_papers": 12,
                "first_year": 2018,
                "last_year": 2025,
            }
        ]
        out = search._fetch_top_co_mentions(cur, entity_id=42, limit=10)
        assert len(out) == 1
        row = out[0]
        assert set(row) == {
            "partner_id",
            "partner_name",
            "partner_entity_type",
            "partner_discipline",
            "n_papers",
            "first_year",
            "last_year",
        }
        assert row["n_papers"] == 12

    @pytest.mark.unit
    def test_limit_is_clamped_to_max(self) -> None:
        cur = MagicMock()
        cur.fetchall.return_value = []
        search._fetch_top_co_mentions(cur, entity_id=42, limit=search.MAX_CO_MENTIONS_LIMIT * 5)
        # The bound query should have been called with MAX_CO_MENTIONS_LIMIT
        _, kwargs_or_params = cur.execute.call_args.args
        assert kwargs_or_params["lim"] == search.MAX_CO_MENTIONS_LIMIT


# ---------------------------------------------------------------------------
# MCP dispatch test — entity_context surfaces co_mentions field
# ---------------------------------------------------------------------------


class TestEntityContextDispatchSurfacesCoMentions:
    @pytest.mark.unit
    @patch("scix.mcp_server._log_query")
    @patch("scix.search.get_entity_context")
    def test_dispatch_passes_co_mentions_limit(
        self,
        mock_gec: MagicMock,
        _mock_log: MagicMock,
    ) -> None:
        mock_gec.return_value = SearchResult(
            papers=[
                {
                    "entity_id": 1,
                    "canonical_name": "X",
                    "entity_type": "method",
                    "discipline": "astrophysics",
                    "source": "test",
                    "identifiers": [],
                    "aliases": [],
                    "properties": {},
                    "relationships": [],
                    "citing_paper_count": 0,
                    "co_mentions": [
                        {
                            "partner_id": 2,
                            "partner_name": "Y",
                            "partner_entity_type": "method",
                            "partner_discipline": None,
                            "n_papers": 4,
                            "first_year": 2020,
                            "last_year": 2024,
                        }
                    ],
                }
            ],
            total=1,
            timing_ms={"query_ms": 0.1},
        )

        out = _dispatch_tool(
            MagicMock(),
            "entity_context",
            {"entity_id": 1, "co_mentions_limit": 5},
        )
        data = json.loads(out)

        # The handler must have forwarded co_mentions_limit
        _, kwargs = mock_gec.call_args
        assert kwargs.get("co_mentions_limit") == 5

        entity = data["papers"][0]
        assert "co_mentions" in entity
        assert entity["co_mentions"][0]["partner_name"] == "Y"

    @pytest.mark.unit
    @patch("scix.mcp_server._log_query")
    @patch("scix.search.get_entity_context")
    def test_dispatch_default_limit_is_used_when_omitted(
        self,
        mock_gec: MagicMock,
        _mock_log: MagicMock,
    ) -> None:
        mock_gec.return_value = SearchResult(
            papers=[{"entity_id": 1, "co_mentions": []}],
            total=1,
            timing_ms={"query_ms": 0.1},
        )
        _dispatch_tool(MagicMock(), "entity_context", {"entity_id": 1})
        _, kwargs = mock_gec.call_args
        assert kwargs.get("co_mentions_limit") == search.DEFAULT_CO_MENTIONS_LIMIT

    @pytest.mark.unit
    @patch("scix.mcp_server._log_query")
    def test_dispatch_rejects_negative_limit(self, _mock_log: MagicMock) -> None:
        out = _dispatch_tool(
            MagicMock(),
            "entity_context",
            {"entity_id": 1, "co_mentions_limit": -1},
        )
        data = json.loads(out)
        assert "error" in data


# ---------------------------------------------------------------------------
# Integration tests — populate_co_mentions.py end-to-end
# ---------------------------------------------------------------------------


@pytest.fixture
def test_conn() -> psycopg.Connection:
    assert TEST_DSN is not None
    c = psycopg.connect(TEST_DSN)
    yield c
    c.rollback()
    c.close()


def _seed_co_mention_fixture(conn: psycopg.Connection) -> dict[str, int]:
    """Idempotent fixture: 4 entities + 5 papers + document_entities rows.

    Returns a mapping of canonical_name → entity id.

    Design (matches the docstring expectations on the populator):

      * (E1, E2) co-mentioned in P1, P2, P3, P4, P5 → n=5, years 2020..2022
      * (E1, E3) co-mentioned in P1, P3            → n=2, years 2020..2022
      * (E2, E3) co-mentioned in P1, P3            → n=2, years 2020..2022
      * (E1, E4) co-mentioned in P5 only           → n=1 → DROPPED by HAVING
      * (E2, E4) co-mentioned in P5 only           → n=1 → DROPPED by HAVING
    """
    fixture_papers = [
        ("0000P1_comen_fix", "Paper 1", 2020),
        ("0000P2_comen_fix", "Paper 2", 2021),
        ("0000P3_comen_fix", "Paper 3", 2022),
        ("0000P4_comen_fix", "Paper 4", 2022),
        ("0000P5_comen_fix", "Paper 5", None),
    ]
    fixture_entities = [
        "CoMen_E1_fix",
        "CoMen_E2_fix",
        "CoMen_E3_fix",
        "CoMen_E4_fix",
    ]
    fixture_doc_entities = [
        ("0000P1_comen_fix", "CoMen_E1_fix"),
        ("0000P1_comen_fix", "CoMen_E2_fix"),
        ("0000P1_comen_fix", "CoMen_E3_fix"),
        ("0000P2_comen_fix", "CoMen_E1_fix"),
        ("0000P2_comen_fix", "CoMen_E2_fix"),
        ("0000P3_comen_fix", "CoMen_E1_fix"),
        ("0000P3_comen_fix", "CoMen_E2_fix"),
        ("0000P3_comen_fix", "CoMen_E3_fix"),
        ("0000P4_comen_fix", "CoMen_E1_fix"),
        ("0000P4_comen_fix", "CoMen_E2_fix"),
        ("0000P5_comen_fix", "CoMen_E1_fix"),
        ("0000P5_comen_fix", "CoMen_E2_fix"),
        ("0000P5_comen_fix", "CoMen_E4_fix"),
    ]

    with conn.cursor() as cur:
        for bibcode, title, year in fixture_papers:
            cur.execute(
                """
                INSERT INTO papers (bibcode, title, year)
                VALUES (%s, %s, %s)
                ON CONFLICT (bibcode) DO UPDATE SET year = EXCLUDED.year
                """,
                (bibcode, title, year),
            )

        ids: dict[str, int] = {}
        for name in fixture_entities:
            cur.execute(
                """
                INSERT INTO entities (canonical_name, entity_type, source)
                VALUES (%s, 'method', 'co_mentions_fixture')
                ON CONFLICT (canonical_name, entity_type, source) DO UPDATE
                    SET updated_at = now()
                RETURNING id
                """,
                (name,),
            )
            row = cur.fetchone()
            assert row is not None
            ids[name] = int(row[0])

        for bibcode, ename in fixture_doc_entities:
            cur.execute(
                """
                INSERT INTO document_entities
                    (bibcode, entity_id, link_type, confidence, match_method, tier)
                VALUES (%s, %s, 'mentions', 1.0, 'co_mentions_fixture_method', 1)
                ON CONFLICT (bibcode, entity_id, link_type, tier) DO NOTHING
                """,
                (bibcode, ids[ename]),
            )
    conn.commit()
    return ids


def _teardown_co_mention_fixture(conn: psycopg.Connection, ids: dict[str, int]) -> None:
    fixture_bibcodes = [
        "0000P1_comen_fix",
        "0000P2_comen_fix",
        "0000P3_comen_fix",
        "0000P4_comen_fix",
        "0000P5_comen_fix",
    ]
    with conn.cursor() as cur:
        cur.execute(
            "DELETE FROM document_entities WHERE bibcode = ANY(%s)",
            (fixture_bibcodes,),
        )
        cur.execute(
            "DELETE FROM co_mentions " "WHERE entity_a_id = ANY(%s) OR entity_b_id = ANY(%s)",
            (list(ids.values()), list(ids.values())),
        )
        cur.execute("DELETE FROM entities WHERE id = ANY(%s)", (list(ids.values()),))
        cur.execute("DELETE FROM papers WHERE bibcode = ANY(%s)", (fixture_bibcodes,))
    conn.commit()


@pytest.mark.integration
@pytest.mark.skipif(
    _integration_skip_reason() is not None,
    reason=_integration_skip_reason() or "",
)
class TestPopulateCoMentionsIntegration:
    def test_populate_full_writes_expected_pairs(self, test_conn: psycopg.Connection) -> None:
        """Run populate_co_mentions.py and assert the produced edges."""
        ids = _seed_co_mention_fixture(test_conn)
        try:
            out = subprocess.run(  # noqa: S603 — fixed argv
                [
                    sys.executable,
                    str(_REPO_ROOT / "scripts" / "populate_co_mentions.py"),
                ],
                cwd=str(_REPO_ROOT),
                env={**os.environ, "SCIX_TEST_DSN": TEST_DSN or ""},
                capture_output=True,
                text=True,
                check=False,
                timeout=180,
            )
            assert out.returncode == 0, (
                f"populate_co_mentions exited {out.returncode}\n"
                f"stdout:\n{out.stdout}\nstderr:\n{out.stderr}"
            )

            with test_conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT entity_a_id, entity_b_id, n_papers, first_year, last_year
                      FROM co_mentions
                     WHERE entity_a_id = ANY(%s) OR entity_b_id = ANY(%s)
                     ORDER BY entity_a_id, entity_b_id
                    """,
                    (list(ids.values()), list(ids.values())),
                )
                rows = cur.fetchall()

            # Build the expected set in canonical (a<b) order.
            expected = sorted(
                [
                    (
                        min(ids["CoMen_E1_fix"], ids["CoMen_E2_fix"]),
                        max(ids["CoMen_E1_fix"], ids["CoMen_E2_fix"]),
                        5,
                        2020,
                        2022,
                    ),
                    (
                        min(ids["CoMen_E1_fix"], ids["CoMen_E3_fix"]),
                        max(ids["CoMen_E1_fix"], ids["CoMen_E3_fix"]),
                        2,
                        2020,
                        2022,
                    ),
                    (
                        min(ids["CoMen_E2_fix"], ids["CoMen_E3_fix"]),
                        max(ids["CoMen_E2_fix"], ids["CoMen_E3_fix"]),
                        2,
                        2020,
                        2022,
                    ),
                ]
            )
            actual = sorted([(r[0], r[1], r[2], r[3], r[4]) for r in rows])
            assert actual == expected, (
                f"unexpected co_mentions rows.\n" f"expected: {expected}\nactual:   {actual}"
            )
        finally:
            _teardown_co_mention_fixture(test_conn, ids)

    def test_get_top_co_mentions_returns_partners_after_populate(
        self, test_conn: psycopg.Connection
    ) -> None:
        ids = _seed_co_mention_fixture(test_conn)
        try:
            subprocess.run(  # noqa: S603
                [
                    sys.executable,
                    str(_REPO_ROOT / "scripts" / "populate_co_mentions.py"),
                ],
                cwd=str(_REPO_ROOT),
                env={**os.environ, "SCIX_TEST_DSN": TEST_DSN or ""},
                capture_output=True,
                text=True,
                check=True,
                timeout=180,
            )
            result = search.get_top_co_mentions(test_conn, ids["CoMen_E1_fix"], limit=10)
            partners = {(r["partner_id"], r["n_papers"]) for r in result.papers}
            assert (ids["CoMen_E2_fix"], 5) in partners
            assert (ids["CoMen_E3_fix"], 2) in partners
            # E4 only has n=1 with E1 → must NOT appear
            assert ids["CoMen_E4_fix"] not in {p["partner_id"] for p in result.papers}
        finally:
            _teardown_co_mention_fixture(test_conn, ids)


# ---------------------------------------------------------------------------
# Schema invariant tests — CHECK constraints
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.skipif(
    _integration_skip_reason() is not None,
    reason=_integration_skip_reason() or "",
)
class TestCoMentionsSchemaConstraints:
    """Migration 063 enforces a<b, n>=2, and first_year<=last_year via CHECKs.

    These tests assert that ad-hoc INSERTs that violate any of those
    constraints are rejected by the database, so consumers don't need to
    re-validate the invariants.
    """

    def test_a_must_be_less_than_b(self, test_conn: psycopg.Connection) -> None:
        with test_conn.cursor() as cur:
            with pytest.raises(psycopg.errors.CheckViolation):
                cur.execute("""
                    INSERT INTO co_mentions (entity_a_id, entity_b_id, n_papers)
                    VALUES (10, 5, 2)
                    """)
        test_conn.rollback()

    def test_n_papers_must_be_at_least_two(self, test_conn: psycopg.Connection) -> None:
        with test_conn.cursor() as cur:
            with pytest.raises(psycopg.errors.CheckViolation):
                cur.execute("""
                    INSERT INTO co_mentions (entity_a_id, entity_b_id, n_papers)
                    VALUES (5, 10, 1)
                    """)
        test_conn.rollback()

    def test_first_year_must_not_exceed_last_year(self, test_conn: psycopg.Connection) -> None:
        with test_conn.cursor() as cur:
            with pytest.raises(psycopg.errors.CheckViolation):
                cur.execute("""
                    INSERT INTO co_mentions
                        (entity_a_id, entity_b_id, n_papers, first_year, last_year)
                    VALUES (5, 10, 2, 2025, 2020)
                    """)
        test_conn.rollback()
