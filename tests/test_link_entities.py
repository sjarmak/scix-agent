"""Tests for src/scix/link_entities.py — document-entity linking pipeline."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, call, patch

import pytest

from scix.link_entities import (
    EntityResolver,
    ResolverMatch,
    _extract_mentions_from_payload,
    get_linking_progress,
    link_entities_batch,
)

# ---------------------------------------------------------------------------
# _extract_mentions_from_payload
# ---------------------------------------------------------------------------


class TestExtractMentionsFromPayload:
    def test_combined_payload(self) -> None:
        payload = {
            "instruments": ["HST", "JWST"],
            "datasets": ["SDSS DR16"],
            "methods": [],
            "observables": ["redshift"],
        }
        mentions = _extract_mentions_from_payload(payload, "entity_extraction_v3")
        texts = [m[0] for m in mentions]
        keys = [m[1] for m in mentions]
        assert "HST" in texts
        assert "JWST" in texts
        assert "SDSS DR16" in texts
        assert "redshift" in texts
        assert "instruments" in keys
        assert "datasets" in keys
        assert "observables" in keys

    def test_per_type_payload(self) -> None:
        payload = {"entities": ["Chandra", "XMM-Newton"]}
        mentions = _extract_mentions_from_payload(payload, "instruments")
        assert len(mentions) == 2
        assert mentions[0] == ("Chandra", "instruments")
        assert mentions[1] == ("XMM-Newton", "instruments")

    def test_empty_payload(self) -> None:
        mentions = _extract_mentions_from_payload({}, "methods")
        assert mentions == []

    def test_whitespace_stripped(self) -> None:
        payload = {"instruments": ["  HST  ", ""]}
        mentions = _extract_mentions_from_payload(payload, "entity_extraction_v3")
        assert len(mentions) == 1
        assert mentions[0][0] == "HST"

    def test_non_string_items_skipped(self) -> None:
        payload = {"instruments": [123, None, "ALMA"]}
        mentions = _extract_mentions_from_payload(payload, "entity_extraction_v3")
        assert len(mentions) == 1
        assert mentions[0][0] == "ALMA"


# ---------------------------------------------------------------------------
# EntityResolver
# ---------------------------------------------------------------------------


class TestEntityResolver:
    def _make_conn_with_entities(
        self,
        entities: list[tuple[int, str]],
        aliases: list[tuple[int, str]],
    ) -> MagicMock:
        """Build a mock connection that returns entities and aliases."""
        conn = MagicMock()
        cursor = MagicMock()
        conn.cursor.return_value.__enter__ = MagicMock(return_value=cursor)
        conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

        # First call: entities, second call: aliases
        cursor.fetchall.side_effect = [entities, aliases]
        return conn

    def test_canonical_match(self) -> None:
        conn = self._make_conn_with_entities(
            entities=[(1, "Hubble Space Telescope")],
            aliases=[],
        )
        resolver = EntityResolver(conn)
        match = resolver.resolve("hubble space telescope")
        assert match is not None
        assert match.entity_id == 1
        assert match.confidence == 1.0
        assert match.match_method == "canonical_exact"

    def test_alias_match(self) -> None:
        conn = self._make_conn_with_entities(
            entities=[(1, "Hubble Space Telescope")],
            aliases=[(1, "HST")],
        )
        resolver = EntityResolver(conn)
        match = resolver.resolve("hst")
        assert match is not None
        assert match.entity_id == 1
        assert match.confidence == 0.9
        assert match.match_method == "alias_exact"

    def test_no_match(self) -> None:
        conn = self._make_conn_with_entities(entities=[], aliases=[])
        resolver = EntityResolver(conn)
        assert resolver.resolve("nonexistent") is None

    def test_canonical_takes_priority_over_alias(self) -> None:
        conn = self._make_conn_with_entities(
            entities=[(1, "ALMA")],
            aliases=[(2, "alma")],
        )
        resolver = EntityResolver(conn)
        match = resolver.resolve("alma")
        assert match is not None
        assert match.entity_id == 1
        assert match.confidence == 1.0

    def test_cache_built_once(self) -> None:
        conn = self._make_conn_with_entities(
            entities=[(1, "Test")],
            aliases=[],
        )
        resolver = EntityResolver(conn)
        resolver.resolve("test")
        resolver.resolve("test")
        # cursor() called once for building cache (two fetchall calls within)
        assert conn.cursor.call_count == 1


# ---------------------------------------------------------------------------
# link_entities_batch
# ---------------------------------------------------------------------------


def _mock_conn_for_linking(
    extraction_rows: list[tuple[str, str, dict]],
    linked_bibcodes: list[str] | None = None,
    entities: list[tuple[int, str]] | None = None,
    aliases: list[tuple[int, str]] | None = None,
) -> MagicMock:
    """Build a mock connection for link_entities_batch tests.

    The mock needs to handle multiple cursor() context manager calls with
    different query results.
    """
    conn = MagicMock()
    linked_bibcodes = linked_bibcodes or []
    entities = entities or []
    aliases = aliases or []

    # Track which query is being executed
    call_results: list[list] = []

    # Call 1: SELECT DISTINCT bibcode FROM extractions
    distinct_bibcodes = list({(r[0],) for r in extraction_rows})
    call_results.append(distinct_bibcodes)

    # Call 2: SELECT DISTINCT bibcode FROM document_entities (resume)
    call_results.append([(b,) for b in linked_bibcodes])

    # Call 3: SELECT bibcode, extraction_type, payload FROM extractions (batch)
    call_results.append(extraction_rows)

    # Call 4: entities cache build — SELECT id, canonical_name FROM entities
    # Call 5: aliases cache build — SELECT entity_id, alias FROM entity_aliases
    # These happen inside the same cursor context
    call_results.append(entities)
    call_results.append(aliases)

    # Call 6: INSERT cursor (for actual inserts) — no fetchall needed

    cursor_mock = MagicMock()
    cursor_mock.fetchall = MagicMock(side_effect=call_results)
    cursor_mock.fetchone = MagicMock(return_value=None)

    ctx = MagicMock()
    ctx.__enter__ = MagicMock(return_value=cursor_mock)
    ctx.__exit__ = MagicMock(return_value=False)
    conn.cursor.return_value = ctx

    return conn


class TestLinkEntitiesBatch:
    def test_basic_linking(self) -> None:
        extraction_rows = [
            ("2024ApJ...1A", "entity_extraction_v3", {"instruments": ["ALMA"]}),
        ]
        conn = _mock_conn_for_linking(
            extraction_rows=extraction_rows,
            entities=[(10, "ALMA")],
            aliases=[],
        )

        result = link_entities_batch(
            conn,
            batch_size=100,
            resume=True,
            extraction_type="entity_extraction_v3",
        )

        assert result["bibcodes_processed"] == 1
        assert result["links_created"] == 1
        assert result["skipped_no_match"] == 0

    def test_no_match_counted(self) -> None:
        extraction_rows = [
            ("2024ApJ...1A", "entity_extraction_v3", {"instruments": ["Unknown Thing"]}),
        ]
        conn = _mock_conn_for_linking(
            extraction_rows=extraction_rows,
            entities=[],
            aliases=[],
        )

        result = link_entities_batch(
            conn,
            batch_size=100,
            resume=True,
            extraction_type="entity_extraction_v3",
        )

        assert result["skipped_no_match"] == 1
        assert result["links_created"] == 0

    def test_resume_skips_linked(self) -> None:
        extraction_rows = [
            ("2024ApJ...1A", "entity_extraction_v3", {"instruments": ["ALMA"]}),
        ]
        conn = _mock_conn_for_linking(
            extraction_rows=extraction_rows,
            linked_bibcodes=["2024ApJ...1A"],
            entities=[(10, "ALMA")],
            aliases=[],
        )

        result = link_entities_batch(
            conn,
            batch_size=100,
            resume=True,
            extraction_type="entity_extraction_v3",
        )

        # Bibcode was already linked, so nothing processed
        assert result["bibcodes_processed"] == 0

    def test_commit_called_per_batch(self) -> None:
        # Two bibcodes, batch_size=1 -> should get 2 commits
        extraction_rows = [
            ("2024A", "entity_extraction_v3", {"instruments": ["ALMA"]}),
            ("2024B", "entity_extraction_v3", {"instruments": ["ALMA"]}),
        ]

        conn = MagicMock()
        cursor_mock = MagicMock()

        # We need to handle multiple cursor contexts with different results
        call_idx = {"i": 0}
        results_sequence = [
            # Call 1: distinct bibcodes
            [("2024A",), ("2024B",)],
            # Call 2: linked bibcodes (resume)
            [],
            # Batch 1: extraction payload for 2024A
            [("2024A", "entity_extraction_v3", {"instruments": ["ALMA"]})],
            # EntityResolver cache: entities
            [(10, "ALMA")],
            # EntityResolver cache: aliases
            [],
            # Batch 1 insert cursor (no fetchall)
            # Batch 2: extraction payload for 2024B
            [("2024B", "entity_extraction_v3", {"instruments": ["ALMA"]})],
            # Batch 2 insert cursor (no fetchall)
        ]

        def fetchall_side_effect() -> list:
            idx = call_idx["i"]
            call_idx["i"] += 1
            if idx < len(results_sequence):
                return results_sequence[idx]
            return []

        cursor_mock.fetchall = MagicMock(side_effect=fetchall_side_effect)
        ctx = MagicMock()
        ctx.__enter__ = MagicMock(return_value=cursor_mock)
        ctx.__exit__ = MagicMock(return_value=False)
        conn.cursor.return_value = ctx

        result = link_entities_batch(
            conn,
            batch_size=1,
            resume=True,
            extraction_type="entity_extraction_v3",
        )

        assert result["bibcodes_processed"] == 2
        # commit called once per batch = 2 times
        assert conn.commit.call_count == 2

    def test_on_conflict_do_nothing_in_sql(self) -> None:
        """Verify the INSERT uses ON CONFLICT DO NOTHING."""
        extraction_rows = [
            ("2024ApJ...1A", "entity_extraction_v3", {"instruments": ["ALMA"]}),
        ]
        conn = _mock_conn_for_linking(
            extraction_rows=extraction_rows,
            entities=[(10, "ALMA")],
            aliases=[],
        )

        link_entities_batch(
            conn,
            batch_size=100,
            resume=True,
            extraction_type="entity_extraction_v3",
        )

        # Find the INSERT call
        cursor = conn.cursor.return_value.__enter__.return_value
        insert_calls = [
            c for c in cursor.execute.call_args_list if "INSERT INTO document_entities" in str(c)
        ]
        assert len(insert_calls) > 0
        sql_text = str(insert_calls[0])
        assert "ON CONFLICT" in sql_text
        assert "DO NOTHING" in sql_text

    def test_dry_run_no_writes(self) -> None:
        extraction_rows = [
            ("2024ApJ...1A", "entity_extraction_v3", {"instruments": ["ALMA"]}),
        ]
        conn = _mock_conn_for_linking(
            extraction_rows=extraction_rows,
            entities=[(10, "ALMA")],
            aliases=[],
        )

        result = link_entities_batch(
            conn,
            batch_size=100,
            resume=True,
            extraction_type="entity_extraction_v3",
            dry_run=True,
        )

        assert result["links_created"] == 1
        assert result["bibcodes_processed"] == 1
        # No commit in dry run
        conn.commit.assert_not_called()

    def test_empty_extractions(self) -> None:
        conn = MagicMock()
        cursor_mock = MagicMock()
        cursor_mock.fetchall.return_value = []
        ctx = MagicMock()
        ctx.__enter__ = MagicMock(return_value=cursor_mock)
        ctx.__exit__ = MagicMock(return_value=False)
        conn.cursor.return_value = ctx

        result = link_entities_batch(
            conn,
            batch_size=100,
            extraction_type="entity_extraction_v3",
        )

        assert result["bibcodes_processed"] == 0
        assert result["links_created"] == 0

    def test_evidence_contains_mention_info(self) -> None:
        extraction_rows = [
            ("2024ApJ...1A", "entity_extraction_v3", {"instruments": ["ALMA"]}),
        ]
        conn = _mock_conn_for_linking(
            extraction_rows=extraction_rows,
            entities=[(10, "ALMA")],
            aliases=[],
        )

        link_entities_batch(
            conn,
            batch_size=100,
            resume=True,
            extraction_type="entity_extraction_v3",
        )

        cursor = conn.cursor.return_value.__enter__.return_value
        insert_calls = [
            c for c in cursor.execute.call_args_list if "INSERT INTO document_entities" in str(c)
        ]
        assert len(insert_calls) > 0
        # The evidence JSON should be the 6th positional arg
        args = insert_calls[0][0][1]  # second arg to execute (params tuple)
        evidence = json.loads(args[5])
        assert evidence["mention"] == "ALMA"
        assert evidence["extraction_type"] == "entity_extraction_v3"
        assert evidence["payload_key"] == "instruments"


# ---------------------------------------------------------------------------
# get_linking_progress
# ---------------------------------------------------------------------------


class TestGetLinkingProgress:
    def test_returns_progress(self) -> None:
        conn = MagicMock()
        cursor_mock = MagicMock()
        cursor_mock.fetchone = MagicMock(side_effect=[(100,), (40,)])
        ctx = MagicMock()
        ctx.__enter__ = MagicMock(return_value=cursor_mock)
        ctx.__exit__ = MagicMock(return_value=False)
        conn.cursor.return_value = ctx

        result = get_linking_progress(conn)
        assert result == {
            "total_bibcodes": 100,
            "linked_bibcodes": 40,
            "pending_bibcodes": 60,
        }


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------


class TestCLI:
    def test_default_args(self) -> None:
        from scripts.link_entities import main

        with (
            patch("scripts.link_entities.get_connection") as mock_conn,
            patch("scripts.link_entities.link_entities_batch") as mock_link,
            patch("scripts.link_entities.get_linking_progress") as mock_progress,
        ):
            mock_link.return_value = {
                "bibcodes_processed": 0,
                "links_created": 0,
                "skipped_no_match": 0,
            }
            mock_progress.return_value = {
                "total_bibcodes": 0,
                "linked_bibcodes": 0,
                "pending_bibcodes": 0,
            }

            main([])

            mock_conn.assert_called_once_with(None)
            mock_link.assert_called_once()
            kwargs = mock_link.call_args[1]
            assert kwargs["batch_size"] == 1000
            assert kwargs["resume"] is False
            assert kwargs["extraction_type"] == "entity_extraction_v3"
            assert kwargs["dry_run"] is False

    def test_custom_args(self) -> None:
        from scripts.link_entities import main

        with (
            patch("scripts.link_entities.get_connection") as mock_conn,
            patch("scripts.link_entities.link_entities_batch") as mock_link,
            patch("scripts.link_entities.get_linking_progress") as mock_progress,
        ):
            mock_link.return_value = {
                "bibcodes_processed": 0,
                "links_created": 0,
                "skipped_no_match": 0,
            }
            mock_progress.return_value = {
                "total_bibcodes": 0,
                "linked_bibcodes": 0,
                "pending_bibcodes": 0,
            }

            main(
                [
                    "--batch-size",
                    "500",
                    "--resume",
                    "--extraction-type",
                    "methods",
                    "--db-url",
                    "dbname=test",
                    "--dry-run",
                ]
            )

            mock_conn.assert_called_once_with("dbname=test")
            kwargs = mock_link.call_args[1]
            assert kwargs["batch_size"] == 500
            assert kwargs["resume"] is True
            assert kwargs["extraction_type"] == "methods"
            assert kwargs["dry_run"] is True
