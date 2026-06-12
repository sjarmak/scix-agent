"""Tests for scripts/backfill_qdrant_filter_fields.py."""
from __future__ import annotations

from unittest.mock import MagicMock, call, patch

import pytest

from scripts.backfill_qdrant_filter_fields import (
    COLLECTION_DEFAULT,
    INDEXED_FIELDS,
    _bibcode_to_point_id,
    _build_payload,
    apply_batch,
    ensure_indexes,
    stream_pg_batches,
)


# ---------------------------------------------------------------------------
# _build_payload
# ---------------------------------------------------------------------------


class TestBuildPayload:
    def _full_row(self) -> dict:
        return {
            "bibcode": "2020ApJ...900..100X",
            "year": 2020,
            "doctype": "article",
            "arxiv_class": ["astro-ph.HE", "astro-ph.GA"],
            "bibstem": ["ApJ"],
            "title": "A test paper",
            "first_author": "Smith, A.",
            "citation_count": 42,
            "is_retracted": False,
            "community_semantic_coarse": 3,
            "community_semantic_medium": 17,
            "pagerank": 0.00012,
        }

    def test_full_row_produces_expected_keys(self) -> None:
        p = _build_payload(self._full_row())
        assert set(p) == {
            "year", "doctype", "arxiv_class", "bibstem",
            "community_semantic_coarse", "community_semantic_medium",
            "title", "first_author", "citation_count", "pagerank",
        }
        # is_retracted=False → absent (absence semantics per ADR-008)
        assert "is_retracted" not in p

    def test_is_retracted_true_is_included(self) -> None:
        row = self._full_row()
        row["is_retracted"] = True
        p = _build_payload(row)
        assert p["is_retracted"] is True

    def test_null_fields_omitted(self) -> None:
        row = self._full_row()
        row["year"] = None
        row["doctype"] = None
        row["arxiv_class"] = None
        row["bibstem"] = None
        row["community_semantic_coarse"] = None
        row["community_semantic_medium"] = None
        row["pagerank"] = None
        p = _build_payload(row)
        for key in ("year", "doctype", "arxiv_class", "bibstem",
                    "community_semantic_coarse", "community_semantic_medium", "pagerank"):
            assert key not in p, f"{key} should be absent when source is NULL"

    def test_empty_list_fields_omitted(self) -> None:
        row = self._full_row()
        row["arxiv_class"] = []
        row["bibstem"] = []
        p = _build_payload(row)
        assert "arxiv_class" not in p
        assert "bibstem" not in p

    def test_year_coerced_to_int(self) -> None:
        row = self._full_row()
        row["year"] = "2019"  # psycopg can return smallint as str in some modes
        p = _build_payload(row)
        assert isinstance(p["year"], int)

    def test_citation_count_coerced_to_int(self) -> None:
        row = self._full_row()
        row["citation_count"] = 7
        p = _build_payload(row)
        assert isinstance(p["citation_count"], int)

    def test_arxiv_class_stored_as_list(self) -> None:
        row = self._full_row()
        p = _build_payload(row)
        assert isinstance(p["arxiv_class"], list)


# ---------------------------------------------------------------------------
# ensure_indexes
# ---------------------------------------------------------------------------


class TestEnsureIndexes:
    def _mock_client(self, existing_schema: dict | None = None) -> MagicMock:
        c = MagicMock()
        c.create_payload_index.return_value = None
        collection_info = MagicMock()
        collection_info.payload_schema = existing_schema or {}
        c.get_collection.return_value = collection_info
        return c

    def test_creates_all_indexed_fields_when_none_exist(self) -> None:
        client = self._mock_client(existing_schema={})
        ensure_indexes(client, "col", dry_run=False)
        assert client.create_payload_index.call_count == len(INDEXED_FIELDS)
        created_names = {
            c.kwargs["field_name"] for c in client.create_payload_index.call_args_list
        }
        assert created_names == {name for name, _ in INDEXED_FIELDS}

    def test_skips_fields_already_in_schema(self) -> None:
        # If "year" already exists, it should not be recreated.
        client = self._mock_client(existing_schema={"year": object()})
        ensure_indexes(client, "col", dry_run=False)
        created_names = {
            c.kwargs["field_name"] for c in client.create_payload_index.call_args_list
        }
        assert "year" not in created_names
        assert len(created_names) == len(INDEXED_FIELDS) - 1

    def test_dry_run_skips_writes(self) -> None:
        client = self._mock_client(existing_schema={})
        ensure_indexes(client, "col", dry_run=True)
        client.create_payload_index.assert_not_called()


# ---------------------------------------------------------------------------
# stream_pg_batches
# ---------------------------------------------------------------------------


class TestStreamPgBatches:
    def _mock_conn(self, pages: list[list[dict]]) -> MagicMock:
        """Build a mock connection that returns pages in sequence."""
        conn = MagicMock()
        cursor_cm = MagicMock()
        conn.cursor.return_value.__enter__ = MagicMock(return_value=cursor_cm)
        conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        cursor_cm.fetchall.side_effect = pages + [[]]
        return conn

    def _rows(self, bibcodes: list[str]) -> list[dict]:
        return [{"bibcode": bc} for bc in bibcodes]

    def test_yields_all_batches(self) -> None:
        page1 = self._rows(["a", "b", "c"])
        page2 = self._rows(["d", "e"])
        conn = self._mock_conn([page1, page2])
        result = list(stream_pg_batches(conn, batch=3, limit=None))
        assert result == [page1, page2]

    def test_limit_stops_early(self) -> None:
        page1 = self._rows(["a", "b", "c"])
        conn = self._mock_conn([page1])
        result = list(stream_pg_batches(conn, batch=10, limit=3))
        assert len(result) == 1
        assert len(result[0]) == 3

    def test_empty_first_page_yields_nothing(self) -> None:
        conn = self._mock_conn([[]])
        result = list(stream_pg_batches(conn, batch=10, limit=None))
        assert result == []

    def test_zero_limit_yields_nothing(self) -> None:
        conn = self._mock_conn([[]])
        result = list(stream_pg_batches(conn, batch=10, limit=0))
        assert result == []


# ---------------------------------------------------------------------------
# apply_batch
# ---------------------------------------------------------------------------


class TestApplyBatch:
    def _row(self, bibcode: str = "2020ApJ...900..100X") -> dict:
        return {
            "bibcode": bibcode,
            "year": 2020,
            "doctype": "article",
            "arxiv_class": ["astro-ph.HE"],
            "bibstem": ["ApJ"],
            "title": "T",
            "first_author": "S",
            "citation_count": 1,
            "is_retracted": False,
            "community_semantic_coarse": 1,
            "community_semantic_medium": 5,
            "pagerank": 0.001,
        }

    def test_calls_set_payload_for_each_row(self) -> None:
        client = MagicMock()
        rows = [self._row("2020A...1"), self._row("2021B...2")]
        apply_batch(client, "col", rows, dry_run=False)
        assert client.set_payload.call_count == 2

    def test_dry_run_skips_set_payload(self) -> None:
        client = MagicMock()
        apply_batch(client, "col", [self._row()], dry_run=True)
        client.set_payload.assert_not_called()

    def test_returns_row_count(self) -> None:
        client = MagicMock()
        rows = [self._row("a"), self._row("b"), self._row("c")]
        count = apply_batch(client, "col", rows, dry_run=False)
        assert count == 3

    def test_set_payload_uses_point_id_list(self) -> None:
        client = MagicMock()
        bibcode = "2020ApJ...900..100X"
        apply_batch(client, "col", [self._row(bibcode)], dry_run=False)
        call_kwargs = client.set_payload.call_args.kwargs
        points = call_kwargs["points"]
        assert points.points == [_bibcode_to_point_id(bibcode)]

    def test_wait_false_for_throughput(self) -> None:
        client = MagicMock()
        apply_batch(client, "col", [self._row()], dry_run=False)
        call_kwargs = client.set_payload.call_args.kwargs
        assert call_kwargs.get("wait") is False

    def test_bibcode_to_point_id_matches_full_load_scheme(self) -> None:
        import uuid
        bibcode = "2020ApJ...900..100X"
        expected = str(uuid.uuid5(uuid.NAMESPACE_URL, bibcode))
        assert _bibcode_to_point_id(bibcode) == expected
