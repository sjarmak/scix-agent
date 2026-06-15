"""Tests for scripts/backfill_qdrant_filter_fields.py."""

from __future__ import annotations

from unittest.mock import MagicMock

from scripts.backfill_qdrant_filter_fields import (
    INDEXED_FIELDS,
    _bibcode_to_point_id,
    _build_payload,
    apply_batch,
    ensure_indexes,
    stream_pg_batches,
    verify_sample,
)


def _make_row(bibcode: str = "2020ApJ...900..100X", **overrides) -> dict:
    return {
        "bibcode": bibcode,
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
        **overrides,
    }


# ---------------------------------------------------------------------------
# _build_payload
# ---------------------------------------------------------------------------


class TestBuildPayload:
    def test_full_row_produces_expected_keys(self) -> None:
        p = _build_payload(_make_row())
        assert set(p) == {
            "year",
            "doctype",
            "arxiv_class",
            "bibstem",
            "community_semantic_coarse",
            "community_semantic_medium",
            "title",
            "first_author",
            "citation_count",
            "pagerank",
        }
        # is_retracted=False → absent (absence semantics per ADR-008)
        assert "is_retracted" not in p

    def test_is_retracted_true_is_included(self) -> None:
        row = _make_row()
        row["is_retracted"] = True
        p = _build_payload(row)
        assert p["is_retracted"] is True

    def test_null_fields_omitted(self) -> None:
        row = _make_row()
        row["year"] = None
        row["doctype"] = None
        row["arxiv_class"] = None
        row["bibstem"] = None
        row["community_semantic_coarse"] = None
        row["community_semantic_medium"] = None
        row["pagerank"] = None
        p = _build_payload(row)
        for key in (
            "year",
            "doctype",
            "arxiv_class",
            "bibstem",
            "community_semantic_coarse",
            "community_semantic_medium",
            "pagerank",
        ):
            assert key not in p, f"{key} should be absent when source is NULL"

    def test_empty_list_fields_omitted(self) -> None:
        row = _make_row()
        row["arxiv_class"] = []
        row["bibstem"] = []
        p = _build_payload(row)
        assert "arxiv_class" not in p
        assert "bibstem" not in p

    def test_year_coerced_to_int(self) -> None:
        row = _make_row()
        row["year"] = "2019"  # coerce defensively — year is always int in practice
        p = _build_payload(row)
        assert isinstance(p["year"], int)

    def test_citation_count_coerced_to_int(self) -> None:
        row = _make_row()
        row["citation_count"] = 7
        p = _build_payload(row)
        assert isinstance(p["citation_count"], int)

    def test_arxiv_class_stored_as_list(self) -> None:
        row = _make_row()
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
        created_names = {c.kwargs["field_name"] for c in client.create_payload_index.call_args_list}
        assert created_names == {name for name, _ in INDEXED_FIELDS}

    def test_skips_fields_already_in_schema(self) -> None:
        # If "year" already exists, it should not be recreated.
        client = self._mock_client(existing_schema={"year": object()})
        ensure_indexes(client, "col", dry_run=False)
        created_names = {c.kwargs["field_name"] for c in client.create_payload_index.call_args_list}
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
        conn = MagicMock()
        cursor_cm = MagicMock()
        conn.cursor.return_value.__enter__ = MagicMock(return_value=cursor_cm)
        conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        cursor_cm.fetchall.side_effect = [page1, []]
        result = list(stream_pg_batches(conn, batch=10, limit=3))
        assert len(result) == 1
        assert len(result[0]) == 3
        # fetch=0 early-break must fire before the second execute, not after.
        assert cursor_cm.execute.call_count == 1

    def test_cursor_advances_to_last_bibcode(self) -> None:
        page1 = self._rows(["a", "b", "c"])
        page2 = self._rows(["d"])
        cursor_cm = MagicMock()
        conn = MagicMock()
        conn.cursor.return_value.__enter__ = MagicMock(return_value=cursor_cm)
        conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        cursor_cm.fetchall.side_effect = [page1, page2, []]
        list(stream_pg_batches(conn, batch=3, limit=None))
        # First call: cursor = "" (start); second call: cursor = "c" (last of page1).
        first_cursor = cursor_cm.execute.call_args_list[0].args[1][0]
        second_cursor = cursor_cm.execute.call_args_list[1].args[1][0]
        assert first_cursor == ""
        assert second_cursor == "c"

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
    def test_calls_set_payload_for_each_row(self) -> None:
        client = MagicMock()
        rows = [_make_row("2020A...1"), _make_row("2021B...2")]
        apply_batch(client, "col", rows, dry_run=False)
        assert client.set_payload.call_count == 2

    def test_dry_run_skips_set_payload(self) -> None:
        client = MagicMock()
        apply_batch(client, "col", [_make_row()], dry_run=True)
        client.set_payload.assert_not_called()

    def test_returns_row_count(self) -> None:
        client = MagicMock()
        rows = [_make_row("a"), _make_row("b"), _make_row("c")]
        count = apply_batch(client, "col", rows, dry_run=False)
        assert count == 3

    def test_returns_written_count_excludes_empty_payload_rows(self) -> None:
        client = MagicMock()
        all_null = _make_row(
            "2020A...null",
            year=None,
            doctype=None,
            arxiv_class=[],
            bibstem=[],
            community_semantic_coarse=None,
            community_semantic_medium=None,
            pagerank=None,
            title=None,
            first_author=None,
            citation_count=None,
            is_retracted=False,
        )
        rows = [all_null, _make_row("2020B...1")]
        count = apply_batch(client, "col", rows, dry_run=False)
        assert count == 1
        assert client.set_payload.call_count == 1

    def test_dry_run_count_excludes_empty_payload_rows(self) -> None:
        # Regression: dry-run must mirror the live skip (empty payload → no
        # set_payload call), so its count matches actual write throughput
        # instead of inflating it with rows the live path drops.
        client = MagicMock()
        all_null = _make_row(
            "2020A...null",
            year=None,
            doctype=None,
            arxiv_class=[],
            bibstem=[],
            community_semantic_coarse=None,
            community_semantic_medium=None,
            pagerank=None,
            title=None,
            first_author=None,
            citation_count=None,
            is_retracted=False,
        )
        rows = [all_null, _make_row("2020B...1")]
        dry_count = apply_batch(client, "col", rows, dry_run=True)
        live_count = apply_batch(client, "col", rows, dry_run=False)
        assert dry_count == 1
        assert dry_count == live_count

    def test_set_payload_uses_point_id_list(self) -> None:
        client = MagicMock()
        bibcode = "2020ApJ...900..100X"
        apply_batch(client, "col", [_make_row(bibcode)], dry_run=False)
        call_kwargs = client.set_payload.call_args.kwargs
        points = call_kwargs["points"]
        assert points.points == [_bibcode_to_point_id(bibcode)]

    def test_wait_false_for_throughput(self) -> None:
        client = MagicMock()
        apply_batch(client, "col", [_make_row()], dry_run=False)
        call_kwargs = client.set_payload.call_args.kwargs
        assert call_kwargs.get("wait") is False

    def test_bibcode_to_point_id_matches_full_load_scheme(self) -> None:
        import uuid

        bibcode = "2020ApJ...900..100X"
        expected = str(uuid.uuid5(uuid.NAMESPACE_URL, bibcode))
        assert _bibcode_to_point_id(bibcode) == expected

    def test_samples_accumulate_up_to_cap(self) -> None:
        client = MagicMock()
        rows = [_make_row(f"2020A...{i}") for i in range(5)]
        samples: list[tuple[str, dict]] = []
        apply_batch(client, "col", rows, dry_run=False, samples=samples, sample_cap=3)
        assert len(samples) == 3
        assert samples[0][0] == "2020A...0"
        assert samples[0][1] == _build_payload(rows[0])


# ---------------------------------------------------------------------------
# verify_sample
# ---------------------------------------------------------------------------


class TestVerifySample:
    def _point(self, bibcode: str, payload: dict) -> MagicMock:
        """Build a mock PointStruct with .id set to the UUID5 for bibcode."""
        p = MagicMock()
        p.id = _bibcode_to_point_id(bibcode)
        p.payload = payload
        return p

    def test_passes_when_payload_applied(self) -> None:
        samples = [("2020A...1", {"year": 2020, "doctype": "article"})]
        client = MagicMock()
        client.retrieve.return_value = [
            self._point("2020A...1", {"year": 2020, "doctype": "article"})
        ]
        assert verify_sample(client, "col", samples, timeout_s=0.1, poll_interval_s=0.01)

    def test_fails_when_payload_never_applied(self) -> None:
        samples = [("2020A...1", {"year": 2020})]
        client = MagicMock()
        client.retrieve.return_value = [self._point("2020A...1", {})]  # no fields written
        assert not verify_sample(client, "col", samples, timeout_s=0.05, poll_interval_s=0.01)

    def test_fails_when_point_missing(self) -> None:
        samples = [("2020A...1", {"year": 2020})]
        client = MagicMock()
        client.retrieve.return_value = []
        assert not verify_sample(client, "col", samples, timeout_s=0.05, poll_interval_s=0.01)

    def test_polls_until_applied(self) -> None:
        samples = [("2020A...1", {"year": 2020})]
        client = MagicMock()
        client.retrieve.side_effect = [
            [self._point("2020A...1", {})],
            [self._point("2020A...1", {"year": 2020})],
        ]
        assert verify_sample(client, "col", samples, timeout_s=5.0, poll_interval_s=0.01)
        assert client.retrieve.call_count == 2

    def test_fails_on_value_mismatch(self) -> None:
        samples = [("2020A...1", {"year": 2020})]
        client = MagicMock()
        client.retrieve.return_value = [self._point("2020A...1", {"year": 1999})]
        assert not verify_sample(client, "col", samples, timeout_s=0.05, poll_interval_s=0.01)

    def test_is_retracted_absence_semantics_via_verify(self) -> None:
        # is_retracted=True must be written and verified; False is omitted (absent = not retracted).
        samples = [("2020A...2", {"is_retracted": True})]
        client = MagicMock()
        client.retrieve.return_value = [self._point("2020A...2", {"is_retracted": True})]
        assert verify_sample(client, "col", samples, timeout_s=0.1, poll_interval_s=0.01)

    def test_passes_without_bibcode_in_payload(self) -> None:
        # verify_sample must not require bibcode in the point's payload — it uses p.id.
        samples = [("2020A...3", {"year": 2022})]
        client = MagicMock()
        # Point has no bibcode key in payload — only the written fields.
        client.retrieve.return_value = [self._point("2020A...3", {"year": 2022})]
        assert verify_sample(client, "col", samples, timeout_s=0.1, poll_interval_s=0.01)
