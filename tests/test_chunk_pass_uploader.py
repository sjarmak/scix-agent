"""Tests for ``scix.extract.chunk_pass.uploader``.

The uploader is the write-side bridge: it takes ``Chunk`` objects + dense
vectors + paper metadata, builds Qdrant payload dicts, upserts the points,
and writes a checkpoint to ``ingest_log``. Every test here uses pure mocks —
no live Qdrant, no Postgres — because the module is a thin orchestrator and
the interesting properties are structural (exact payload keys, deterministic
ids, checkpoint round-trip).
"""
from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

qdrant_client = pytest.importorskip("qdrant_client")
from qdrant_client.http import models as qm  # noqa: E402

from scix.extract.chunk_pass.chunker import Chunk  # noqa: E402
from scix.extract.chunk_pass.collection import (  # noqa: E402
    CHUNKS_COLLECTION,
    chunk_point_id,
)
from scix.extract.chunk_pass.uploader import (  # noqa: E402
    PAYLOAD_KEYS,
    _checkpoint_key,
    assemble_payload,
    is_batch_done,
    record_checkpoint,
    upsert_chunks,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _make_chunk(
    *,
    bibcode: str = "2024ApJ...999..001A",
    section_idx: int = 0,
    chunk_id: int = 0,
    char_offset: int = 0,
    n_tokens: int = 256,
) -> Chunk:
    return Chunk(
        bibcode=bibcode,
        section_idx=section_idx,
        section_heading="Methods",
        section_level=1,
        char_offset=char_offset,
        chunk_id=chunk_id,
        n_tokens=n_tokens,
        text="lorem ipsum dolor sit amet",
    )


def _make_stub_conn() -> tuple[MagicMock, MagicMock]:
    """Return ``(conn, cursor)`` mocks wired so ``with conn.cursor() as cur``
    yields the cursor. Callers configure ``cursor.fetchone.return_value``."""
    conn = MagicMock(name="conn")
    cursor = MagicMock(name="cursor")
    cm = MagicMock(name="cursor_cm")
    cm.__enter__.return_value = cursor
    cm.__exit__.return_value = False
    conn.cursor.return_value = cm
    return conn, cursor


# ---------------------------------------------------------------------------
# assemble_payload
# ---------------------------------------------------------------------------


class TestAssemblePayload:
    def test_returns_exact_key_set(self) -> None:
        """The payload dict must contain EXACTLY the PRD-mandated keys —
        no extras (would bloat Qdrant storage), no missing (would break
        downstream filters / facet code)."""
        chunk = _make_chunk()
        payload = assemble_payload(
            chunk,
            year=2024,
            arxiv_class="astro-ph.GA",
            community_id_med=42,
            doctype="article",
            section_heading_norm="methods",
        )

        expected = {
            "bibcode",
            "year",
            "arxiv_class",
            "community_id_med",
            "section_heading_norm",
            "doctype",
            "section_idx",
            "char_offset",
            "chunk_id",
            "n_tokens",
        }
        assert set(payload.keys()) == expected
        assert set(PAYLOAD_KEYS) == expected

    def test_values_pass_through(self) -> None:
        chunk = _make_chunk(
            bibcode="2024ApJ...999..001A",
            section_idx=3,
            chunk_id=7,
            char_offset=2048,
            n_tokens=384,
        )
        payload = assemble_payload(
            chunk,
            year=2024,
            arxiv_class="astro-ph.GA",
            community_id_med=42,
            doctype="article",
            section_heading_norm="methods",
        )

        assert payload == {
            "bibcode": "2024ApJ...999..001A",
            "year": 2024,
            "arxiv_class": "astro-ph.GA",
            "community_id_med": 42,
            "section_heading_norm": "methods",
            "doctype": "article",
            "section_idx": 3,
            "char_offset": 2048,
            "chunk_id": 7,
            "n_tokens": 384,
        }

    def test_nullable_fields_kept_as_none(self) -> None:
        """``year``, ``arxiv_class``, ``community_id_med``, ``doctype`` may all
        be NULL in ``papers`` — the payload should preserve None rather than
        coerce to a sentinel that would corrupt filtered search."""
        chunk = _make_chunk()
        payload = assemble_payload(
            chunk,
            year=None,
            arxiv_class=None,
            community_id_med=None,
            doctype=None,
            section_heading_norm="introduction",
        )
        assert payload["year"] is None
        assert payload["arxiv_class"] is None
        assert payload["community_id_med"] is None
        assert payload["doctype"] is None
        assert payload["section_heading_norm"] == "introduction"


# ---------------------------------------------------------------------------
# upsert_chunks
# ---------------------------------------------------------------------------


class TestUpsertChunks:
    def test_calls_client_upsert_with_pointstructs(self) -> None:
        client = MagicMock()
        chunks = [_make_chunk(chunk_id=0), _make_chunk(chunk_id=1)]
        vectors = [[0.1] * 768, [0.2] * 768]
        payloads = [
            assemble_payload(
                c,
                year=2024,
                arxiv_class="astro-ph.GA",
                community_id_med=1,
                doctype="article",
                section_heading_norm="methods",
            )
            for c in chunks
        ]

        n = upsert_chunks(
            client,
            CHUNKS_COLLECTION,
            chunks,
            vectors,
            payloads,
            parser_version="v1.0.0",
        )

        assert n == 2
        assert client.upsert.call_count == 1
        kwargs = client.upsert.call_args.kwargs
        assert kwargs["collection_name"] == CHUNKS_COLLECTION
        assert kwargs["wait"] is False

        points = kwargs["points"]
        assert len(points) == 2
        for point, chunk, vector, payload in zip(points, chunks, vectors, payloads):
            assert isinstance(point, qm.PointStruct)
            assert point.id == chunk_point_id(chunk.bibcode, "v1.0.0", chunk.chunk_id)
            assert list(point.vector) == vector
            assert dict(point.payload) == payload

    def test_returns_zero_on_empty_input(self) -> None:
        client = MagicMock()
        n = upsert_chunks(client, CHUNKS_COLLECTION, [], [], [], parser_version="v1.0.0")
        assert n == 0
        client.upsert.assert_not_called()

    def test_length_mismatch_raises(self) -> None:
        client = MagicMock()
        chunks = [_make_chunk()]
        with pytest.raises(ValueError, match="same length"):
            upsert_chunks(
                client,
                CHUNKS_COLLECTION,
                chunks,
                vectors=[[0.0] * 768, [0.0] * 768],  # mismatched
                payloads=[{}],
                parser_version="v1.0.0",
            )

    def test_idempotent_point_ids(self) -> None:
        """Re-uploading the same chunk + vector must produce the same point
        id — that's how idempotency is guaranteed (Qdrant overwrites on id
        collision)."""
        client = MagicMock()
        chunk = _make_chunk(chunk_id=5)
        vector = [0.42] * 768
        payload = assemble_payload(
            chunk,
            year=2024,
            arxiv_class="astro-ph.GA",
            community_id_med=1,
            doctype="article",
            section_heading_norm="methods",
        )

        upsert_chunks(client, CHUNKS_COLLECTION, [chunk], [vector], [payload], parser_version="v1")
        first_id = client.upsert.call_args.kwargs["points"][0].id

        client.reset_mock()
        upsert_chunks(client, CHUNKS_COLLECTION, [chunk], [vector], [payload], parser_version="v1")
        second_id = client.upsert.call_args.kwargs["points"][0].id

        assert first_id == second_id
        # And matches the canonical helper directly.
        assert first_id == chunk_point_id(chunk.bibcode, "v1", chunk.chunk_id)

    def test_parser_version_changes_point_id(self) -> None:
        """Different parser versions must yield different ids — bumping the
        parser is how we force reindex without manual deletion."""
        client = MagicMock()
        chunk = _make_chunk()
        vector = [0.0] * 768
        payload: dict[str, Any] = {}

        upsert_chunks(client, CHUNKS_COLLECTION, [chunk], [vector], [payload], parser_version="v1")
        id_v1 = client.upsert.call_args.kwargs["points"][0].id

        client.reset_mock()
        upsert_chunks(client, CHUNKS_COLLECTION, [chunk], [vector], [payload], parser_version="v2")
        id_v2 = client.upsert.call_args.kwargs["points"][0].id

        assert id_v1 != id_v2


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------


class TestCheckpointKey:
    def test_format(self) -> None:
        assert _checkpoint_key("2024ApJ...999..001A") == "chunk_pass:2024ApJ...999..001A"

    def test_distinct_per_bibcode(self) -> None:
        a = _checkpoint_key("2024ApJ...001")
        b = _checkpoint_key("2024ApJ...002")
        assert a != b


class TestIsBatchDone:
    def test_complete_returns_true(self) -> None:
        conn, cursor = _make_stub_conn()
        cursor.fetchone.return_value = ("complete",)
        assert is_batch_done(conn, "chunk_pass:bib1") is True
        cursor.execute.assert_called_once_with(
            "SELECT status FROM ingest_log WHERE filename = %s",
            ("chunk_pass:bib1",),
        )

    def test_missing_returns_false(self) -> None:
        conn, cursor = _make_stub_conn()
        cursor.fetchone.return_value = None
        assert is_batch_done(conn, "chunk_pass:bib1") is False

    def test_in_progress_returns_false(self) -> None:
        conn, cursor = _make_stub_conn()
        cursor.fetchone.return_value = ("in_progress",)
        assert is_batch_done(conn, "chunk_pass:bib1") is False


class TestRecordCheckpoint:
    def test_executes_upsert_with_zero_edges_and_status_complete(self) -> None:
        conn, cursor = _make_stub_conn()
        record_checkpoint(conn, "chunk_pass:bib1", records_loaded=42)

        assert cursor.execute.call_count == 1
        sql, params = cursor.execute.call_args.args
        assert "INSERT INTO ingest_log" in sql
        assert "ON CONFLICT (filename) DO UPDATE" in sql
        # (key, records_loaded, edges_loaded=0, status='complete')
        assert params == ("chunk_pass:bib1", 42, 0, "complete")

    def test_status_override(self) -> None:
        conn, cursor = _make_stub_conn()
        record_checkpoint(conn, "chunk_pass:bib1", records_loaded=0, status="failed")
        _, params = cursor.execute.call_args.args
        assert params[3] == "failed"


class TestCheckpointRoundTrip:
    """``record_checkpoint`` followed by ``is_batch_done`` for the same key
    must return True. Modeled with a single in-memory dict standing in for
    the ``ingest_log`` table — that's the smallest stub that exercises both
    code paths together."""

    def _stub_conn_with_table(self) -> tuple[MagicMock, dict[str, tuple]]:
        table: dict[str, tuple] = {}
        conn = MagicMock(name="conn")
        cursor = MagicMock(name="cursor")

        def execute(sql: str, params: tuple) -> None:
            sql_norm = " ".join(sql.split())
            if sql_norm.startswith("INSERT INTO ingest_log"):
                key, records_loaded, edges_loaded, status = params
                table[key] = (records_loaded, edges_loaded, status)
            elif sql_norm.startswith("SELECT status FROM ingest_log"):
                (key,) = params
                row = table.get(key)
                cursor._next_fetch = (row[2],) if row else None
            else:  # pragma: no cover — defensive
                raise AssertionError(f"unexpected SQL: {sql_norm}")

        def fetchone() -> Any:
            return cursor._next_fetch

        cursor.execute.side_effect = execute
        cursor.fetchone.side_effect = fetchone
        cursor._next_fetch = None

        cm = MagicMock()
        cm.__enter__.return_value = cursor
        cm.__exit__.return_value = False
        conn.cursor.return_value = cm
        return conn, table

    def test_record_then_is_batch_done_true(self) -> None:
        conn, table = self._stub_conn_with_table()
        key = _checkpoint_key("2024ApJ...999..001A")

        # Pre-condition: not done.
        assert is_batch_done(conn, key) is False

        record_checkpoint(conn, key, records_loaded=128)
        assert table[key] == (128, 0, "complete")

        assert is_batch_done(conn, key) is True

    def test_record_failed_then_is_batch_done_false(self) -> None:
        conn, _ = self._stub_conn_with_table()
        key = _checkpoint_key("2024ApJ...999..001A")
        record_checkpoint(conn, key, records_loaded=0, status="failed")
        assert is_batch_done(conn, key) is False
