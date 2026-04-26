"""Tests for ``scix.extract.chunk_pass.pipeline``.

Every test uses pure mocks — no live Postgres, no live Qdrant, no
HuggingFace model downloads — because the pipeline is a thin orchestrator
and the interesting properties are structural (resumability, idempotency,
dry-run semantics, log fields, deterministic ordering).
"""
from __future__ import annotations

import logging
from typing import Any
from unittest.mock import MagicMock

import pytest

qdrant_client = pytest.importorskip("qdrant_client")

from scix.extract.chunk_pass import (  # noqa: E402
    BatchStats,
    CHUNKS_COLLECTION,
    Chunk,
    INDUSEmbedder,
    chunk_point_id,
    ensure_collection,
    iter_chunks,
    normalize_heading,
    run,
)
from scix.extract.chunk_pass.pipeline import (  # noqa: E402
    PaperWithMeta,
    iter_paper_batches,
    process_batch,
)
from scix.extract.chunk_pass.uploader import _checkpoint_key  # noqa: E402


# ---------------------------------------------------------------------------
# Stub infrastructure
# ---------------------------------------------------------------------------


class StubCursor:
    """Mimics ``psycopg.Cursor`` for context-manager use.

    Wired by ``StubConn``: the conn keeps a ``script`` list of canned row
    sets; each ``execute`` pops the next entry. ``fetchone`` returns the
    first row of the most-recent ``execute`` call's result, ``fetchall``
    returns the full list.
    """

    def __init__(self, conn: "StubConn") -> None:
        self._conn = conn
        self._last_rows: list[tuple] = []

    def __enter__(self) -> "StubCursor":
        return self

    def __exit__(self, *exc: Any) -> bool:
        return False

    def execute(self, sql: str, params: tuple = ()) -> None:
        self._conn.executed.append((sql, tuple(params)))
        # Resolve next result. Either a callable (params -> rows) or a
        # static list of rows. Callable lets us implement is_batch_done
        # which inspects the lookup key.
        if self._conn.script:
            entry = self._conn.script.pop(0)
        else:
            entry = []
        if callable(entry):
            self._last_rows = list(entry(sql, params))
        else:
            self._last_rows = list(entry)

    def fetchall(self) -> list[tuple]:
        return list(self._last_rows)

    def fetchone(self) -> tuple | None:
        return self._last_rows[0] if self._last_rows else None


class StubConn:
    """Mimics ``psycopg.Connection`` for the chunk-pass pipeline.

    ``script`` is a queue of canned cursor results; each entry is either a
    list of row tuples (returned verbatim) or a callable ``(sql, params) ->
    rows`` (used for ``is_batch_done``-style lookups that depend on the key).
    ``commit_count`` tracks how many times the pipeline finalized a batch.
    """

    def __init__(self, script: list[Any] | None = None) -> None:
        self.script: list[Any] = list(script or [])
        self.executed: list[tuple[str, tuple]] = []
        self.commit_count = 0

    def cursor(self, *_: Any, **__: Any) -> StubCursor:
        return StubCursor(self)

    def commit(self) -> None:
        self.commit_count += 1


class StubTokenizer:
    """Whitespace tokenizer; honors ``encode_with_offsets`` strategy in chunker."""

    def __call__(self, *_: Any, **__: Any) -> dict[str, list]:
        # Force fall-through to encode_with_offsets path by missing
        # offset_mapping intentionally.
        raise TypeError("stub does not implement HF call signature")

    def encode_with_offsets(self, text: str) -> tuple[list[int], list[tuple[int, int]]]:
        offsets: list[tuple[int, int]] = []
        i = 0
        n = len(text)
        while i < n:
            while i < n and text[i].isspace():
                i += 1
            if i >= n:
                break
            start = i
            while i < n and not text[i].isspace():
                i += 1
            offsets.append((start, i))
        return list(range(len(offsets))), offsets

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:  # noqa: ARG002
        return list(range(len(text.split())))


class StubEmbedder:
    """Implements the slice of :class:`INDUSEmbedder` used by the pipeline."""

    def __init__(self) -> None:
        self._tokenizer = StubTokenizer()
        self.encode_calls: list[list[str]] = []

    @property
    def tokenizer(self) -> StubTokenizer:
        return self._tokenizer

    def encode_batch(self, texts: list[str]) -> list[list[float]]:
        self.encode_calls.append(list(texts))
        return [[float(i) for _ in range(768)] for i, _ in enumerate(texts)]


class StubQdrant:
    """Records every ``upsert`` call; used to assert idempotency + dry-run."""

    def __init__(self) -> None:
        self.upsert_calls: list[dict[str, Any]] = []

    def upsert(self, collection_name: str, points: list, wait: bool = False) -> None:
        self.upsert_calls.append(
            {"collection_name": collection_name, "points": list(points), "wait": wait}
        )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_sections(n_words: int = 200) -> list[dict]:
    """Build a single-section paper with ``n_words`` whitespace tokens.

    With the default chunker (window=512, stride=64), a 200-word section
    produces exactly one chunk.
    """
    return [
        {
            "heading": "Methods",
            "level": 1,
            "text": " ".join(f"word{i}" for i in range(n_words)),
            "offset": 0,
        }
    ]


def _paper_row(
    bibcode: str,
    *,
    n_words: int = 200,
    year: int | None = 2024,
    arxiv_class: list[str] | None = None,
    community_id_med: int | None = 7,
    doctype: str | None = "article",
) -> tuple:
    return (
        bibcode,
        _make_sections(n_words),
        year,
        arxiv_class if arxiv_class is not None else ["astro-ph.GA"],
        community_id_med,
        doctype,
    )


def _no_checkpoint(_sql: str, _params: tuple) -> list[tuple]:
    """Cursor result for ``is_batch_done`` lookups with no prior checkpoint."""
    return []


def _existing_checkpoint(_sql: str, _params: tuple) -> list[tuple]:
    """Cursor result for ``is_batch_done`` lookups already complete."""
    return [("complete",)]


# ---------------------------------------------------------------------------
# Public API re-export tests
# ---------------------------------------------------------------------------


class TestPackageReexports:
    def test_required_symbols_exposed(self) -> None:
        # Per AC 2: from scix.extract.chunk_pass import run, INDUSEmbedder, ...
        assert callable(run)
        assert isinstance(CHUNKS_COLLECTION, str)
        assert callable(ensure_collection)
        assert callable(normalize_heading)
        assert callable(chunk_point_id)
        assert callable(iter_chunks)
        # Class symbols
        assert Chunk is not None
        assert INDUSEmbedder is not None
        assert BatchStats is not None


# ---------------------------------------------------------------------------
# BatchStats shape
# ---------------------------------------------------------------------------


class TestBatchStats:
    def test_frozen_with_required_fields(self) -> None:
        stats = BatchStats(
            papers_seen=1,
            papers_with_chunks=1,
            chunks_emitted=2,
            chunks_uploaded=2,
            elapsed_chunk_s=0.1,
            elapsed_embed_s=0.2,
            elapsed_upload_s=0.3,
        )
        assert stats.papers_seen == 1
        assert stats.papers_with_chunks == 1
        assert stats.chunks_emitted == 2
        assert stats.chunks_uploaded == 2
        assert stats.elapsed_chunk_s == 0.1
        assert stats.elapsed_embed_s == 0.2
        assert stats.elapsed_upload_s == 0.3
        # Frozen → mutation raises.
        with pytest.raises(Exception):  # noqa: BLE001 — dataclasses.FrozenInstanceError
            stats.papers_seen = 99  # type: ignore[misc]


# ---------------------------------------------------------------------------
# iter_paper_batches
# ---------------------------------------------------------------------------


class TestIterPaperBatches:
    def test_empty_source_yields_zero_batches(self) -> None:
        # AC 13: empty source -> 0 batches.
        conn = StubConn(script=[[]])
        out = list(iter_paper_batches(conn, batch_size=5))
        assert out == []

    def test_max_papers_one_yields_one_paper(self) -> None:
        # AC 13: max_papers=1 -> exactly 1 paper processed.
        conn = StubConn(script=[[_paper_row("2024A.....1A")]])
        out = list(iter_paper_batches(conn, batch_size=5, max_papers=1))
        assert len(out) == 1
        assert len(out[0]) == 1
        assert out[0][0].bibcode == "2024A.....1A"

    def test_since_bibcode_passed_as_watermark(self) -> None:
        # AC 13: since_bibcode -> only papers strictly greater.
        conn = StubConn(script=[[_paper_row("2024Z.....9Z")]])
        out = list(iter_paper_batches(conn, batch_size=5, since_bibcode="2024A.....1A", max_papers=1))
        assert len(out) == 1
        # The first executed query used the watermark in its first param.
        sql, params = conn.executed[0]
        assert params[0] == "2024A.....1A"

    def test_paginates_until_empty(self) -> None:
        conn = StubConn(
            script=[
                [_paper_row("2024A.....1A"), _paper_row("2024A.....2A")],
                [_paper_row("2024A.....3A")],
                [],
            ]
        )
        batches = list(iter_paper_batches(conn, batch_size=2))
        assert [p.bibcode for batch in batches for p in batch] == [
            "2024A.....1A",
            "2024A.....2A",
            "2024A.....3A",
        ]
        # Second iteration must use the last bibcode of batch 1 as watermark.
        assert conn.executed[1][1][0] == "2024A.....2A"

    def test_missing_metadata_yields_none(self) -> None:
        # AC 5: missing metadata fields just become None; pipeline never raises.
        row = (
            "2024A.....1A",
            _make_sections(),
            None,  # year
            None,  # arxiv_class
            None,  # community_id_med
            None,  # doctype
        )
        conn = StubConn(script=[[row]])
        batches = list(iter_paper_batches(conn, batch_size=5, max_papers=1))
        paper = batches[0][0]
        assert paper.year is None
        assert paper.arxiv_class is None
        assert paper.community_id_med is None
        assert paper.doctype is None

    def test_arxiv_class_first_element_promoted(self) -> None:
        row = (
            "2024A.....1A",
            _make_sections(),
            2024,
            ["astro-ph.GA", "astro-ph.SR"],
            7,
            "article",
        )
        conn = StubConn(script=[[row]])
        paper = next(iter(iter_paper_batches(conn, batch_size=1, max_papers=1)))[0]
        assert paper.arxiv_class == "astro-ph.GA"

    def test_arxiv_class_empty_array_is_none(self) -> None:
        row = ("2024A.....1A", _make_sections(), 2024, [], 7, "article")
        conn = StubConn(script=[[row]])
        paper = next(iter(iter_paper_batches(conn, batch_size=1, max_papers=1)))[0]
        assert paper.arxiv_class is None

    def test_invalid_batch_size_raises(self) -> None:
        conn = StubConn(script=[])
        with pytest.raises(ValueError):
            list(iter_paper_batches(conn, batch_size=0))


# ---------------------------------------------------------------------------
# run() — full pipeline tests
# ---------------------------------------------------------------------------


class TestRun:
    def test_processes_and_uploads(self, caplog: pytest.LogCaptureFixture) -> None:
        # AC 6, 10, 11 — happy path.
        caplog.set_level(logging.INFO, logger="scix.extract.chunk_pass.pipeline")
        conn = StubConn(
            script=[
                [_paper_row("2024A.....1A")],  # iter_paper_batches batch 1
                _no_checkpoint,                # is_batch_done lookup
                [],                            # iter_paper_batches batch 2 (drain)
            ]
        )
        embedder = StubEmbedder()
        client = StubQdrant()

        totals = run(conn, embedder, client, batch_size=5, max_papers=1)

        assert totals.papers_seen == 1
        assert totals.papers_with_chunks == 1
        assert totals.chunks_emitted >= 1
        assert totals.chunks_uploaded == totals.chunks_emitted
        # Embedder + qdrant called.
        assert embedder.encode_calls, "embedder should have been invoked"
        assert client.upsert_calls, "qdrant.upsert should have been invoked"
        # AC 11: commit per batch (one batch processed -> one commit).
        assert conn.commit_count == 1
        # AC 10: log fields present.
        log_text = "\n".join(r.getMessage() for r in caplog.records)
        assert "papers=" in log_text
        assert "chunks=" in log_text
        assert "elapsed_chunk_s=" in log_text
        assert "elapsed_embed_s=" in log_text
        assert "elapsed_upload_s=" in log_text

    def test_dry_run_skips_upload_and_checkpoint(self) -> None:
        # AC 7: dry_run=True -> no qdrant upsert, no checkpoint, no commit.
        conn = StubConn(
            script=[
                [_paper_row("2024A.....1A")],
                _no_checkpoint,
                [],
            ]
        )
        embedder = StubEmbedder()
        client = StubQdrant()

        totals = run(conn, embedder, client, batch_size=5, max_papers=1, dry_run=True)

        assert totals.chunks_emitted >= 1  # still emits
        assert totals.chunks_uploaded == 0  # no upload
        assert client.upsert_calls == []
        # No checkpoint INSERT, no commit (record_checkpoint never called).
        assert conn.commit_count == 0
        # Embedder still ran (we count chunks).
        assert embedder.encode_calls, "embedder should still encode in dry-run"

    def test_resumability_skips_completed_batch(self) -> None:
        # AC 8: pre-checkpointed batch -> embedder + qdrant NOT called.
        conn = StubConn(
            script=[
                [_paper_row("2024A.....1A")],   # iter_paper_batches batch 1
                _existing_checkpoint,            # is_batch_done -> 'complete'
                [],                              # iter_paper_batches drain
            ]
        )
        embedder = StubEmbedder()
        client = StubQdrant()

        totals = run(conn, embedder, client, batch_size=5, max_papers=1)

        assert totals.papers_seen == 0
        assert totals.chunks_emitted == 0
        assert totals.chunks_uploaded == 0
        assert embedder.encode_calls == []
        assert client.upsert_calls == []
        assert conn.commit_count == 0

    def test_idempotency_second_run_zero(self) -> None:
        # AC 9: re-running with all batches checkpointed yields 0 new uploads.
        conn = StubConn(
            script=[
                [_paper_row("2024A.....1A")],   # batch 1
                _existing_checkpoint,            # batch 1 already done
                [_paper_row("2024A.....2A")],   # batch 2
                _existing_checkpoint,            # batch 2 already done
                [],                              # drain
            ]
        )
        embedder = StubEmbedder()
        client = StubQdrant()

        totals = run(conn, embedder, client, batch_size=1, max_papers=2)

        assert totals.chunks_uploaded == 0
        assert client.upsert_calls == []

    def test_first_bibcode_used_as_checkpoint_key(self) -> None:
        captured: dict[str, str] = {}

        def capture_done(sql: str, params: tuple) -> list[tuple]:
            captured["key"] = params[0]
            return []  # not done

        conn = StubConn(
            script=[
                [_paper_row("2024A.....1A"), _paper_row("2024A.....2A")],
                capture_done,
                [],
            ]
        )
        embedder = StubEmbedder()
        client = StubQdrant()

        run(conn, embedder, client, batch_size=2, max_papers=2)

        assert captured["key"] == _checkpoint_key("2024A.....1A")

    def test_payload_assembly_uses_paper_metadata(self) -> None:
        # Per-chunk payload must carry the joined paper metadata, not None.
        conn = StubConn(
            script=[
                [_paper_row("2024A.....1A", year=2023, community_id_med=99, doctype="proceedings")],
                _no_checkpoint,
                [],
            ]
        )
        embedder = StubEmbedder()
        client = StubQdrant()

        run(conn, embedder, client, batch_size=5, max_papers=1)

        assert client.upsert_calls, "expected one upsert call"
        points = client.upsert_calls[0]["points"]
        assert points, "expected at least one point"
        payload = points[0].payload
        assert payload["bibcode"] == "2024A.....1A"
        assert payload["year"] == 2023
        assert payload["arxiv_class"] == "astro-ph.GA"
        assert payload["community_id_med"] == 99
        assert payload["doctype"] == "proceedings"
        assert payload["section_heading_norm"] == "methods"

    def test_uses_default_collection_name(self) -> None:
        conn = StubConn(
            script=[
                [_paper_row("2024A.....1A")],
                _no_checkpoint,
                [],
            ]
        )
        embedder = StubEmbedder()
        client = StubQdrant()
        run(conn, embedder, client, batch_size=5, max_papers=1)
        assert client.upsert_calls[0]["collection_name"] == CHUNKS_COLLECTION

    def test_no_papers_no_commits_no_calls(self) -> None:
        conn = StubConn(script=[[]])
        embedder = StubEmbedder()
        client = StubQdrant()
        totals = run(conn, embedder, client, batch_size=5)
        assert totals == BatchStats()
        assert embedder.encode_calls == []
        assert client.upsert_calls == []
        assert conn.commit_count == 0


# ---------------------------------------------------------------------------
# process_batch (direct)
# ---------------------------------------------------------------------------


class TestProcessBatch:
    def test_empty_batch_returns_zero(self) -> None:
        embedder = StubEmbedder()
        stats = process_batch(StubConn(), embedder, StubQdrant(), [])
        assert stats == BatchStats()
        assert embedder.encode_calls == []

    def test_paper_too_short_yields_no_chunks(self) -> None:
        # 5 words is below stride=64, so chunker returns nothing.
        paper = PaperWithMeta(
            bibcode="2024A.....1A",
            sections=[{"heading": "Methods", "level": 1, "text": "a b c d e", "offset": 0}],
            year=2024,
            arxiv_class="astro-ph.GA",
            community_id_med=1,
            doctype="article",
        )
        embedder = StubEmbedder()
        client = StubQdrant()
        stats = process_batch(StubConn(), embedder, client, [paper])
        assert stats.papers_seen == 1
        assert stats.papers_with_chunks == 0
        assert stats.chunks_emitted == 0
        assert stats.chunks_uploaded == 0
        assert embedder.encode_calls == []
        assert client.upsert_calls == []
