"""End-to-end ingestion tests using real data files."""

from __future__ import annotations

import gzip
import json
import os
from pathlib import Path

import psycopg
import pytest

from scix.ingest import IngestPipeline, discover_files, open_jsonl

DSN = os.environ.get("SCIX_DSN", "dbname=scix")
DATA_DIR = Path("ads_metadata_by_year_picard")
SMALL_FILE = DATA_DIR / "ads_metadata_2026_full.jsonl.gz"


@pytest.fixture()
def conn():
    with psycopg.connect(DSN) as c:
        yield c


def _wipe_tables(conn) -> None:
    """Delete all rows from tables that depend on papers (in FK order), then papers."""
    with conn.cursor() as cur:
        cur.execute("DELETE FROM citation_edges")
        cur.execute("DELETE FROM paper_embeddings")
        cur.execute("DELETE FROM extractions")
        cur.execute("DELETE FROM papers")
        cur.execute("DELETE FROM ingest_log")
    conn.commit()


@pytest.fixture()
def clean_db(conn):
    """Clean papers, edges, and ingest_log before/after test."""
    _wipe_tables(conn)
    yield
    _wipe_tables(conn)


class TestOpenJsonl:
    def test_open_gzip(self) -> None:
        with open_jsonl(SMALL_FILE) as f:
            line = f.readline()
            rec = json.loads(line)
            assert "bibcode" in rec

    def test_open_plain_jsonl(self, tmp_path: Path) -> None:
        p = tmp_path / "test.jsonl"
        p.write_text('{"bibcode": "test"}\n')
        with open_jsonl(p) as f:
            assert json.loads(f.readline())["bibcode"] == "test"

    def test_open_xz(self, tmp_path: Path) -> None:
        import lzma

        p = tmp_path / "test.jsonl.xz"
        with lzma.open(p, "wt") as f:
            f.write('{"bibcode": "test_xz"}\n')
        with open_jsonl(p) as f:
            assert json.loads(f.readline())["bibcode"] == "test_xz"

    def test_unknown_format_raises(self, tmp_path: Path) -> None:
        p = tmp_path / "test.csv"
        p.write_text("data")
        with pytest.raises(ValueError, match="Unknown file format"):
            open_jsonl(p)


class TestDiscoverFiles:
    def test_finds_real_files(self) -> None:
        files = discover_files(DATA_DIR)
        assert len(files) >= 1
        assert any("2026" in f.name for f in files)

    def test_empty_dir(self, tmp_path: Path) -> None:
        assert discover_files(tmp_path) == []


class TestIngestPipelineE2E:
    @pytest.mark.skipif(not SMALL_FILE.exists(), reason="2026 data file not found")
    def test_ingest_2026_file(self, conn, clean_db) -> None:
        """Ingest the real 2026 file (21 records) end-to-end."""
        pipeline = IngestPipeline(data_dir=DATA_DIR, batch_size=100)
        pipeline.run(drop_indexes=False, single_file=SMALL_FILE)

        with conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM papers")
            paper_count = cur.fetchone()[0]
            assert paper_count == 21

            # Verify a known record
            cur.execute("SELECT title, year FROM papers WHERE bibcode = '2026KIsMS.336..114S'")
            row = cur.fetchone()
            assert row is not None
            assert "copper monosulfide" in row[0].lower()
            assert row[1] == 2026

            # Check ingest log
            cur.execute("SELECT status, records_loaded FROM ingest_log WHERE filename = %s",
                        (SMALL_FILE.name,))
            log_row = cur.fetchone()
            assert log_row[0] == "complete"
            assert log_row[1] == 21

    @pytest.mark.skipif(not SMALL_FILE.exists(), reason="2026 data file not found")
    def test_resumability_skips_complete(self, conn, clean_db) -> None:
        """Re-running should skip already-complete files."""
        pipeline = IngestPipeline(data_dir=DATA_DIR, batch_size=100)
        pipeline.run(drop_indexes=False, single_file=SMALL_FILE)

        with conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM papers")
            first_count = cur.fetchone()[0]

        # Run again
        pipeline.run(drop_indexes=False, single_file=SMALL_FILE)

        with conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM papers")
            second_count = cur.fetchone()[0]

        assert first_count == second_count  # no duplicates

    def test_ingest_synthetic_file(self, conn, clean_db, tmp_path: Path) -> None:
        """Ingest a synthetic file with known records including references."""
        records = [
            {
                "bibcode": "2024test...001A",
                "title": ["Test Paper A"],
                "year": "2024",
                "author": ["Author, A."],
                "first_author": "Author, A.",
                "doctype": "article",
                "reference": ["2024test...002B", "2024test...003C"],
                "citation_count": 5,
            },
            {
                "bibcode": "2024test...002B",
                "title": ["Test Paper B"],
                "year": "2024",
                "author": ["Author, B."],
                "first_author": "Author, B.",
                "doctype": "article",
            },
        ]
        filepath = tmp_path / "test_synthetic.jsonl.gz"
        with gzip.open(filepath, "wt") as f:
            for rec in records:
                f.write(json.dumps(rec) + "\n")

        pipeline = IngestPipeline(data_dir=tmp_path, batch_size=100)
        pipeline.run(drop_indexes=False, single_file=filepath)

        with conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM papers")
            assert cur.fetchone()[0] == 2

            # Check citation edges from reference[]
            cur.execute("SELECT count(*) FROM citation_edges WHERE source_bibcode = '2024test...001A'")
            assert cur.fetchone()[0] == 2

            cur.execute("SELECT target_bibcode FROM citation_edges WHERE source_bibcode = '2024test...001A' ORDER BY target_bibcode")
            targets = [r[0] for r in cur.fetchall()]
            assert targets == ["2024test...002B", "2024test...003C"]
