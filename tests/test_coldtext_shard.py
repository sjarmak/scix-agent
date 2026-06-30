"""Unit tests for the ADR-016 cold-text shard codec/reader/writer.

No database required — shards are built from synthetic records.
"""

from __future__ import annotations

import sqlite3

import pytest

from scix.coldtext import (
    SHARD_SCHEMA_VERSION,
    ColdTextReader,
    sections_checksum,
    shard_path,
    write_shard,
)
from scix.coldtext.shard import _decode, _encode


def _rec(bibcode: str, sections: list[dict]) -> dict:
    return {
        "bibcode": bibcode,
        "source": "arxiv",
        "parser_version": "grobid-0.8.0",
        "canonical_bibcode": None,
        "suppressed_by_publisher": False,
        "sections": sections,
        "figures": [],
        "tables": [],
        "equations": [],
    }


SAMPLE_SECTIONS = [
    {"heading": "Introduction", "text": "We study X.", "level": 1, "offset": 0},
    {"heading": "Methods", "text": "We measured Y.", "level": 1, "offset": 11},
]


def test_encode_decode_roundtrip():
    obj = {"a": [1, 2, 3], "b": "ünïcode", "c": None}
    assert _decode(_encode(obj)) == obj


def test_encode_none_is_empty_and_decodes_none():
    assert _encode(None) == b""
    assert _decode(b"") is None
    assert _decode(None) is None


def test_checksum_stable_under_key_order():
    a = {"heading": "M", "text": "t", "level": 1, "offset": 0}
    b = {"offset": 0, "level": 1, "text": "t", "heading": "M"}
    assert sections_checksum([a]) == sections_checksum([b])


def test_checksum_changes_with_content():
    assert sections_checksum(SAMPLE_SECTIONS) != sections_checksum(SAMPLE_SECTIONS[:1])


def test_shard_path():
    assert shard_path("/mnt/x", 2020).name == "papers_fulltext_2020.sqlite"


def test_write_and_read_roundtrip(tmp_path):
    path = shard_path(tmp_path, 2020)
    recs = [_rec(f"2020bib..{i:04d}", SAMPLE_SECTIONS) for i in range(5)]
    stats = write_shard(path, 2020, recs)

    assert stats.rows == 5
    assert stats.path == path
    assert path.exists()

    with ColdTextReader(path) as r:
        assert r.row_count() == 5
        secs = r.fetch_sections("2020bib..0003")
        assert secs == SAMPLE_SECTIONS
        assert r.fetch_sections("does.not.exist") is None
        rec = r.fetch_record("2020bib..0000")
        assert rec is not None
        assert rec["parser_version"] == "grobid-0.8.0"
        assert rec["suppressed_by_publisher"] == 0  # bool -> int
        assert rec["figures"] == []  # [] round-trips as []
        assert sorted(r.iter_bibcodes())[0] == "2020bib..0000"


def test_empty_json_columns_roundtrip_as_none(tmp_path):
    # [] encodes to a non-empty blob, so it round-trips as []; only missing/None
    # decodes to None. Guard the distinction explicitly.
    path = shard_path(tmp_path, 2021)
    rec = _rec("2021bib..0001", SAMPLE_SECTIONS)
    rec["figures"] = [{"id": "f1"}]
    write_shard(path, 2021, [rec])
    with ColdTextReader(path) as r:
        got = r.fetch_record("2021bib..0001")
        assert got["figures"] == [{"id": "f1"}]
        # equations was [] -> round-trips as []
        assert got["equations"] == []


def test_missing_bibcode_raises(tmp_path):
    path = shard_path(tmp_path, 2022)
    with pytest.raises(ValueError, match="bibcode"):
        write_shard(path, 2022, [{"sections": []}])


def test_atomic_rebuild_leaves_no_tmp(tmp_path):
    path = shard_path(tmp_path, 2023)
    write_shard(path, 2023, [_rec("2023bib..0001", SAMPLE_SECTIONS)])
    tmp = path.with_suffix(path.suffix + ".tmp")
    assert not tmp.exists()


def test_reader_is_readonly(tmp_path):
    path = shard_path(tmp_path, 2024)
    write_shard(path, 2024, [_rec("2024bib..0001", SAMPLE_SECTIONS)])
    with ColdTextReader(path) as r:
        with pytest.raises(sqlite3.OperationalError):
            r._conn.execute("DELETE FROM fulltext")


def test_meta_written(tmp_path):
    path = shard_path(tmp_path, 2020)
    write_shard(path, 2020, [_rec("2020bib..0001", SAMPLE_SECTIONS)])
    with ColdTextReader(path) as r:
        meta = dict(r._conn.execute("SELECT key, value FROM shard_meta"))
    assert meta["schema_version"] == str(SHARD_SCHEMA_VERSION)
    assert meta["year"] == "2020"
    assert meta["rows"] == "1"


def test_missing_shard_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        ColdTextReader(tmp_path / "nope.sqlite")


def test_corrupt_blob_returns_none(tmp_path):
    # A truncated/garbage blob (e.g. partial NFS read) must degrade to None,
    # not raise into the read path.
    path = shard_path(tmp_path, 2020)
    write_shard(path, 2020, [_rec("2020bib..0001", SAMPLE_SECTIONS)])
    w = sqlite3.connect(path)
    w.execute(
        "UPDATE fulltext SET sections = ? WHERE bibcode = ?",
        (b"not-valid-zlib", "2020bib..0001"),
    )
    w.commit()
    w.close()
    with ColdTextReader(path) as r:
        assert r.fetch_sections("2020bib..0001") is None
