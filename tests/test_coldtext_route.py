"""Unit tests for the ADR-016 Phase 1a cold read route.

DB-free: Postgres year-resolution is faked; the shard is a real tmp SQLite.
"""

from __future__ import annotations

import pytest

from scix.coldtext import route, shard_path, write_shard


class _FakeCursor:
    def __init__(self, lookup):
        self._lookup = lookup
        self._row = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql, params):
        self._row = self._lookup(params[0])

    def fetchone(self):
        return self._row


class FakeConn:
    """Maps bibcode -> year; a bibcode absent from the map has no PG row."""

    def __init__(self, years: dict[str, int | None]):
        self._years = years

    def cursor(self, **_kw):
        def lookup(bibcode):
            if bibcode not in self._years:
                return None
            return (self._years[bibcode],)

        return _FakeCursor(lookup)


SECTIONS = [{"heading": "Intro", "text": "x", "level": 1, "offset": 0}]


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    monkeypatch.setenv("SCIX_COLDTEXT_ROOT", str(tmp_path))
    route.clear_cache()
    yield
    route.clear_cache()


def _make_shard(tmp_path, year, bibcode):
    write_shard(
        shard_path(tmp_path, year),
        year,
        [{"bibcode": bibcode, "sections": SECTIONS, "figures": [], "tables": [], "equations": []}],
    )


def test_cold_hit_returns_sections(tmp_path):
    _make_shard(tmp_path, 2010, "2010bib..0001")
    conn = FakeConn({"2010bib..0001": 2010})
    assert route.fetch_sections_cold(conn, "2010bib..0001") == SECTIONS


def test_year_resolves_but_no_shard(tmp_path):
    conn = FakeConn({"1999bib..0001": 1999})  # no shard built for 1999
    assert route.fetch_sections_cold(conn, "1999bib..0001") is None


def test_bibcode_unknown_to_pg(tmp_path):
    _make_shard(tmp_path, 2010, "2010bib..0001")
    conn = FakeConn({})  # PG returns no row
    assert route.fetch_sections_cold(conn, "2010bib..0001") is None


def test_bibcode_in_shard_missing(tmp_path):
    _make_shard(tmp_path, 2010, "2010bib..0001")
    conn = FakeConn({"2010bib..9999": 2010})  # year has a shard, bibcode absent
    assert route.fetch_sections_cold(conn, "2010bib..9999") is None


def test_null_year_resolves_to_zero_sentinel():
    conn = FakeConn({"noyear..0001": None})
    assert route._resolve_year(conn, "noyear..0001") == 0


def test_no_coldtext_root(monkeypatch):
    monkeypatch.setenv("SCIX_COLDTEXT_ROOT", "/nonexistent/coldtext/root")
    route.clear_cache()
    conn = FakeConn({"2010bib..0001": 2010})
    assert route.coldtext_root() is None
    assert route.fetch_sections_cold(conn, "2010bib..0001") is None


def test_reader_cache_reused(tmp_path):
    _make_shard(tmp_path, 2010, "2010bib..0001")
    conn = FakeConn({"2010bib..0001": 2010})
    route.fetch_sections_cold(conn, "2010bib..0001")
    assert 2010 in route._readers
    r1 = route._readers[2010]
    route.fetch_sections_cold(conn, "2010bib..0001")
    assert route._readers[2010] is r1  # same cached reader


def test_absent_shard_is_cached(tmp_path):
    # Root present, no shard for the year -> cached as None (don't re-stat).
    conn = FakeConn({"1999bib..0001": 1999})
    assert route.fetch_sections_cold(conn, "1999bib..0001") is None
    assert 1999 in route._readers
    assert route._readers[1999] is None


def test_no_mount_not_cached(monkeypatch):
    monkeypatch.setenv("SCIX_COLDTEXT_ROOT", "/nonexistent/root")
    route.clear_cache()
    assert route._reader_for_year(2010) is None
    assert 2010 not in route._readers  # may appear later -> not cached


def test_transient_open_error_not_cached(tmp_path):
    # Restore only ColdTextReader (not the autouse env) so recovery still points
    # at the tmp root, not the real /mnt mount.
    _make_shard(tmp_path, 2010, "2010bib..0001")
    orig = route.ColdTextReader

    def boom(_path):
        raise OSError("ESTALE: stale NFS file handle")

    route.ColdTextReader = boom
    try:
        assert route._reader_for_year(2010) is None
        assert 2010 not in route._readers  # transient error -> retry next call
    finally:
        route.ColdTextReader = orig

    route.clear_cache()
    conn = FakeConn({"2010bib..0001": 2010})
    assert route.fetch_sections_cold(conn, "2010bib..0001") == SECTIONS
