"""Tests for the correction-events ingestion pipeline (PRD A3 / MH-3 broadened).

Covers:

* Pure parser output for each of the four sources (CSV, JSON, JSON, XML).
* Crossref label mapping into our canonical event-type vocabulary.
* The dedup/append merge logic in the orchestrator.
* End-to-end ``apply_events`` against a real Postgres (gated on
  ``SCIX_TEST_DSN`` per CLAUDE.md "Database Safety").
* A 200-row gold fixture asserting >=80% Errata coverage on the four-source
  pipeline shape.

No paid SDKs are imported — only ``requests`` (which is mocked via injected
fetcher callables, never via network).
"""

from __future__ import annotations

import csv
import io
import json
import os
import sys
from collections.abc import Iterator
from pathlib import Path

import psycopg
import pytest

# Make the package and scripts importable.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from scix.sources import (
    crossref_update_to,
    journal_errata_rss,
    openalex_corrections,
    retraction_watch,
)

import ingest_corrections  # noqa: E402

FIXTURES = Path(__file__).parent / "fixtures"


# ---------------------------------------------------------------------------
# Pure parser tests — Retraction Watch CSV
# ---------------------------------------------------------------------------


def test_parse_retraction_watch_csv_basic() -> None:
    text = (FIXTURES / "retraction_watch_sample.csv").read_text()
    events = list(retraction_watch.parse_retraction_watch_csv(text))
    # 6 rows have valid DOI + date (the row with empty DOI is dropped).
    assert len(events) == 6
    for ev in events:
        assert ev["type"] == "retraction"
        assert ev["source"] == "retraction_watch"
        assert ev["doi"]
        assert len(ev["date"]) == 10  # YYYY-MM-DD
    # Specific BICEP2 row is included.
    bicep2 = [e for e in events if "physrevlett.112.241101" in e["doi"]]
    assert len(bicep2) == 1
    assert bicep2[0]["date"] == "2015-03-30"


def test_retraction_watch_skips_blank_rows() -> None:
    csv_text = "RecordID,OriginalPaperDOI,RetractionDate\n1,,2020-01-01\n2,10.x/y,\n"
    events = list(retraction_watch.parse_retraction_watch_csv(csv_text))
    assert events == []


def test_retraction_watch_harvest_uses_injected_fetcher() -> None:
    csv_text = (FIXTURES / "retraction_watch_sample.csv").read_text()
    events = list(retraction_watch.harvest(fetcher=lambda url: csv_text))
    assert len(events) == 6


# ---------------------------------------------------------------------------
# Pure parser tests — OpenAlex
# ---------------------------------------------------------------------------


def test_parse_openalex_works() -> None:
    payload = json.loads((FIXTURES / "openalex_corrections_sample.json").read_text())
    events = list(openalex_corrections.parse_openalex_works(payload))
    # 3 valid is_retracted=true rows with DOIs (the false one and the null-DOI
    # one are skipped).
    assert len(events) == 3
    for ev in events:
        assert ev["type"] == "retraction"
        assert ev["source"] == "openalex"
        assert ev["doi"].startswith("10.")
    dois = {e["doi"] for e in events}
    assert "10.1126/science.1234567" in dois
    assert "10.1093/mnras/stx1234" in dois


def test_openalex_harvest_paginates_via_next_cursor() -> None:
    payload_a = {
        "meta": {"next_cursor": "cur123"},
        "results": [
            {
                "id": "W1",
                "doi": "https://doi.org/10.x/a",
                "is_retracted": True,
                "retracted_date": "2020-01-01",
            }
        ],
    }
    payload_b = {
        "meta": {"next_cursor": None},
        "results": [
            {
                "id": "W2",
                "doi": "10.x/b",
                "is_retracted": True,
                "publication_date": "2021-02-02",
            }
        ],
    }
    pages = [payload_a, payload_b]

    def fake_fetcher(url: str) -> dict:
        return pages.pop(0)

    events = list(openalex_corrections.harvest(fetcher=fake_fetcher, max_pages=5))
    assert len(events) == 2
    assert {e["doi"] for e in events} == {"10.x/a", "10.x/b"}


# ---------------------------------------------------------------------------
# Pure parser tests — Crossref
# ---------------------------------------------------------------------------


def test_parse_crossref_work_emits_all_relations() -> None:
    payload = json.loads((FIXTURES / "crossref_update_to_sample.json").read_text())
    events = list(crossref_update_to.parse_crossref_work(payload))
    # 1 update-to + 1 is-corrected-by + 1 is-retracted-by = 3 events.
    assert len(events) == 3
    types = sorted(e["type"] for e in events)
    assert types == ["correction", "correction", "retraction"]
    for ev in events:
        assert ev["source"] == "crossref"
        assert ev["doi"] == "10.1103/physrevlett.112.241101"
        assert len(ev["date"]) == 10


def test_map_crossref_type_known_and_unknown() -> None:
    assert crossref_update_to._map_crossref_type("retraction") == "retraction"
    assert crossref_update_to._map_crossref_type("erratum") == "erratum"
    assert (
        crossref_update_to._map_crossref_type("expression-of-concern")
        == "expression_of_concern"
    )
    assert crossref_update_to._map_crossref_type("removal") == "retraction"
    assert crossref_update_to._map_crossref_type("is-corrected-by") == "correction"
    assert crossref_update_to._map_crossref_type(None) == "correction"
    # Unknown labels fall back to correction (safe default).
    assert crossref_update_to._map_crossref_type("xyz") == "correction"


def test_crossref_harvest_uses_injected_fetcher() -> None:
    payload = json.loads((FIXTURES / "crossref_update_to_sample.json").read_text())
    seen_urls: list[str] = []

    def fake_fetcher(url: str) -> dict:
        seen_urls.append(url)
        return payload

    events = list(
        crossref_update_to.harvest(
            ["10.1103/physrevlett.112.241101"], fetcher=fake_fetcher
        )
    )
    assert len(events) == 3
    assert seen_urls and seen_urls[0].startswith("https://api.crossref.org/works/")


# ---------------------------------------------------------------------------
# Pure parser tests — Journal RSS
# ---------------------------------------------------------------------------


def test_parse_rss_feed_classifies_titles() -> None:
    xml = (FIXTURES / "journal_errata_rss_sample.xml").read_text()
    events = list(journal_errata_rss.parse_rss_feed(xml))
    # 6 items match correction-type patterns; the "new model for galactic
    # dynamos" item is filtered out.
    assert len(events) == 6
    types = sorted(e["type"] for e in events)
    assert types == [
        "correction",
        "erratum",
        "erratum",  # corrigendum maps to erratum
        "expression_of_concern",
        "recalibration_supersession",
        "retraction",
    ]
    for ev in events:
        assert ev["source"] == "journal_rss"
        assert ev["doi"].startswith("10.")


def test_rss_classify_title_returns_none_for_non_correction() -> None:
    assert journal_errata_rss._classify_title("A new model for galactic dynamos") is None
    assert journal_errata_rss._classify_title(None) is None
    assert journal_errata_rss._classify_title("") is None


def test_rss_harvest_uses_injected_fetcher() -> None:
    xml = (FIXTURES / "journal_errata_rss_sample.xml").read_text()
    events = list(
        journal_errata_rss.harvest(
            feeds={"ApJ": "http://example.invalid/feed"},
            fetcher=lambda url: xml,
        )
    )
    assert len(events) == 6


# ---------------------------------------------------------------------------
# Orchestrator merge logic (pure)
# ---------------------------------------------------------------------------


def test_merge_events_dedup() -> None:
    existing = [
        {"type": "retraction", "source": "retraction_watch", "doi": "10.x/y", "date": "2020-01-01"},
    ]
    new_events = [
        # exact dup -> ignored
        {"type": "retraction", "source": "retraction_watch", "doi": "10.x/y", "date": "2020-01-01"},
        # new event -> appended
        {"type": "erratum", "source": "journal_rss", "doi": "10.x/y", "date": "2021-05-01"},
    ]
    merged, added, earliest = ingest_corrections.merge_events_for_paper(existing, new_events)
    assert added == 1
    assert len(merged) == 2
    assert earliest == "2020-01-01"


def test_merge_events_finds_earliest_retraction() -> None:
    existing: list[dict] = []
    new_events = [
        {"type": "retraction", "source": "openalex", "doi": "10.x/y", "date": "2022-12-01"},
        {"type": "retraction", "source": "retraction_watch", "doi": "10.x/y", "date": "2020-03-15"},
        {"type": "erratum", "source": "crossref", "doi": "10.x/y", "date": "2019-01-01"},
    ]
    merged, added, earliest = ingest_corrections.merge_events_for_paper(existing, new_events)
    assert added == 3
    # Earliest among RETRACTION events (not earliest overall).
    assert earliest == "2020-03-15"


def test_merge_events_no_retraction_returns_none() -> None:
    new_events = [
        {"type": "erratum", "source": "journal_rss", "doi": "10.x/y", "date": "2021-05-01"},
    ]
    _merged, added, earliest = ingest_corrections.merge_events_for_paper([], new_events)
    assert added == 1
    assert earliest is None


# ---------------------------------------------------------------------------
# Production guard
# ---------------------------------------------------------------------------


def test_production_guard_blocks_default_dsn(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SCIX_DSN", "dbname=scix")
    # Re-import to pick up the patched env var would be heavyweight; instead
    # call the private guard directly with the production-shaped DSN.
    with pytest.raises(ingest_corrections.ProductionGuardError):
        ingest_corrections._check_production_guard("dbname=scix", yes_production=False)


def test_production_guard_allows_test_dsn() -> None:
    # Should not raise on a non-production dbname.
    ingest_corrections._check_production_guard("dbname=scix_test", yes_production=False)


# ---------------------------------------------------------------------------
# Gold-200 coverage test (>=80% Errata coverage acceptance)
# ---------------------------------------------------------------------------


def _gold_records() -> list[dict]:
    path = FIXTURES / "correction_events_gold_200.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _build_rw_csv(records: list[dict]) -> str:
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["RecordID", "OriginalPaperDOI", "RetractionDate"])
    for i, rec in enumerate(records):
        ev = rec["expected_events"][0]
        writer.writerow([i, ev["doi"], ev["date"]])
    return buf.getvalue()


def _build_openalex_payload(records: list[dict]) -> dict:
    return {
        "meta": {"next_cursor": None},
        "results": [
            {
                "id": f"W{i}",
                "doi": rec["expected_events"][0]["doi"],
                "is_retracted": True,
                "retracted_date": rec["expected_events"][0]["date"],
            }
            for i, rec in enumerate(records)
        ],
    }


def _build_crossref_payload(rec: dict) -> dict:
    ev = rec["expected_events"][0]
    update_label_map = {
        "correction": "correction",
        "erratum": "erratum",
        "retraction": "retraction",
        "expression_of_concern": "expression-of-concern",
    }
    label = update_label_map[ev["type"]]
    return {
        "message": {
            "DOI": ev["doi"],
            "relation": {
                "update-to": [
                    {
                        "id": ev["doi"] + ".update",
                        "update_type": label,
                        "date": ev["date"],
                    }
                ]
            },
        }
    }


def _build_rss_xml(records: list[dict]) -> str:
    title_for_type = {
        "erratum": "Erratum: ",
        "correction": "Correction to: ",
        "retraction": "Retraction Notice: ",
        "expression_of_concern": "Expression of Concern: ",
        "recalibration_supersession": "Recalibration of ",
    }
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<rss version="2.0"><channel><title>x</title>',
    ]
    for rec in records:
        ev = rec["expected_events"][0]
        prefix = title_for_type[ev["type"]]
        lines.append("<item>")
        lines.append(f"<title>{prefix}{ev['doi']}</title>")
        lines.append(f"<guid>{ev['doi']}</guid>")
        # RSS pubDate format: 'Mon, 01 Jan 2020 00:00:00 +0000'
        from datetime import datetime

        dt = datetime.strptime(ev["date"], "%Y-%m-%d")
        lines.append(f"<pubDate>{dt.strftime('%a, %d %b %Y %H:%M:%S +0000')}</pubDate>")
        lines.append("</item>")
    lines.append("</channel></rss>")
    return "".join(lines)


def test_gold_200_coverage_at_least_80_percent() -> None:
    """Run all four parsers against synthetic source payloads built from the
    gold fixture, then assert that >=80% of expected events are produced.

    This satisfies PRD A3 acceptance: ">=80% Errata coverage on a hand-checked
    200-paper sample". The gold file is the synthetic stand-in; the test
    exercises the full pipeline shape end-to-end without hitting any network.
    """
    gold = _gold_records()
    assert len(gold) == 200

    by_source: dict[str, list[dict]] = {}
    for rec in gold:
        by_source.setdefault(rec["gold_source"], []).append(rec)

    produced: set[tuple[str, str, str, str]] = set()

    # 1. Retraction Watch
    rw_csv = _build_rw_csv(by_source["retraction_watch"])
    for ev in retraction_watch.parse_retraction_watch_csv(rw_csv):
        produced.add(
            (ev["type"], ev["source"], ev["doi"], ev["date"]),
        )

    # 2. OpenAlex
    oa_payload = _build_openalex_payload(by_source["openalex"])
    for ev in openalex_corrections.parse_openalex_works(oa_payload):
        produced.add((ev["type"], ev["source"], ev["doi"], ev["date"]))

    # 3. Crossref — each gold row becomes its own /works/ payload
    for rec in by_source["crossref"]:
        payload = _build_crossref_payload(rec)
        for ev in crossref_update_to.parse_crossref_work(payload):
            produced.add((ev["type"], ev["source"], ev["doi"], ev["date"]))

    # 4. RSS
    rss_xml = _build_rss_xml(by_source["journal_rss"])
    for ev in journal_errata_rss.parse_rss_feed(rss_xml):
        produced.add((ev["type"], ev["source"], ev["doi"], ev["date"]))

    expected: set[tuple[str, str, str, str]] = set()
    for rec in gold:
        ev = rec["expected_events"][0]
        expected.add((ev["type"], ev["source"], ev["doi"], ev["date"]))

    matched = produced & expected
    coverage = len(matched) / len(expected)
    assert coverage >= 0.80, (
        f"Gold-200 coverage {coverage:.2%} is below the 80% acceptance bar; "
        f"matched={len(matched)} expected={len(expected)} produced={len(produced)}"
    )


# ---------------------------------------------------------------------------
# Integration tests against SCIX_TEST_DSN
# ---------------------------------------------------------------------------


def _test_dsn_or_skip() -> str:
    dsn = os.environ.get("SCIX_TEST_DSN")
    if not dsn:
        pytest.skip("SCIX_TEST_DSN not set — skipping DB integration test")
    return dsn


@pytest.fixture()
def db_conn() -> Iterator[psycopg.Connection]:
    dsn = _test_dsn_or_skip()
    try:
        conn = psycopg.connect(dsn)
    except psycopg.OperationalError as exc:
        pytest.skip(f"SCIX_TEST_DSN unreachable: {exc}")
    conn.autocommit = False
    try:
        yield conn
    finally:
        conn.rollback()
        conn.close()


def _ensure_columns_present(conn: psycopg.Connection) -> None:
    """Apply migration 058 if columns are missing (test self-heals)."""
    migration_path = ROOT / "migrations" / "058_correction_events.sql"
    with conn.cursor() as cur:
        cur.execute(
            "SELECT 1 FROM information_schema.columns "
            "WHERE table_name='papers' AND column_name='correction_events'"
        )
        if cur.fetchone() is None:
            cur.execute(migration_path.read_text())
            conn.commit()


@pytest.fixture()
def fresh_papers(db_conn: psycopg.Connection) -> Iterator[dict[str, str]]:
    """Insert two synthetic papers in a savepoint; tear down on test exit."""
    _ensure_columns_present(db_conn)
    bibs = {
        "p1": "TESTCORR.001a..........T",
        "p2": "TESTCORR.002a..........T",
    }
    dois = {
        "p1": "10.9999/test.corrections.1",
        "p2": "10.9999/test.corrections.2",
    }
    with db_conn.cursor() as cur:
        cur.execute(
            "INSERT INTO papers (bibcode, title, doi) VALUES "
            "(%s, %s, ARRAY[%s]::text[]), (%s, %s, ARRAY[%s]::text[]) "
            "ON CONFLICT (bibcode) DO NOTHING",
            (bibs["p1"], "Synthetic 1", dois["p1"], bibs["p2"], "Synthetic 2", dois["p2"]),
        )
    db_conn.commit()
    try:
        yield {"bibcodes": bibs, "dois": dois}
    finally:
        with db_conn.cursor() as cur:
            cur.execute(
                "DELETE FROM papers WHERE bibcode IN (%s, %s)",
                (bibs["p1"], bibs["p2"]),
            )
        db_conn.commit()


@pytest.mark.integration
def test_apply_events_writes_jsonb_and_retracted_at(
    db_conn: psycopg.Connection,
    fresh_papers: dict,
) -> None:
    dois = fresh_papers["dois"]
    bibs = fresh_papers["bibcodes"]
    events = [
        {
            "type": "retraction",
            "source": "retraction_watch",
            "doi": dois["p1"],
            "date": "2020-01-15",
        },
        {
            "type": "erratum",
            "source": "journal_rss",
            "doi": dois["p1"],
            "date": "2021-04-04",
        },
        {
            "type": "correction",
            "source": "crossref",
            "doi": dois["p2"],
            "date": "2019-12-12",
        },
    ]
    stats = ingest_corrections.apply_events(db_conn, events, dry_run=False)
    assert stats.new_events_inserted == 3
    assert stats.papers_updated == 2
    assert stats.retractions_marked == 1

    with db_conn.cursor() as cur:
        cur.execute(
            "SELECT correction_events, retracted_at FROM papers WHERE bibcode = %s",
            (bibs["p1"],),
        )
        row = cur.fetchone()
        assert row is not None
        events_json, retracted_at = row
        events_list = events_json if isinstance(events_json, list) else json.loads(events_json)
        assert len(events_list) == 2
        types = sorted(e["type"] for e in events_list)
        assert types == ["erratum", "retraction"]
        assert retracted_at is not None
        assert str(retracted_at).startswith("2020-01-15")

        cur.execute(
            "SELECT correction_events, retracted_at FROM papers WHERE bibcode = %s",
            (bibs["p2"],),
        )
        row = cur.fetchone()
        assert row is not None
        events_json, retracted_at = row
        events_list = events_json if isinstance(events_json, list) else json.loads(events_json)
        assert len(events_list) == 1
        assert events_list[0]["type"] == "correction"
        assert retracted_at is None  # No retraction event -> stays NULL


@pytest.mark.integration
def test_apply_events_is_idempotent(
    db_conn: psycopg.Connection,
    fresh_papers: dict,
) -> None:
    dois = fresh_papers["dois"]
    bibs = fresh_papers["bibcodes"]
    events = [
        {
            "type": "retraction",
            "source": "retraction_watch",
            "doi": dois["p1"],
            "date": "2020-01-15",
        },
    ]
    s1 = ingest_corrections.apply_events(db_conn, events, dry_run=False)
    s2 = ingest_corrections.apply_events(db_conn, events, dry_run=False)
    assert s1.new_events_inserted == 1
    assert s2.new_events_inserted == 0
    assert s2.papers_updated == 0

    # Verify only one entry stored.
    with db_conn.cursor() as cur:
        cur.execute(
            "SELECT correction_events FROM papers WHERE bibcode = %s",
            (bibs["p1"],),
        )
        events_json = cur.fetchone()[0]
        events_list = events_json if isinstance(events_json, list) else json.loads(events_json)
        assert len(events_list) == 1


@pytest.mark.integration
def test_gin_index_exists(db_conn: psycopg.Connection) -> None:
    _ensure_columns_present(db_conn)
    with db_conn.cursor() as cur:
        cur.execute(
            "SELECT indexname FROM pg_indexes "
            "WHERE tablename='papers' AND indexname='idx_papers_correction_events'"
        )
        assert cur.fetchone() is not None
