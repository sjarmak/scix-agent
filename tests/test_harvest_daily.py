"""Tests for scripts/harvest_daily.py (bead scix_experiments-d4c1).

Pins the two ways the old watermark harvest lost records:
  * ADS numFound flapped between replicas mid-harvest (4710 vs 2918 on
    2026-09-23) and the loop stopped at 3000 because it trusted the last page;
  * ADS backdates entry_date to the arXiv date but indexes days later, so a
    record could land behind an already-advanced watermark and never be seen.

DB-free: ADS and the papers-table lookup are faked.
"""

from __future__ import annotations

import gzip
import json
import sys
from datetime import date
from pathlib import Path

import pytest

sys.path.insert(0, "scripts")

from harvest_daily import (  # noqa: E402
    HarvestIncomplete,
    harvest,
    list_window_bibcodes,
    window_query,
)


class FakeADS:
    """Serves a fixed bibcode set through the ADS search params contract.

    ``num_found`` lets a test make numFound flap per request, and
    ``drop_pages`` lets a test make chosen listing requests come back short,
    the way a stale replica does.
    """

    def __init__(self, bibcodes, num_found=None, drop_pages=()):
        self.bibcodes = sorted(bibcodes)
        self.num_found = num_found or (lambda call: len(self.bibcodes))
        self.drop_pages = set(drop_pages)
        self.calls: list[dict] = []

    def __call__(self, params: dict) -> tuple[list[dict], int]:
        call = len(self.calls)
        self.calls.append(dict(params))
        q = params["q"]
        if q.startswith("bibcode:("):
            wanted = [w.strip('"') for w in q[len("bibcode:(") : -1].split(" OR ")]
            docs = [{"bibcode": b, "title": [f"T {b}"]} for b in wanted if b in self.bibcodes]
            return docs, len(docs)
        start, rows = params["start"], params["rows"]
        page = [] if call in self.drop_pages else self.bibcodes[start : start + rows]
        return [{"bibcode": b} for b in page], self.num_found(call)


def _bibs(n: int) -> list[str]:
    return [f"2026arXiv{i:08d}X" for i in range(n)]


def test_window_query_reaches_back_window_days():
    assert window_query(date(2026, 9, 23), 14) == "entdate:[2026-09-09 TO 2026-09-23]"


def test_listing_is_sorted_so_offset_paging_is_stable():
    ads = FakeADS(_bibs(5))
    list_window_bibcodes(ads, "entdate:[x TO y]", rows=2)
    pages = [c for c in ads.calls if c["rows"]]
    assert pages and all(c["sort"] == "bibcode asc" and c["fl"] == "bibcode" for c in pages)


def test_flapping_num_found_does_not_truncate_listing():
    # The 2026-09-23 shape: 4710 real records, some replicas report 2918.
    bibs = _bibs(4710)
    ads = FakeADS(bibs, num_found=lambda call: 2918 if call % 2 else 4710)
    assert list_window_bibcodes(ads, "q", rows=100) == set(bibs)


def test_short_page_is_healed_by_a_second_pass():
    bibs = _bibs(10)
    ads = FakeADS(bibs, drop_pages={4})  # 3 count probes, then pass 1's second page is empty
    assert list_window_bibcodes(ads, "q", rows=3) == set(bibs)


def test_listing_that_never_completes_raises():
    bibs = _bibs(10)
    ads = FakeADS(bibs, drop_pages=set(range(4, 1000)))  # only the first page ever serves
    with pytest.raises(HarvestIncomplete, match="3/10"):
        list_window_bibcodes(ads, "q", rows=3, max_passes=3)


def test_harvest_fetches_only_bibcodes_missing_from_db(tmp_path: Path):
    bibs = _bibs(7)
    in_db = set(bibs[:4])
    ads = FakeADS(bibs)
    out = harvest(
        tmp_path,
        window_days=14,
        fetch=ads,
        existing=lambda bs: in_db & set(bs),
        today=date(2026, 9, 23),
        fetch_batch=2,
    )
    assert out == tmp_path / "ads_daily_2026-09-23.jsonl.gz"
    with gzip.open(out, "rt") as f:
        written = [json.loads(line)["bibcode"] for line in f]
    assert sorted(written) == bibs[4:]
    fetched = [c for c in ads.calls if c["q"].startswith("bibcode:(")]
    assert [c["rows"] for c in fetched] == [4, 2]
    assert not list(tmp_path.glob("*.tmp"))


def test_harvest_catches_backdated_record_inside_window(tmp_path: Path):
    # A record ADS indexed today but stamped 11 days ago is still in the window
    # and still missing from the DB, so it is harvested.
    late = "2026arXiv260911515S"
    ads = FakeADS([late, "2026arXiv260925396X"])
    out = harvest(
        tmp_path,
        window_days=14,
        fetch=ads,
        existing=lambda bs: {"2026arXiv260925396X"} & set(bs),
        today=date(2026, 9, 23),
    )
    with gzip.open(out, "rt") as f:
        assert [json.loads(line)["bibcode"] for line in f] == [late]


def test_harvest_with_nothing_missing_writes_no_file(tmp_path: Path):
    bibs = _bibs(3)
    out = harvest(
        tmp_path,
        window_days=14,
        fetch=FakeADS(bibs),
        existing=lambda bs: set(bs),
        today=date(2026, 9, 23),
    )
    assert out is None
    assert not list(tmp_path.iterdir())


def test_unfetchable_bibcode_is_reported_not_fatal(tmp_path: Path, caplog):
    bibs = _bibs(3)
    ads = FakeADS(bibs)
    real = ads.__call__

    def fetch(params):
        docs, n = real(params)
        if params["q"].startswith("bibcode:("):
            docs = [d for d in docs if d["bibcode"] != bibs[1]]
        return docs, n

    out = harvest(tmp_path, 14, fetch=fetch, existing=lambda bs: set(), today=date(2026, 9, 23))
    with gzip.open(out, "rt") as f:
        assert len(f.readlines()) == 2
    assert bibs[1] in caplog.text


def test_backlog_over_cap_fetches_arxiv_first(tmp_path: Path, caplog):
    arxiv = ["2026arXiv260911515S", "2026arXiv260925396X"]
    datasets = ["2026arch.data....1A", "2026arch.data....2A", "2026euvi.data....1A"]
    ads = FakeADS(arxiv + datasets)
    out = harvest(
        tmp_path, 14, fetch=ads, existing=lambda bs: set(), today=date(2026, 9, 23), max_fetch=3
    )
    with gzip.open(out, "rt") as f:
        written = sorted(json.loads(line)["bibcode"] for line in f)
    assert written == sorted(arxiv + ["2026arch.data....1A"])
    assert "deferring 2" in caplog.text


def test_pass_served_by_one_stale_replica_is_not_trusted():
    # Every listing request hits a replica that only has the first 5 of 10
    # records and says numFound=5; one count probe reaches a current replica.
    bibs = _bibs(10)
    calls = []

    def fetch(params):
        calls.append(params)
        visible = bibs if len(calls) == 2 else bibs[:5]
        page = visible[params["start"] : params["start"] + params["rows"]]
        return [{"bibcode": b} for b in page], len(visible)

    with pytest.raises(HarvestIncomplete, match="5/10"):
        list_window_bibcodes(fetch, "q", rows=3)


def test_record_fetch_quotes_bibcodes_and_drops_unrequested_hits(tmp_path: Path):
    bib = "2026A&A...700A..12X"
    seen = []

    def fetch(params):
        seen.append(params["q"])
        if params["q"].startswith("bibcode:("):
            return [{"bibcode": bib}, {"bibcode": "2025ApJ...1..1Z"}], 2
        page = [{"bibcode": bib}][params["start"] : params["start"] + params["rows"]]
        return page, 1

    out = harvest(tmp_path, 14, fetch=fetch, existing=lambda bs: set(), today=date(2026, 9, 23))
    assert f'bibcode:("{bib}")' in seen
    with gzip.open(out, "rt") as f:
        assert [json.loads(line)["bibcode"] for line in f] == [bib]
