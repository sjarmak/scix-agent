"""Unit tests for ``scripts/viz/build_section_coverage.py``.

Exercises the pure-function layer (heading classifier roll-up, decade merge,
payload reshaper, synthetic generator, serializer) — the DB path is covered by
the smoke run that ships ``data/viz/section_coverage.json``.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT_PATH = _REPO_ROOT / "scripts" / "viz" / "build_section_coverage.py"


def _import_script():
    spec = importlib.util.spec_from_file_location("build_section_coverage", _SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["build_section_coverage"] = mod
    spec.loader.exec_module(mod)
    return mod


bsc = _import_script()


# ---------------------------------------------------------------------------
# summarize_paper
# ---------------------------------------------------------------------------


def test_summarize_paper_classifies_headings() -> None:
    paper = bsc.summarize_paper(["Introduction", "Methods", "Results"])
    assert paper.section_count == 3
    assert paper.roles_present == frozenset({"background", "method", "result"})


def test_summarize_paper_none_heading_counts_as_section_and_other() -> None:
    # A section with no heading still counts toward section_count and
    # classifies as the catch-all "other" role.
    paper = bsc.summarize_paper([None, "Conclusions"])
    assert paper.section_count == 2
    assert "other" in paper.roles_present
    assert "conclusion" in paper.roles_present


def test_summarize_paper_empty() -> None:
    paper = bsc.summarize_paper([])
    assert paper.section_count == 0
    assert paper.roles_present == frozenset()


# ---------------------------------------------------------------------------
# _merge_decades
# ---------------------------------------------------------------------------


def test_merge_decades_zero_fills_missing_side() -> None:
    merged = bsc._merge_decades({1990: 100, 2000: 200}, {1990: 400, 2010: 50})
    by_decade = {d.decade: d for d in merged}
    assert by_decade[1990].fulltext == 100 and by_decade[1990].total == 400
    assert by_decade[2000].fulltext == 200 and by_decade[2000].total == 0
    assert by_decade[2010].fulltext == 0 and by_decade[2010].total == 50
    # Ascending order.
    assert [d.decade for d in merged] == [1990, 2000, 2010]


# ---------------------------------------------------------------------------
# build_payload
# ---------------------------------------------------------------------------


def _samples(*role_sets: set[str]) -> list:
    return [
        bsc.PaperSections(roles_present=frozenset(rs), section_count=len(rs) or 1)
        for rs in role_sets
    ]


def test_build_payload_role_presence_pct() -> None:
    samples = _samples(
        {"background", "method"},
        {"background"},
        {"result"},
        {"background", "result"},
    )
    decades = [bsc.DecadeCoverage(decade=2010, fulltext=5, total=10)]
    payload = bsc.build_payload(
        samples, decades, corpus_papers=100, fulltext_papers=40,
        year_min=1990, year_max=2020,
    )
    roles = {r["role"]: r for r in payload["roles"]}
    # background in 3/4 papers, result in 2/4.
    assert roles["background"]["papers_with"] == 3
    assert roles["background"]["present_pct"] == pytest.approx(0.75)
    assert roles["result"]["present_pct"] == pytest.approx(0.5)
    assert roles["conclusion"]["present_pct"] == 0.0
    # All five canonical roles present in display order.
    assert [r["role"] for r in payload["roles"]] == [
        "background", "method", "result", "conclusion", "other"
    ]


def test_build_payload_corpus_fractions_and_decade_pct() -> None:
    samples = _samples({"method"})
    decades = [bsc.DecadeCoverage(decade=2000, fulltext=3, total=12)]
    payload = bsc.build_payload(
        samples, decades, corpus_papers=1000, fulltext_papers=460,
        year_min=1990, year_max=2010,
    )
    assert payload["fulltext_fraction"] == pytest.approx(0.46)
    assert payload["sample_size"] == 1
    assert payload["decades"][0]["fulltext_pct"] == pytest.approx(0.25)


def test_build_payload_empty_sample_no_division_by_zero() -> None:
    payload = bsc.build_payload(
        [], [], corpus_papers=0, fulltext_papers=0, year_min=2000, year_max=2010,
    )
    assert payload["fulltext_fraction"] == 0.0
    assert payload["mean_sections_per_paper"] == 0.0
    assert payload["median_sections_per_paper"] == 0
    assert all(r["present_pct"] == 0.0 for r in payload["roles"])


# ---------------------------------------------------------------------------
# synthesize + serialize
# ---------------------------------------------------------------------------


def test_synthesize_is_deterministic_and_shaped() -> None:
    a = bsc.synthesize(500, 1980, 2024)
    b = bsc.synthesize(500, 1980, 2024)
    samples_a, decades_a, corpus_a, fulltext_a = a
    samples_b, decades_b, corpus_b, fulltext_b = b
    assert len(samples_a) == 500
    assert corpus_a == corpus_b and fulltext_a == fulltext_b
    assert [d.decade for d in decades_a] == [d.decade for d in decades_b]
    # fulltext never exceeds total per decade.
    assert all(d.fulltext <= d.total for d in decades_a)
    assert fulltext_a <= corpus_a


def test_serialize_roundtrip(tmp_path: Path) -> None:
    samples, decades, corpus, fulltext = bsc.synthesize(100, 1990, 2020)
    payload = bsc.build_payload(
        samples, decades, corpus_papers=corpus, fulltext_papers=fulltext,
        year_min=1990, year_max=2020,
    )
    out = tmp_path / "section_coverage.json"
    bsc.serialize(payload, out)
    loaded = json.loads(out.read_text())
    assert loaded["roles"] == payload["roles"]
    assert loaded["decades"] == payload["decades"]


def test_main_synthetic_writes_file(tmp_path: Path) -> None:
    out = tmp_path / "sc.json"
    rc = bsc.main(["--synthetic", "--sample-size", "50", "--output", str(out)])
    assert rc == 0
    payload = json.loads(out.read_text())
    assert payload["sample_size"] == 50
    assert len(payload["roles"]) == 5


def test_main_rejects_inverted_year_window(tmp_path: Path) -> None:
    rc = bsc.main(
        ["--synthetic", "--year-min", "2020", "--year-max", "1990",
         "--output", str(tmp_path / "x.json")]
    )
    assert rc == 2
