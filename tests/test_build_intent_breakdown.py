"""Unit tests for ``scripts/viz/build_intent_breakdown.py``.

Pure-aggregation tests; no DB and no live HTTP. Coverage targets the four
output panels (totals/coverage, top papers, by-year, community ranking) and
the synthetic + dry-run path through ``main()``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from viz.build_intent_breakdown import (
    INTENTS,
    TOTAL_CITATION_EDGES,
    CommunityRow,
    Config,
    IntentDataset,
    TopPaper,
    YearRow,
    _synthetic_dataset,
    aggregate_by_year,
    coverage_dict,
    load_community_labels,
    main,
    rank_communities,
    serialize_to_json,
    to_payload,
)

# ---------------------------------------------------------------------------
# coverage_dict
# ---------------------------------------------------------------------------


def test_coverage_dict_pct_matches_real_corpus_share() -> None:
    totals = {"method": 623_599, "background": 141_427, "result_comparison": 60_089}
    cov = coverage_dict(totals)
    assert cov["classified_edges"] == 825_115
    assert cov["total_edges"] == TOTAL_CITATION_EDGES
    # 825115 / 299_397_468 ≈ 0.002756
    assert cov["pct_classified"] == pytest.approx(0.002756, abs=1e-5)


def test_coverage_dict_handles_zero_classified() -> None:
    cov = coverage_dict({"method": 0, "background": 0, "result_comparison": 0})
    assert cov["classified_edges"] == 0
    assert cov["pct_classified"] == 0.0


# ---------------------------------------------------------------------------
# aggregate_by_year
# ---------------------------------------------------------------------------


def test_aggregate_by_year_sums_buckets_and_sorts_ascending() -> None:
    rows = [
        (2003, "method", 100),
        (2003, "background", 50),
        (2001, "method", 10),
        (2003, "result_comparison", 25),
        (2001, "background", 5),
    ]
    out = aggregate_by_year(rows)
    assert tuple(r.year for r in out) == (2001, 2003)
    assert out[0] == YearRow(year=2001, method=10, background=5, result_comparison=0)
    assert out[1] == YearRow(year=2003, method=100, background=50, result_comparison=25)
    assert out[1].total == 175


def test_aggregate_by_year_skips_unknown_intent() -> None:
    rows = [
        (2003, "method", 5),
        (2003, "garbage", 99),  # ignored
        (2003, "background", 1),
    ]
    out = aggregate_by_year(rows)
    assert out[0].method == 5
    assert out[0].background == 1
    assert out[0].total == 6  # garbage excluded


def test_aggregate_by_year_empty_returns_empty_tuple() -> None:
    assert aggregate_by_year([]) == ()


# ---------------------------------------------------------------------------
# rank_communities
# ---------------------------------------------------------------------------


def test_rank_communities_sorts_by_method_ratio_desc() -> None:
    # c=0: 90% method (high ratio)
    # c=1: 50% method
    # c=2: below threshold (gets dropped)
    rows = [
        (0, "method", 90),
        (0, "background", 5),
        (0, "result_comparison", 5),
        (1, "method", 50),
        (1, "background", 30),
        (1, "result_comparison", 20),
        (2, "method", 5),
        (2, "background", 2),
    ]
    out = rank_communities(rows, top_n=10, min_volume=20)
    assert tuple(r.community_id for r in out) == (0, 1)
    assert out[0].method_ratio == pytest.approx(0.9)
    assert out[1].method_ratio == pytest.approx(0.5)


def test_rank_communities_truncates_to_top_n() -> None:
    rows = []
    for c in range(5):
        rows.extend(
            [
                (c, "method", 100 - c),  # ratio decreases monotonically with c
                (c, "background", 50),
            ]
        )
    out = rank_communities(rows, top_n=3, min_volume=10)
    assert len(out) == 3
    # Verify ordering: c=0 has highest method count, ratio is highest there too
    assert out[0].community_id == 0
    assert out[0].method_ratio >= out[1].method_ratio >= out[2].method_ratio


def test_rank_communities_ties_break_by_total_then_id() -> None:
    # Two communities with identical method_ratio (1.0) — bigger volume wins,
    # then lower id wins among equal volumes.
    rows = [
        (3, "method", 50),
        (1, "method", 50),
        (2, "method", 100),  # bigger
    ]
    out = rank_communities(rows, top_n=10, min_volume=10)
    assert tuple(r.community_id for r in out) == (2, 1, 3)


def test_rank_communities_drops_below_min_volume() -> None:
    rows = [
        (0, "method", 5),
        (0, "background", 4),  # total 9, below threshold
        (1, "method", 100),
    ]
    out = rank_communities(rows, top_n=10, min_volume=10)
    assert tuple(r.community_id for r in out) == (1,)


def test_rank_communities_attaches_labels_when_provided() -> None:
    labels = {0: ("alpha", "beta"), 1: ("gamma",)}
    rows = [(0, "method", 50), (1, "method", 50)]
    out = rank_communities(rows, top_n=10, min_volume=10, labels=labels)
    by_id = {r.community_id: r for r in out}
    assert by_id[0].terms == ("alpha", "beta")
    assert by_id[1].terms == ("gamma",)


def test_rank_communities_rejects_negative_args() -> None:
    with pytest.raises(ValueError, match="top_n"):
        rank_communities([], top_n=-1, min_volume=0)
    with pytest.raises(ValueError, match="min_volume"):
        rank_communities([], top_n=1, min_volume=-1)


# ---------------------------------------------------------------------------
# load_community_labels
# ---------------------------------------------------------------------------


def test_load_community_labels_returns_none_for_missing(tmp_path: Path) -> None:
    assert load_community_labels(None) is None
    assert load_community_labels(tmp_path / "does_not_exist.json") is None


def test_load_community_labels_parses_real_shape(tmp_path: Path) -> None:
    payload = {
        "resolution": "medium",
        "communities": [
            {"community_id": 0, "n_sampled": 10, "terms": ["energy", "heat"]},
            {"community_id": 1, "n_sampled": 5, "terms": ["wave"]},
            {"community_id": 99, "terms": []},
        ],
    }
    p = tmp_path / "labels.json"
    p.write_text(json.dumps(payload))
    out = load_community_labels(p)
    assert out is not None
    assert out[0] == ("energy", "heat")
    assert out[1] == ("wave",)
    assert out[99] == ()


# ---------------------------------------------------------------------------
# Synthetic dataset & serialization
# ---------------------------------------------------------------------------


def _make_config(
    output: Path,
    *,
    dry_run: bool = False,
    synthetic: bool = True,
) -> Config:
    return Config(
        output=output,
        dsn="dbname=ignored",
        resolution="medium",
        top_papers=25,
        top_communities=20,
        min_community_volume=200,
        dry_run=dry_run,
        synthetic=synthetic,
        labels_path=None,
    )


def test_synthetic_dataset_shape(tmp_path: Path) -> None:
    cfg = _make_config(tmp_path / "out.json")
    ds = _synthetic_dataset(cfg)
    assert isinstance(ds, IntentDataset)
    assert set(ds.totals) == set(INTENTS)
    assert ds.coverage["total_edges"] == TOTAL_CITATION_EDGES
    assert all(isinstance(p, TopPaper) for p in ds.top_method)
    assert all(isinstance(p, TopPaper) for p in ds.top_background)
    assert all(isinstance(r, YearRow) for r in ds.by_year)
    assert all(isinstance(r, CommunityRow) for r in ds.communities_method)
    # Synthetic includes one community below volume threshold (total=60) — dropped.
    assert all(r.total >= cfg.min_community_volume for r in ds.communities_method)


def test_to_payload_is_json_serializable(tmp_path: Path) -> None:
    ds = _synthetic_dataset(_make_config(tmp_path / "out.json"))
    payload = to_payload(ds)
    text = json.dumps(payload)  # must not raise
    parsed = json.loads(text)
    assert "totals" in parsed
    assert "by_year" in parsed
    assert "communities_method" in parsed
    # Numeric totals add up
    assert (
        parsed["totals"]["method"]
        + parsed["totals"]["background"]
        + parsed["totals"]["result_comparison"]
        == parsed["coverage"]["classified_edges"]
    )


def test_serialize_to_json_writes_file(tmp_path: Path) -> None:
    out = tmp_path / "viz" / "citation_intent.json"
    ds = _synthetic_dataset(_make_config(out))
    serialize_to_json(ds, out)
    assert out.exists()
    parsed = json.loads(out.read_text())
    assert parsed["resolution"] == "medium"


# ---------------------------------------------------------------------------
# main() — synthetic CLI path
# ---------------------------------------------------------------------------


def test_main_synthetic_dry_run_returns_zero_no_write(tmp_path: Path) -> None:
    out = tmp_path / "missing.json"
    rc = main(["--synthetic", "--dry-run", "--output", str(out)])
    assert rc == 0
    assert not out.exists()


def test_main_synthetic_writes_file(tmp_path: Path) -> None:
    out = tmp_path / "cits.json"
    rc = main(["--synthetic", "--output", str(out)])
    assert rc == 0
    parsed = json.loads(out.read_text())
    assert "totals" in parsed
    assert "top_method" in parsed
    assert "by_year" in parsed
    assert "communities_method" in parsed
