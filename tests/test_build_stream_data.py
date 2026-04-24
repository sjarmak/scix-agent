"""Unit tests for ``scripts/viz/build_stream_data.py``.

Exercises the pure-function layer (SQL builder, payload reshaper, label
loader, synthetic generator) — the DB path is covered by the smoke run
that ships ``data/viz/stream.<res>.json``.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT_PATH = _REPO_ROOT / "scripts" / "viz" / "build_stream_data.py"


def _import_script():
    """Load build_stream_data.py as a module without going through scripts/."""
    spec = importlib.util.spec_from_file_location("build_stream_data", _SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["build_stream_data"] = mod
    spec.loader.exec_module(mod)
    return mod


bsd = _import_script()


# ---------------------------------------------------------------------------
# SQL builder
# ---------------------------------------------------------------------------


def test_build_sql_uses_allowlist_column() -> None:
    """The SQL must interpolate the resolution column from the allowlist
    only — never raw user input."""
    sql = bsd._build_sql("medium")
    assert "community_semantic_medium" in sql
    assert "community_semantic_coarse" not in sql
    assert "community_semantic_fine" not in sql
    # Year window must be a bound parameter, not a literal.
    assert "BETWEEN %s AND %s" in sql


def test_build_sql_rejects_unknown_resolution() -> None:
    with pytest.raises(ValueError, match="unknown resolution"):
        bsd._build_sql("ultra-fine")


# ---------------------------------------------------------------------------
# Synthetic loader
# ---------------------------------------------------------------------------


def test_synthetic_cells_cover_full_window() -> None:
    """Every (community, year) combination should produce exactly one cell."""
    cells = bsd.load_cells_synthetic("coarse", 2010, 2014)
    by_cid = {}
    for c in cells:
        by_cid.setdefault(c.community_id, []).append(c.year)
    # 20 coarse communities × 5 years
    assert len(by_cid) == 20
    for years in by_cid.values():
        assert sorted(years) == [2010, 2011, 2012, 2013, 2014]


def test_synthetic_cells_are_deterministic() -> None:
    """Same args → same data so tests stay stable across runs."""
    a = bsd.load_cells_synthetic("coarse", 2010, 2012)
    b = bsd.load_cells_synthetic("coarse", 2010, 2012)
    assert a == b


# ---------------------------------------------------------------------------
# build_payload — the reshaping layer
# ---------------------------------------------------------------------------


def test_build_payload_pivots_cells_into_dense_counts() -> None:
    cells = [
        bsd.Cell(community_id=0, year=2020, count=10),
        bsd.Cell(community_id=0, year=2021, count=20),
        bsd.Cell(community_id=1, year=2020, count=5),
        # No 2021 entry for cid=1 — payload should zero-fill it.
    ]
    payload = bsd.build_payload(cells, "coarse", 2020, 2021)

    assert payload["resolution"] == "coarse"
    assert payload["years"] == [2020, 2021]
    by_cid = {c["community_id"]: c for c in payload["communities"]}
    assert by_cid[0]["counts"] == [10, 20]
    assert by_cid[0]["total"] == 30
    assert by_cid[1]["counts"] == [5, 0]
    assert by_cid[1]["total"] == 5


def test_build_payload_sorts_by_total_descending() -> None:
    """Largest community first so D3 stack draws big bands at the base."""
    cells = [
        bsd.Cell(community_id=10, year=2020, count=1),
        bsd.Cell(community_id=11, year=2020, count=999),
        bsd.Cell(community_id=12, year=2020, count=42),
    ]
    payload = bsd.build_payload(cells, "coarse", 2020, 2020)
    cids = [c["community_id"] for c in payload["communities"]]
    assert cids == [11, 12, 10]


def test_build_payload_drops_out_of_window_cells() -> None:
    cells = [
        bsd.Cell(community_id=0, year=1999, count=1000),  # before window
        bsd.Cell(community_id=0, year=2030, count=1000),  # after window
        bsd.Cell(community_id=0, year=2020, count=5),
    ]
    payload = bsd.build_payload(cells, "coarse", 2020, 2024)
    assert payload["communities"][0]["counts"] == [5, 0, 0, 0, 0]
    assert payload["communities"][0]["total"] == 5


def test_build_payload_skips_zero_total_communities() -> None:
    """A community that lands entirely outside the window contributes nothing."""
    cells = [bsd.Cell(community_id=0, year=2030, count=1000)]
    payload = bsd.build_payload(cells, "coarse", 2020, 2024)
    assert payload["communities"] == []


def test_build_payload_uses_label_when_present() -> None:
    cells = [bsd.Cell(community_id=0, year=2020, count=1)]
    labels = {0: "energy / heat / thermal"}
    payload = bsd.build_payload(cells, "coarse", 2020, 2020, labels=labels)
    assert payload["communities"][0]["label"] == "energy / heat / thermal"


def test_build_payload_falls_back_to_community_id_label() -> None:
    cells = [bsd.Cell(community_id=42, year=2020, count=1)]
    payload = bsd.build_payload(cells, "coarse", 2020, 2020)
    assert payload["communities"][0]["label"] == "community 42"


def test_build_payload_rejects_inverted_year_window() -> None:
    with pytest.raises(ValueError, match="year_min > year_max"):
        bsd.build_payload([], "coarse", 2024, 2020)


# ---------------------------------------------------------------------------
# Label loader
# ---------------------------------------------------------------------------


def test_load_labels_handles_missing_file(tmp_path: Path) -> None:
    """Missing label file is fine — returns empty dict."""
    assert bsd.load_labels(tmp_path / "does_not_exist.json") == {}


def test_load_labels_handles_none() -> None:
    assert bsd.load_labels(None) == {}


def test_load_labels_parses_compute_community_labels_schema(tmp_path: Path) -> None:
    src = tmp_path / "labels.json"
    src.write_text(
        json.dumps(
            {
                "resolution": "coarse",
                "communities": [
                    {"community_id": 0, "n_sampled": 100, "terms": ["energy", "heat", "thermal", "x", "y"]},
                    {"community_id": 1, "n_sampled": 50, "terms": []},  # filtered out
                    {"community_id": 2, "terms": ["plant", "soil", "water"]},
                ],
            }
        )
    )
    out = bsd.load_labels(src)
    assert out[0] == "energy / heat / thermal"
    assert 1 not in out  # empty terms list filtered
    assert out[2] == "plant / soil / water"


def test_load_labels_handles_corrupt_json(tmp_path: Path) -> None:
    src = tmp_path / "broken.json"
    src.write_text("{ this is not valid json")
    assert bsd.load_labels(src) == {}


# ---------------------------------------------------------------------------
# End-to-end synthetic
# ---------------------------------------------------------------------------


def test_main_synthetic_writes_valid_json(tmp_path: Path) -> None:
    out = tmp_path / "stream.json"
    rc = bsd.main(
        [
            "--synthetic",
            "--resolution",
            "coarse",
            "--year-min",
            "2010",
            "--year-max",
            "2014",
            "--output",
            str(out),
        ]
    )
    assert rc == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["resolution"] == "coarse"
    assert payload["years"] == [2010, 2011, 2012, 2013, 2014]
    assert len(payload["communities"]) == 20
    # Every community has counts aligned with years
    for c in payload["communities"]:
        assert len(c["counts"]) == 5
        assert c["total"] == sum(c["counts"])
