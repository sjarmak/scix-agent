"""Unit tests for ``scripts/viz/build_temporal_sankey_data.py``.

Each test maps 1:1 to a sub-point of acceptance criterion 5 in the
unit-v2-sankey-data work unit spec. No database required.
"""

from __future__ import annotations

import pytest

from scripts.viz.build_temporal_sankey_data import (
    PaperRow,
    aggregate,
    decade_of,
    main,
    validate_sankey_data,
)


# ---------------------------------------------------------------------------
# 5(1) — schema validation on a hand-built minimal dataset
# ---------------------------------------------------------------------------


def test_validate_sankey_data_accepts_minimal_valid_payload() -> None:
    raw = {
        "nodes": [
            {"id": "1990-0", "decade": 1990, "community_id": 0, "paper_count": 5},
            {"id": "2000-0", "decade": 2000, "community_id": 0, "paper_count": 7},
        ],
        "links": [
            {"source": "1990-0", "target": "2000-0", "value": 7},
        ],
    }
    sd = validate_sankey_data(raw)
    assert len(sd.nodes) == 2
    assert len(sd.links) == 1
    assert sd.nodes[0].id == "1990-0"
    assert sd.links[0].value == 7


def test_validate_sankey_data_rejects_missing_keys() -> None:
    # Missing 'links' top-level key.
    with pytest.raises(ValueError, match="links"):
        validate_sankey_data({"nodes": []})

    # Missing 'value' in a link.
    bad = {
        "nodes": [{"id": "1990-0", "decade": 1990, "community_id": 0, "paper_count": 1}],
        "links": [{"source": "1990-0", "target": "2000-0"}],
    }
    with pytest.raises(ValueError, match="value"):
        validate_sankey_data(bad)

    # Wrong type on a node field.
    bad_type = {
        "nodes": [{"id": "x", "decade": "1990", "community_id": 0, "paper_count": 1}],
        "links": [],
    }
    with pytest.raises(ValueError, match="decade"):
        validate_sankey_data(bad_type)


# ---------------------------------------------------------------------------
# 5(2) — correct decade bucketing
# ---------------------------------------------------------------------------


def test_decade_of_boundary_years() -> None:
    # Spec-prescribed examples.
    assert decade_of(1999) == 1990
    assert decade_of(2000) == 2000
    # Additional boundaries, including the synthetic-data range ends.
    assert decade_of(1990) == 1990
    assert decade_of(2009) == 2000
    assert decade_of(2010) == 2010
    assert decade_of(2025) == 2020


# ---------------------------------------------------------------------------
# 5(3) — top-flows cap is respected
# ---------------------------------------------------------------------------


def test_top_flows_cap_limits_links() -> None:
    # Construct rows such that the same 50 communities appear in each of 5
    # decades. Each adjacent decade pair contributes 50 persistent-community
    # links -> 50 * 4 = 200 candidate links. Cap to 10.
    n_communities = 50
    decades = [1980, 1990, 2000, 2010, 2020]
    rows: list[PaperRow] = []
    counter = 0
    for decade in decades:
        for c in range(n_communities):
            # Stagger paper counts so sort is stable and deterministic.
            paper_count = (decade // 10) + c + 1
            for _ in range(paper_count):
                rows.append(
                    PaperRow(
                        bibcode=f"b-{counter:08d}",
                        year=decade + 1,  # lands in `decade` bucket
                        community_id=c,
                    )
                )
                counter += 1

    top_n = 10
    result = aggregate(rows, top_flows=top_n)

    total_candidates = n_communities * (len(decades) - 1)
    assert total_candidates > top_n, "precondition: candidate count must exceed top_n"
    assert len(result.links) == top_n

    # Links are sorted by value desc — verify the invariant holds.
    values = [lk.value for lk in result.links]
    assert values == sorted(values, reverse=True)


# ---------------------------------------------------------------------------
# 5(4) — --dry-run does not write any file
# ---------------------------------------------------------------------------


def test_dry_run_does_not_write_output(tmp_path) -> None:
    out = tmp_path / "sankey.json"
    rc = main(
        [
            "--dry-run",
            "--synthetic",
            "--output",
            str(out),
        ]
    )
    assert rc == 0
    assert not out.exists(), f"dry-run should not create {out}"
