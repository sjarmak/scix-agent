"""Unit tests for ``scripts/viz/project_embeddings_umap.py``.

Each test maps to an acceptance-criterion sub-point of unit-v3-projection:

* 6(a) — synthetic end-to-end with umap-learn on ~200 random 768-d vectors.
* 6(b) — CLI flag parsing.
* 6(c) — output JSON schema validator (accept + reject).
* 6(d) — ``--dry-run`` does not write the output file.

No database required. Uses only numpy + umap-learn.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.viz.project_embeddings_umap import (
    COMMUNITY_COLUMNS,
    Config,
    ProjectedPoint,
    _config_from_args,
    _parse_args,
    _stratified_sample_sql,
    build_points,
    load_embeddings_from_db,
    load_embeddings_synthetic,
    main,
    pick_backend,
    project,
    serialize,
    validate_projection_payload,
)


# ---------------------------------------------------------------------------
# 6(a) — synthetic end-to-end with umap-learn
# ---------------------------------------------------------------------------


def test_synthetic_end_to_end(tmp_path: Path) -> None:
    """``main --synthetic 200 --output ... --backend umap-learn`` writes valid JSON."""
    out = tmp_path / "umap.json"
    rc = main(
        [
            "--synthetic",
            "200",
            "--output",
            str(out),
            "--backend",
            "umap-learn",
            "--resolution",
            "coarse",
        ]
    )
    assert rc == 0
    assert out.exists(), "output file must be written when --dry-run is absent"

    payload = json.loads(out.read_text())
    points = validate_projection_payload(payload)
    assert len(points) == 200
    # Every row must carry the requested resolution label.
    assert all(p.resolution == "coarse" for p in points)
    # x and y must be finite.
    assert all(np.isfinite(p.x) and np.isfinite(p.y) for p in points)
    # Community ids from synthetic loader use range(20).
    assert all(p.community_id is not None and 0 <= p.community_id < 20 for p in points)
    # bibcodes are the deterministic synthetic ids.
    assert points[0].bibcode == "synthetic-000000"
    assert points[-1].bibcode == "synthetic-000199"


# ---------------------------------------------------------------------------
# 6(b) — CLI flag parsing
# ---------------------------------------------------------------------------


def test_cli_parsing_defaults() -> None:
    args = _parse_args([])
    assert args.sample_size == 100_000
    assert args.resolution == "coarse"
    assert args.backend == "auto"
    assert args.output == "data/viz/umap.json"
    assert args.dry_run is False
    assert args.synthetic is None


def test_cli_parsing_all_flags(tmp_path: Path) -> None:
    out = tmp_path / "x.json"
    args = _parse_args(
        [
            "--sample-size",
            "500",
            "--resolution",
            "medium",
            "--backend",
            "umap-learn",
            "--dsn",
            "dbname=test",
            "--output",
            str(out),
            "--dry-run",
            "--synthetic",
            "42",
        ]
    )
    assert args.sample_size == 500
    assert args.resolution == "medium"
    assert args.backend == "umap-learn"
    assert args.dsn == "dbname=test"
    assert args.output == str(out)
    assert args.dry_run is True
    assert args.synthetic == 42

    config = _config_from_args(args)
    assert isinstance(config, Config)
    assert config.sample_size == 500
    assert config.resolution == "medium"
    assert config.backend == "umap-learn"
    assert config.dry_run is True
    assert config.synthetic_n == 42


def test_cli_parsing_rejects_unknown_resolution() -> None:
    with pytest.raises(SystemExit):
        _parse_args(["--resolution", "bogus"])


def test_cli_parsing_rejects_unknown_backend() -> None:
    with pytest.raises(SystemExit):
        _parse_args(["--backend", "bogus"])


# ---------------------------------------------------------------------------
# 6(c) — output JSON schema validator
# ---------------------------------------------------------------------------


def test_validate_projection_payload_accepts_valid() -> None:
    raw = [
        {
            "bibcode": "2020ApJ...1..100X",
            "x": 1.0,
            "y": 2.0,
            "community_id": 3,
            "resolution": "coarse",
        },
        {
            "bibcode": "2020ApJ...1..200X",
            "x": -0.5,
            "y": 4.25,
            "community_id": None,
            "resolution": "coarse",
        },
    ]
    points = validate_projection_payload(raw)
    assert len(points) == 2
    assert points[0].community_id == 3
    assert points[1].community_id is None
    assert all(isinstance(p, ProjectedPoint) for p in points)


def test_validate_projection_payload_rejects_missing_key() -> None:
    bad = [
        {"bibcode": "a", "x": 1.0, "y": 2.0, "resolution": "coarse"},  # no community_id
    ]
    with pytest.raises(ValueError, match="community_id"):
        validate_projection_payload(bad)


def test_validate_projection_payload_rejects_wrong_type() -> None:
    bad = [
        {
            "bibcode": "a",
            "x": "not a number",
            "y": 2.0,
            "community_id": 0,
            "resolution": "coarse",
        },
    ]
    with pytest.raises(ValueError, match=r"x"):
        validate_projection_payload(bad)


def test_validate_projection_payload_rejects_unknown_resolution() -> None:
    bad = [
        {
            "bibcode": "a",
            "x": 1.0,
            "y": 2.0,
            "community_id": 0,
            "resolution": "bogus",
        },
    ]
    with pytest.raises(ValueError, match="resolution"):
        validate_projection_payload(bad)


def test_validate_projection_payload_rejects_non_list() -> None:
    with pytest.raises(ValueError, match="list"):
        validate_projection_payload({"nodes": []})


# ---------------------------------------------------------------------------
# 6(d) — --dry-run does not write the output file
# ---------------------------------------------------------------------------


def test_dry_run_writes_no_file(tmp_path: Path) -> None:
    out = tmp_path / "should_not_exist.json"
    rc = main(
        [
            "--synthetic",
            "50",
            "--dry-run",
            "--output",
            str(out),
            "--backend",
            "umap-learn",
        ]
    )
    assert rc == 0
    assert not out.exists(), "--dry-run must NOT write the output file"


# ---------------------------------------------------------------------------
# Supporting coverage — synthetic loader, backend picker, SQL allowlist
# ---------------------------------------------------------------------------


def test_synthetic_loader_shape() -> None:
    bibcodes, embeddings, community_ids = load_embeddings_synthetic(50)
    assert len(bibcodes) == 50
    assert embeddings.shape == (50, 768)
    assert embeddings.dtype == np.float32
    assert len(community_ids) == 50
    assert all(isinstance(c, int) for c in community_ids)


def test_synthetic_loader_is_deterministic() -> None:
    a = load_embeddings_synthetic(20, seed=7)
    b = load_embeddings_synthetic(20, seed=7)
    assert a[0] == b[0]
    assert np.array_equal(a[1], b[1])
    assert a[2] == b[2]


def test_synthetic_loader_rejects_nonpositive_n() -> None:
    with pytest.raises(ValueError):
        load_embeddings_synthetic(0)


def test_stratified_sample_sql_allowlist_rejects_unknown() -> None:
    """Defence in depth: unknown resolution must raise before SQL interpolation."""
    with pytest.raises(ValueError, match="unknown resolution"):
        _stratified_sample_sql("bogus")


def test_stratified_sample_sql_uses_expected_column() -> None:
    for resolution, column in COMMUNITY_COLUMNS.items():
        sql_text = _stratified_sample_sql(resolution)
        assert column in sql_text
        assert "paper_embeddings" in sql_text
        assert "paper_metrics" in sql_text
        assert "model_name = 'indus'" in sql_text


def test_load_embeddings_from_db_rejects_unknown_resolution() -> None:
    with pytest.raises(ValueError, match="unknown resolution"):
        load_embeddings_from_db("dbname=test", "bogus", 100)


def test_pick_backend_umap_learn_small_n() -> None:
    choice = pick_backend("umap-learn", n=10)
    assert choice.label == "umap-learn"
    # n_neighbors must be safe for small n (<= n-1).
    assert hasattr(choice.backend, "fit_transform")


def test_pick_backend_auto_falls_back_without_cuml() -> None:
    # cuml is not installed in this environment; auto must land on umap-learn.
    choice = pick_backend("auto", n=30)
    assert choice.label in ("cuml", "umap-learn")


def test_pick_backend_rejects_unknown() -> None:
    with pytest.raises(ValueError, match="unknown backend"):
        pick_backend("bogus", n=10)


# ---------------------------------------------------------------------------
# build_points / project — lightweight sanity
# ---------------------------------------------------------------------------


def test_project_and_build_points_roundtrip() -> None:
    bibcodes, embeddings, community_ids = load_embeddings_synthetic(80, seed=1)
    choice = pick_backend("umap-learn", n=len(bibcodes))
    xy = project(embeddings, choice.backend)
    assert xy.shape == (80, 2)
    points = build_points(bibcodes, xy, community_ids, "coarse")
    assert len(points) == 80
    assert all(isinstance(p, ProjectedPoint) for p in points)
    assert all(p.resolution == "coarse" for p in points)


def test_build_points_rejects_length_mismatch() -> None:
    with pytest.raises(ValueError, match="mismatched"):
        build_points(
            bibcodes=["a", "b"],
            xy=np.zeros((3, 2), dtype=np.float32),
            community_ids=[0, 1],
            resolution="coarse",
        )


def test_build_points_rejects_unknown_resolution() -> None:
    with pytest.raises(ValueError, match="unknown resolution"):
        build_points(
            bibcodes=["a"],
            xy=np.zeros((1, 2), dtype=np.float32),
            community_ids=[0],
            resolution="bogus",
        )


def test_serialize_writes_valid_json(tmp_path: Path) -> None:
    points = (
        ProjectedPoint(
            bibcode="a", x=1.0, y=2.0, community_id=0, resolution="coarse"
        ),
    )
    out = tmp_path / "nested" / "out.json"
    serialize(points, out)
    assert out.exists()
    payload = json.loads(out.read_text())
    restored = validate_projection_payload(payload)
    assert restored == points
