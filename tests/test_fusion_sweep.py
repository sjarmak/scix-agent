"""Unit tests for the pure fusion primitives in ``scripts/fusion_sweep.py``.

These exercise only the DB-free, model-free fusion math, so they run in CI
without Postgres, Qdrant, or torch — mirroring ``test_eval_retrieval_50q.py``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
for sub in ("src", "scripts"):
    p = str(_REPO_ROOT / sub)
    if p not in sys.path:
        sys.path.insert(0, p)

import fusion_sweep as sweep  # noqa: E402
from fusion_sweep import (  # noqa: E402
    BenchmarkQuery,
    Lane,
    QueryLanes,
    build_sweep,
    enforce_coverage_gate,
    evaluate_config,
    fuse_bm25_only,
    fuse_dense_only,
    fuse_dense_prior,
    fuse_naive_rrf,
    fuse_rank_cutoff_rrf,
    fuse_weighted_sum,
    load_benchmark_queries,
    min_max_normalize,
    paired_bootstrap_delta,
    render_markdown,
    split_queries,
)

DENSE: Lane = [("A", 0.90), ("B", 0.50), ("C", 0.10)]
BM25: Lane = [("C", 3.0), ("D", 2.0), ("A", 1.0)]


# --- min_max_normalize ------------------------------------------------------


def test_min_max_normalize_maps_endpoints() -> None:
    norm = min_max_normalize(DENSE)
    assert norm["A"] == pytest.approx(1.0)
    assert norm["C"] == pytest.approx(0.0)
    assert 0.0 < norm["B"] < 1.0


def test_min_max_normalize_constant_lane_is_one() -> None:
    # No spread → every member is fully relevant within the lane.
    assert min_max_normalize([("X", 5.0), ("Y", 5.0)]) == {"X": 1.0, "Y": 1.0}


def test_min_max_normalize_empty() -> None:
    assert min_max_normalize([]) == {}


# --- reference passthroughs -------------------------------------------------


def test_dense_only_preserves_dense_order() -> None:
    assert fuse_dense_only(DENSE, BM25) == ["A", "B", "C"]


def test_bm25_only_preserves_bm25_order() -> None:
    assert fuse_bm25_only(DENSE, BM25) == ["C", "D", "A"]


# --- weighted sum -----------------------------------------------------------


def test_weighted_sum_full_dense_weight_recovers_dense_order() -> None:
    # w_dense=1.0 → BM25-exclusive doc D (norm_dense 0) sinks to the bottom.
    out = fuse_weighted_sum(DENSE, BM25, w_dense=1.0)
    assert out[:3] == ["A", "B", "C"]
    assert out[-1] == "D"


def test_weighted_sum_full_bm25_weight_recovers_bm25_order() -> None:
    out = fuse_weighted_sum(DENSE, BM25, w_dense=0.0)
    assert out[0] == "C"  # BM25 top
    assert out.index("D") < out.index("B")  # D (bm25) outranks B (dense-only)


def test_weighted_sum_is_a_permutation_of_the_union() -> None:
    out = fuse_weighted_sum(DENSE, BM25, w_dense=0.7)
    assert sorted(out) == ["A", "B", "C", "D"]


# --- rank-cutoff RRF --------------------------------------------------------


def test_rank_cutoff_zero_is_dense_only() -> None:
    # cutoff=0 → BM25 contributes nothing; result is the dense ranking.
    assert fuse_rank_cutoff_rrf(DENSE, BM25, cutoff=0) == ["A", "B", "C"]


def test_rank_cutoff_limits_bm25_tail() -> None:
    # cutoff=1 → only BM25's top doc (C) can be boosted; D (bm25 rank 2) is
    # excluded from BM25's contribution, so it can only appear via dense (it
    # never does) → D absent from the fused list.
    out = fuse_rank_cutoff_rrf(DENSE, BM25, cutoff=1)
    assert "D" not in out
    assert set(out) == {"A", "B", "C"}


# --- dense prior ------------------------------------------------------------


def test_dense_prior_keeps_dense_as_anchor() -> None:
    # Small lambda → every dense doc outranks the BM25-exclusive doc D.
    out = fuse_dense_prior(DENSE, BM25, lam=0.05)
    assert out[:3] == ["A", "B", "C"]
    assert out[-1] == "D"


def test_dense_prior_larger_lambda_can_promote_shared_docs() -> None:
    # A is in both lanes; a bigger lambda only strengthens it, never demotes
    # it below a dense-only doc with lower dense score.
    out = fuse_dense_prior(DENSE, BM25, lam=0.2)
    assert out[0] == "A"


# --- naive RRF / sweep grid -------------------------------------------------


def test_naive_rrf_rewards_cross_lane_agreement() -> None:
    # A and C appear in both lanes; they should outrank single-lane docs.
    out = fuse_naive_rrf(DENSE, BM25, k=60)
    assert set(out[:2]) == {"A", "C"}


def test_build_sweep_has_all_strategies_and_is_deterministic() -> None:
    names = [c.name for c in build_sweep()]
    assert "dense_only" in names
    assert "bm25_only" in names
    assert any(n.startswith("naive_rrf(") for n in names)
    assert any(n.startswith("weighted_sum(") for n in names)
    assert any(n.startswith("rank_cutoff_rrf(") for n in names)
    assert any(n.startswith("dense_prior(") for n in names)
    # Late-binding closure guard: each swept config must capture its own param.
    assert len(names) == len(set(names))
    assert names == [c.name for c in build_sweep()]


# --- report rendering -------------------------------------------------------


def _block(ndcg: float) -> dict:
    """A minimal results block: flat nDCG across overall + every decile."""
    overall = {"ndcg_at_10": ndcg, "mrr_at_10": ndcg, "recall_at_50": ndcg}
    return {
        "overall": dict(overall),
        "by_decile": {str(decile): dict(overall) for decile in range(10)},
    }


def test_render_rejects_multiple_confirmation_hybrids() -> None:
    results = {
        "dense_only": _block(0.50),
        "bm25_only": _block(0.20),
        "hybrid-a": _block(0.60),
        "hybrid-b": _block(0.70),
    }

    with pytest.raises(ValueError, match="exactly one tuning-selected hybrid"):
        render_markdown(results, queries_path="g.jsonl", n_queries=50, k=10)


def test_render_flags_premise_inversion_when_bm25_beats_dense() -> None:
    # dense < bm25 → the "naive RRF hurts dense" regime does not hold; the
    # report must say so instead of crowning a hybrid "winner".
    results = {
        "dense_only": _block(0.04),
        "bm25_only": _block(0.08),
        "weighted_sum(w_dense=0.5)": _block(0.07),
    }
    md = render_markdown(results, queries_path="g.jsonl", n_queries=50, k=10)
    assert "Premise does NOT reproduce" in md
    assert "Winning fusion" not in md


def test_render_declares_winner_only_when_dense_dominates() -> None:
    results = {
        "dense_only": _block(0.50),
        "bm25_only": _block(0.20),
        "weighted_sum(w_dense=0.5)": _block(0.60),
    }
    md = render_markdown(results, queries_path="g.jsonl", n_queries=50, k=10)
    assert "Winning fusion: `weighted_sum(w_dense=0.5)`" in md
    assert "Premise does NOT reproduce" not in md


def test_render_uses_one_consistent_headline_config() -> None:
    # The per-bucket table header must name the SAME config the verdict crowns,
    # never a second different "winner" (the rows[0]-vs-verdict mismatch bug).
    results = {
        "dense_only": _block(0.50),
        "bm25_only": _block(0.20),
        "weighted_sum(w_dense=0.5)": _block(0.60),
    }
    md = render_markdown(results, queries_path="g.jsonl", n_queries=50, k=10)
    header = next(ln for ln in md.splitlines() if ln.startswith("## Per-decile"))
    assert "weighted_sum(w_dense=0.5)" in header


def test_render_keeps_tuning_selected_hybrid_when_confirmation_loses() -> None:
    results = {
        "dense_only": _block(0.60),
        "bm25_only": _block(0.20),
        "weighted_sum(w_dense=0.5)": _block(0.50),
    }

    md = render_markdown(results, queries_path="g.jsonl", n_queries=50, k=10)

    header = next(ln for ln in md.splitlines() if ln.startswith("## Per-decile"))
    assert "weighted_sum(w_dense=0.5)" in header
    assert "dense_only` at" not in md


# --- publication-valid benchmark protocol ---------------------------------


def _query(index: int, decile: int) -> BenchmarkQuery:
    return BenchmarkQuery(
        query_id=f"q-{index}",
        query=f"title {index}",
        bucket="recall_decile",
        discipline="test",
        gold_bibcodes=(f"SELF-{index}", f"NEIGHBOR-{index}"),
        decile=decile,
    )


def test_split_queries_is_deterministic_stratified_and_disjoint() -> None:
    queries = [_query(decile * 4 + offset, decile) for decile in range(10) for offset in range(4)]
    tuning, confirmation = split_queries(queries, tuning_fraction=0.5, salt="frozen")
    tuning_again, confirmation_again = split_queries(
        list(reversed(queries)), tuning_fraction=0.5, salt="frozen"
    )

    assert [q.query_id for q in tuning] == [q.query_id for q in tuning_again]
    assert [q.query_id for q in confirmation] == [q.query_id for q in confirmation_again]
    assert {q.query_id for q in tuning}.isdisjoint(q.query_id for q in confirmation)
    assert {q.decile for q in tuning} == set(range(10))
    assert {q.decile for q in confirmation} == set(range(10))


def test_loader_requires_and_preserves_all_source_deciles(tmp_path: Path) -> None:
    path = tmp_path / "gold.jsonl"
    source_rows = [
        {
            "query": f"exact title {decile}",
            "bucket": "recall_decile",
            "discipline": "test",
            "gold_bibcodes": [f"SELF-{decile}"],
            "decile": decile,
        }
        for decile in range(10)
    ]
    path.write_text("\n".join(json.dumps(row) for row in source_rows), encoding="utf-8")

    loaded = load_benchmark_queries(path)

    assert [query.decile for query in loaded] == list(range(10))
    assert all(query.query_id for query in loaded)


def test_loader_rejects_missing_decile_strata(tmp_path: Path) -> None:
    path = tmp_path / "gold.jsonl"
    path.write_text(
        json.dumps(
            {
                "query": "exact title",
                "bucket": "recall_decile",
                "discipline": "test",
                "gold_bibcodes": ["SELF"],
                "decile": 0,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="all deciles"):
        load_benchmark_queries(path)


def test_loader_rejects_duplicate_query_text(tmp_path: Path) -> None:
    rows = [
        {
            "query": f"exact title {decile}",
            "bucket": "recall_decile",
            "discipline": "test",
            "gold_bibcodes": [f"SELF-{decile}"],
            "decile": decile,
        }
        for decile in range(10)
    ]
    duplicate_query = dict(rows[0])
    duplicate_query["gold_bibcodes"] = ["DIFFERENT-GOLD"]
    rows.append(duplicate_query)
    path = tmp_path / "gold.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate query text"):
        load_benchmark_queries(path)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("query", 17, "query must be a non-empty string"),
        ("discipline", "", "discipline must be a non-empty string"),
        ("gold_bibcodes", ["SELF", ""], "must contain non-empty strings"),
        ("gold_bibcodes", ["SELF", "SELF"], "must not contain duplicates"),
    ],
)
def test_loader_rejects_malformed_boundary_values(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    rows = [
        {
            "query": f"exact title {decile}",
            "bucket": "recall_decile",
            "discipline": "test",
            "gold_bibcodes": [f"SELF-{decile}"],
            "decile": decile,
        }
        for decile in range(10)
    ]
    rows[0][field] = value
    path = tmp_path / "gold.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_benchmark_queries(path)


def test_evaluate_config_preserves_per_query_scores_and_all_deciles() -> None:
    queries = [_query(decile, decile) for decile in range(10)]
    lanes = [
        QueryLanes(query=q, dense=[(q.gold_bibcodes[0], 1.0)], bm25=[], error=None) for q in queries
    ]

    result = evaluate_config(build_sweep()[0], lanes, k=10)

    assert len(result["per_query"]) == 10
    assert set(result["by_decile"]) == {str(decile) for decile in range(10)}
    assert all(row["query_id"].startswith("q-") for row in result["per_query"])


def test_coverage_gate_rejects_errors_instead_of_scoring_them_as_zero() -> None:
    good = QueryLanes(query=_query(0, 0), dense=[], bm25=[], error=None)
    failed = QueryLanes(query=_query(1, 0), dense=[], bm25=[], error="TimeoutError: timed out")

    with pytest.raises(RuntimeError, match="coverage 50.00% is below required 100.00%"):
        enforce_coverage_gate([good, failed], minimum=1.0)


def test_paired_bootstrap_delta_reports_difference_and_interval() -> None:
    candidate = [
        {"query_id": "a", "ndcg_at_10": 0.8},
        {"query_id": "b", "ndcg_at_10": 0.6},
    ]
    baseline = [
        {"query_id": "a", "ndcg_at_10": 0.5},
        {"query_id": "b", "ndcg_at_10": 0.5},
    ]

    uncertainty = paired_bootstrap_delta(candidate, baseline, seed=7, samples=200)

    assert uncertainty["n_paired"] == 2
    assert uncertainty["mean_delta"] == pytest.approx(0.2)
    assert uncertainty["ci95_low"] <= uncertainty["mean_delta"] <= uncertainty["ci95_high"]


def test_paired_bootstrap_delta_rejects_nonpositive_sample_count() -> None:
    with pytest.raises(ValueError, match="samples must be positive"):
        paired_bootstrap_delta([], [], samples=0)


def test_qdrant_endpoint_requires_http_and_strips_credentials_and_path() -> None:
    endpoint_with_credentials = "https://user:secret" + "@qdrant.example:6333/private?token=x"
    assert sweep._safe_qdrant_endpoint(endpoint_with_credentials) == "https://qdrant.example:6333"
    with pytest.raises(ValueError, match=r"HTTP\(S\) URL"):
        sweep._safe_qdrant_endpoint("qdrant.example:6333")


def test_render_labels_known_item_protocol_and_holdout_confirmation() -> None:
    results = {
        "dense_only": _block(0.50),
        "bm25_only": _block(0.20),
        "weighted_sum(w_dense=0.5)": _block(0.60),
    }
    md = render_markdown(results, queries_path="recall.jsonl", n_queries=1200, k=10)

    assert "known-item retrieval diagnostic" in md
    assert "held-out confirmation" in md


def test_main_runs_tuning_confirmation_and_writes_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    queries_path = tmp_path / "gold.jsonl"
    markdown_path = tmp_path / "report.md"
    json_path = tmp_path / "report.json"
    source_rows = [
        {
            "query": f"exact title {decile}-{offset}",
            "bucket": "recall_decile",
            "discipline": "test",
            "gold_bibcodes": [f"SELF-{decile}-{offset}", f"N-{decile}-{offset}"],
            "decile": decile,
        }
        for decile in range(10)
        for offset in range(2)
    ]
    queries_path.write_text("\n".join(json.dumps(row) for row in source_rows), encoding="utf-8")

    class FakeConnection:
        closed = False

        def close(self) -> None:
            self.closed = True

    connection = FakeConnection()
    monkeypatch.setattr("scix.db.get_connection", lambda: connection)

    def fake_compute_lanes(conn: object, queries: list[BenchmarkQuery], *, pool: int):
        assert conn is connection
        assert pool == 100
        return [
            QueryLanes(
                query=query,
                dense=[(query.gold_bibcodes[0], 1.0)],
                bm25=[(query.gold_bibcodes[1], 1.0)],
                error=None,
            )
            for query in queries
        ]

    provenance = {
        "gold": {"sha256": "gold-hash"},
        "git_revision": "git-sha",
        "model": {"huggingface_id": "model-id", "revision": "model-sha"},
        "qdrant": {"endpoint": "http://qdrant", "collection": "collection"},
        "corpus": {
            "database": "scix_test",
            "schema_migration": 74,
            "paper_count_estimate": 20,
        },
    }
    monkeypatch.setattr(sweep, "compute_lanes", fake_compute_lanes)
    monkeypatch.setattr(sweep, "collect_provenance", lambda conn, path: provenance)

    result = sweep.main(
        [
            "--queries",
            str(queries_path),
            "--output",
            str(markdown_path),
            "--json-output",
            str(json_path),
        ]
    )

    assert result == 0
    assert connection.closed is True
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["protocol"]["n_tuning"] == 10
    assert payload["protocol"]["n_confirmation"] == 10
    assert payload["coverage"]["observed"] == 1.0
    selected = payload["protocol"]["selected_config"]
    assert len(payload["confirmation_results"][selected]["per_query"]) == 10
    assert set(payload["confirmation_results"][selected]["by_decile"]) == {
        str(decile) for decile in range(10)
    }
    markdown = markdown_path.read_text(encoding="utf-8")
    assert "Paired uncertainty on held-out confirmation" in markdown
    assert "Gold SHA-256: `gold-hash`" in markdown


def test_main_dry_run_needs_no_gold_db_or_model(tmp_path: Path) -> None:
    markdown_path = tmp_path / "dry.md"
    json_path = tmp_path / "dry.json"

    result = sweep.main(
        [
            "--dry-run",
            "--queries",
            str(tmp_path / "absent.jsonl"),
            "--output",
            str(markdown_path),
            "--json-output",
            str(json_path),
        ]
    )

    assert result == 0
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["dry_run"] is True
    assert payload["n_queries"] == 0
    assert payload["n_configs"] == len(build_sweep())
