"""Tests for the three-lane retrieval health prober."""

from __future__ import annotations

import pathlib
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import ANY, Mock

import psycopg
import pytest

from tests.helpers import get_test_dsn

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import check_retrieval_health as crh  # noqa: E402


def _search_result(
    *,
    lexical_ms: float = 2.0,
    body_ms: float = 3.0,
    vector_ms: float = 4.0,
    dropped: list[str] | None = None,
):
    metadata = {} if dropped is None else {"dropped_lanes": dropped}
    return SimpleNamespace(
        papers=[{"bibcode": "2020ApJ...known"}],
        timing_ms={
            "lexical_ms": lexical_ms,
            "body_lexical_ms": body_ms,
            "vector_ms": vector_ms,
        },
        metadata=metadata,
    )


class _CollectionInfo:
    def __init__(self, status: object = "green", points_count: int = 1000) -> None:
        self.status = status
        self.points_count = points_count


def test_probe_exercises_all_lanes_and_collection() -> None:
    search_fn = Mock(return_value=_search_result())
    qdrant = Mock()
    qdrant.get_collection.return_value = _CollectionInfo()

    results = crh.probe_retrieval(
        object(),
        [0.1] * 768,
        qdrant,
        query="galaxy formation",
        expected_points=998,
        point_tolerance=5,
        search_fn=search_fn,
    )

    assert all(result.ok for result in results)
    assert [result.name for result in results] == [
        "lexical",
        "body_bm25",
        "dense",
        "qdrant_collection",
    ]
    search_fn.assert_called_once_with(
        ANY,
        "galaxy formation",
        [0.1] * 768,
        model_name="indus",
        include_body=True,
    )
    qdrant.get_collection.assert_called_once_with(crh.INDUS_COLLECTION)


@pytest.mark.parametrize(
    ("timing_key", "result_name"),
    [
        ("lexical_ms", "lexical"),
        ("body_lexical_ms", "body_bm25"),
        ("vector_ms", "dense"),
    ],
)
def test_zero_lane_timing_fails(timing_key: str, result_name: str) -> None:
    timings = {"lexical_ms": 2.0, "body_lexical_ms": 3.0, "vector_ms": 4.0}
    timings[timing_key] = 0.0
    search = _search_result(
        lexical_ms=timings["lexical_ms"],
        body_ms=timings["body_lexical_ms"],
        vector_ms=timings["vector_ms"],
    )
    qdrant = Mock()
    qdrant.get_collection.return_value = _CollectionInfo()

    results = crh.probe_retrieval(
        object(),
        [0.1] * 768,
        qdrant,
        query="galaxy formation",
        expected_points=1000,
        point_tolerance=0,
        search_fn=Mock(return_value=search),
    )

    failed = {result.name: result for result in results if not result.ok}
    assert result_name in failed
    assert "must be > 0" in failed[result_name].detail


def test_dropped_lane_fails_even_when_timings_are_nonzero() -> None:
    qdrant = Mock()
    qdrant.get_collection.return_value = _CollectionInfo()
    results = crh.probe_retrieval(
        object(),
        [0.1] * 768,
        qdrant,
        query="galaxy formation",
        expected_points=1000,
        point_tolerance=0,
        search_fn=Mock(return_value=_search_result(dropped=["body_bm25"])),
    )
    by_name = {result.name: result for result in results}
    assert by_name["body_bm25"].ok is False
    assert "dropped_lanes" in by_name["body_bm25"].detail


@pytest.mark.parametrize(
    ("status", "points", "expected", "tolerance", "detail"),
    [
        ("yellow", 1000, 1000, 0, "status='yellow'"),
        ("green", 989, 1000, 10, "outside tolerance"),
    ],
)
def test_collection_health_failures(
    status: str, points: int, expected: int, tolerance: int, detail: str
) -> None:
    qdrant = Mock()
    qdrant.get_collection.return_value = _CollectionInfo(status, points)
    results = crh.probe_retrieval(
        object(),
        [0.1] * 768,
        qdrant,
        query="galaxy formation",
        expected_points=expected,
        point_tolerance=tolerance,
        search_fn=Mock(return_value=_search_result()),
    )
    collection = results[-1]
    assert collection.ok is False
    assert detail in collection.detail


def test_status_enum_value_is_normalized() -> None:
    qdrant = Mock()
    qdrant.get_collection.return_value = _CollectionInfo(SimpleNamespace(value="green"), 1000)
    results = crh.probe_retrieval(
        object(),
        [0.1] * 768,
        qdrant,
        query="galaxy formation",
        expected_points=1000,
        point_tolerance=0,
        search_fn=Mock(return_value=_search_result()),
    )
    assert results[-1].ok is True


def test_render_is_cron_readable() -> None:
    text = crh.render(
        [
            crh.ProbeResult("lexical", True, "2.0ms"),
            crh.ProbeResult("dense", False, "timing 0ms"),
        ]
    )
    assert "PASS lexical: 2.0ms" in text
    assert "FAIL dense: timing 0ms" in text
    assert "retrieval health: 1/2 checks FAILED" in text


class _ConnectionContext:
    def __init__(self, connection: object) -> None:
        self.connection = connection

    def __enter__(self) -> object:
        return self.connection

    def __exit__(self, *exc: object) -> None:
        return None


def _patch_successful_main(monkeypatch: pytest.MonkeyPatch, results: list[crh.ProbeResult]) -> None:
    monkeypatch.setenv("QDRANT_URL", "http://qdrant.test:6333")
    monkeypatch.setattr(crh, "_embed_query", Mock(return_value=[0.1] * 768))
    monkeypatch.setattr(crh, "get_connection", Mock(return_value=_ConnectionContext(object())))
    monkeypatch.setattr(crh, "query_expected_points", Mock(return_value=1000))
    monkeypatch.setattr(crh, "dense_client", Mock(return_value=object()))
    monkeypatch.setattr(crh, "probe_retrieval", Mock(return_value=results))


def test_main_refuses_production_without_explicit_flag() -> None:
    assert crh.main(["--dsn", "dbname=scix"]) == 2


def test_main_refuses_missing_qdrant(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("QDRANT_URL", raising=False)
    assert crh.main(["--dsn", "dbname=scix_test"]) == 2


def test_main_refuses_negative_point_tolerance() -> None:
    assert crh.main(["--dsn", "dbname=scix_test", "--point-tolerance", "-1"]) == 2


def test_main_returns_zero_for_healthy_probe(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    results = [crh.ProbeResult("lexical", True, "lexical_ms=2.00ms")]
    _patch_successful_main(monkeypatch, results)
    assert crh.main(["--dsn", "dbname=scix_test"]) == 0
    assert "PASS lexical" in capsys.readouterr().out


def test_main_returns_one_for_breach(monkeypatch: pytest.MonkeyPatch) -> None:
    results = [crh.ProbeResult("dense", False, "vector_ms=0ms")]
    _patch_successful_main(monkeypatch, results)
    assert crh.main(["--dsn", "dbname=scix_test"]) == 1


def test_main_reuses_distinct_notification_channel(monkeypatch: pytest.MonkeyPatch) -> None:
    results = [crh.ProbeResult("dense", False, "vector_ms=0ms")]
    _patch_successful_main(monkeypatch, results)
    notify = Mock(return_value="created")
    monkeypatch.setattr(crh.pipeline_health, "notify", notify)
    assert crh.main(["--dsn", "dbname=scix_test", "--notify"]) == 1
    assert notify.call_args.kwargs["label"] == "retrieval-health"
    assert notify.call_args.kwargs["title"] == crh.NOTIFY_TITLE
    assert "check_retrieval_health.py" in notify.call_args.kwargs["reproduce_command"]


def test_main_returns_three_when_notification_channel_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    results = [crh.ProbeResult("dense", False, "vector_ms=0ms")]
    _patch_successful_main(monkeypatch, results)
    monkeypatch.setattr(
        crh.pipeline_health,
        "notify",
        Mock(side_effect=crh.pipeline_health.NotifyError("unreachable")),
    )
    report_failure = Mock()
    monkeypatch.setattr(crh.pipeline_health, "report_notification_failure", report_failure)
    assert crh.main(["--dsn", "dbname=scix_test", "--notify"]) == 3
    report_failure.assert_called_once()


def test_main_notifies_on_probe_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("QDRANT_URL", "http://qdrant.test:6333")
    monkeypatch.setattr(crh, "_embed_query", Mock(side_effect=RuntimeError("model missing")))
    notify = Mock(return_value="created")
    monkeypatch.setattr(crh.pipeline_health, "notify", notify)

    assert crh.main(["--dsn", "dbname=scix_test", "--notify"]) == 1

    alert_results = notify.call_args.args[0]
    assert [(result.name, result.ok) for result in alert_results] == [("probe_execution", False)]
    assert "model missing" in alert_results[0].detail


def test_main_notifies_when_qdrant_is_unconfigured(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("QDRANT_URL", raising=False)
    notify = Mock(return_value="created")
    monkeypatch.setattr(crh.pipeline_health, "notify", notify)

    assert crh.main(["--dsn", "dbname=scix_test", "--notify"]) == 1

    alert_results = notify.call_args.args[0]
    assert [(result.name, result.ok) for result in alert_results] == [("probe_execution", False)]
    assert "QDRANT_URL is unset" in alert_results[0].detail


def test_embed_query_uses_indus_mean_pooling(monkeypatch: pytest.MonkeyPatch) -> None:
    import scix.embed

    load_model = Mock(return_value=("model", "tokenizer"))
    embed_batch = Mock(return_value=[[0.5] * 768])
    monkeypatch.setattr(scix.embed, "load_model", load_model)
    monkeypatch.setattr(scix.embed, "embed_batch", embed_batch)
    assert crh._embed_query("galaxy formation") == [0.5] * 768
    load_model.assert_called_once_with("indus", device="cpu")
    embed_batch.assert_called_once_with(
        "model", "tokenizer", ["galaxy formation"], batch_size=1, pooling="mean"
    )


@pytest.mark.integration
def test_expected_point_count_query_executes_against_test_schema() -> None:
    dsn = get_test_dsn()
    if dsn is None:
        pytest.skip("SCIX_TEST_DSN not set")
    with psycopg.connect(dsn) as conn:
        row = conn.execute("select to_regclass('public.indus_qdrant_synced')").fetchone()
        if row is None or row[0] is None:
            pytest.skip("indus_qdrant_synced is absent from the test schema")
        assert crh.query_expected_points(conn) >= 0


def test_cli_help_e2e() -> None:
    completed = subprocess.run(
        [sys.executable, str(SCRIPTS_DIR / "check_retrieval_health.py"), "--help"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert completed.returncode == 0
    assert "three retrieval lanes" in completed.stdout
