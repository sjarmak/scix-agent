"""Unit tests for ``scripts/eval_entity_value_props.py``.

Tests cover:
- Gold-set YAML parsing + schema validation.
- Retrieval + judge plumbing with stub backends.
- Aggregation math: per-prop mean / stderr, overall score.
- JSONL and markdown writer shapes.
- ``claude -p`` subprocess dispatcher parses the CLI's JSON envelope.
- ``subprocess.run`` is mocked — no real Claude invocation.
- CLI dry-run exit code + prop listing.
"""

from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

# `eval_entity_value_props` lives in scripts/ (pythonpath configured in
# pyproject.toml), so this import works at test time.
import eval_entity_value_props as eevp

# ---------------------------------------------------------------------------
# Gold-set loader
# ---------------------------------------------------------------------------


GOLD_YAML = """
prop: alias_expansion
description: test set
queries:
  - id: t-1
    query: "HST observations of M31"
    expectation: "Should include 'Hubble Space Telescope' papers."
    tags: [x]
  - id: t-2
    query: "JWST NIRSpec"
    expectation: "Should include full-name papers."
    alias: JWST
    canonical: James Webb Space Telescope
"""


def test_load_gold_set_parses_queries_and_extras(tmp_path: Path) -> None:
    p = tmp_path / "alias_expansion.yaml"
    p.write_text(GOLD_YAML, encoding="utf-8")

    gold = eevp.load_gold_set(p)

    assert len(gold) == 2
    assert gold[0].prop == "alias_expansion"
    assert gold[0].query_id == "t-1"
    assert gold[1].extra["alias"] == "JWST"
    assert gold[1].extra["canonical"] == "James Webb Space Telescope"


def test_load_gold_set_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        eevp.load_gold_set(tmp_path / "nope.yaml")


def test_load_gold_set_rejects_unknown_prop(tmp_path: Path) -> None:
    p = tmp_path / "bad.yaml"
    p.write_text("prop: not_a_real_prop\nqueries:\n  - id: x\n    query: y\n", encoding="utf-8")
    with pytest.raises(ValueError, match="prop"):
        eevp.load_gold_set(p)


def test_load_gold_set_rejects_empty_queries(tmp_path: Path) -> None:
    p = tmp_path / "empty.yaml"
    p.write_text("prop: alias_expansion\nqueries: []\n", encoding="utf-8")
    with pytest.raises(ValueError, match="queries"):
        eevp.load_gold_set(p)


def test_shipped_gold_sets_have_at_least_10_queries() -> None:
    """Acceptance: at least 10 curated queries per prop (6 x >=10)."""
    gold_dir = Path(__file__).resolve().parent.parent / "data" / "eval" / "entity_value_props"
    for prop in eevp.PROPS:
        yaml_path = gold_dir / f"{prop}.yaml"
        assert yaml_path.exists(), f"missing gold set {yaml_path}"
        gold = eevp.load_gold_set(yaml_path)
        assert len(gold) >= 10, f"{prop} has only {len(gold)} queries; need >= 10"


# ---------------------------------------------------------------------------
# Stubs: retrieval + judge
# ---------------------------------------------------------------------------


def _fixture_gold() -> dict[str, list[eevp.GoldQuery]]:
    return {
        "alias_expansion": [
            eevp.GoldQuery(
                prop="alias_expansion",
                query_id="a-1",
                query="HST M31",
                expectation="expand to Hubble",
            ),
            eevp.GoldQuery(
                prop="alias_expansion",
                query_id="a-2",
                query="JWST NIRSpec",
                expectation="expand to James Webb",
            ),
            eevp.GoldQuery(
                prop="alias_expansion",
                query_id="a-3",
                query="Chandra clusters",
                expectation="expand to Chandra X-ray Observatory",
            ),
        ],
        "disambiguation": [
            eevp.GoldQuery(
                prop="disambiguation",
                query_id="d-1",
                query="Hubble mission UV",
                expectation="mission sense",
            ),
        ],
    }


def test_run_eval_calls_judge_once_per_query() -> None:
    gold = _fixture_gold()
    retrieval = eevp.StubRetrievalBackend(max_docs=2)
    judge = eevp.StubJudge(fixed_score=2, rationale="ok")

    results = eevp.run_eval(gold, retrieval, judge)

    expected_n = sum(len(v) for v in gold.values())
    assert len(results) == expected_n
    assert len(judge.calls) == expected_n
    for r in results:
        assert r.retrieval_count == 2
        assert r.rubric_score == 2
        assert r.judge_rationale == "ok"


def test_run_eval_preserves_prop_order() -> None:
    gold = _fixture_gold()
    retrieval = eevp.StubRetrievalBackend()
    judge = eevp.StubJudge()

    results = eevp.run_eval(gold, retrieval, judge)
    # alias_expansion comes before disambiguation in eevp.PROPS
    seen_props = [r.prop for r in results]
    first_disambig = seen_props.index("disambiguation")
    last_alias = max(i for i, p in enumerate(seen_props) if p == "alias_expansion")
    assert last_alias < first_disambig


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def _make_results(prop: str, scores: list[int]) -> list[eevp.JudgeResult]:
    return [
        eevp.JudgeResult(
            prop=prop,
            query_id=f"{prop[:3]}-{i}",
            query=f"q{i}",
            rubric_score=s,
            judge_rationale="r",
            retrieval_count=1,
        )
        for i, s in enumerate(scores)
    ]


def test_summarize_prop_mean_and_stderr() -> None:
    results = _make_results("alias_expansion", [0, 1, 2, 3])
    summary = eevp.summarize_prop("alias_expansion", results)

    assert summary.n == 4
    assert summary.mean == pytest.approx(1.5)
    # sample stdev of [0,1,2,3] is sqrt(5/3) ≈ 1.290994 ; stderr = stdev / sqrt(4)
    expected_stderr = math.sqrt(5.0 / 3.0) / 2.0
    assert summary.stderr == pytest.approx(expected_stderr, rel=1e-9)


def test_summarize_prop_single_query_stderr_zero() -> None:
    summary = eevp.summarize_prop("alias_expansion", _make_results("alias_expansion", [2]))
    assert summary.n == 1
    assert summary.mean == pytest.approx(2.0)
    assert summary.stderr == 0.0


def test_summarize_prop_empty_is_zero() -> None:
    summary = eevp.summarize_prop("alias_expansion", [])
    assert summary.n == 0
    assert summary.mean == 0.0
    assert summary.stderr == 0.0


def test_summarize_prop_filters_by_prop() -> None:
    mixed = _make_results("alias_expansion", [1, 2, 3]) + _make_results("disambiguation", [0, 0])
    summary = eevp.summarize_prop("alias_expansion", mixed)
    assert summary.n == 3
    assert summary.mean == pytest.approx(2.0)


def test_overall_score_is_mean_of_prop_means_not_query_count_weighted() -> None:
    summaries = [
        eevp.PropSummary("alias_expansion", n=100, mean=3.0, stderr=0.0, scores=(3,) * 100),
        eevp.PropSummary("disambiguation", n=2, mean=0.0, stderr=0.0, scores=(0, 0)),
    ]
    # If this were query-weighted it would be (100*3 + 2*0)/102 = 2.94.
    # Prop-weighted = (3 + 0)/2 = 1.5.
    assert eevp.overall_score(summaries) == pytest.approx(1.5)


def test_overall_score_skips_empty_props() -> None:
    summaries = [
        eevp.PropSummary("alias_expansion", n=2, mean=3.0, stderr=0.0, scores=(3, 3)),
        eevp.PropSummary("disambiguation", n=0, mean=0.0, stderr=0.0, scores=()),
    ]
    assert eevp.overall_score(summaries) == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# JSONL + report writers
# ---------------------------------------------------------------------------


def test_write_jsonl_has_required_fields(tmp_path: Path) -> None:
    results = _make_results("alias_expansion", [2, 1])
    path = tmp_path / "out.jsonl"

    eevp.write_jsonl(path, results)

    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 2
    for row in rows:
        # Per acceptance #4: prop, query, rubric_score, judge_rationale must be present.
        assert "prop" in row
        assert "query" in row
        assert "rubric_score" in row
        assert "judge_rationale" in row


def test_render_report_includes_table_and_overall(tmp_path: Path) -> None:
    results = _make_results("alias_expansion", [2, 3])
    summaries = [eevp.summarize_prop("alias_expansion", results)]
    body = eevp.render_report(
        summaries, {"alias_expansion": results}, run_timestamp="20260421T000000"
    )

    assert "Entity Enrichment Value Props Eval" in body
    assert "Overall score:" in body
    assert "| Prop | N | Mean | StdErr |" in body
    assert "alias_expansion" in body
    assert "## alias_expansion" in body  # per-prop section
    assert "ali-0" in body  # per-query row rendered (see _make_results id scheme)


def test_write_report_creates_parent_dirs(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "dir" / "report.md"
    eevp.write_report(target, "hello\n")
    assert target.read_text(encoding="utf-8") == "hello\n"


# ---------------------------------------------------------------------------
# ClaudeSubprocessJudge — mocked subprocess.run
# ---------------------------------------------------------------------------


def _completed_process(stdout: str, returncode: int = 0, stderr: str = "") -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(
        args=["claude", "-p", "--output-format=json", "..."],
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
    )


def test_claude_subprocess_judge_parses_cli_envelope() -> None:
    envelope = {
        "type": "result",
        "result": '{"score": 3, "rationale": "Strong alias expansion — HST→Hubble hit."}',
    }
    gold = eevp.GoldQuery(
        prop="alias_expansion", query_id="a-1", query="HST M31", expectation="expand HST"
    )
    with patch("shutil.which", return_value="/bin/claude"), patch(
        "subprocess.run", return_value=_completed_process(json.dumps(envelope))
    ) as mocked_run:
        judge = eevp.ClaudeSubprocessJudge()
        score, rationale = judge.judge(gold, [eevp.RetrievalDoc("b", "T")])

    assert score == 3
    assert "HST" in rationale
    mocked_run.assert_called_once()
    # Confirm invocation shape so a reviewer can trust "--output-format=json" is in the argv.
    argv = mocked_run.call_args[0][0]
    assert argv[0] == "claude"
    assert "-p" in argv
    assert "--output-format=json" in argv


def test_claude_subprocess_judge_parses_bare_json_stdout() -> None:
    """If the CLI emits the assistant reply directly (no envelope), we still parse it."""
    raw = '{"score": 1, "rationale": "partial"}'
    gold = eevp.GoldQuery(
        prop="disambiguation", query_id="d-1", query="Hubble mission", expectation=""
    )
    with patch("shutil.which", return_value="/bin/claude"), patch(
        "subprocess.run", return_value=_completed_process(raw)
    ):
        judge = eevp.ClaudeSubprocessJudge()
        score, rationale = judge.judge(gold, [])

    assert score == 1
    assert rationale == "partial"


def test_claude_subprocess_judge_raises_on_missing_binary() -> None:
    with patch("shutil.which", return_value=None):
        with pytest.raises(FileNotFoundError, match="claude"):
            eevp.ClaudeSubprocessJudge()


def test_claude_subprocess_judge_raises_on_nonzero_exit() -> None:
    gold = eevp.GoldQuery(
        prop="alias_expansion", query_id="a-1", query="x", expectation="y"
    )
    with patch("shutil.which", return_value="/bin/claude"), patch(
        "subprocess.run", return_value=_completed_process("", returncode=2, stderr="boom")
    ):
        judge = eevp.ClaudeSubprocessJudge()
        with pytest.raises(RuntimeError, match="exited 2"):
            judge.judge(gold, [])


def test_parse_judge_stdout_rejects_out_of_range() -> None:
    with pytest.raises(RuntimeError, match="out of"):
        eevp._parse_judge_stdout('{"score": 5, "rationale": "nope"}')


def test_parse_judge_stdout_rejects_non_integer() -> None:
    with pytest.raises(RuntimeError, match="score must be int"):
        eevp._parse_judge_stdout('{"score": "high", "rationale": "nope"}')


def test_parse_judge_stdout_handles_fenced_block() -> None:
    raw = 'Here is my verdict:\n```json\n{"score": 2, "rationale": "ok"}\n```\n'
    score, rationale = eevp._parse_judge_stdout(raw)
    assert score == 2
    assert rationale == "ok"


# ---------------------------------------------------------------------------
# End-to-end: eval + JSONL + report with mocked judge
# ---------------------------------------------------------------------------


def test_end_to_end_writes_jsonl_and_markdown(tmp_path: Path) -> None:
    gold = _fixture_gold()
    retrieval = eevp.StubRetrievalBackend()
    judge = eevp.StubJudge(fixed_score=2, rationale="stubbed")

    artifact = tmp_path / "artifact.jsonl"
    jsonl_fh = artifact.open("w", encoding="utf-8")

    def _append(result: eevp.JudgeResult) -> None:
        jsonl_fh.write(
            json.dumps(
                {
                    "prop": result.prop,
                    "query": result.query_id,
                    "rubric_score": result.rubric_score,
                    "judge_rationale": result.judge_rationale,
                }
            )
            + "\n"
        )

    try:
        results = eevp.run_eval(gold, retrieval, judge, on_result=_append)
    finally:
        jsonl_fh.close()

    summaries = [eevp.summarize_prop(p, results) for p in eevp.PROPS]
    per_prop: dict[str, list[eevp.JudgeResult]] = {p: [] for p in eevp.PROPS}
    for r in results:
        per_prop[r.prop].append(r)

    report_path = tmp_path / "report.md"
    eevp.write_report(report_path, eevp.render_report(summaries, per_prop, run_timestamp="ts"))

    # JSONL assertions.
    rows = [json.loads(line) for line in artifact.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == sum(len(v) for v in gold.values())
    assert all("rubric_score" in r for r in rows)

    # Report assertions.
    body = report_path.read_text(encoding="utf-8")
    assert "Overall score" in body
    assert "alias_expansion" in body
    # Overall = mean of per-prop means. alias has 3 queries all 2.0; disambiguation has 1 query 2.0;
    # four other props have 0 queries → skipped. Overall = 2.00.
    assert "2.00" in body


# ---------------------------------------------------------------------------
# CLI — dry-run path
# ---------------------------------------------------------------------------


def test_cli_dry_run_lists_all_props(capsys: pytest.CaptureFixture[str]) -> None:
    rc = eevp.main(
        [
            "--props",
            "all",
            "--db",
            "dbname=scix_test",
            "--dry-run",
            "--gold-dir",
            str(Path(__file__).resolve().parent.parent / "data" / "eval" / "entity_value_props"),
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    for prop in eevp.PROPS:
        assert prop in out


def test_cli_resolves_short_prop_names(capsys: pytest.CaptureFixture[str]) -> None:
    rc = eevp.main(
        [
            "--props",
            "alias",
            "disambig",
            "--db",
            "dbname=scix_test",
            "--dry-run",
            "--gold-dir",
            str(Path(__file__).resolve().parent.parent / "data" / "eval" / "entity_value_props"),
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "alias_expansion" in out
    assert "disambiguation" in out
    # Not-requested props should not appear in the per-prop listing
    # (they will still appear in the "props:" header but not as "  - <prop>:").
    assert "  - type_filter:" not in out


def test_cli_rejects_unknown_prop(capsys: pytest.CaptureFixture[str]) -> None:
    # argparse's choices= validation fires first — we get SystemExit from argparse,
    # not our custom unknown-prop message.
    with pytest.raises(SystemExit):
        eevp.main(["--props", "not_a_prop"])


def test_no_anthropic_sdk_import_in_script() -> None:
    """Acceptance #3: ensure the harness never imports the paid Anthropic SDK."""
    script_path = Path(__file__).resolve().parent.parent / "scripts" / "eval_entity_value_props.py"
    source = script_path.read_text(encoding="utf-8")
    assert "import anthropic" not in source
    assert "from anthropic" not in source


# ---------------------------------------------------------------------------
# CommunityExpansionBackend — SQL pivot wired into the eval only
# ---------------------------------------------------------------------------


class _FakeCursor:
    """Minimal psycopg-like cursor that returns canned tuples per query pattern."""

    def __init__(self, handlers: list[tuple[str, Any]]) -> None:
        self._handlers = handlers
        self._last_result: list[tuple[Any, ...]] = []
        self.executed: list[tuple[str, tuple[Any, ...]]] = []

    def __enter__(self) -> "_FakeCursor":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def execute(self, sql: str, params: tuple[Any, ...] = ()) -> None:
        self.executed.append((sql, params))
        for pattern, result in self._handlers:
            if pattern in sql:
                self._last_result = result if isinstance(result, list) else [result]
                return
        self._last_result = []

    def fetchone(self) -> tuple[Any, ...] | None:
        return self._last_result[0] if self._last_result else None

    def fetchall(self) -> list[tuple[Any, ...]]:
        return list(self._last_result)


class _FakeConn:
    def __init__(self, cursor: _FakeCursor) -> None:
        self._cursor = cursor
        self.closed = False

    def cursor(self) -> _FakeCursor:
        return self._cursor

    def close(self) -> None:
        self.closed = True


def _community_query_gold(seed: str = "Hubble Space Telescope") -> eevp.GoldQuery:
    return eevp.GoldQuery(
        prop="community_expansion",
        query_id="comm-001",
        query="papers related to this community",
        expectation="should surface community siblings",
        extra={"seed_entity": seed, "community_label": "HST ecosystem"},
    )


def _non_community_gold() -> eevp.GoldQuery:
    return eevp.GoldQuery(
        prop="alias_expansion",
        query_id="a-1",
        query="HST observations",
        expectation="HST→Hubble",
    )


def test_community_backend_delegates_non_community_props() -> None:
    inner = eevp.StubRetrievalBackend(max_docs=2)
    backend = eevp.CommunityExpansionBackend(inner=inner, dsn="dbname=scix_test")
    # Do not wire _conn — delegation path must not touch the DB.
    docs = backend.retrieve(_non_community_gold(), top_k=5)
    assert len(docs) == 2
    assert docs[0].bibcode.startswith("stub.alias_expansion.")


def test_community_backend_returns_empty_when_no_seed_entity() -> None:
    inner = eevp.StubRetrievalBackend()
    backend = eevp.CommunityExpansionBackend(inner=inner, dsn="dbname=scix_test")
    gold = eevp.GoldQuery(
        prop="community_expansion",
        query_id="comm-x",
        query="q",
        expectation="e",
        extra={},  # no seed_entity
    )
    assert backend.retrieve(gold) == []


def test_community_backend_returns_empty_when_seed_unresolvable() -> None:
    inner = eevp.StubRetrievalBackend()
    cursor = _FakeCursor(
        handlers=[
            ("FROM entities e", []),  # canonical-name lookup → no match
            ("FROM entity_aliases a", []),  # alias lookup → no match
        ]
    )
    backend = eevp.CommunityExpansionBackend(inner=inner, dsn="dbname=scix_test")
    backend._conn = _FakeConn(cursor)
    assert backend.retrieve(_community_query_gold(seed="Nonexistent Mission")) == []


def test_community_backend_resolves_via_alias_when_canonical_miss() -> None:
    inner = eevp.StubRetrievalBackend()
    cursor = _FakeCursor(
        handlers=[
            ("FROM entities e", []),  # canonical miss
            ("FROM entity_aliases a", [(42, 100)]),  # alias hit → entity_id=42
            ("FROM document_entities de", [(99, 500)]),  # modal community_id=99
            (
                "FROM papers p",
                [
                    ("2024X....42...1A", "Paper 42", "abstract 42"),
                    ("2024X....42...2B", "Paper 43", ""),
                ],
            ),
        ]
    )
    backend = eevp.CommunityExpansionBackend(inner=inner, dsn="dbname=scix_test")
    backend._conn = _FakeConn(cursor)

    docs = backend.retrieve(_community_query_gold(seed="JWST"), top_k=2)
    assert len(docs) == 2
    assert docs[0].bibcode == "2024X....42...1A"
    assert docs[0].title == "Paper 42"
    # Verify the sibling query excluded the seed entity (param[0]) and used the community id (param[1]).
    siblings_call = [c for c in cursor.executed if "FROM papers p" in c[0]][0]
    assert siblings_call[1][0] == 42  # seed_entity_id
    assert siblings_call[1][1] == 99  # community_id
    assert siblings_call[1][2] == 2  # top_k


def test_community_backend_returns_empty_when_no_semantic_community() -> None:
    inner = eevp.StubRetrievalBackend()
    cursor = _FakeCursor(
        handlers=[
            ("FROM entities e", [(42, 100)]),  # canonical hit
            ("FROM document_entities de", []),  # no community assigned
        ]
    )
    backend = eevp.CommunityExpansionBackend(inner=inner, dsn="dbname=scix_test")
    backend._conn = _FakeConn(cursor)
    assert backend.retrieve(_community_query_gold()) == []


def test_community_backend_picks_highest_paper_count_on_ambiguous_canonical() -> None:
    """Two entities named 'LIGO' — prefer the one with more linked papers."""
    inner = eevp.StubRetrievalBackend()
    cursor = _FakeCursor(
        handlers=[
            # Canonical-name lookup returns the highest-paper-count row
            # by virtue of the ORDER BY n DESC LIMIT 1 — so we simulate
            # just the winning row here.
            ("FROM entities e", [(1588891, 1036)]),
            ("FROM document_entities de", [(77, 300)]),
            ("FROM papers p", [("2024LIGO..1", "GW paper", "ab")]),
        ]
    )
    backend = eevp.CommunityExpansionBackend(inner=inner, dsn="dbname=scix_test")
    backend._conn = _FakeConn(cursor)
    docs = backend.retrieve(_community_query_gold(seed="LIGO"))
    assert len(docs) == 1
    # Confirm the SQL carries an ORDER BY n DESC so multi-match rows
    # are disambiguated by paper-link count.
    canonical_sql = [c for c in cursor.executed if "FROM entities e" in c[0]][0][0]
    assert "ORDER BY n DESC" in canonical_sql


def test_community_backend_close_cascades_to_inner() -> None:
    class _Closable:
        def __init__(self) -> None:
            self.closed = False

        def retrieve(self, gold, *, top_k=10):  # pragma: no cover - unused
            return []

        def close(self) -> None:
            self.closed = True

    inner = _Closable()
    cursor = _FakeCursor(handlers=[])
    backend = eevp.CommunityExpansionBackend(inner=inner, dsn="dbname=scix_test")
    backend._conn = _FakeConn(cursor)
    backend.close()
    assert inner.closed is True
    assert backend._conn is None


# ---------------------------------------------------------------------------
# SpecificEntityBackend — entity_id-scoped retrieval wired into the eval
# ---------------------------------------------------------------------------


def _specific_query_gold(entity_name: str = "Hubble Space Telescope") -> eevp.GoldQuery:
    return eevp.GoldQuery(
        prop="specific_entity",
        query_id="spec-001",
        query="papers mentioning this specific entity",
        expectation="all results should mention entity_id=X",
        extra={"entity_name": entity_name, "entity_type": "mission"},
    )


def test_specific_backend_delegates_non_specific_props() -> None:
    inner = eevp.StubRetrievalBackend(max_docs=2)
    backend = eevp.SpecificEntityBackend(inner=inner, dsn="dbname=scix_test")
    docs = backend.retrieve(_non_community_gold(), top_k=5)
    assert len(docs) == 2
    assert docs[0].bibcode.startswith("stub.alias_expansion.")


def test_specific_backend_returns_empty_when_no_entity_name() -> None:
    inner = eevp.StubRetrievalBackend()
    backend = eevp.SpecificEntityBackend(inner=inner, dsn="dbname=scix_test")
    gold = eevp.GoldQuery(
        prop="specific_entity",
        query_id="spec-x",
        query="q",
        expectation="e",
        extra={},  # no entity_name
    )
    assert backend.retrieve(gold) == []


def test_specific_backend_returns_empty_when_entity_unresolvable() -> None:
    inner = eevp.StubRetrievalBackend()
    cursor = _FakeCursor(
        handlers=[
            # Resolver runs the union-CTE query (typed pass and fallback) — both empty.
            ("WITH candidates", []),
        ]
    )
    backend = eevp.SpecificEntityBackend(inner=inner, dsn="dbname=scix_test")
    backend._conn = _FakeConn(cursor)
    assert backend.retrieve(_specific_query_gold("Nonexistent Mission")) == []


def test_specific_backend_resolves_via_union_of_canonical_and_alias() -> None:
    """Resolver does a single union-CTE search across canonical_name and aliases."""
    inner = eevp.StubRetrievalBackend()
    cursor = _FakeCursor(
        handlers=[
            ("WITH candidates", [(42, 100)]),  # union hit -> entity_id=42
            (
                "FROM document_entities de\n                      JOIN papers p",
                [
                    ("2024X....42...1A", "Paper 42", "abstract 42"),
                    ("2024X....42...2B", "Paper 43", ""),
                ],
            ),
        ]
    )
    backend = eevp.SpecificEntityBackend(inner=inner, dsn="dbname=scix_test")
    backend._conn = _FakeConn(cursor)

    docs = backend.retrieve(_specific_query_gold("JWST"), top_k=2)
    assert len(docs) == 2
    assert docs[0].bibcode == "2024X....42...1A"
    papers_call = [
        c for c in cursor.executed if "FROM document_entities de\n                      JOIN papers p" in c[0]
    ][0]
    assert papers_call[1][0] == 42  # entity_id
    assert papers_call[1][1] == 2  # top_k


def test_specific_backend_returns_papers_filtered_by_entity_id() -> None:
    inner = eevp.StubRetrievalBackend()
    cursor = _FakeCursor(
        handlers=[
            ("WITH candidates", [(1588866, 5000)]),  # union hit, JWST-like
            (
                "FROM document_entities de\n                      JOIN papers p",
                [
                    ("2023JWST.001", "Paper A", "abs A"),
                    ("2023JWST.002", "Paper B", "abs B"),
                    ("2023JWST.003", "Paper C", ""),
                ],
            ),
        ]
    )
    backend = eevp.SpecificEntityBackend(inner=inner, dsn="dbname=scix_test")
    backend._conn = _FakeConn(cursor)
    docs = backend.retrieve(_specific_query_gold("James Webb Space Telescope"), top_k=3)
    assert [d.bibcode for d in docs] == ["2023JWST.001", "2023JWST.002", "2023JWST.003"]
    papers_sql = [
        c for c in cursor.executed if "FROM document_entities de\n                      JOIN papers p" in c[0]
    ][0][0]
    assert "pagerank DESC NULLS LAST" in papers_sql
    assert "de.entity_id = %s" in papers_sql


def test_specific_backend_resolver_applies_entity_type_hint() -> None:
    """Type hint should restrict the candidate set in the first query pass."""
    inner = eevp.StubRetrievalBackend()
    cursor = _FakeCursor(
        handlers=[
            ("WITH candidates", [(1588887, 13006)]),  # typed hit
            ("FROM document_entities de\n                      JOIN papers p", [("2024A.1", "ALMA paper", "")]),
        ]
    )
    backend = eevp.SpecificEntityBackend(inner=inner, dsn="dbname=scix_test")
    backend._conn = _FakeConn(cursor)
    backend.retrieve(_specific_query_gold("ALMA"), top_k=1)  # gold has entity_type="mission"; resolver should pass it through

    resolver_calls = [c for c in cursor.executed if "WITH candidates" in c[0]]
    # First call: typed pass with entity_type as 3rd param.
    typed_call = resolver_calls[0]
    assert "e.entity_type = %s" in typed_call[0]
    assert typed_call[1][2] == "mission"


def test_specific_backend_resolver_falls_back_when_typed_search_empty() -> None:
    """If typed search yields nothing, untyped search runs as a fallback."""
    inner = eevp.StubRetrievalBackend()
    cursor = _FakeCursor(
        handlers=[
            ("WITH candidates", []),  # both passes share the same handler match → both empty
        ]
    )
    backend = eevp.SpecificEntityBackend(inner=inner, dsn="dbname=scix_test")
    backend._conn = _FakeConn(cursor)
    backend.retrieve(_specific_query_gold("Mystery"), top_k=1)

    resolver_calls = [c for c in cursor.executed if "WITH candidates" in c[0]]
    # Two passes ran: one typed, one untyped.
    assert len(resolver_calls) == 2
    assert "e.entity_type = %s" in resolver_calls[0][0]
    assert "e.entity_type = %s" not in resolver_calls[1][0]


def test_specific_backend_close_cascades_to_inner() -> None:
    class _Closable:
        def __init__(self) -> None:
            self.closed = False

        def retrieve(self, gold, *, top_k=10):  # pragma: no cover - unused
            return []

        def close(self) -> None:
            self.closed = True

    inner = _Closable()
    cursor = _FakeCursor(handlers=[])
    backend = eevp.SpecificEntityBackend(inner=inner, dsn="dbname=scix_test")
    backend._conn = _FakeConn(cursor)
    backend.close()
    assert inner.closed is True
    assert backend._conn is None
