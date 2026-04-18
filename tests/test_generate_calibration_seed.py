"""Unit tests for scripts/generate_calibration_seed.py.

Covers:
- YAML loading (happy path + validation errors)
- Seed row generation with stub dispatcher + stub candidate source
- CSV schema stability + atomic write
- needs_human_review flag serialization
- Failed dispatcher calls are dropped from CSV but counted in stats
"""

from __future__ import annotations

import asyncio
import csv
import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

# Import the script module by path so we don't need to add scripts/ to sys.path
# in production contexts.
_SCRIPT_PATH = REPO_ROOT / "scripts" / "generate_calibration_seed.py"
_spec = importlib.util.spec_from_file_location("generate_calibration_seed", _SCRIPT_PATH)
assert _spec is not None and _spec.loader is not None
gcs = importlib.util.module_from_spec(_spec)
sys.modules["generate_calibration_seed"] = gcs
_spec.loader.exec_module(gcs)


from scix.eval.persona_judge import (  # noqa: E402
    JudgeScore,
    JudgeTriple,
    StubDispatcher,
)

# ---------------------------------------------------------------------------
# YAML loading
# ---------------------------------------------------------------------------


class TestLoadQuerySpecs:
    def test_loads_real_calibration_yaml(self) -> None:
        """The shipped calibration_queries.yaml must always load cleanly."""
        specs, top_k = gcs.load_query_specs(gcs.DEFAULT_QUERIES_YAML)
        assert len(specs) == 50
        assert top_k >= 1
        lanes = {s.lane for s in specs}
        assert lanes == {
            "alias",
            "ontology",
            "disambiguation",
            "entity_type",
            "specific_entity",
            "community",
        }

    def test_rejects_missing_queries_key(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.yaml"
        bad.write_text("version: 1\n")
        with pytest.raises(ValueError, match="queries"):
            gcs.load_query_specs(bad)

    def test_rejects_duplicate_ids(self, tmp_path: Path) -> None:
        bad = tmp_path / "dup.yaml"
        bad.write_text(
            "version: 1\n"
            "queries:\n"
            "  - {id: a, lane: alias, query: q1}\n"
            "  - {id: a, lane: alias, query: q2}\n"
        )
        with pytest.raises(ValueError, match="duplicate"):
            gcs.load_query_specs(bad)

    def test_rejects_missing_fields(self, tmp_path: Path) -> None:
        bad = tmp_path / "partial.yaml"
        bad.write_text("version: 1\nqueries:\n  - {id: a, lane: alias}\n")
        with pytest.raises(ValueError, match="query"):
            gcs.load_query_specs(bad)


# ---------------------------------------------------------------------------
# generate_seed_rows with stubs
# ---------------------------------------------------------------------------


class TestGenerateSeedRows:
    def _specs(self) -> list[gcs.QuerySpec]:
        return [
            gcs.QuerySpec(id="q1", lane="alias", query="HST brown dwarfs"),
            gcs.QuerySpec(id="q2", lane="ontology", query="M-type asteroids"),
        ]

    def test_happy_path_rows_match_candidates(self) -> None:
        source = gcs.StubCandidateSource(per_query=3)
        dispatcher = StubDispatcher(fixed_score=2, reason="stub")

        rows, stats = asyncio.run(
            gcs.generate_seed_rows(
                specs=self._specs(),
                top_k=3,
                candidate_source=source,
                dispatcher=dispatcher,
            )
        )

        assert stats.n_queries == 2
        assert stats.n_candidates == 6  # 2 queries * 3
        assert stats.n_scored == 6
        assert stats.n_failed == 0
        assert len(rows) == 6
        for row in rows:
            assert row.draft_score == 2
            assert row.lane in {"alias", "ontology"}
            assert row.bibcode.startswith("STUB")
            assert row.snippet_preview
            # Stub doesn't set needs_human_review, so it stays None.
            assert row.needs_human_review is None

    def test_lane_counts_track_queries_not_candidates(self) -> None:
        source = gcs.StubCandidateSource(per_query=5)
        dispatcher = StubDispatcher(fixed_score=1)

        _, stats = asyncio.run(
            gcs.generate_seed_rows(
                specs=self._specs(),
                top_k=5,
                candidate_source=source,
                dispatcher=dispatcher,
            )
        )
        assert stats.lanes == {"alias": 1, "ontology": 1}

    def test_failed_scores_are_dropped_and_counted(self) -> None:
        class _AlwaysFail:
            async def judge(self, triple: JudgeTriple) -> JudgeScore:
                return JudgeScore(score=-1, reason="boom", triple=triple)

        rows, stats = asyncio.run(
            gcs.generate_seed_rows(
                specs=self._specs(),
                top_k=2,
                candidate_source=gcs.StubCandidateSource(per_query=2),
                dispatcher=_AlwaysFail(),
            )
        )
        assert rows == []
        assert stats.n_failed == 4
        assert stats.n_scored == 0

    def test_preserves_needs_human_review_from_dispatcher(self) -> None:
        class _Flagging:
            async def judge(self, triple: JudgeTriple) -> JudgeScore:
                # Flag every odd index
                flag = triple.bibcode.endswith("1")
                return JudgeScore(score=2, reason="", needs_human_review=flag)

        rows, _ = asyncio.run(
            gcs.generate_seed_rows(
                specs=[gcs.QuerySpec(id="q1", lane="alias", query="x")],
                top_k=3,
                candidate_source=gcs.StubCandidateSource(per_query=3),
                dispatcher=_Flagging(),
            )
        )
        flags = [r.needs_human_review for r in rows]
        assert any(flags) and any(f is False for f in flags)


# ---------------------------------------------------------------------------
# CSV write
# ---------------------------------------------------------------------------


class TestWriteSeedCsv:
    def _row(self, **overrides) -> gcs.SeedRow:
        defaults = {
            "query_id": "q1",
            "lane": "alias",
            "query": "HST brown dwarfs",
            "bibcode": "2024ABC",
            "title": "A paper",
            "draft_score": 2,
            "needs_human_review": False,
            "snippet_preview": "Title: A paper Abstract: ...",
        }
        defaults.update(overrides)
        return gcs.SeedRow(**defaults)

    def test_writes_header_and_rows(self, tmp_path: Path) -> None:
        path = tmp_path / "seed.csv"
        gcs.write_seed_csv(path, [self._row(), self._row(bibcode="2024DEF")])
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            assert reader.fieldnames == list(gcs.CSV_COLUMNS)
            rows = list(reader)
        assert len(rows) == 2
        assert rows[0]["bibcode"] == "2024ABC"
        assert rows[0]["needs_human_review"] == "false"
        assert rows[0]["draft_score"] == "2"

    def test_serializes_none_review_as_empty(self, tmp_path: Path) -> None:
        path = tmp_path / "seed.csv"
        gcs.write_seed_csv(path, [self._row(needs_human_review=None)])
        with path.open("r", encoding="utf-8", newline="") as f:
            row = next(csv.DictReader(f))
        assert row["needs_human_review"] == ""

    def test_serializes_true_review_flag(self, tmp_path: Path) -> None:
        path = tmp_path / "seed.csv"
        gcs.write_seed_csv(path, [self._row(needs_human_review=True)])
        with path.open("r", encoding="utf-8", newline="") as f:
            row = next(csv.DictReader(f))
        assert row["needs_human_review"] == "true"

    def test_csv_injection_quoting(self, tmp_path: Path) -> None:
        """Query strings with commas / quotes / newlines must round-trip."""
        tricky = 'query with, comma and "quoted" term\nnewline'
        path = tmp_path / "seed.csv"
        gcs.write_seed_csv(path, [self._row(query=tricky)])
        with path.open("r", encoding="utf-8", newline="") as f:
            row = next(csv.DictReader(f))
        assert row["query"] == tricky

    def test_atomic_write_overwrites_existing(self, tmp_path: Path) -> None:
        path = tmp_path / "seed.csv"
        gcs.write_seed_csv(path, [self._row(bibcode="A")])
        gcs.write_seed_csv(path, [self._row(bibcode="B")])
        with path.open("r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 1
        assert rows[0]["bibcode"] == "B"
        # No leftover tmp files.
        tmps = list(tmp_path.glob(".seed.csv.*.tmp"))
        assert tmps == []

    def test_creates_parent_directory(self, tmp_path: Path) -> None:
        path = tmp_path / "nested" / "dir" / "seed.csv"
        gcs.write_seed_csv(path, [self._row()])
        assert path.exists()
