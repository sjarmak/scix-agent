"""Tests for scripts/audit_paper_embeddings.py (ADR-015 pre-flight).

Pure-function tests only (soak clock, verdict, report rendering) — they need
no DB and still cover real logic.

The integration leg was removed on 2026-07-28: it ran the audit against a
database that still had ``paper_embeddings``, and ADR-015 dropped that table
(migration 074 records the retirement). The script audits readiness to perform
a drop that has already happened, so there is nothing left for it to inspect;
retiring the script itself is tracked separately.
"""

from __future__ import annotations

import datetime as _dt
import pathlib
import sys

# scripts/ is not a package — make it importable like the other script tests.
REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import audit_paper_embeddings as ape  # noqa: E402

# ---------------------------------------------------------------------------
# Soak clock — pure
# ---------------------------------------------------------------------------


class TestSoakClock:
    def test_day_11_not_satisfied(self) -> None:
        clock = ape.soak_clock(_dt.date(2026, 6, 22))  # 11 days after 2026-06-11
        assert clock.days_elapsed == 11
        assert clock.clears_on == _dt.date(2026, 7, 11)
        assert clock.satisfied is False
        assert clock.days_remaining == 19

    def test_day_30_boundary_satisfied(self) -> None:
        clock = ape.soak_clock(_dt.date(2026, 7, 11))
        assert clock.days_elapsed == 30
        assert clock.satisfied is True
        assert clock.days_remaining == 0

    def test_well_past_soak(self) -> None:
        clock = ape.soak_clock(_dt.date(2026, 9, 1))
        assert clock.satisfied is True
        assert clock.days_remaining == 0


# ---------------------------------------------------------------------------
# Reclaim accounting + verdict — pure
# ---------------------------------------------------------------------------


def _pg(
    indus_rows: int = 32_000_000,
    idx_bytes: tuple[int, int] = (120 * 1024**3, 30 * 1024**3),
    row_estimate: int = 50 * 1024**3,
    others: dict | None = None,
) -> ape.PgFacts:
    counts = {"indus": indus_rows, **(others or {"specter2": 20_000})}
    indexes = [
        ape.IndexFact("idx_embed_hnsw_indus", idx_bytes[0] > 0, idx_bytes[0]),
        ape.IndexFact("idx_embed_hnsw_indus_hv", idx_bytes[1] > 0, idx_bytes[1]),
    ]
    return ape.PgFacts(
        row_counts_by_model=counts,
        indus_rows=indus_rows,
        table_total_bytes=250 * 1024**3,
        indus_indexes=indexes,
        indus_row_bytes_estimate=row_estimate,
    )


class TestReclaim:
    def test_stage1_sums_present_indexes_only(self) -> None:
        pg = _pg(idx_bytes=(120 * 1024**3, 0))  # halfvec index absent
        assert pg.stage1_reclaim_bytes == 120 * 1024**3

    def test_stage2_is_the_row_estimate(self) -> None:
        pg = _pg(row_estimate=42 * 1024**3)
        assert pg.stage2_reclaim_bytes_estimate == 42 * 1024**3


class TestVerdict:
    def test_parity_ok_and_soak_satisfied_is_ready(self) -> None:
        pg = _pg(indus_rows=32_000_000)
        qd = ape.QdrantFacts(available=True, collection="c", points_count=32_000_000)
        clock = ape.soak_clock(_dt.date(2026, 9, 1))
        v = ape.compute_verdict(pg, qd, clock)
        assert v.parity_ok is True
        assert v.soak_satisfied is True
        assert v.stage1_ready is True
        assert v.notes == []

    def test_soak_active_blocks_even_with_parity(self) -> None:
        pg = _pg(indus_rows=32_000_000)
        qd = ape.QdrantFacts(available=True, collection="c", points_count=32_000_000)
        clock = ape.soak_clock(_dt.date(2026, 6, 22))  # mid-soak
        v = ape.compute_verdict(pg, qd, clock)
        assert v.parity_ok is True
        assert v.stage1_ready is False
        assert any("SOAK HOLD ACTIVE" in n for n in v.notes)

    def test_count_mismatch_blocks(self) -> None:
        pg = _pg(indus_rows=32_000_000)
        qd = ape.QdrantFacts(available=True, collection="c", points_count=31_999_000)
        clock = ape.soak_clock(_dt.date(2026, 9, 1))
        v = ape.compute_verdict(pg, qd, clock)
        assert v.parity_ok is False
        assert v.stage1_ready is False
        assert any("MISMATCH" in n for n in v.notes)

    def test_qdrant_unavailable_withholds_parity(self) -> None:
        pg = _pg()
        qd = ape.QdrantFacts(
            available=False, collection="c", points_count=None, reason="QDRANT_URL not set"
        )
        clock = ape.soak_clock(_dt.date(2026, 9, 1))
        v = ape.compute_verdict(pg, qd, clock)
        assert v.parity_ok is None
        assert v.stage1_ready is False
        assert any("UNVERIFIED" in n for n in v.notes)


class TestRender:
    def test_report_contains_verdict_and_sections(self) -> None:
        pg = _pg()
        qd = ape.QdrantFacts(
            available=True, collection="scix_indus_v2_papers_s1", points_count=32_000_000
        )
        clock = ape.soak_clock(_dt.date(2026, 6, 22))
        nas = [
            ape.NasArtifact("INDUS row archive", pathlib.Path("/mnt/x"), False, None, 0),
            ape.NasArtifact("Qdrant snapshot", pathlib.Path("/mnt/y"), False, None, 0),
        ]
        v = ape.compute_verdict(pg, qd, clock)
        report = ape.render_report(pg, qd, clock, nas, v)
        assert "VERDICT (Stage 1" in report
        assert "NOT READY" in report  # mid-soak
        assert "Soak clock" in report
        assert "Reclaimable bytes" in report
        assert "idx_embed_hnsw_indus" in report
        assert "specter2" in report  # other models reported as retained/out-of-scope


# ---------------------------------------------------------------------------
# NAS stat — pure (tmp path stands in for /mnt)
# ---------------------------------------------------------------------------


class TestNasStatus:
    def test_absent_dir(self, tmp_path: pathlib.Path) -> None:
        art = ape.nas_artifact_status("x", tmp_path / "nope")
        assert art.exists is False
        assert art.file_count == 0
        assert art.newest_mtime is None

    def test_present_dir_with_file(self, tmp_path: pathlib.Path) -> None:
        (tmp_path / "a.zst").write_text("data")
        art = ape.nas_artifact_status("x", tmp_path)
        assert art.exists is True
        assert art.file_count == 1
        assert art.newest_mtime is not None
