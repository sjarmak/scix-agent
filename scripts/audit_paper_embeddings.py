#!/usr/bin/env python3
"""Read-only pre-flight for the INDUS ``paper_embeddings`` offload (ADR-015).

Produces the go/no-go evidence the operator needs *before* dropping the INDUS
HNSW indexes (Stage 1) or reclaiming the INDUS rows (Stage 2):

  * PG INDUS row count vs the Qdrant collection ``points_count`` — the derived
    cache must fully cover the source before the source's indexes are dropped.
  * exact reclaimable bytes per INDUS index (the Stage-1 disk yield) and an
    estimate for the row/column footprint (the Stage-2 yield).
  * presence/age of a NAS archive of the INDUS rows + a Qdrant snapshot on NAS.
  * the soak clock (days since the 2026-06-11 ADR-013 cutover) so the rollback
    hold is explicit.

This script **only reads**. It issues ``SELECT``/``pg_*_size`` calls and one
read-only Qdrant ``GET``; it never writes PG, Qdrant, or NAS. It is gated by
``is_production_dsn`` + ``--allow-prod`` and offers an optional
``--require-batch-scope`` self-check per repo convention (it is lightweight —
a handful of COUNT/size queries — so the scope check is opt-in, not forced).

Usage::

    # against scix_test (schema only, no data) — CI / smoke
    SCIX_TEST_DSN=dbname=scix_test python scripts/audit_paper_embeddings.py --dsn dbname=scix_test

    # against prod, read-only, under scix-batch
    QDRANT_URL=http://127.0.0.1:6633 scix-batch \
        python scripts/audit_paper_embeddings.py --allow-prod --require-batch-scope

See: docs/ADR/015_offload_drop_paper_embeddings_indus.md,
     docs/prd/qdrant_nas_migration.md, results/qdrant_ram_relief_options.md
"""

from __future__ import annotations

import argparse
import datetime as _dt
import logging
import os
import pathlib
import sys
from dataclasses import dataclass

import psycopg

from scix.db import DEFAULT_DSN, get_connection, is_production_dsn, redact_dsn

logger = logging.getLogger(__name__)

# --- Constants tied to ADR-013 / ADR-015 ------------------------------------
INDUS_MODEL = "indus"
PROD_QDRANT_COLLECTION = "scix_indus_v2_papers_s1"  # search.py _QDRANT_DENSE_COLLECTIONS["indus"]
INDUS_INDEXES = ("idx_embed_hnsw_indus", "idx_embed_hnsw_indus_hv")
CUTOVER_DATE = _dt.date(2026, 6, 11)  # ADR-013 dense-lane flip to production
SOAK_DAYS = 30  # PRD qdrant_nas_migration.md: pgvector retained ≥30 days post-cutover

# NAS archive locations the audit looks for (existence + mtime only; never written here).
NAS_ROW_ARCHIVE_DIR = pathlib.Path("/mnt/scix_offload/paper_embeddings_indus_archive")
NAS_QDRANT_SNAPSHOT_DIR = pathlib.Path("/mnt/qdrant_snapshots") / PROD_QDRANT_COLLECTION

# Read-only safety: bound every query so a planner surprise can't pin the box.
_STMT_TIMEOUT_MS = int(os.environ.get("SCIX_AUDIT_STMT_TIMEOUT_MS", "120000"))


# ---------------------------------------------------------------------------
# Soak clock (pure, tested)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SoakClock:
    today: _dt.date
    cutover: _dt.date
    soak_days: int

    @property
    def days_elapsed(self) -> int:
        return (self.today - self.cutover).days

    @property
    def clears_on(self) -> _dt.date:
        return self.cutover + _dt.timedelta(days=self.soak_days)

    @property
    def satisfied(self) -> bool:
        return self.days_elapsed >= self.soak_days

    @property
    def days_remaining(self) -> int:
        return max(0, self.soak_days - self.days_elapsed)


def soak_clock(
    today: _dt.date, cutover: _dt.date = CUTOVER_DATE, soak_days: int = SOAK_DAYS
) -> SoakClock:
    return SoakClock(today=today, cutover=cutover, soak_days=soak_days)


# ---------------------------------------------------------------------------
# PG facts (read-only)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class IndexFact:
    name: str
    exists: bool
    bytes: int  # 0 when absent


@dataclass(frozen=True)
class PgFacts:
    row_counts_by_model: dict[str, int]
    indus_rows: int
    table_total_bytes: int  # pg_total_relation_size(paper_embeddings): heap+toast+all indexes
    indus_indexes: list[IndexFact]
    indus_row_bytes_estimate: int  # embedding + embedding_hv column bytes for indus rows (estimate)

    @property
    def stage1_reclaim_bytes(self) -> int:
        """Bytes freed to the OS by Stage 1 (DROP INDEX) — exact, immediate."""
        return sum(ix.bytes for ix in self.indus_indexes if ix.exists)

    @property
    def stage2_reclaim_bytes_estimate(self) -> int:
        """Additional bytes recoverable by Stage 2 (row/column reclaim) — estimate."""
        return self.indus_row_bytes_estimate


_COUNT_BY_MODEL_SQL = """
    SELECT model_name, count(*)
    FROM paper_embeddings
    GROUP BY model_name
"""

# Sampled average column size × indus count. Sampling keeps this read cheap on a
# 32M-row table; the result is explicitly reported as an estimate, not a measurement.
_INDUS_ROW_BYTES_SQL = """
    WITH sample AS (
        SELECT
            COALESCE(pg_column_size(embedding), 0)    AS emb_sz,
            COALESCE(pg_column_size(embedding_hv), 0) AS hv_sz
        FROM paper_embeddings
        WHERE model_name = %(model)s
        LIMIT %(sample)s
    )
    SELECT
        COALESCE(avg(emb_sz), 0) + COALESCE(avg(hv_sz), 0) AS avg_row_bytes,
        count(*)                                           AS sampled
    FROM sample
"""


def _index_bytes(conn: psycopg.Connection, index_name: str) -> IndexFact:
    """Exact on-disk size of an index, or exists=False when it is absent."""
    with conn.cursor() as cur:
        cur.execute("SELECT to_regclass(%s)", (index_name,))
        if cur.fetchone()[0] is None:
            return IndexFact(name=index_name, exists=False, bytes=0)
        cur.execute("SELECT pg_relation_size(%s)", (index_name,))
        return IndexFact(name=index_name, exists=True, bytes=int(cur.fetchone()[0]))


def fetch_pg_facts(conn: psycopg.Connection, *, sample: int = 5000) -> PgFacts:
    """Collect all read-only PG facts for the report (no writes)."""
    with conn.cursor() as cur:
        cur.execute(f"SET statement_timeout = {_STMT_TIMEOUT_MS}")

        cur.execute(_COUNT_BY_MODEL_SQL)
        counts = {str(model): int(n) for model, n in cur.fetchall()}

        cur.execute("SELECT pg_total_relation_size('paper_embeddings')")
        table_total = int(cur.fetchone()[0])

        cur.execute(_INDUS_ROW_BYTES_SQL, {"model": INDUS_MODEL, "sample": sample})
        avg_row_bytes, sampled = cur.fetchone()

    indus_rows = counts.get(INDUS_MODEL, 0)
    row_bytes_estimate = int(round(float(avg_row_bytes) * indus_rows)) if sampled else 0
    indexes = [_index_bytes(conn, name) for name in INDUS_INDEXES]

    return PgFacts(
        row_counts_by_model=counts,
        indus_rows=indus_rows,
        table_total_bytes=table_total,
        indus_indexes=indexes,
        indus_row_bytes_estimate=row_bytes_estimate,
    )


# ---------------------------------------------------------------------------
# Qdrant facts (read-only) + NAS archive presence
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class QdrantFacts:
    available: bool
    collection: str
    points_count: int | None
    reason: str = ""


def fetch_qdrant_facts(
    collection: str, *, url: str | None = None, timeout: float = 30.0
) -> QdrantFacts:
    """Read the Qdrant collection ``points_count`` (read-only GET).

    Never raises: an unreachable/unconfigured Qdrant degrades to
    ``available=False`` with a reason, so the PG-side report still renders and
    the verdict simply withholds the parity check.
    """
    url = url or os.environ.get("QDRANT_URL")
    if not url:
        return QdrantFacts(
            available=False, collection=collection, points_count=None, reason="QDRANT_URL not set"
        )
    try:
        from qdrant_client import QdrantClient
    except ImportError:
        return QdrantFacts(
            available=False,
            collection=collection,
            points_count=None,
            reason="qdrant-client not installed",
        )
    try:
        client = QdrantClient(url=url, timeout=timeout)
        info = client.get_collection(collection)
    except Exception as exc:  # noqa: BLE001 — read-only probe; report, don't crash
        return QdrantFacts(
            available=False,
            collection=collection,
            points_count=None,
            reason=f"{type(exc).__name__}: {exc}",
        )
    return QdrantFacts(
        available=True, collection=collection, points_count=int(info.points_count or 0)
    )


@dataclass(frozen=True)
class NasArtifact:
    label: str
    path: pathlib.Path
    exists: bool
    newest_mtime: _dt.datetime | None
    file_count: int


def nas_artifact_status(label: str, path: pathlib.Path) -> NasArtifact:
    """Report presence/age of a NAS artifact dir (read-only stat; never writes)."""
    if not path.exists():
        return NasArtifact(label=label, path=path, exists=False, newest_mtime=None, file_count=0)
    files = [p for p in path.rglob("*") if p.is_file()]
    newest = None
    if files:
        ts = max(p.stat().st_mtime for p in files)
        newest = _dt.datetime.fromtimestamp(ts)
    return NasArtifact(
        label=label, path=path, exists=True, newest_mtime=newest, file_count=len(files)
    )


# ---------------------------------------------------------------------------
# Verdict + report (pure, tested)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Verdict:
    parity_ok: bool | None  # None = could not check (Qdrant unavailable)
    soak_satisfied: bool
    stage1_ready: bool
    notes: list[str]


def compute_verdict(pg: PgFacts, qd: QdrantFacts, clock: SoakClock) -> Verdict:
    notes: list[str] = []

    if not qd.available or qd.points_count is None:
        parity_ok: bool | None = None
        notes.append(f"PG↔Qdrant parity UNVERIFIED ({qd.reason or 'qdrant unavailable'}).")
    else:
        parity_ok = pg.indus_rows == qd.points_count
        if not parity_ok:
            notes.append(
                f"PG↔Qdrant COUNT MISMATCH: pg indus rows={pg.indus_rows:,} "
                f"vs qdrant points={qd.points_count:,} (Δ={pg.indus_rows - qd.points_count:+,}). "
                "Stage 1 must NOT proceed until the derived cache covers the source."
            )

    if not clock.satisfied:
        notes.append(
            f"SOAK HOLD ACTIVE: day {clock.days_elapsed} of {clock.soak_days} "
            f"(clears {clock.clears_on.isoformat()}, {clock.days_remaining} days remaining). "
            "Stage 1 is blocked absent documented evidence-based early-retirement sign-off "
            "(see ADR-015 §soak hold)."
        )

    stage1_ready = bool(parity_ok) and clock.satisfied
    return Verdict(
        parity_ok=parity_ok,
        soak_satisfied=clock.satisfied,
        stage1_ready=stage1_ready,
        notes=notes,
    )


def _gib(n: int) -> str:
    return f"{n / 1024**3:.1f} GiB"


def render_report(
    pg: PgFacts, qd: QdrantFacts, clock: SoakClock, nas: list[NasArtifact], verdict: Verdict
) -> str:
    lines: list[str] = []
    lines.append("# paper_embeddings INDUS offload — audit pre-flight (ADR-015)\n")
    lines.append(
        f"- Generated for cutover {clock.cutover.isoformat()}, " f"as-of {clock.today.isoformat()}"
    )
    lines.append("")

    verdict_word = "READY" if verdict.stage1_ready else "NOT READY"
    lines.append(f"## VERDICT (Stage 1, drop INDUS indexes): **{verdict_word}**\n")
    if verdict.notes:
        for note in verdict.notes:
            lines.append(f"- {note}")
    else:
        lines.append("- All Stage-1 preconditions satisfied.")
    lines.append("")

    lines.append("## Soak clock (ADR-013 / PRD 30-day hold)\n")
    lines.append(
        f"- cutover: {clock.cutover.isoformat()} · day {clock.days_elapsed} of "
        f"{clock.soak_days} · clears {clock.clears_on.isoformat()} · "
        f"satisfied: **{clock.satisfied}**"
    )
    lines.append("")

    lines.append("## PG ↔ Qdrant parity\n")
    lines.append(f"- PG `paper_embeddings` rows where model_name='indus': **{pg.indus_rows:,}**")
    if qd.available:
        lines.append(f"- Qdrant `{qd.collection}` points_count: **{qd.points_count:,}**")
        lines.append(f"- parity: **{verdict.parity_ok}**")
    else:
        lines.append(f"- Qdrant `{qd.collection}`: unavailable ({qd.reason}) — parity UNVERIFIED")
    other = {m: n for m, n in pg.row_counts_by_model.items() if m != INDUS_MODEL}
    if other:
        lines.append(
            "- other models in table (out of scope, retained): "
            + ", ".join(f"{m}={n:,}" for m, n in sorted(other.items()))
        )
    lines.append("")

    lines.append("## Reclaimable bytes\n")
    lines.append(
        f"- `paper_embeddings` total (heap+toast+all indexes): **{_gib(pg.table_total_bytes)}**"
    )
    lines.append("- **Stage 1 — DROP INDEX (exact, freed to OS at commit, no VACUUM FULL):**")
    for ix in pg.indus_indexes:
        state = _gib(ix.bytes) if ix.exists else "absent"
        lines.append(f"  - `{ix.name}`: {state}")
    lines.append(f"  - **Stage-1 total reclaim: {_gib(pg.stage1_reclaim_bytes)}**")
    lines.append(
        f"- Stage 2 — row/column reclaim (estimate; needs pg_repack/VACUUM FULL): "
        f"~{_gib(pg.stage2_reclaim_bytes_estimate)}"
    )
    lines.append("")

    lines.append("## NAS archives (read-only stat)\n")
    for art in nas:
        if art.exists:
            age = art.newest_mtime.isoformat(timespec="seconds") if art.newest_mtime else "no files"
            lines.append(
                f"- {art.label}: present at `{art.path}` ({art.file_count} files, newest {age})"
            )
        else:
            lines.append(f"- {art.label}: **absent** at `{art.path}`")
    lines.append("")
    lines.append(
        "> Stage 1 keeps the INDUS rows, so it needs no row archive — the in-PG rows "
        "remain the rollback source. Stage 2 (row reclaim) requires a verified NAS row "
        "archive before it may proceed. See ADR-015."
    )
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run_audit(
    conn: psycopg.Connection, *, collection: str, today: _dt.date, sample: int = 5000
) -> tuple[str, Verdict]:
    pg = fetch_pg_facts(conn, sample=sample)
    qd = fetch_qdrant_facts(collection)
    clock = soak_clock(today)
    nas = [
        nas_artifact_status("INDUS row archive", NAS_ROW_ARCHIVE_DIR),
        nas_artifact_status("Qdrant snapshot", NAS_QDRANT_SNAPSHOT_DIR),
    ]
    verdict = compute_verdict(pg, qd, clock)
    return render_report(pg, qd, clock, nas, verdict), verdict


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dsn", default=DEFAULT_DSN, help="PostgreSQL DSN (read-only)")
    parser.add_argument(
        "--allow-prod",
        action="store_true",
        help="Required to run against a production DSN (read-only, but gated).",
    )
    parser.add_argument(
        "--require-batch-scope",
        action="store_true",
        help="Refuse to run unless invoked under systemd-run scope "
        "(SYSTEMD_SCOPE in env); per CLAUDE.md convention. Optional — "
        "this script is read-only and lightweight.",
    )
    parser.add_argument(
        "--collection",
        default=PROD_QDRANT_COLLECTION,
        help=f"Qdrant collection to compare (default: {PROD_QDRANT_COLLECTION}).",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=5000,
        help="Row sample for the Stage-2 byte estimate (default: 5000).",
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=None,
        help="Optional path to write the markdown report (also printed to stdout).",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if is_production_dsn(args.dsn) and not args.allow_prod:
        logger.error(
            "Refusing to run against production DSN %s — pass --allow-prod (read-only).",
            redact_dsn(args.dsn),
        )
        return 2

    if args.require_batch_scope and "SYSTEMD_SCOPE" not in os.environ:
        sys.stderr.write(
            "ERROR: --require-batch-scope set but SYSTEMD_SCOPE not in environment.\n"
            "       Run via: scix-batch python scripts/audit_paper_embeddings.py ...\n"
        )
        return 2

    today = _dt.date.today()
    conn = get_connection(args.dsn)
    try:
        report, verdict = run_audit(
            conn, collection=args.collection, today=today, sample=args.sample
        )
    finally:
        conn.close()

    print(report)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report, encoding="utf-8")
        logger.info("wrote %s", args.output)

    # Exit 0 always (read-only pre-flight). The READY/NOT-READY verdict is in the
    # report; a non-green parity/soak state is a finding, not a script failure.
    return 0


if __name__ == "__main__":
    sys.exit(main())
