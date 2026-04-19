#!/usr/bin/env python3
"""Giant-component-only citation community recompute (PRD M3).

Extracts the giant component of the citation graph, runs Leiden at three
resolutions, and UPDATEs ``paper_metrics.community_id_coarse/medium/fine``.
Papers outside the giant component (isolated nodes + small-component
nodes) are UPDATEd to NULL — they are covered by the semantic (M2) and
taxonomic signals instead.

Usage::

    SCIX_TEST_DSN="dbname=scix_test" \\
    python scripts/recompute_citation_communities.py \\
        --resolution-coarse 0.001 \\
        --resolution-medium 0.01 \\
        --resolution-fine 0.1 \\
        --seed 42

Production runs must pass ``--allow-prod``.

Outputs:
    ``logs/leiden_recompute/run_meta.json`` — run metadata including the
    giant-component size, community counts, largest-community sizes, NMI
    against the semantic signal when populated, wall-clock, git SHA, and
    seed. The coarse-level ``largest_community_size <= 10% of giant`` is
    recorded as a boolean invariant.
"""

from __future__ import annotations

import argparse
import gc
import io
import json
import logging
import os
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psycopg

# Make src/scix importable when run directly from a checkout.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from scix.db import DEFAULT_DSN, is_production_dsn, redact_dsn  # noqa: E402
from scix.graph_metrics import (  # noqa: E402
    compute_leiden,
    compute_nmi,
    extract_giant_component,
    load_graph,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("recompute_citation_communities")

DEFAULT_LOG_DIR = _REPO_ROOT / "logs" / "leiden_recompute"
STAGING_TABLE_SQL = """
    CREATE TEMP TABLE tmp_citation_communities (
        bibcode TEXT PRIMARY KEY,
        community_coarse INTEGER,
        community_medium INTEGER,
        community_fine INTEGER
    ) ON COMMIT DROP
"""
NULL_STAGING_SQL = """
    CREATE TEMP TABLE tmp_null_bibcodes (
        bibcode TEXT PRIMARY KEY
    ) ON COMMIT DROP
"""
UPDATE_GIANT_SQL = """
    UPDATE paper_metrics pm
    SET community_id_coarse = tc.community_coarse,
        community_id_medium = tc.community_medium,
        community_id_fine   = tc.community_fine,
        updated_at = NOW()
    FROM tmp_citation_communities tc
    WHERE pm.bibcode = tc.bibcode
"""
UPDATE_NULL_SQL = """
    UPDATE paper_metrics pm
    SET community_id_coarse = NULL,
        community_id_medium = NULL,
        community_id_fine   = NULL,
        updated_at = NOW()
    FROM tmp_null_bibcodes t
    WHERE pm.bibcode = t.bibcode
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_dsn(cli_dsn: str | None) -> str:
    """Resolve effective DSN: CLI flag > SCIX_TEST_DSN > SCIX_DSN/default."""
    if cli_dsn:
        return cli_dsn
    test_dsn = os.environ.get("SCIX_TEST_DSN")
    if test_dsn:
        return test_dsn
    return DEFAULT_DSN


def _git_sha() -> str | None:
    """Return current git SHA of the repo, or None if unavailable."""
    try:
        out = subprocess.run(  # noqa: S603  — fixed argv, not user input
            ["git", "rev-parse", "HEAD"],
            cwd=str(_REPO_ROOT),
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None
    return None


def _summarize_membership(
    membership: list[int],
    giant_i2b: dict[int, str],
) -> tuple[int, int, dict[str, int]]:
    """Return (n_communities, largest_community_size, {bibcode: cid})."""
    counts = Counter(membership)
    n_comm = len(counts)
    largest = max(counts.values()) if counts else 0
    by_bib = {giant_i2b[vid]: cid for vid, cid in enumerate(membership)}
    return n_comm, largest, by_bib


def _copy_giant_rows(
    conn: psycopg.Connection,
    giant_i2b: dict[int, str],
    coarse: list[int],
    medium: list[int],
    fine: list[int],
) -> None:
    """COPY (bibcode, coarse, medium, fine) tuples into tmp_citation_communities."""
    buf = io.StringIO()
    for vid, bib in giant_i2b.items():
        buf.write(f"{bib}\t{coarse[vid]}\t{medium[vid]}\t{fine[vid]}\n")
    buf.seek(0)
    with conn.cursor() as cur:
        with cur.copy(
            "COPY tmp_citation_communities "
            "(bibcode, community_coarse, community_medium, community_fine) FROM STDIN"
        ) as copy:
            while chunk := buf.read(8192):
                copy.write(chunk.encode("utf-8"))


def _copy_null_rows(conn: psycopg.Connection, bibcodes: set[str]) -> None:
    """COPY bibcode rows into tmp_null_bibcodes."""
    if not bibcodes:
        return
    buf = io.StringIO()
    for bib in bibcodes:
        buf.write(f"{bib}\n")
    buf.seek(0)
    with conn.cursor() as cur:
        with cur.copy("COPY tmp_null_bibcodes (bibcode) FROM STDIN") as copy:
            while chunk := buf.read(8192):
                copy.write(chunk.encode("utf-8"))


def _compute_nmi_vs_semantic(
    conn: psycopg.Connection,
    giant_bib_to_community: dict[str, int],
) -> float | None:
    """Compute NMI of the given giant-component membership vs community_semantic_coarse.

    Returns None if the semantic signal is unpopulated for every
    giant-component paper (M2 not yet run).
    """
    if not giant_bib_to_community:
        return None

    bibcodes = list(giant_bib_to_community.keys())

    # Fetch semantic-coarse labels for all giant-component bibcodes.
    with conn.cursor() as cur:
        cur.execute(
            "SELECT bibcode, community_semantic_coarse "
            "FROM paper_metrics "
            "WHERE bibcode = ANY(%s) "
            "  AND community_semantic_coarse IS NOT NULL",
            (bibcodes,),
        )
        rows = cur.fetchall()

    if not rows:
        return None

    citation_labels: list[int] = []
    semantic_labels: list[int] = []
    for bib, sem in rows:
        citation_labels.append(giant_bib_to_community[bib])
        semantic_labels.append(int(sem))

    return compute_nmi(citation_labels, semantic_labels)


# ---------------------------------------------------------------------------
# Main flow
# ---------------------------------------------------------------------------


def run(
    dsn: str,
    resolution_coarse: float,
    resolution_medium: float,
    resolution_fine: float,
    seed: int,
    log_dir: Path,
) -> dict[str, Any]:
    """Execute the recompute pipeline and return the run_meta dict.

    The returned dict is also written to ``<log_dir>/run_meta.json``.
    """
    t0 = time.perf_counter()
    started_at = datetime.now(timezone.utc).isoformat()
    log_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Connecting to %s", redact_dsn(dsn))
    conn = psycopg.connect(dsn)
    try:
        # 1. Load full citation graph.
        # keep_bibcode_to_id=False drops the 32M-entry bibcode→vid dict inside
        # load_graph() once edges are resolved — it is not referenced after
        # that point here, and retaining it pushes peak RSS into OOM territory
        # during igraph.Graph() construction on the 32M/298M prod graph.
        logger.info("Loading citation graph...")
        graph, b2i, i2b = load_graph(conn, keep_bibcode_to_id=False)

        # 2. Count degree-0 nodes for run_meta (pure metric; does not mutate graph).
        #    This avoids the memory cost of materialising an intermediate subgraph
        #    via filter_isolated_nodes(): extract_giant_component() already lumps
        #    degree-0 nodes into small_bibcodes, and downstream _copy_null_rows
        #    unions isolated + small anyway — so the filter step is redundant.
        logger.info("Counting isolated nodes...")
        n_isolated = sum(1 for d in graph.degree() if d == 0)
        logger.info("Found %d isolated nodes", n_isolated)

        # 3. Extract giant component directly from full graph.
        logger.info("Extracting giant component...")
        giant, giant_b2i, giant_i2b, small_bibcodes = extract_giant_component(
            graph, b2i, i2b
        )
        # Free the full-graph representation as soon as possible — peak RSS during
        # induced_subgraph() + giant_i2b construction was what OOM'd on prod (32M
        # nodes / 298M edges).
        del graph, b2i, i2b
        gc.collect()

        giant_n = giant.vcount()
        logger.info(
            "Giant component: %d nodes (isolated=%d, non-giant total=%d)",
            giant_n,
            n_isolated,
            len(small_bibcodes),
        )

        # 4. Leiden at each resolution.
        if giant_n == 0:
            logger.warning("Giant component is empty — skipping Leiden")
            coarse: list[int] = []
            medium: list[int] = []
            fine: list[int] = []
        else:
            logger.info("Running Leiden (coarse=%.6f)...", resolution_coarse)
            coarse = compute_leiden(giant, resolution=resolution_coarse, seed=seed)
            logger.info("Running Leiden (medium=%.6f)...", resolution_medium)
            medium = compute_leiden(giant, resolution=resolution_medium, seed=seed)
            logger.info("Running Leiden (fine=%.6f)...", resolution_fine)
            fine = compute_leiden(giant, resolution=resolution_fine, seed=seed)

        n_coarse, largest_coarse, coarse_by_bib = _summarize_membership(coarse, giant_i2b)
        n_medium, largest_medium, _ = _summarize_membership(medium, giant_i2b)
        n_fine, largest_fine, _ = _summarize_membership(fine, giant_i2b)

        largest_coarse_pct = (
            (largest_coarse / giant_n * 100.0) if giant_n > 0 else 0.0
        )
        invariant_largest_coarse_ok = largest_coarse_pct <= 10.0 or giant_n == 0

        # 5. Stage + UPDATE paper_metrics.
        # Must run inside a single transaction so the TEMP tables survive
        # long enough to drive the UPDATEs. ``load_graph`` left the
        # connection in an idle-in-transaction state (server-side cursor);
        # commit it to start a clean transaction for the writes.
        conn.commit()
        with conn.transaction():
            with conn.cursor() as cur:
                cur.execute(STAGING_TABLE_SQL)
                cur.execute(NULL_STAGING_SQL)
            if giant_n > 0:
                _copy_giant_rows(conn, giant_i2b, coarse, medium, fine)
            _copy_null_rows(conn, small_bibcodes)
            with conn.cursor() as cur:
                cur.execute(UPDATE_GIANT_SQL)
                giant_updated = cur.rowcount
                cur.execute(UPDATE_NULL_SQL)
                null_updated = cur.rowcount
        logger.info(
            "Updated paper_metrics: %d giant-component rows, %d NULL rows",
            giant_updated,
            null_updated,
        )

        # 6. NMI vs semantic signal (may be None if M2 not yet populated).
        try:
            nmi_coarse = _compute_nmi_vs_semantic(conn, coarse_by_bib)
            nmi_medium = _compute_nmi_vs_semantic(
                conn, {giant_i2b[vid]: cid for vid, cid in enumerate(medium)}
            )
            nmi_fine = _compute_nmi_vs_semantic(
                conn, {giant_i2b[vid]: cid for vid, cid in enumerate(fine)}
            )
        except psycopg.Error as exc:
            logger.warning("NMI computation failed: %s", exc)
            nmi_coarse = nmi_medium = nmi_fine = None

        if nmi_coarse is None:
            logger.warning(
                "community_semantic_coarse not populated — skipping NMI"
            )
    finally:
        conn.close()

    wall_clock = time.perf_counter() - t0
    run_meta: dict[str, Any] = {
        "run_id": started_at,
        "started_at": started_at,
        "git_sha": _git_sha(),
        "seed": seed,
        "partition_type": "modularity",
        "resolutions": {
            "coarse": resolution_coarse,
            "medium": resolution_medium,
            "fine": resolution_fine,
        },
        "giant_component_size": giant_n,
        "isolated_node_count": n_isolated,
        "small_component_node_count": len(small_bibcodes) - n_isolated,
        "n_communities": {
            "coarse": n_coarse,
            "medium": n_medium,
            "fine": n_fine,
        },
        "largest_community_size": {
            "coarse": largest_coarse,
            "medium": largest_medium,
            "fine": largest_fine,
        },
        "largest_coarse_pct_of_giant": round(largest_coarse_pct, 4),
        "nmi_vs_semantic": {
            "coarse": nmi_coarse,
            "medium": nmi_medium,
            "fine": nmi_fine,
        },
        "wall_clock_seconds": round(wall_clock, 3),
        "invariants": {
            "largest_coarse_le_10pct": invariant_largest_coarse_ok,
        },
    }

    out_path = log_dir / "run_meta.json"
    out_path.write_text(json.dumps(run_meta, indent=2, sort_keys=True) + "\n")
    logger.info("Wrote run meta: %s", out_path)

    return run_meta


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resolution-coarse", type=float, default=0.001)
    parser.add_argument("--resolution-medium", type=float, default=0.01)
    parser.add_argument("--resolution-fine", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--dsn",
        default=None,
        help="PostgreSQL DSN (defaults to SCIX_TEST_DSN env, then SCIX_DSN).",
    )
    parser.add_argument(
        "--allow-prod",
        action="store_true",
        help="Required to run against production DSN (dbname=scix).",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=DEFAULT_LOG_DIR,
        help="Directory for run_meta.json (default: logs/leiden_recompute/).",
    )
    args = parser.parse_args(argv)

    dsn = _resolve_dsn(args.dsn)

    if is_production_dsn(dsn) and not args.allow_prod:
        logger.error(
            "Refusing to write to production DSN %s — pass --allow-prod to override",
            redact_dsn(dsn),
        )
        return 2

    run_meta = run(
        dsn=dsn,
        resolution_coarse=args.resolution_coarse,
        resolution_medium=args.resolution_medium,
        resolution_fine=args.resolution_fine,
        seed=args.seed,
        log_dir=args.log_dir,
    )

    logger.info(
        "Summary: giant=%d, n_comm=(%d, %d, %d), largest_coarse=%d (%.2f%%)",
        run_meta["giant_component_size"],
        run_meta["n_communities"]["coarse"],
        run_meta["n_communities"]["medium"],
        run_meta["n_communities"]["fine"],
        run_meta["largest_community_size"]["coarse"],
        run_meta["largest_coarse_pct_of_giant"],
    )
    if not run_meta["invariants"]["largest_coarse_le_10pct"]:
        logger.error(
            "INVARIANT FAILED: largest coarse community is %.2f%% of giant "
            "(> 10%% limit)",
            run_meta["largest_coarse_pct_of_giant"],
        )
        return 1

    logger.info("Recompute complete in %.1fs", run_meta["wall_clock_seconds"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
