#!/usr/bin/env python3
"""Semantic community assignment via minibatch k-means on INDUS embeddings.

Streams INDUS vectors from ``paper_embeddings WHERE model_name='indus'`` using
a server-side cursor, runs ``sklearn.cluster.MiniBatchKMeans`` at three
resolutions (coarse/medium/fine), stages per-resolution assignments into a
TEMP table, then updates ``paper_metrics.community_semantic_{coarse,medium,fine}``
in a single pass per resolution.

Memory: 32M rows x 768 dims x float32 = ~98 GB uncompressed. We NEVER
materialize the full set — both the training pass (``partial_fit``) and the
prediction pass (``predict``) consume the cursor in bounded batches of
``batch_size`` rows. Silhouette score is computed on a stratified sample
(<= ``silhouette_sample_size`` points) held in RAM only.

Safety: writes are blocked against production DSNs unless ``--allow-prod`` is
passed. Default DSN resolves from ``SCIX_TEST_DSN`` so forgetting to set the
test env variable does not silently mutate production.

Usage:
    SCIX_TEST_DSN=dbname=scix_test \\
    python scripts/compute_semantic_communities.py \\
        --k-coarse 20 --k-medium 200 --k-fine 2000 --seed 42

Outputs:
    results/semantic_communities.json       — silhouette + wall-clock + RSS
    logs/semantic_communities/run_meta.json — run metadata for audit
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import logging
import math
import os
import resource
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Optional, Sequence

import numpy as np
import psycopg
from pgvector.psycopg import register_vector

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from scix.db import is_production_dsn, redact_dsn  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("compute_semantic_communities")


RESULTS_DIR = REPO_ROOT / "results"
LOGS_DIR = REPO_ROOT / "logs" / "semantic_communities"

DEFAULT_SILHOUETTE_SAMPLE = 10_000
RESOLUTION_NAMES: tuple[str, ...] = ("coarse", "medium", "fine")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ResolutionSpec:
    """One clustering resolution: a label + target number of clusters."""

    name: str  # one of coarse/medium/fine
    k: int


@dataclass
class ResolutionResult:
    """Per-resolution training + prediction outputs."""

    name: str
    k: int
    n_rows: int
    wall_clock_s: float
    silhouette: Optional[float]
    sample_n: int


# ---------------------------------------------------------------------------
# Vector parsing helpers
# ---------------------------------------------------------------------------


def _parse_pgvector_text(text: str) -> np.ndarray:
    """Parse pgvector text form ``'[f1,f2,...]'`` into a 1-D float32 ndarray."""
    inner = text.strip()
    if inner.startswith("[") and inner.endswith("]"):
        inner = inner[1:-1]
    return np.fromstring(inner, sep=",", dtype=np.float32)


# ---------------------------------------------------------------------------
# Server-side cursor streaming
# ---------------------------------------------------------------------------


_STREAM_SQL_BASE = (
    "SELECT bibcode, embedding FROM paper_embeddings "
    "WHERE model_name = 'indus'"
)

def _build_stream_sql(row_limit: Optional[int]) -> str:
    """Return the stream SQL optionally capped by ``row_limit``.

    ``row_limit`` is intended for validation / small-scale test runs. In
    production leave it as ``None`` to process the full corpus.
    """
    if row_limit is not None and row_limit > 0:
        return f"{_STREAM_SQL_BASE} LIMIT {int(row_limit)}"
    return _STREAM_SQL_BASE


# Intentionally no ORDER BY: on 32M rows x 768 dims the sort
# materializes the full result in backend memory (~tens of GB) before
# streaming. The staging TEMP table uses bibcode as primary key so row
# order does not affect correctness of the bulk UPDATE.


def _iter_indus_batches(
    conn: psycopg.Connection,
    batch_size: int,
    cursor_name: str,
    row_limit: Optional[int] = None,
) -> Iterator[tuple[list[str], np.ndarray]]:
    """Yield ``(bibcodes, vectors)`` batches from ``paper_embeddings``.

    Uses a server-side cursor so the driver never buffers more than
    ``batch_size`` rows at once. ``vectors`` is a ``(len(bibcodes), 768)``
    ``float32`` matrix.

    Requires ``pgvector.psycopg.register_vector`` to have been called on
    the connection so embeddings arrive as numpy arrays via the binary
    pgvector wire format. Text-form parsing was ~100x slower at 32M-row
    scale.

    The connection must be in ``autocommit=False`` mode — psycopg3 named
    cursors refuse to execute in autocommit.
    """
    if conn.autocommit:
        raise RuntimeError(
            "Server-side cursor requires autocommit=False; caller must wrap "
            "this iterator in a read-only transaction"
        )

    with conn.cursor(name=cursor_name) as cur:
        cur.itersize = batch_size
        cur.execute(_build_stream_sql(row_limit))

        batch_bibs: list[str] = []
        batch_vecs: list[np.ndarray] = []
        for bibcode, emb in cur:
            if emb is None:
                continue
            batch_bibs.append(bibcode)
            # np.array (not np.asarray) forces a copy — detaches the
            # ndarray from the libpq result buffer so the buffer can be
            # freed on the next FETCH instead of being retained for the
            # lifetime of the iteration.
            batch_vecs.append(np.array(emb, dtype=np.float32, copy=True))
            if len(batch_bibs) >= batch_size:
                yield batch_bibs, np.vstack(batch_vecs)
                batch_bibs = []
                batch_vecs = []
        if batch_bibs:
            yield batch_bibs, np.vstack(batch_vecs)


# ---------------------------------------------------------------------------
# Silhouette reservoir — stratified by cluster id
# ---------------------------------------------------------------------------


class _SilhouetteReservoir:
    """Stratified reservoir of (vector, label) pairs for silhouette_score.

    Keeps up to ``per_cluster_cap`` samples per cluster id. After all batches
    have been fed, the reservoir can be materialized via ``materialize()``.
    """

    def __init__(
        self,
        k: int,
        total_cap: int,
        seed: int,
    ) -> None:
        self._k = k
        self._total_cap = total_cap
        self._per_cluster_cap = max(1, math.ceil(total_cap / max(k, 1)))
        self._rng = np.random.default_rng(seed)
        self._buckets: dict[int, list[np.ndarray]] = {}

    def extend(self, vecs: np.ndarray, labels: np.ndarray) -> None:
        for vec, lbl in zip(vecs, labels):
            cid = int(lbl)
            bucket = self._buckets.setdefault(cid, [])
            if len(bucket) < self._per_cluster_cap:
                bucket.append(np.asarray(vec, dtype=np.float32))
            else:
                # Reservoir replacement so later batches aren't under-represented
                j = int(self._rng.integers(0, self._per_cluster_cap * 2))
                if j < self._per_cluster_cap:
                    bucket[j] = np.asarray(vec, dtype=np.float32)

    def materialize(self) -> tuple[np.ndarray, np.ndarray]:
        vecs: list[np.ndarray] = []
        labels: list[int] = []
        for cid, bucket in self._buckets.items():
            vecs.extend(bucket)
            labels.extend([cid] * len(bucket))
        if not vecs:
            return np.zeros((0, 1), dtype=np.float32), np.zeros((0,), dtype=np.int32)
        return np.vstack(vecs), np.asarray(labels, dtype=np.int32)


# ---------------------------------------------------------------------------
# Staging table helpers
# ---------------------------------------------------------------------------


_STAGING_NAME = "_semcomm_staging"


def _create_staging(conn: psycopg.Connection) -> None:
    """Create the per-resolution staging TEMP table. ON COMMIT DROP."""
    with conn.cursor() as cur:
        cur.execute(
            f"CREATE TEMP TABLE IF NOT EXISTS {_STAGING_NAME} ("
            f"  bibcode TEXT PRIMARY KEY,"
            f"  cluster_id INT NOT NULL"
            f") ON COMMIT DROP"
        )
        cur.execute(f"TRUNCATE {_STAGING_NAME}")


def _copy_assignments(
    conn: psycopg.Connection,
    bibcodes: Sequence[str],
    labels: np.ndarray,
) -> None:
    """COPY a batch of (bibcode, cluster_id) into the staging table."""
    if len(bibcodes) == 0:
        return
    with conn.cursor() as cur:
        with cur.copy(f"COPY {_STAGING_NAME} (bibcode, cluster_id) FROM STDIN") as cp:
            for bib, lbl in zip(bibcodes, labels):
                cp.write_row((bib, int(lbl)))


def _merge_resolution(conn: psycopg.Connection, resolution_name: str) -> int:
    """Update paper_metrics.community_semantic_<name> from staging.

    Performs one UPDATE covering existing paper_metrics rows, followed by an
    INSERT covering bibcodes that do not yet have a paper_metrics row. Both
    statements are driven by the staging table so we never enumerate the full
    embedding set in Python.
    """
    column = f"community_semantic_{resolution_name}"
    with conn.cursor() as cur:
        cur.execute(
            f"UPDATE paper_metrics "
            f"   SET {column} = s.cluster_id, "
            f"       updated_at = now() "
            f"  FROM {_STAGING_NAME} s "
            f" WHERE paper_metrics.bibcode = s.bibcode"
        )
        updated = cur.rowcount

        cur.execute(
            f"INSERT INTO paper_metrics (bibcode, {column}) "
            f"SELECT s.bibcode, s.cluster_id "
            f"  FROM {_STAGING_NAME} s "
            f"  LEFT JOIN paper_metrics pm ON pm.bibcode = s.bibcode "
            f" WHERE pm.bibcode IS NULL "
            f"ON CONFLICT (bibcode) DO UPDATE SET "
            f"  {column} = EXCLUDED.{column}, "
            f"  updated_at = now()"
        )
        inserted = cur.rowcount
    return (updated or 0) + (inserted or 0)


# ---------------------------------------------------------------------------
# One resolution — train, predict, stage, merge
# ---------------------------------------------------------------------------


def _run_resolution(
    conn: psycopg.Connection,
    spec: ResolutionSpec,
    seed: int,
    batch_size: int,
    silhouette_sample: int,
    train_max_rows: int,
    row_limit: Optional[int] = None,
) -> ResolutionResult:
    """Train MiniBatchKMeans, predict assignments, stage + merge into paper_metrics."""
    # Local import keeps the CLI fast for --help and makes missing-sklearn
    # failures point at the install target rather than module import time.
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.metrics import silhouette_score

    logger.info(
        "Resolution %s: training MiniBatchKMeans k=%d seed=%d batch_size=%d",
        spec.name,
        spec.k,
        seed,
        batch_size,
    )
    t0 = time.perf_counter()

    kmeans = MiniBatchKMeans(
        n_clusters=spec.k,
        random_state=seed,
        batch_size=batch_size,
        n_init=1,
        reassignment_ratio=0.01,
        max_no_improvement=None,
    )

    # --- Pass 1: training via partial_fit over streamed batches ---
    # Caller must have set autocommit=False before calling this function.
    # MiniBatchKMeans converges long before 32M samples; cap the training
    # pass at ``train_max_rows`` and use the fitted model to predict
    # labels over the full corpus in pass 2.
    carry_vecs: Optional[np.ndarray] = None
    n_rows_train = 0
    with conn.transaction():
        for _bibs, vecs in _iter_indus_batches(
            conn, batch_size, f"sc_{spec.name}_train", row_limit=row_limit
        ):
            n_rows_train += len(_bibs)
            if carry_vecs is not None:
                vecs = np.vstack([carry_vecs, vecs])
                carry_vecs = None
            if len(vecs) < spec.k:
                # Too few samples to run partial_fit — carry into next batch.
                carry_vecs = vecs
                continue
            kmeans.partial_fit(vecs)
            if n_rows_train >= train_max_rows:
                logger.info(
                    "Resolution %s: training cap reached at %d rows (>= %d) — "
                    "proceeding to predict pass",
                    spec.name, n_rows_train, train_max_rows,
                )
                break
        if carry_vecs is not None and len(carry_vecs) > 0:
            # Final tail: pad by re-fitting alongside already-seen centroids.
            # partial_fit refuses n_samples < n_clusters, so we only proceed
            # when the tail is large enough; otherwise the tail is lost to
            # training (the already-fitted model still predicts them).
            if len(carry_vecs) >= spec.k:
                kmeans.partial_fit(carry_vecs)
    if n_rows_train == 0:
        logger.warning(
            "Resolution %s: no INDUS embeddings found — skipping", spec.name
        )
        return ResolutionResult(
            name=spec.name,
            k=spec.k,
            n_rows=0,
            wall_clock_s=time.perf_counter() - t0,
            silhouette=None,
            sample_n=0,
        )

    # --- Pass 2: predict + COPY to staging, one transaction per resolution ---
    reservoir = _SilhouetteReservoir(
        k=spec.k, total_cap=silhouette_sample, seed=seed
    )
    n_rows_predict = 0
    with conn.transaction():
        _create_staging(conn)
        batch_counter = 0
        for bibs, vecs in _iter_indus_batches(
            conn, batch_size, f"sc_{spec.name}_predict", row_limit=row_limit
        ):
            labels = kmeans.predict(vecs)
            _copy_assignments(conn, bibs, labels)
            reservoir.extend(vecs, labels)
            n_rows_predict += len(bibs)
            # Drop references to the batch numpy arrays so the next
            # iteration's allocations can reuse the freed pages. Without
            # this the interpreter hangs on to each batch until the
            # next reassignment at the top of the for-loop, which is
            # too late when the allocator has already expanded the
            # python heap.
            del labels, vecs, bibs
            batch_counter += 1
            if batch_counter % 50 == 0:
                gc.collect()
        n_merged = _merge_resolution(conn, spec.name)
        logger.info(
            "Resolution %s: merged %d paper_metrics rows from %d staged",
            spec.name,
            n_merged,
            n_rows_predict,
        )

    # --- Silhouette on stratified sample ---
    sample_vecs, sample_labels = reservoir.materialize()
    silhouette: Optional[float] = None
    if len(sample_vecs) >= 2 and len(np.unique(sample_labels)) >= 2:
        # ``silhouette_score`` needs at least 2 clusters; guard against the
        # pathological case where every sampled point landed in one cluster.
        silhouette = float(
            silhouette_score(sample_vecs, sample_labels, metric="cosine")
        )

    wall_clock = time.perf_counter() - t0
    logger.info(
        "Resolution %s: done in %.2fs — silhouette=%s sample_n=%d",
        spec.name,
        wall_clock,
        f"{silhouette:.4f}" if silhouette is not None else "n/a",
        len(sample_vecs),
    )
    return ResolutionResult(
        name=spec.name,
        k=spec.k,
        n_rows=n_rows_predict,
        wall_clock_s=wall_clock,
        silhouette=silhouette,
        sample_n=int(len(sample_vecs)),
    )


# ---------------------------------------------------------------------------
# Observability — git sha, RSS, run metadata
# ---------------------------------------------------------------------------


def _git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        pass
    return "unknown"


def _peak_rss_mb() -> float:
    """Read peak resident set size in MB. Linux-accurate via /proc/self/status."""
    try:
        with open("/proc/self/status", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("VmHWM:"):
                    # Format: "VmHWM:\t    12345 kB"
                    kb = int(line.split()[1])
                    return kb / 1024.0
    except OSError:
        pass
    # Fallback: getrusage returns KB on Linux, bytes on BSD/macOS.
    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return ru / (1024.0 * 1024.0)
    return ru / 1024.0


def _count_indus_rows(conn: psycopg.Connection) -> int:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT COUNT(*) FROM paper_embeddings WHERE model_name = 'indus'"
        )
        row = cur.fetchone()
        return int(row[0]) if row else 0


# ---------------------------------------------------------------------------
# DSN resolution + safety gate
# ---------------------------------------------------------------------------


def _resolve_dsn(cli_dsn: Optional[str]) -> str:
    """Resolve the effective DSN.

    Precedence: ``--dsn`` CLI flag → ``SCIX_TEST_DSN`` env var. We deliberately
    do NOT fall back to ``SCIX_DSN`` (which often points at production) —
    callers must pass ``--allow-prod`` and provide the DSN explicitly to target
    production.
    """
    if cli_dsn:
        return cli_dsn
    test_dsn = os.environ.get("SCIX_TEST_DSN")
    if test_dsn:
        return test_dsn
    raise SystemExit(
        "No DSN resolved: pass --dsn or set SCIX_TEST_DSN. "
        "SCIX_DSN fallback is intentionally disabled to prevent accidental "
        "production writes — pass --allow-prod + --dsn explicitly for prod."
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k-coarse", type=int, default=20)
    parser.add_argument("--k-medium", type=int, default=200)
    parser.add_argument("--k-fine", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=10_000)
    parser.add_argument(
        "--row-limit",
        type=int,
        default=None,
        help=(
            "Cap total rows streamed from paper_embeddings (both train and "
            "predict). Intended for validation / small-scale test runs; "
            "leave unset for a full-corpus run."
        ),
    )
    parser.add_argument(
        "--train-max-rows",
        type=int,
        default=1_500_000,
        help=(
            "Cap training-pass rows per resolution. MiniBatchKMeans converges "
            "long before 32M samples; predict pass still covers the full "
            "corpus. 1.5M is ~750x k=2000 — sample ratio well above the "
            "10-100x rule."
        ),
    )
    parser.add_argument(
        "--silhouette-sample",
        type=int,
        default=DEFAULT_SILHOUETTE_SAMPLE,
        help="Target size of the stratified silhouette sample.",
    )
    parser.add_argument(
        "--dsn",
        default=None,
        help="PostgreSQL DSN. Defaults to SCIX_TEST_DSN env var.",
    )
    parser.add_argument(
        "--allow-prod",
        action="store_true",
        help="Required to write to a production DSN (see is_production_dsn).",
    )
    parser.add_argument(
        "--results-path",
        default=str(RESULTS_DIR / "semantic_communities.json"),
    )
    parser.add_argument(
        "--run-meta-path",
        default=str(LOGS_DIR / "run_meta.json"),
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)

    dsn = _resolve_dsn(args.dsn)
    if is_production_dsn(dsn) and not args.allow_prod:
        logger.error(
            "Refusing to write to production DSN %s — pass --allow-prod to override",
            redact_dsn(dsn),
        )
        return 2

    logger.info(
        "Connecting to %s (k_coarse=%d k_medium=%d k_fine=%d seed=%d batch_size=%d)",
        redact_dsn(dsn),
        args.k_coarse,
        args.k_medium,
        args.k_fine,
        args.seed,
        args.batch_size,
    )

    run_id = str(uuid.uuid4())
    started_at = datetime.now(timezone.utc).isoformat()
    dsn_hash = hashlib.sha256(dsn.encode("utf-8")).hexdigest()[:12]

    # Ensure both the default dirs AND any caller-supplied custom output
    # paths have their parent directories in place (tmpdir paths in tests
    # often point outside RESULTS_DIR / LOGS_DIR).
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    Path(args.results_path).parent.mkdir(parents=True, exist_ok=True)
    Path(args.run_meta_path).parent.mkdir(parents=True, exist_ok=True)

    specs = [
        ResolutionSpec("coarse", args.k_coarse),
        ResolutionSpec("medium", args.k_medium),
        ResolutionSpec("fine",   args.k_fine),
    ]

    results: list[ResolutionResult] = []

    # Count once up front using a short-lived connection.
    with psycopg.connect(dsn) as conn:
        conn.autocommit = False
        register_vector(conn)
        n_indus = _count_indus_rows(conn)
        logger.info("paper_embeddings(model_name='indus'): %d rows", n_indus)

    # One connection per resolution. Under autocommit=False the outer
    # psycopg.connect wraps everything in a SINGLE implicit transaction,
    # so nothing commits until the connection closes — `with
    # conn.transaction()` inside _run_resolution only creates savepoints.
    # Running all three resolutions in one tx means (a) none of the work
    # is durable until the final exit, and (b) cross-resolution state
    # (cursor bookkeeping, kmeans fit history, gc-detached reservoirs)
    # accumulates in the python process. A fresh connection per
    # resolution commits cleanly on close and resets both ends.
    for spec in specs:
        with psycopg.connect(dsn) as conn:
            conn.autocommit = False
            register_vector(conn)
            result = _run_resolution(
                conn,
                spec,
                seed=args.seed,
                batch_size=args.batch_size,
                silhouette_sample=args.silhouette_sample,
                train_max_rows=args.train_max_rows,
                row_limit=args.row_limit,
            )
            # Explicit commit before close so the resolution's UPDATE
            # lands immediately in the database (visible to other
            # connections) and backend-side state is released.
            conn.commit()
        results.append(result)
        gc.collect()

    peak_rss_mb = _peak_rss_mb()
    finished_at = datetime.now(timezone.utc).isoformat()

    results_payload = {
        "run_id": run_id,
        "seed": args.seed,
        "peak_rss_mb": peak_rss_mb,
        "resolutions": {
            r.name: {
                "k": r.k,
                "n_rows": r.n_rows,
                "wall_clock_s": r.wall_clock_s,
                "silhouette": r.silhouette,
                "sample_n": r.sample_n,
            }
            for r in results
        },
    }
    Path(args.results_path).write_text(
        json.dumps(results_payload, indent=2) + "\n", encoding="utf-8"
    )
    logger.info("Wrote %s", args.results_path)

    run_meta = {
        "run_id": run_id,
        "git_sha": _git_sha(),
        "started_at": started_at,
        "finished_at": finished_at,
        "params": {
            "k_coarse": args.k_coarse,
            "k_medium": args.k_medium,
            "k_fine": args.k_fine,
            "seed": args.seed,
            "batch_size": args.batch_size,
            "silhouette_sample_size": args.silhouette_sample,
            "dsn_redacted": redact_dsn(dsn),
            "dsn_hash_12": dsn_hash,
            "allow_prod": args.allow_prod,
        },
        "n_indus_rows": n_indus,
        "peak_rss_mb": peak_rss_mb,
        "resolutions": results_payload["resolutions"],
    }
    Path(args.run_meta_path).write_text(
        json.dumps(run_meta, indent=2) + "\n", encoding="utf-8"
    )
    logger.info("Wrote %s", args.run_meta_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
