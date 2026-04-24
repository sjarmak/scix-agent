#!/usr/bin/env python3
"""Project INDUS paper embeddings to 2-D with UMAP for visualization.

Read/write tool: samples papers stratified by ``paper_metrics.community_semantic_*``,
pulls the halfvec(768) INDUS embedding from ``paper_embeddings``, runs UMAP
(cuML when available, else umap-learn), writes ``(bibcode, x, y, community_id,
resolution)`` rows into ``paper_umap_2d`` (upsert), and dumps a static JSON
for hosting.

Usage
-----

Smoke-test mode (no DB required) — synthetic embeddings, writes a JSON file::

    python scripts/viz/project_embeddings_umap.py \\
        --synthetic 2000 \\
        --output /tmp/umap_test.json \\
        --backend umap-learn

Smoke-test mode with no file write (just print summary)::

    python scripts/viz/project_embeddings_umap.py \\
        --synthetic 2000 --dry-run

Production::

    python scripts/viz/project_embeddings_umap.py \\
        --sample-size 100000 --resolution coarse \\
        --output data/viz/umap.json

Output JSON schema (top-level array)::

    [
      {"bibcode": "2020ApJ...900..100X",
       "x": 1.234, "y": -5.678,
       "community_id": 17,
       "resolution": "coarse"},
      ...
    ]

Semantics of ``--dry-run`` vs ``--synthetic``:

* ``--synthetic N`` — skip the DB entirely; generate ``N`` deterministic
  768-d synthetic vectors + community ids for testing.
* ``--dry-run`` — do all computation but DO NOT write the output file AND
  DO NOT write rows to the DB.

The script uses a stratified sample (per-community cap via
``ROW_NUMBER() OVER (PARTITION BY community_id ORDER BY random())``) so
small communities are not drowned out by large ones. The per-community cap
is derived from ``sample_size`` divided by an estimate of 20 coarse
communities — tuned empirically; see ``_stratified_sample_sql``.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Protocol, Sequence

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from scix.db import DEFAULT_DSN, is_production_dsn, redact_dsn  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("project_embeddings_umap")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


DEFAULT_SAMPLE_SIZE = 100_000
DEFAULT_RESOLUTION = "coarse"
DEFAULT_BACKEND = "auto"
DEFAULT_OUTPUT_REL = Path("data/viz/umap.json")

RESOLUTIONS: tuple[str, ...] = ("coarse", "medium", "fine")
BACKENDS: tuple[str, ...] = ("auto", "cuml", "umap-learn")

# Allowlist mapping --resolution -> paper_metrics column. Used as a
# defence-in-depth check before interpolating the column into SQL.
COMMUNITY_COLUMNS: dict[str, str] = {
    "coarse": "community_semantic_coarse",
    "medium": "community_semantic_medium",
    "fine": "community_semantic_fine",
}

# INDUS embedding dimensionality (halfvec(768) post migration 053).
_EMBEDDING_DIM = 768

# Heuristic community count per resolution — matches the minibatch-kmeans
# K values from migration 051. Used to estimate a per-community row cap for
# the stratified SQL sample.
_RESOLUTION_K: dict[str, int] = {"coarse": 20, "medium": 200, "fine": 2000}

# Server-side cursor batch size for the DB loader.
_DB_FETCH_BATCH = 10_000


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProjectedPoint:
    """One row of the output dataset."""

    bibcode: str
    x: float
    y: float
    community_id: Optional[int]
    resolution: str


@dataclass(frozen=True)
class Config:
    """Resolved CLI configuration."""

    sample_size: int
    resolution: str
    backend: str
    dsn: str
    output: Path
    dry_run: bool
    synthetic_n: Optional[int]


# ---------------------------------------------------------------------------
# Backend protocol
# ---------------------------------------------------------------------------


class UMAPBackend(Protocol):
    """Minimal interface every UMAP backend must expose."""

    def fit_transform(self, X: np.ndarray) -> np.ndarray:  # pragma: no cover - Protocol
        ...


@dataclass(frozen=True)
class _BackendChoice:
    """Backend instance + human-readable label for logging."""

    backend: UMAPBackend
    label: str


# ---------------------------------------------------------------------------
# Synthetic loader — deterministic, no DB
# ---------------------------------------------------------------------------


def load_embeddings_synthetic(
    n: int, seed: int = 42, resolution: str = "coarse"
) -> tuple[list[str], np.ndarray, list[Optional[int]]]:
    """Generate ``n`` deterministic 768-d synthetic embeddings.

    Returns
    -------
    bibcodes
        ``[f"synthetic-{i:06d}" for i in range(n)]``.
    embeddings
        ``np.ndarray`` of shape ``(n, 768)``, dtype ``float32``, drawn from
        a standard normal distribution seeded by ``seed``.
    community_ids
        Length-``n`` list of integer community ids drawn from
        ``range(K)``, where ``K`` matches the resolution's expected
        community cardinality (``_RESOLUTION_K``: coarse=20, medium=200,
        fine=2000). This keeps the synthetic placeholder shape consistent
        with prod when the visualization is exercised end-to-end without a
        live DB.
    """
    if n <= 0:
        raise ValueError(f"synthetic n must be > 0, got {n}")

    k = _RESOLUTION_K.get(resolution, 20)
    rng = np.random.default_rng(seed)
    bibcodes = [f"synthetic-{i:06d}" for i in range(n)]
    embeddings = rng.standard_normal(size=(n, _EMBEDDING_DIM)).astype(np.float32)
    community_ids: list[Optional[int]] = [int(i % k) for i in range(n)]
    return bibcodes, embeddings, community_ids


# ---------------------------------------------------------------------------
# DB loader — stratified sample, server-side cursor
# ---------------------------------------------------------------------------


def _per_community_cap(sample_size: int, resolution: str) -> int:
    """Heuristic per-community row cap for a stratified sample.

    Uses the documented K value per resolution (migration 051). For coarse
    (K=20) and ``sample_size=100000`` the cap lands at 5000 rows per
    community. ``max(1, ...)`` guards against zero-rounding on tiny samples.
    """
    k = _RESOLUTION_K.get(resolution, 20)
    return max(1, math.ceil(sample_size / k))


def _stratified_sample_sql(resolution: str) -> str:
    """Return a parametrised stratified-sample SQL (psycopg %s placeholders).

    Sampling strategy (documented in module docstring): window-function cap
    of ``per_community`` rows per community, ranked by ``random()``. Simple,
    deterministic per connection, and applies the stratification in SQL so
    Python never has to materialise the full paper_embeddings join.
    """
    if resolution not in COMMUNITY_COLUMNS:
        raise ValueError(
            f"unknown resolution {resolution!r}; must be one of {RESOLUTIONS}"
        )
    column = COMMUNITY_COLUMNS[resolution]
    # Column name comes from the hard-coded allowlist — safe to interpolate.
    return (
        "WITH ranked AS (\n"
        "    SELECT pe.bibcode,\n"
        "           pe.embedding,\n"
        f"           pm.{column} AS community_id,\n"
        f"           ROW_NUMBER() OVER (PARTITION BY pm.{column} "
        "ORDER BY random()) AS rn\n"
        "    FROM paper_embeddings pe\n"
        "    JOIN paper_metrics pm USING (bibcode)\n"
        "    WHERE pe.model_name = 'indus'\n"
        "      AND pe.embedding IS NOT NULL\n"
        f"      AND pm.{column} IS NOT NULL\n"
        ")\n"
        "SELECT bibcode, embedding, community_id\n"
        "FROM ranked\n"
        "WHERE rn <= %s\n"
        "LIMIT %s"
    )


def _coerce_embedding(raw: object) -> np.ndarray:
    """Convert a psycopg halfvec payload into a 1-D float32 numpy array.

    pgvector's psycopg3 adapter returns a ``list[float]`` (or
    ``numpy.ndarray``). If the adapter isn't registered on the connection,
    we see the raw text form ``"[0.1, 0.2, ...]"`` and fall back to
    ``json.loads`` (the halfvec text representation is JSON-compatible).
    """
    if isinstance(raw, np.ndarray):
        arr = raw.astype(np.float32, copy=False)
    elif isinstance(raw, (list, tuple)):
        arr = np.asarray(raw, dtype=np.float32)
    elif isinstance(raw, str):
        arr = np.asarray(json.loads(raw), dtype=np.float32)
    else:
        raise TypeError(f"unsupported embedding payload type: {type(raw).__name__}")
    if arr.shape != (_EMBEDDING_DIM,):
        raise ValueError(
            f"expected {_EMBEDDING_DIM}-d embedding, got shape {arr.shape}"
        )
    return arr


def load_embeddings_from_db(
    dsn: str, resolution: str, sample_size: int
) -> tuple[list[str], np.ndarray, list[Optional[int]]]:
    """Stream a stratified sample of INDUS embeddings from the database.

    Uses a server-side named cursor so the intermediate ranked set does not
    have to fit in client RAM. Returns Python-native types (bibcodes, a
    packed ``(n, 768)`` float32 array, and a list of community ids) matching
    :func:`load_embeddings_synthetic`.

    Raises ValueError if ``resolution`` is not in the allowlist.
    """
    if resolution not in COMMUNITY_COLUMNS:
        raise ValueError(
            f"unknown resolution {resolution!r}; must be one of {RESOLUTIONS}"
        )
    if sample_size <= 0:
        raise ValueError(f"sample_size must be > 0, got {sample_size}")

    import psycopg  # lazy import so --synthetic and tests don't need a live DB

    per_community = _per_community_cap(sample_size, resolution)
    sql_text = _stratified_sample_sql(resolution)
    logger.info(
        "DB sample: resolution=%s per_community=%d cap=%d",
        resolution,
        per_community,
        sample_size,
    )

    bibcodes: list[str] = []
    embedding_rows: list[np.ndarray] = []
    community_ids: list[Optional[int]] = []

    with psycopg.connect(dsn) as conn:
        with conn.cursor(name="umap_sample_cursor") as cur:
            cur.execute(sql_text, (per_community, sample_size))
            while True:
                batch = cur.fetchmany(_DB_FETCH_BATCH)
                if not batch:
                    break
                for bibcode, embedding, community_id in batch:
                    bibcodes.append(str(bibcode))
                    embedding_rows.append(_coerce_embedding(embedding))
                    community_ids.append(
                        int(community_id) if community_id is not None else None
                    )

    if not embedding_rows:
        logger.warning("DB sample returned zero rows")
        return [], np.empty((0, _EMBEDDING_DIM), dtype=np.float32), []

    embeddings = np.vstack(embedding_rows).astype(np.float32, copy=False)
    return bibcodes, embeddings, community_ids


# ---------------------------------------------------------------------------
# Backend selection & projection
# ---------------------------------------------------------------------------


def _n_neighbors_for(n: int) -> int:
    """Pick a safe ``n_neighbors`` for datasets of any size.

    UMAP's default is 15; for tiny datasets we must cap at ``n - 1`` to avoid
    a kNN-graph error. We also keep a floor of 2 so UMAP has at least one
    neighbour to work with.
    """
    return max(2, min(15, n - 1))


def pick_backend(name: str, n: int) -> _BackendChoice:
    """Return a UMAP backend instance + label.

    * ``name="auto"`` tries cuML first, falls back to umap-learn on
      ``ImportError``.
    * ``name="cuml"`` requires cuML (raises ``ImportError`` otherwise).
    * ``name="umap-learn"`` uses the CPU umap-learn package.

    The choice is logged at INFO level.
    """
    if name not in BACKENDS:
        raise ValueError(f"unknown backend {name!r}; must be one of {BACKENDS}")

    n_neighbors = _n_neighbors_for(n)
    random_state = 42

    def _build_cuml() -> _BackendChoice:
        import cuml  # lazy, not required for tests

        backend = cuml.UMAP(
            n_components=2,
            n_neighbors=n_neighbors,
            random_state=random_state,
        )
        logger.info("UMAP backend: cuml (n_neighbors=%d)", n_neighbors)
        return _BackendChoice(backend=backend, label="cuml")

    def _build_umap_learn() -> _BackendChoice:
        import umap  # lazy

        backend = umap.UMAP(
            n_components=2,
            n_neighbors=n_neighbors,
            random_state=random_state,
        )
        logger.info("UMAP backend: umap-learn (n_neighbors=%d)", n_neighbors)
        return _BackendChoice(backend=backend, label="umap-learn")

    if name == "cuml":
        return _build_cuml()
    if name == "umap-learn":
        return _build_umap_learn()

    # auto
    try:
        return _build_cuml()
    except ImportError:
        logger.info("cuml unavailable, falling back to umap-learn")
        return _build_umap_learn()


def project(embeddings: np.ndarray, backend: UMAPBackend) -> np.ndarray:
    """Run UMAP ``fit_transform`` and return an ``(n, 2)`` float array.

    Handles cuML's cupy-array return by calling ``.get()`` when present.
    """
    if embeddings.ndim != 2 or embeddings.shape[1] != _EMBEDDING_DIM:
        raise ValueError(
            f"embeddings must be (n, {_EMBEDDING_DIM}); got {embeddings.shape}"
        )
    if embeddings.shape[0] == 0:
        return np.empty((0, 2), dtype=np.float32)

    xy = backend.fit_transform(embeddings)
    # cuML returns a cupy array — convert to numpy if needed.
    if hasattr(xy, "get") and not isinstance(xy, np.ndarray):
        xy = xy.get()
    xy_np = np.asarray(xy, dtype=np.float32)
    if xy_np.ndim != 2 or xy_np.shape[1] != 2:
        raise ValueError(f"UMAP output must be (n, 2); got {xy_np.shape}")
    return xy_np


# ---------------------------------------------------------------------------
# Point construction, serialization, validation
# ---------------------------------------------------------------------------


def build_points(
    bibcodes: Sequence[str],
    xy: np.ndarray,
    community_ids: Sequence[Optional[int]],
    resolution: str,
) -> tuple[ProjectedPoint, ...]:
    """Assemble immutable ProjectedPoint tuples from projection outputs."""
    if resolution not in COMMUNITY_COLUMNS:
        raise ValueError(
            f"unknown resolution {resolution!r}; must be one of {RESOLUTIONS}"
        )
    if not (len(bibcodes) == xy.shape[0] == len(community_ids)):
        raise ValueError(
            "mismatched lengths: "
            f"bibcodes={len(bibcodes)} xy={xy.shape[0]} communities={len(community_ids)}"
        )
    return tuple(
        ProjectedPoint(
            bibcode=str(bibcode),
            x=float(xy[i, 0]),
            y=float(xy[i, 1]),
            community_id=(int(cid) if cid is not None else None),
            resolution=resolution,
        )
        for i, (bibcode, cid) in enumerate(zip(bibcodes, community_ids))
    )


def _point_to_dict(p: ProjectedPoint) -> dict[str, object]:
    return {
        "bibcode": p.bibcode,
        "x": p.x,
        "y": p.y,
        "community_id": p.community_id,
        "resolution": p.resolution,
    }


def serialize(points: Iterable[ProjectedPoint], path: Path) -> None:
    """Write projected points to ``path`` as a JSON array (UTF-8, trailing nl)."""
    payload = [_point_to_dict(p) for p in points]
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=False) + "\n"
    path.write_text(text, encoding="utf-8")


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise ValueError(msg)


def validate_projection_payload(obj: object) -> tuple[ProjectedPoint, ...]:
    """Type-check a raw JSON payload against the projection schema.

    Returns a tuple of ProjectedPoint on success; raises ``ValueError`` with
    a descriptive message on any structural or type mismatch.
    """
    _require(isinstance(obj, list), "top-level must be a list")
    assert isinstance(obj, list)  # narrow for type checker
    points: list[ProjectedPoint] = []
    for i, raw in enumerate(obj):
        _require(isinstance(raw, dict), f"obj[{i}] must be a dict")
        assert isinstance(raw, dict)
        for key in ("bibcode", "x", "y", "community_id", "resolution"):
            _require(key in raw, f"obj[{i}] missing key '{key}'")
        _require(isinstance(raw["bibcode"], str), f"obj[{i}].bibcode must be str")
        _require(
            isinstance(raw["x"], (int, float)) and not isinstance(raw["x"], bool),
            f"obj[{i}].x must be a number",
        )
        _require(
            isinstance(raw["y"], (int, float)) and not isinstance(raw["y"], bool),
            f"obj[{i}].y must be a number",
        )
        cid = raw["community_id"]
        _require(
            cid is None or (isinstance(cid, int) and not isinstance(cid, bool)),
            f"obj[{i}].community_id must be int or null",
        )
        _require(
            isinstance(raw["resolution"], str) and raw["resolution"] in RESOLUTIONS,
            f"obj[{i}].resolution must be one of {RESOLUTIONS}",
        )
        points.append(
            ProjectedPoint(
                bibcode=raw["bibcode"],
                x=float(raw["x"]),
                y=float(raw["y"]),
                community_id=(int(cid) if cid is not None else None),
                resolution=raw["resolution"],
            )
        )
    return tuple(points)


# ---------------------------------------------------------------------------
# DB upsert
# ---------------------------------------------------------------------------


_UPSERT_SQL = (
    "INSERT INTO paper_umap_2d (bibcode, x, y, community_id, resolution) "
    "VALUES (%s, %s, %s, %s, %s) "
    "ON CONFLICT (bibcode) DO UPDATE SET "
    "    x = EXCLUDED.x, "
    "    y = EXCLUDED.y, "
    "    community_id = EXCLUDED.community_id, "
    "    resolution = EXCLUDED.resolution, "
    "    projected_at = now()"
)


def write_to_db(dsn: str, points: Sequence[ProjectedPoint]) -> int:
    """Upsert projected points into ``paper_umap_2d``; return row count written."""
    if not points:
        logger.info("no points to upsert")
        return 0

    import psycopg  # lazy

    rows = [
        (p.bibcode, p.x, p.y, p.community_id, p.resolution)
        for p in points
    ]
    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.executemany(_UPSERT_SQL, rows)
        conn.commit()
    logger.info("upserted %d rows into paper_umap_2d", len(rows))
    return len(rows)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Project INDUS paper embeddings to 2-D with UMAP.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=DEFAULT_SAMPLE_SIZE,
        help=f"Target sample size (default: {DEFAULT_SAMPLE_SIZE}).",
    )
    parser.add_argument(
        "--resolution",
        choices=RESOLUTIONS,
        default=DEFAULT_RESOLUTION,
        help=f"Semantic-community resolution (default: {DEFAULT_RESOLUTION}).",
    )
    parser.add_argument(
        "--backend",
        choices=BACKENDS,
        default=DEFAULT_BACKEND,
        help=f"UMAP backend (default: {DEFAULT_BACKEND}).",
    )
    parser.add_argument(
        "--dsn",
        default=DEFAULT_DSN,
        help="PostgreSQL DSN (default: scix.db.DEFAULT_DSN).",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_REL),
        help=f"Output JSON path (default: {DEFAULT_OUTPUT_REL}).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do all computation but do NOT write the JSON output or DB rows.",
    )
    parser.add_argument(
        "--synthetic",
        type=int,
        default=None,
        metavar="N",
        help="Generate N deterministic synthetic embeddings instead of querying the DB.",
    )
    return parser.parse_args(argv)


def _resolve_output_path(raw: str) -> Path:
    """Resolve output path. Relative paths are anchored at repo root."""
    p = Path(raw)
    if p.is_absolute():
        return p
    return _REPO_ROOT / p


def _config_from_args(args: argparse.Namespace) -> Config:
    return Config(
        sample_size=int(args.sample_size),
        resolution=args.resolution,
        backend=args.backend,
        dsn=args.dsn,
        output=_resolve_output_path(args.output),
        dry_run=bool(args.dry_run),
        synthetic_n=(int(args.synthetic) if args.synthetic is not None else None),
    )


def _summary_log(
    config: Config,
    n_points: int,
    backend_label: str,
) -> None:
    logger.info(
        "summary: n_points=%d backend=%s resolution=%s dry_run=%s synthetic=%s output=%s",
        n_points,
        backend_label,
        config.resolution,
        config.dry_run,
        config.synthetic_n,
        config.output,
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entrypoint. Returns a process exit code."""
    args = _parse_args(argv)
    try:
        config = _config_from_args(args)
    except ValueError as e:
        logger.error("invalid config: %s", e)
        return 2

    # Defence in depth: resolution already constrained by argparse choices,
    # but re-check against the allowlist dict the SQL uses.
    if config.resolution not in COMMUNITY_COLUMNS:
        logger.error("resolution %r not in allowlist", config.resolution)
        return 2
    if config.sample_size <= 0:
        logger.error("--sample-size must be > 0, got %d", config.sample_size)
        return 2

    # 1. Load embeddings (synthetic or DB).
    if config.synthetic_n is not None:
        logger.info(
            "loading %d synthetic embeddings (deterministic, no DB)",
            config.synthetic_n,
        )
        bibcodes, embeddings, community_ids = load_embeddings_synthetic(
            config.synthetic_n, resolution=config.resolution
        )
    else:
        logger.info(
            "loading from DB dsn=%s resolution=%s sample_size=%d",
            redact_dsn(config.dsn),
            config.resolution,
            config.sample_size,
        )
        if is_production_dsn(config.dsn) and not config.dry_run:
            logger.info(
                "DSN appears to point at production; DB upsert will occur "
                "unless --dry-run is set"
            )
        bibcodes, embeddings, community_ids = load_embeddings_from_db(
            config.dsn, config.resolution, config.sample_size
        )

    n = embeddings.shape[0]
    if n == 0:
        logger.warning("no embeddings loaded; nothing to project")
        _summary_log(config, n_points=0, backend_label="none")
        return 0

    # 2. Pick backend + project.
    choice = pick_backend(config.backend, n)
    xy = project(embeddings, choice.backend)

    # 3. Build immutable points.
    points = build_points(bibcodes, xy, community_ids, config.resolution)
    _summary_log(config, n_points=len(points), backend_label=choice.label)

    # 4. Dry-run: no file, no DB.
    if config.dry_run:
        logger.info("--dry-run set: skipping file write and DB upsert")
        return 0

    # 5. Always write the JSON file (agent-navigable static host target).
    serialize(points, config.output)
    logger.info("wrote %s (%d points)", config.output, len(points))

    # 6. DB upsert only when NOT synthetic (synthetic bibcodes violate FK).
    if config.synthetic_n is None:
        try:
            write_to_db(config.dsn, points)
        except Exception:  # pragma: no cover - DB errors are operational
            logger.exception("write_to_db failed; JSON output already written")
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
