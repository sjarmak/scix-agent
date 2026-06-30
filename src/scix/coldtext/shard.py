"""Read-only cold-text shards for ADR-016.

One SQLite file per harvest year, keyed by ``bibcode``, holding the bulky
``papers_fulltext`` JSONB (``sections``/``figures``/``tables``/``equations``)
that has been sealed out of Postgres for closed years. JSON payloads are
zlib-compressed. ``sections_tsv`` is deliberately absent — it remains in
Postgres so section BM25 keeps serving sealed years.

Design constraints (all from ADR-016):

- **stdlib only** — ``sqlite3`` + ``zlib`` + ``json``; no new runtime dep, so
  the seal tooling runs in any venv and the MCP server's read path needs nothing
  extra.
- **read-only over NFS** — shards are opened with ``mode=ro&immutable=1``. They
  are write-once: built on DS or directly on NAS, then never mutated. Read-only
  SQLite over NFS is safe (the NFS hazard is concurrent *writers*).
- **point lookup by bibcode** — the read path is ``fetch(bibcode)``; the PRIMARY
  KEY index serves it without a scan.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import zlib
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

SHARD_SCHEMA_VERSION = 1

# JSONB columns sealed into the shard. ``sections`` is the only one the current
# read path consumes, but the rest are preserved so a sealed year is not lossy
# (re-deriving them means re-running the GROBID parser — expensive).
_JSON_COLS: tuple[str, ...] = ("sections", "figures", "tables", "equations")
_SCALAR_COLS: tuple[str, ...] = (
    "source",
    "parser_version",
    "canonical_bibcode",
    "suppressed_by_publisher",
)


def shard_path(root: str | Path, year: int) -> Path:
    """Path of the shard file for ``year`` under ``root``."""
    return Path(root) / f"papers_fulltext_{int(year)}.sqlite"


def _encode(obj: object) -> bytes:
    """Compact-JSON then zlib-compress. ``None`` round-trips as empty bytes."""
    if obj is None:
        return b""
    return zlib.compress(json.dumps(obj, separators=(",", ":"), ensure_ascii=False).encode("utf-8"))


def _decode(blob: bytes | None) -> object:
    """Inverse of :func:`_encode`. Empty/absent blob decodes to ``None``."""
    if not blob:
        return None
    return json.loads(zlib.decompress(blob).decode("utf-8"))


def sections_checksum(sections: object) -> str:
    """Stable SHA-256 over canonicalized ``sections`` JSON.

    Keys are sorted so the digest is independent of dict ordering, letting the
    verifier compare a shard row against its Postgres source byte-for-byte at the
    semantic level.
    """
    canonical = json.dumps(sections, separators=(",", ":"), sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ShardStats:
    """Outcome of :func:`write_shard`."""

    year: int
    rows: int
    path: Path


def _connect_write(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    # Bulk-load pragmas; the file is fsync'd once at close. Safe because a shard
    # is rebuilt from scratch on failure, never appended to.
    conn.execute("PRAGMA journal_mode = OFF")
    conn.execute("PRAGMA synchronous = OFF")
    return conn


def _create_schema(conn: sqlite3.Connection) -> None:
    cols = ", ".join(f"{c} BLOB" for c in _JSON_COLS)
    conn.execute(
        f"""
        CREATE TABLE fulltext (
            bibcode TEXT PRIMARY KEY,
            source TEXT,
            parser_version TEXT,
            canonical_bibcode TEXT,
            suppressed_by_publisher INTEGER,
            {cols}
        )
        """
    )
    conn.execute("CREATE TABLE shard_meta (key TEXT PRIMARY KEY, value TEXT)")


def _row_tuple(rec: Mapping[str, object]) -> tuple[object, ...]:
    if "bibcode" not in rec or not rec["bibcode"]:
        raise ValueError("cold-text record missing bibcode")
    suppressed = rec.get("suppressed_by_publisher")
    scalar = (
        rec.get("source"),
        rec.get("parser_version"),
        rec.get("canonical_bibcode"),
        None if suppressed is None else int(bool(suppressed)),
    )
    blobs = tuple(_encode(rec.get(c)) for c in _JSON_COLS)
    return (rec["bibcode"], *scalar, *blobs)


def write_shard(
    path: str | Path,
    year: int,
    rows: Iterable[Mapping[str, object]],
    *,
    batch_size: int = 1000,
) -> ShardStats:
    """Build a cold-text shard for ``year`` from an iterable of records.

    Each record is a mapping with at least ``bibcode`` and the ``papers_fulltext``
    columns. The file is written atomically: build to ``<path>.tmp`` then rename,
    so a partial build never shadows a good shard.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    if tmp.exists():
        tmp.unlink()

    conn: sqlite3.Connection | None = None
    count = 0
    try:
        conn = _connect_write(tmp)
        _create_schema(conn)
        insert = (
            "INSERT INTO fulltext (bibcode, "
            + ", ".join(_SCALAR_COLS)
            + ", "
            + ", ".join(_JSON_COLS)
            + ") VALUES ("
            + ", ".join(["?"] * (1 + len(_SCALAR_COLS) + len(_JSON_COLS)))
            + ")"
        )
        batch: list[tuple[object, ...]] = []
        for rec in rows:
            batch.append(_row_tuple(rec))
            if len(batch) >= batch_size:
                conn.executemany(insert, batch)
                count += len(batch)
                batch = []
        if batch:
            conn.executemany(insert, batch)
            count += len(batch)
        conn.executemany(
            "INSERT INTO shard_meta (key, value) VALUES (?, ?)",
            [
                ("schema_version", str(SHARD_SCHEMA_VERSION)),
                ("year", str(int(year))),
                ("rows", str(count)),
            ],
        )
        conn.commit()
    finally:
        if conn is not None:
            conn.close()

    tmp.replace(path)
    return ShardStats(year=int(year), rows=count, path=path)


class ColdTextReader:
    """Read-only accessor for a single cold-text shard.

    Opened with ``mode=ro&immutable=1`` so it is safe over NFS and never takes a
    write lock. Use as a context manager or call :meth:`close`.
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        if not self._path.exists():
            raise FileNotFoundError(f"cold-text shard not found: {self._path}")
        uri = f"file:{self._path}?mode=ro&immutable=1"
        # check_same_thread=False so a cached reader can be shared across the
        # MCP server's request threads; the DB is read-only/immutable and
        # sqlite3 serializes access internally, so this is safe.
        self._conn = sqlite3.connect(uri, uri=True, check_same_thread=False)

    def __enter__(self) -> ColdTextReader:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        self._conn.close()

    def row_count(self) -> int:
        (n,) = self._conn.execute("SELECT count(*) FROM fulltext").fetchone()
        return int(n)

    def fetch_sections(self, bibcode: str) -> object | None:
        """Decoded ``sections`` JSON for ``bibcode``, or ``None`` if absent.

        Matches the shape the Postgres path returns: a list of
        ``{text, heading, level, offset}`` dicts.
        """
        cur = self._conn.execute("SELECT sections FROM fulltext WHERE bibcode = ?", (bibcode,))
        row = cur.fetchone()
        if row is None:
            return None
        try:
            return _decode(row[0])
        except (zlib.error, json.JSONDecodeError, UnicodeDecodeError) as exc:
            # A corrupt/truncated blob (e.g. partial NFS read) must not crash the
            # read path — degrade to None so the caller falls back to Postgres /
            # body parsing. Verify still catches it (checksum mismatch).
            logger.error(
                "cold-text shard corrupt for bibcode %s in %s: %s",
                bibcode,
                self._path,
                exc,
            )
            return None

    def fetch_record(self, bibcode: str) -> dict[str, object] | None:
        """Full decoded record for ``bibcode`` (all JSON + scalar columns)."""
        cols = (
            "bibcode",
            *_SCALAR_COLS,
            *_JSON_COLS,
        )
        cur = self._conn.execute(
            f"SELECT {', '.join(cols)} FROM fulltext WHERE bibcode = ?",
            (bibcode,),
        )
        row = cur.fetchone()
        if row is None:
            return None
        rec: dict[str, object] = {"bibcode": row[0]}
        offset = 1
        for i, c in enumerate(_SCALAR_COLS):
            rec[c] = row[offset + i]
        offset += len(_SCALAR_COLS)
        for i, c in enumerate(_JSON_COLS):
            rec[c] = _decode(row[offset + i])
        return rec

    def iter_bibcodes(self) -> Iterator[str]:
        for (b,) in self._conn.execute("SELECT bibcode FROM fulltext"):
            yield b
