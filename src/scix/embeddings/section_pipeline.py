"""Section-level embedding pipeline.

Reads ``papers_fulltext.sections`` (JSONB array of {heading, level, text, offset})
in bibcode-sorted chunks, encodes each section's ``heading + '\\n' + text`` via
a local open-weight model (default ``nomic-ai/nomic-embed-text-v1.5`` at 1024
Matryoshka dimensions), and writes the resulting halfvec rows into
``section_embeddings`` via psycopg COPY.

Resumability
------------
Each section gets a stable ``section_text_sha256 = sha256(heading + '\\n' + text)``.
Before encoding, the pipeline looks up rows already stored for the in-batch
bibcodes and skips any (bibcode, section_index) whose stored sha matches the
computed sha. This makes the run safe to restart at any point and idempotent
across re-encodes when section content is unchanged.

Why a local model
-----------------
Per project policy (see memory note ``feedback_no_paid_apis``) the encoder MUST
be a local open-weight model. This module deliberately does NOT import any
paid-API SDK. The import-policy is enforced in unit tests by grepping the source
file. The model is loaded lazily on first encode so import-time has no GPU /
torch / transformers cost — unit tests run on CPU-only CI without paying for
``sentence_transformers``.

Storage shape
-------------
The target table ``section_embeddings`` is created upstream by the parser PRD
(expected migration ``061_section_embeddings.sql``)::

    section_embeddings (
        bibcode             TEXT,
        section_index       INT,
        section_heading     TEXT,
        section_text_sha256 TEXT,
        embedding           halfvec(1024),
        PRIMARY KEY (bibcode, section_index)
    )

The full-corpus encode is upstream-blocked by the parser PRD; this module ships
the code + unit tests so the encode is ready to fire as soon as the migration
lands.

CLI
---
The pipeline is invoked via::

    python -m scix.embeddings.section_pipeline \\
        --model nomic-ai/nomic-embed-text-v1.5 \\
        --dimensions 1024 \\
        --batch-size 32 \\
        --start-bibcode 2020ApJ...900..001S \\
        --end-bibcode 2021ApJ...920..999Z

In production this is wrapped by ``scix-batch --mem-high 20G --mem-max 25G``
to enforce memory isolation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from typing import Any, Iterable, Iterator, Sequence

import psycopg

from scix.db import get_connection

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_MODEL: str = "nomic-ai/nomic-embed-text-v1.5"
DEFAULT_DIMENSIONS: int = 1024
DEFAULT_BATCH_SIZE: int = 32

# nomic-embed-text-v1.5 expects task-prefix sentinels on inputs. For indexed
# documents the prefix is "search_document: ". For queries it would be
# "search_query: ". We only emit document embeddings here.
NOMIC_DOC_PREFIX: str = "search_document: "

# Module-global model cache so a single process pays the load cost once.
_MODEL_CACHE: dict[str, Any] = {}


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def compute_section_sha(heading: str | None, text: str | None) -> str:
    """Return ``sha256((heading or '') + '\\n' + (text or '')).hexdigest()``.

    Stable across calls and processes — used as the resumability key in
    ``section_embeddings.section_text_sha256``.
    """
    h = heading or ""
    t = text or ""
    payload = f"{h}\n{t}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def format_halfvec(vector: Sequence[float]) -> str:
    """Format a float sequence as the pgvector text literal ``[v1,v2,...]``.

    The same text format is accepted by both ``vector`` and ``halfvec`` columns
    on COPY and INSERT — pgvector parses the literal and casts to the column's
    storage precision (32-bit float vs 16-bit half).
    """
    return "[" + ",".join(repr(float(v)) for v in vector) + "]"


def _section_text_for_encode(heading: str | None, text: str | None) -> str:
    """Concatenate heading + body for encode input.

    The heading carries non-trivial signal ("Methods", "Results", ...) so it
    must be part of the encoded text. We use a newline separator to match the
    sha-key payload.
    """
    return f"{heading or ''}\n{text or ''}"


# ---------------------------------------------------------------------------
# DB I/O
# ---------------------------------------------------------------------------


def iter_sections(
    conn: Any,
    start_bibcode: str | None = None,
    end_bibcode: str | None = None,
) -> Iterator[tuple[str, int, str, str]]:
    """Yield ``(bibcode, section_index, heading, text)`` from papers_fulltext.

    Iteration order is bibcode ASC, section_index ASC. ``section_index`` is
    0-based per paper. Sections whose ``text`` is empty/whitespace-only are
    silently dropped — they would produce zero-information embeddings.

    Inputs:
        conn: psycopg connection (or any duck-typed object whose ``cursor()``
            context manager yields rows ``(bibcode, sections_jsonb)``).
        start_bibcode / end_bibcode: optional half-open range filter,
            ``[start, end]`` inclusive on both ends. Either may be None.
    """
    sql = (
        "SELECT bibcode, sections FROM papers_fulltext "
        "WHERE (%(start)s IS NULL OR bibcode >= %(start)s) "
        "  AND (%(end)s   IS NULL OR bibcode <= %(end)s) "
        "ORDER BY bibcode"
    )
    params = {"start": start_bibcode, "end": end_bibcode}

    with conn.cursor() as cur:
        cur.execute(sql, params)
        for bibcode, sections in cur:
            # ``sections`` may arrive as a Python list (psycopg JSONB adapter)
            # or as a JSON string; tolerate both for test/mock convenience.
            if isinstance(sections, (str, bytes)):
                sections = json.loads(sections)
            if not isinstance(sections, list):
                continue
            for idx, section in enumerate(sections):
                if not isinstance(section, dict):
                    continue
                heading = section.get("heading") or ""
                text = section.get("text") or ""
                if not text.strip():
                    continue
                yield bibcode, idx, heading, text


def existing_shas(
    conn: Any,
    bibcodes: Sequence[str],
) -> dict[tuple[str, int], str]:
    """Return ``{(bibcode, section_index): section_text_sha256}`` for rows
    already stored in ``section_embeddings`` for the given bibcodes.

    Empty input -> empty dict (no DB round-trip).
    """
    if not bibcodes:
        return {}

    sql = (
        "SELECT bibcode, section_index, section_text_sha256 "
        "FROM section_embeddings "
        "WHERE bibcode = ANY(%(bibcodes)s)"
    )
    out: dict[tuple[str, int], str] = {}
    with conn.cursor() as cur:
        cur.execute(sql, {"bibcodes": list(bibcodes)})
        for bibcode, section_index, sha in cur:
            out[(bibcode, int(section_index))] = sha
    return out


def filter_unchanged(
    rows: Sequence[tuple[str, int, str, str, str]],
    stored: dict[tuple[str, int], str],
) -> list[tuple[str, int, str, str, str]]:
    """Drop rows whose stored sha already matches the computed sha.

    ``rows`` items are ``(bibcode, section_index, heading, text, sha)``.
    """
    survivors: list[tuple[str, int, str, str, str]] = []
    for row in rows:
        bibcode, idx, _heading, _text, sha = row
        if stored.get((bibcode, idx)) == sha:
            continue
        survivors.append(row)
    return survivors


def write_batch_copy(
    conn: Any,
    rows: Sequence[tuple[str, int, str, str, Sequence[float]]],
) -> int:
    """COPY-write ``section_embeddings`` rows.

    ``rows`` items are ``(bibcode, section_index, heading, sha, vector)``.

    Uses TEMP staging + INSERT ON CONFLICT for idempotent upsert. The halfvec
    is shipped as the text literal ``[v1,v2,...]`` (same format pgvector
    accepts on INSERT). Returns the number of rows COPYed in.
    """
    if not rows:
        return 0

    with conn.cursor() as cur:
        cur.execute(
            "CREATE TEMP TABLE IF NOT EXISTS _section_embed_staging ("
            "  bibcode TEXT, "
            "  section_index INT, "
            "  section_heading TEXT, "
            "  section_text_sha256 TEXT, "
            "  embedding halfvec"
            ") ON COMMIT DELETE ROWS"
        )

        with cur.copy(
            "COPY _section_embed_staging "
            "  (bibcode, section_index, section_heading, section_text_sha256, embedding) "
            "FROM STDIN"
        ) as copy:
            for bibcode, section_index, heading, sha, vector in rows:
                copy.write_row(
                    (
                        bibcode,
                        int(section_index),
                        heading,
                        sha,
                        format_halfvec(vector),
                    )
                )

        cur.execute(
            "INSERT INTO section_embeddings "
            "  (bibcode, section_index, section_heading, section_text_sha256, embedding) "
            "SELECT bibcode, section_index, section_heading, section_text_sha256, embedding "
            "FROM _section_embed_staging "
            "ON CONFLICT (bibcode, section_index) DO UPDATE SET "
            "  section_heading     = EXCLUDED.section_heading, "
            "  section_text_sha256 = EXCLUDED.section_text_sha256, "
            "  embedding           = EXCLUDED.embedding"
        )
        written = cur.rowcount

    conn.commit()
    return written


# ---------------------------------------------------------------------------
# Model loading + encoding (lazy)
# ---------------------------------------------------------------------------


def _load_model(model_name: str) -> Any:
    """Lazy-load a sentence_transformers model.

    Imports ``sentence_transformers`` *inside* this function so module import
    is GPU-free and CPU-only test environments work. Cached in module-global
    ``_MODEL_CACHE`` for process lifetime.
    """
    cached = _MODEL_CACHE.get(model_name)
    if cached is not None:
        return cached

    try:
        # Local import — keeps unit tests fast and CPU-only.
        from sentence_transformers import SentenceTransformer  # type: ignore
    except ImportError as exc:  # pragma: no cover - exercised only on real run
        raise ImportError(
            "sentence-transformers is required to load embedding models. "
            "Install via the `search` extra: pip install -e .[search]"
        ) from exc

    logger.info("Loading embedding model %s", model_name)
    model = SentenceTransformer(model_name, trust_remote_code=True)
    _MODEL_CACHE[model_name] = model
    return model


def encode_batch(
    model: Any,
    texts: Sequence[str],
    dimensions: int,
) -> list[list[float]]:
    """Encode a batch of texts to ``dimensions``-dim vectors.

    Caller is responsible for prepending the nomic ``search_document: ``
    prefix (or whatever sentinel a different local model expects). The model
    object must duck-type ``model.encode(texts, ...)`` returning an
    array-like of shape ``(len(texts), dimensions)``.

    ``dimensions`` is forwarded as a Matryoshka truncation hint where
    supported. Vectors are converted to plain Python ``list[list[float]]``
    so they can be serialized for COPY without importing numpy here.
    """
    if not texts:
        return []

    # ``truncate_dim`` is the sentence_transformers v3 kwarg for Matryoshka
    # truncation. Older versions silently ignore it; we trim post-hoc below
    # to preserve the contract.
    raw = model.encode(
        list(texts),
        normalize_embeddings=True,
        truncate_dim=dimensions,
    )

    out: list[list[float]] = []
    for vec in raw:
        # Tolerate numpy arrays, torch tensors (via .tolist if available),
        # and plain lists.
        if hasattr(vec, "tolist"):
            vec = vec.tolist()
        vec = list(vec)
        if len(vec) > dimensions:
            vec = vec[:dimensions]
        out.append([float(x) for x in vec])
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    """Build the argparse parser. Exposed for unit tests."""
    parser = argparse.ArgumentParser(
        prog="scix.embeddings.section_pipeline",
        description=(
            "Encode papers_fulltext.sections into section_embeddings using a "
            "local open-weight model (default nomic-embed-text-v1.5)."
        ),
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"HuggingFace model id (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Encode/COPY batch size (default: {DEFAULT_BATCH_SIZE}).",
    )
    parser.add_argument(
        "--start-bibcode",
        default=None,
        help="Optional inclusive lower bound on bibcode range.",
    )
    parser.add_argument(
        "--end-bibcode",
        default=None,
        help="Optional inclusive upper bound on bibcode range.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Plan the run without loading the model or writing to the DB.",
    )
    parser.add_argument(
        "--dimensions",
        type=int,
        default=DEFAULT_DIMENSIONS,
        help=f"Output vector dimensions (Matryoshka truncation, default: {DEFAULT_DIMENSIONS}).",
    )
    parser.add_argument(
        "--dsn",
        default=None,
        help="Postgres DSN (defaults to SCIX_DSN env or the libpq default).",
    )
    return parser


def _chunked(it: Iterable, size: int) -> Iterator[list]:
    """Split an iterable into lists of length ``size`` (last list may be shorter)."""
    if size <= 0:
        raise ValueError("size must be positive")
    buf: list = []
    for item in it:
        buf.append(item)
        if len(buf) >= size:
            yield buf
            buf = []
    if buf:
        yield buf


def _process_batch(
    conn: Any,
    model: Any,
    batch: list[tuple[str, int, str, str]],
    dimensions: int,
) -> int:
    """Encode + write one batch. Skips rows whose stored sha already matches.

    Returns the number of rows written.
    """
    # Compute shas first so the skip-existing check is cheap.
    rows_with_sha: list[tuple[str, int, str, str, str]] = [
        (bibcode, idx, heading, text, compute_section_sha(heading, text))
        for bibcode, idx, heading, text in batch
    ]

    bibcodes = sorted({r[0] for r in rows_with_sha})
    stored = existing_shas(conn, bibcodes)
    survivors = filter_unchanged(rows_with_sha, stored)
    if not survivors:
        return 0

    texts_for_encode = [
        NOMIC_DOC_PREFIX + _section_text_for_encode(heading, text)
        for _bibcode, _idx, heading, text, _sha in survivors
    ]
    vectors = encode_batch(model, texts_for_encode, dimensions=dimensions)

    if len(vectors) != len(survivors):
        raise RuntimeError(
            f"Encoder returned {len(vectors)} vectors for {len(survivors)} inputs"
        )

    write_rows: list[tuple[str, int, str, str, Sequence[float]]] = [
        (bibcode, idx, heading, sha, vec)
        for (bibcode, idx, heading, _text, sha), vec in zip(survivors, vectors)
    ]
    return write_batch_copy(conn, write_rows)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point. Returns process exit code."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    parser = _build_parser()
    args = parser.parse_args(argv)

    logger.info(
        "section-embed start: model=%s dims=%d batch=%d range=[%s, %s] dry_run=%s",
        args.model,
        args.dimensions,
        args.batch_size,
        args.start_bibcode,
        args.end_bibcode,
        args.dry_run,
    )

    if args.dry_run:
        logger.info("dry-run: skipping model load and DB writes")
        return 0

    model = _load_model(args.model)

    conn = get_connection(args.dsn) if args.dsn is not None else get_connection()
    try:
        total_written = 0
        total_seen = 0
        for batch in _chunked(
            iter_sections(conn, args.start_bibcode, args.end_bibcode),
            args.batch_size,
        ):
            total_seen += len(batch)
            written = _process_batch(conn, model, batch, args.dimensions)
            total_written += written
            logger.info(
                "batch: seen=%d written=%d (cum seen=%d written=%d)",
                len(batch),
                written,
                total_seen,
                total_written,
            )
    finally:
        conn.close()

    logger.info("section-embed done: seen=%d written=%d", total_seen, total_written)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI passthrough
    sys.exit(main())
