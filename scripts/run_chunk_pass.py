#!/usr/bin/env python3
"""Run the INDUS chunk-embedding ingest pass over papers (PRD: chunk-embeddings-build).

Mirrors :mod:`scripts.run_ner_pass` in shape: argparse, optional batch-scope
guard, lazy connection bring-up, optional ``--dry-run`` that bypasses Qdrant,
and a final TOTAL log line summarising :class:`scix.extract.chunk_pass.BatchStats`.

ALWAYS wrap in scix-batch (see CLAUDE.md memory rule on systemd-oomd):

    scix-batch --mem-high 16G --mem-max 24G \\
        python scripts/run_chunk_pass.py \\
            --batch-size 200 \\
            --inference-batch 64

Other examples:

    # Smoke run: first 50 papers, no Qdrant writes (no QDRANT_URL needed)
    python scripts/run_chunk_pass.py --max-papers 50 --dry-run

    # Resume from a watermark bibcode (rarely needed — checkpoints handle this)
    scix-batch python scripts/run_chunk_pass.py \\
        --since-bibcode 2020ApJ...900....1S

The pipeline is resumable: every batch is checkpointed in ``ingest_log``,
so a killed run picks up at the next un-checkpointed batch on rerun.

Environment:
  * ``QDRANT_URL`` — required unless ``--dry-run`` is passed; the script exits
    with status ``2`` if it is missing.
  * ``SCIX_DSN`` — Postgres DSN; can be overridden with ``--dsn``.
  * ``SYSTEMD_SCOPE`` — required when ``--require-batch-scope`` is set.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from scix.db import get_connection  # noqa: E402
from scix.extract.chunk_pass import (  # noqa: E402
    CHUNKS_COLLECTION,
    INDUSEmbedder,
    run,
)
from scix.extract.chunk_pass.embedder import DEFAULT_INFERENCE_BATCH  # noqa: E402
from scix.extract.chunk_pass.pipeline import (  # noqa: E402
    DEFAULT_BATCH_SIZE,
    DEFAULT_PARSER_VERSION,
)

logger = logging.getLogger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Papers per checkpointed batch (default: {DEFAULT_BATCH_SIZE}).",
    )
    p.add_argument(
        "--inference-batch",
        type=int,
        default=DEFAULT_INFERENCE_BATCH,
        help=f"INDUS encoder batch size (default: {DEFAULT_INFERENCE_BATCH}).",
    )
    p.add_argument(
        "--since-bibcode",
        default=None,
        help="Resume watermark — only process bibcodes strictly greater than this.",
    )
    p.add_argument(
        "--max-papers",
        type=int,
        default=None,
        help="Cap total papers processed (for sample / dev runs).",
    )
    p.add_argument(
        "--collection",
        default=CHUNKS_COLLECTION,
        help=f"Qdrant collection name (default: {CHUNKS_COLLECTION}).",
    )
    p.add_argument(
        "--parser-version",
        default=DEFAULT_PARSER_VERSION,
        help=(
            "Parser-version stamp — written to chunk point ids and payloads. "
            f"Default: {DEFAULT_PARSER_VERSION}."
        ),
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Chunk + embed but skip Qdrant upsert and ingest_log checkpoint.",
    )
    p.add_argument(
        "--require-batch-scope",
        action="store_true",
        help="Refuse to run unless invoked under systemd-run scope (CLAUDE.md rule).",
    )
    p.add_argument("--dsn", default=None, help="Database DSN; defaults to SCIX_DSN.")
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.require_batch_scope and "SYSTEMD_SCOPE" not in os.environ:
        sys.stderr.write(
            "ERROR: --require-batch-scope set but SYSTEMD_SCOPE not in environment.\n"
            "       Run via: scix-batch python scripts/run_chunk_pass.py ...\n"
        )
        return 2

    qdrant_url = os.environ.get("QDRANT_URL")
    if not args.dry_run and not qdrant_url:
        sys.stderr.write(
            "ERROR: QDRANT_URL is not set. Either export QDRANT_URL=... or pass --dry-run.\n"
        )
        return 2

    embedder = INDUSEmbedder(inference_batch=args.inference_batch)

    qdrant_client: object | None
    if args.dry_run:
        # Pipeline gates Qdrant access on ``dry_run=True``; passing None is
        # safe and avoids needing a live URL for smoke runs.
        qdrant_client = None
    else:
        from qdrant_client import QdrantClient  # local import — heavy

        qdrant_client = QdrantClient(url=qdrant_url, timeout=30.0)
        # Ensure the collection exists before the first batch — first-time
        # production runs auto-create. Idempotent: re-running on an existing
        # collection just re-asserts the payload indexes.
        from scix.extract.chunk_pass import ensure_collection

        ensure_collection(qdrant_client, collection_name=args.collection)

    conn = get_connection(args.dsn)
    try:
        if args.dry_run:
            logger.info("DRY RUN — Qdrant upsert and ingest_log checkpoint suppressed.")

        totals = run(
            conn,
            embedder,
            qdrant_client,
            batch_size=args.batch_size,
            since_bibcode=args.since_bibcode,
            max_papers=args.max_papers,
            parser_version=args.parser_version,
            dry_run=args.dry_run,
            collection_name=args.collection,
        )
        logger.info(
            "TOTAL: papers_seen=%d papers_with_chunks=%d chunks_emitted=%d "
            "chunks_uploaded=%d elapsed_chunk_s=%.2f elapsed_embed_s=%.2f "
            "elapsed_upload_s=%.2f",
            totals.papers_seen,
            totals.papers_with_chunks,
            totals.chunks_emitted,
            totals.chunks_uploaded,
            totals.elapsed_chunk_s,
            totals.elapsed_embed_s,
            totals.elapsed_upload_s,
        )
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
