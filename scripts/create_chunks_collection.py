#!/usr/bin/env python3
"""Create (or verify) the ``scix_chunks_v1`` Qdrant collection.

Idempotent bootstrap for the chunk-embeddings index described in
``docs/prd/prd_chunk_embeddings_build.md``. Reads ``QDRANT_URL`` from the
environment and calls
:func:`scix.extract.chunk_pass.collection.ensure_collection`, then prints the
resulting summary as JSON to stdout.

Exit codes:
    0  success — collection + payload indexes are in place
    2  ``QDRANT_URL`` is unset (refuse to run)
    1  any other error
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import pathlib
import sys

# Make the ``src`` layout importable when run as a plain script.
_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from scix.extract.chunk_pass.collection import (  # noqa: E402
    CHUNKS_COLLECTION,
    ensure_collection,
)

logger = logging.getLogger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--collection",
        type=str,
        default=CHUNKS_COLLECTION,
        help=f"Collection name to create (default: {CHUNKS_COLLECTION})",
    )
    p.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="Qdrant client request timeout in seconds (default: 30)",
    )
    p.add_argument("--verbose", "-v", action="store_true", default=False)
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    qdrant_url = os.environ.get("QDRANT_URL")
    if not qdrant_url:
        logger.error("QDRANT_URL is not set; refusing to run")
        return 2

    try:
        from qdrant_client import QdrantClient
    except ImportError:
        logger.error(
            "qdrant-client is not installed — install via the 'search' extra"
        )
        return 1

    client = QdrantClient(url=qdrant_url, timeout=args.timeout)
    try:
        result = ensure_collection(client, collection_name=args.collection)
    except Exception as exc:  # noqa: BLE001
        logger.error("ensure_collection failed: %s", exc)
        return 1

    payload = {
        "collection": args.collection,
        **result,
    }
    print(json.dumps(payload, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
