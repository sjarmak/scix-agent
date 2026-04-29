#!/usr/bin/env python3
"""Harvest HuggingFace Hub scientific-tagged models into entity_dictionary.

Walks the HF Hub model index and ingests two overlapping subsets:

1. Models tagged with one of the scientific topic tags listed in
   ``SCIENTIFIC_TAGS`` (e.g. medical, biology, chemistry, physics, astronomy,
   climate, earth, geoscience, scientific).
2. Models whose tags include an ``arxiv:<id>`` reference where ``<id>``
   matches an arxiv identifier already in the ``papers`` table — i.e. the
   model is associated with a paper in our corpus.

Each match is upserted into ``entity_dictionary`` with
``entity_type='software'`` and ``source='huggingface'``. Metadata captures
``tags``, ``arxiv_id`` (list), and ``downloads`` for downstream ranking.

The ``--mode`` flag selects between two retrieval strategies:

* ``tags-only`` — issues one ``list_models`` call per scientific tag.
  Fast but limited to the explicitly-tagged subset (~10–15k models).
* ``full`` (default) — paginates the entire HF Hub index, classifies each
  model client-side. Slower (~minutes) but recovers SciBERT/BioBERT-class
  models that lack scientific tags but reference an in-corpus arxiv paper.
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Iterator

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from scix.db import get_connection
from scix.dictionary import bulk_load
from scix.harvest_utils import HarvestRunLog

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SCIENTIFIC_TAGS: frozenset[str] = frozenset(
    {
        "medical",
        "biology",
        "chemistry",
        "physics",
        "astronomy",
        "climate",
        "earth",
        "geoscience",
        "scientific",
    }
)

# Matches an HF tag of the form "arxiv:1903.10676" or "arxiv:cs/0101001".
# HF historically also uses "arxiv:abs/1903.10676" (rare).
_ARXIV_TAG_RE = re.compile(
    r"^arxiv:(?:abs/)?([0-9]{4}\.[0-9]{4,5}|[a-z\-]+/[0-9]{7})$",
    re.IGNORECASE,
)

# Matches an arxiv id embedded in an ADS ``identifier`` array entry.
# Examples: "arXiv:2108.03126", "10.48550/arXiv.2108.03126",
#           "2021arXiv210803126L" (intentionally not matched — bibcode form).
_ARXIV_ID_FROM_IDENTIFIER_RE = re.compile(
    r"arXiv[:.](?:abs/)?([0-9]{4}\.[0-9]{4,5}|[a-z\-]+/[0-9]{7})",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# HF API client (lazy)
# ---------------------------------------------------------------------------

_hf_api: Any | None = None


def _get_hf_api() -> Any:
    """Return a shared ``HfApi`` instance.

    Lazily imports ``huggingface_hub`` so unit tests can patch
    ``_get_hf_api`` without paying the import cost.
    """
    global _hf_api
    if _hf_api is None:
        from huggingface_hub import HfApi  # local import — lazy

        _hf_api = HfApi(user_agent="scix-experiments/1.0")
    return _hf_api


# ---------------------------------------------------------------------------
# Arxiv-id helpers
# ---------------------------------------------------------------------------


def extract_arxiv_id_from_tag(tag: str) -> str | None:
    """Return the normalised arxiv id from an HF tag, or ``None``.

    The returned id is lowercase and stripped of the ``arxiv:`` prefix.
    Old-style ids (e.g. ``cs/0101001``) are preserved verbatim.
    """
    m = _ARXIV_TAG_RE.match(tag.strip()) if tag else None
    return m.group(1).lower() if m else None


def extract_arxiv_ids_from_identifier(identifier: Iterable[str] | None) -> set[str]:
    """Extract all arxiv ids from a paper ``identifier`` array.

    Looks for ``arXiv:<id>`` and ``10.48550/arXiv.<id>`` shapes.
    Bibcode-form ids (``2021arXiv210803126L``) are deliberately ignored —
    HF tags never use that shape, so matching them would only inflate
    false positives.
    """
    if not identifier:
        return set()
    ids: set[str] = set()
    for raw in identifier:
        if not raw:
            continue
        for m in _ARXIV_ID_FROM_IDENTIFIER_RE.finditer(str(raw)):
            ids.add(m.group(1).lower())
    return ids


def fetch_corpus_arxiv_ids(conn: Any) -> set[str]:
    """Return the set of arxiv ids referenced by papers in the corpus.

    Uses a server-side cursor — the ``papers`` table has ~32M rows and
    ~1.6M arxiv-tagged ones; materialising the full list in client memory
    is fine (~few hundred MB of strings) but the cursor avoids loading the
    full result set into a single fetch buffer.
    """
    arxiv_ids: set[str] = set()
    with conn.cursor(name="hf_corpus_arxiv") as cur:
        cur.itersize = 50_000
        cur.execute(
            """
            SELECT identifier FROM papers
            WHERE EXISTS (
                SELECT 1 FROM unnest(identifier) i
                WHERE i ILIKE 'arXiv:%' OR i ILIKE '%arXiv.%'
            )
            """
        )
        for (idents,) in cur:
            arxiv_ids |= extract_arxiv_ids_from_identifier(idents)
    logger.info("Loaded %d unique arxiv ids from corpus", len(arxiv_ids))
    return arxiv_ids


# ---------------------------------------------------------------------------
# Classification + parsing
# ---------------------------------------------------------------------------


def classify_model(
    tags: Iterable[str],
    *,
    scientific_tags: frozenset[str],
    corpus_arxiv_ids: frozenset[str] | set[str],
) -> tuple[bool, list[str], list[str]]:
    """Decide whether a model is in-scope and extract its arxiv refs.

    Returns ``(matched, matched_scientific_tags, matched_arxiv_ids)``.

    * ``matched`` is True if the model has at least one scientific tag OR
      at least one arxiv tag matching ``corpus_arxiv_ids``.
    * ``matched_scientific_tags`` is the subset of ``tags`` that are in
      ``scientific_tags``.
    * ``matched_arxiv_ids`` is the subset of arxiv ids extracted from
      ``tags`` that are in ``corpus_arxiv_ids`` (preserved in tag order
      with duplicates removed).
    """
    sci: list[str] = []
    arxiv_corpus: list[str] = []
    seen_arxiv: set[str] = set()
    for tag in tags or ():
        if not tag:
            continue
        if tag in scientific_tags:
            if tag not in sci:
                sci.append(tag)
            continue
        aid = extract_arxiv_id_from_tag(tag)
        if aid and aid in corpus_arxiv_ids and aid not in seen_arxiv:
            arxiv_corpus.append(aid)
            seen_arxiv.add(aid)
    matched = bool(sci) or bool(arxiv_corpus)
    return matched, sci, arxiv_corpus


def _coerce_int(value: Any) -> int | None:
    """Best-effort int coercion that returns ``None`` on missing/invalid."""
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def model_to_entry(
    model: Any,
    *,
    scientific_tags: frozenset[str],
    corpus_arxiv_ids: frozenset[str] | set[str],
) -> dict[str, Any] | None:
    """Convert an HF ``ModelInfo`` to an ``entity_dictionary`` entry.

    Returns ``None`` if the model does not match any inclusion criterion
    or lacks a usable model id.
    """
    model_id = getattr(model, "id", None) or getattr(model, "modelId", None)
    if not model_id:
        return None
    model_id = str(model_id).strip()
    if not model_id:
        return None

    tags: list[str] = list(getattr(model, "tags", None) or [])
    matched, sci_tags, arxiv_ids = classify_model(
        tags,
        scientific_tags=scientific_tags,
        corpus_arxiv_ids=corpus_arxiv_ids,
    )
    if not matched:
        return None

    # canonical_name = full HF id ("org/model"); alias = short name.
    aliases: list[str] = []
    if "/" in model_id:
        short = model_id.split("/", 1)[1]
        if short and short != model_id:
            aliases.append(short)

    # Capture all arxiv ids on the model (not just corpus matches) so the
    # downstream linker can choose to expand the corpus later.
    all_arxiv_ids: list[str] = []
    seen: set[str] = set()
    for tag in tags:
        aid = extract_arxiv_id_from_tag(tag)
        if aid and aid not in seen:
            all_arxiv_ids.append(aid)
            seen.add(aid)

    metadata: dict[str, Any] = {
        "tags": tags,
        "arxiv_id": all_arxiv_ids,
        "arxiv_id_in_corpus": arxiv_ids,
        "downloads": _coerce_int(getattr(model, "downloads", None)) or 0,
        "matched_scientific_tags": sci_tags,
    }
    pipeline_tag = getattr(model, "pipeline_tag", None)
    if pipeline_tag:
        metadata["pipeline_tag"] = str(pipeline_tag)
    library_name = getattr(model, "library_name", None)
    if library_name:
        metadata["library_name"] = str(library_name)

    return {
        "canonical_name": model_id,
        "entity_type": "software",
        "source": "huggingface",
        "external_id": model_id,
        "aliases": aliases,
        "metadata": metadata,
    }


# ---------------------------------------------------------------------------
# Iteration strategies
# ---------------------------------------------------------------------------


def iter_models_tags_only(
    api: Any,
    scientific_tags: Iterable[str],
) -> Iterator[Any]:
    """Yield models matching any tag in ``scientific_tags``, deduped by id."""
    seen: set[str] = set()
    for tag in scientific_tags:
        for m in api.list_models(filter=tag):
            mid = getattr(m, "id", None) or getattr(m, "modelId", None)
            if not mid or mid in seen:
                continue
            seen.add(mid)
            yield m


def iter_models_full(api: Any) -> Iterator[Any]:
    """Yield every model on the HF Hub, sorted by downloads (descending).

    Sorting by downloads lets early termination return the highest-impact
    models first if the harvest is interrupted.
    """
    yield from api.list_models(sort="downloads")


# ---------------------------------------------------------------------------
# Harvest pipeline
# ---------------------------------------------------------------------------


def harvest_models(
    api: Any,
    *,
    mode: str,
    scientific_tags: frozenset[str],
    corpus_arxiv_ids: frozenset[str] | set[str],
    limit: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Iterate HF models, classify, and return entity_dictionary entries.

    Args:
        api: HfApi instance.
        mode: ``"tags-only"`` or ``"full"``.
        scientific_tags: Set of scientific topic tags to match.
        corpus_arxiv_ids: Set of arxiv ids present in our papers table.
        limit: Optional cap on number of models scanned (for testing).

    Returns:
        A tuple ``(entries, counts)`` where ``counts`` includes
        ``scanned``, ``matched``, ``via_tag``, ``via_arxiv``, ``via_both``.
    """
    if mode == "tags-only":
        iterator = iter_models_tags_only(api, scientific_tags)
    elif mode == "full":
        iterator = iter_models_full(api)
    else:
        raise ValueError(f"Unknown mode {mode!r}; expected 'tags-only' or 'full'")

    counts = {
        "scanned": 0,
        "matched": 0,
        "via_tag": 0,
        "via_arxiv": 0,
        "via_both": 0,
    }
    entries_by_id: dict[str, dict[str, Any]] = {}
    for model in iterator:
        counts["scanned"] += 1
        if limit is not None and counts["scanned"] > limit:
            break
        entry = model_to_entry(
            model,
            scientific_tags=scientific_tags,
            corpus_arxiv_ids=corpus_arxiv_ids,
        )
        if entry is None:
            continue
        # Dedup by canonical_name — tags-only mode already dedupes,
        # full mode never duplicates, but be defensive.
        existing = entries_by_id.get(entry["canonical_name"])
        if existing is not None:
            continue
        entries_by_id[entry["canonical_name"]] = entry
        counts["matched"] += 1

        meta = entry["metadata"]
        has_tag = bool(meta.get("matched_scientific_tags"))
        has_arxiv = bool(meta.get("arxiv_id_in_corpus"))
        if has_tag and has_arxiv:
            counts["via_both"] += 1
        elif has_tag:
            counts["via_tag"] += 1
        else:
            counts["via_arxiv"] += 1

        if counts["scanned"] % 50_000 == 0:
            logger.info(
                "Progress: scanned=%d matched=%d (tag=%d arxiv=%d both=%d)",
                counts["scanned"],
                counts["matched"],
                counts["via_tag"],
                counts["via_arxiv"],
                counts["via_both"],
            )

    return list(entries_by_id.values()), counts


def run_pipeline(
    *,
    mode: str = "full",
    dsn: str | None = None,
    limit: int | None = None,
) -> int:
    """Run the full HuggingFace harvest pipeline.

    Steps:
        1. Load corpus arxiv ids (one read of ``papers``).
        2. Iterate the HF Hub model index per ``mode``.
        3. Bulk-load matches into ``entity_dictionary``.
        4. Record run in ``harvest_runs``.

    Args:
        mode: ``"tags-only"`` for fast tag-only scan or ``"full"`` for the
            full Hub index walk with arxiv-id cross-reference.
        dsn: Database connection string. Uses SCIX_DSN or default if None.
        limit: Optional cap on number of models scanned (for testing only).

    Returns:
        Number of entries upserted into entity_dictionary.
    """
    t0 = time.monotonic()

    conn = get_connection(dsn)
    run_log = HarvestRunLog(conn, "huggingface")
    try:
        run_log.start(config={"mode": mode, "limit": limit})

        corpus_arxiv_ids: set[str]
        if mode == "tags-only":
            # Tag-only mode does not need the corpus — the arxiv-id cross
            # reference is a no-op when ``classify_model`` only sees
            # scientific tags. Skip the DB read for speed.
            corpus_arxiv_ids = set()
            logger.info("Mode 'tags-only' — skipping corpus arxiv-id load")
        else:
            corpus_arxiv_ids = fetch_corpus_arxiv_ids(conn)

        api = _get_hf_api()
        entries, counts = harvest_models(
            api,
            mode=mode,
            scientific_tags=SCIENTIFIC_TAGS,
            corpus_arxiv_ids=corpus_arxiv_ids,
            limit=limit,
        )

        if not entries:
            logger.warning("No HuggingFace models matched — nothing to load")
            run_log.complete(records_fetched=counts["scanned"], records_upserted=0)
            return 0

        upserted = bulk_load(conn, entries)
        # Skip the post-harvest agent-view refresh — it touches several
        # large materialized views and frequently contends with other
        # concurrently running harvesters in this repo. Refreshes are
        # idempotent and can be triggered separately when convenient.
        run_log.complete(
            records_fetched=counts["scanned"],
            records_upserted=upserted,
            counts=counts,
            refresh_views=False,
        )
    except Exception as exc:
        try:
            run_log.fail(str(exc))
        except Exception:
            logger.warning("Failed to mark harvest_run failed")
        raise
    finally:
        conn.close()

    elapsed = time.monotonic() - t0
    logger.info(
        "HuggingFace harvest complete: scanned=%d matched=%d upserted=%d in %.1fs "
        "(via_tag=%d via_arxiv=%d via_both=%d)",
        counts["scanned"],
        counts["matched"],
        upserted,
        elapsed,
        counts["via_tag"],
        counts["via_arxiv"],
        counts["via_both"],
    )
    return upserted


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    """Parse arguments and run the HuggingFace harvest pipeline."""
    parser = argparse.ArgumentParser(
        description=(
            "Harvest scientific-tagged HuggingFace Hub models into "
            "entity_dictionary as software entities."
        ),
    )
    parser.add_argument(
        "--mode",
        choices=("tags-only", "full"),
        default="full",
        help=(
            "Iteration strategy. 'tags-only' is fast but limited to the "
            "scientific tag set (~10–15k models). 'full' (default) walks "
            "the entire Hub index and additionally includes models whose "
            "arxiv: tag matches a paper in the corpus."
        ),
    )
    parser.add_argument(
        "--dsn",
        default=None,
        help="Database connection string (uses SCIX_DSN env var if not provided)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on models scanned (for testing/dev only).",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    count = run_pipeline(mode=args.mode, dsn=args.dsn, limit=args.limit)
    logger.info("Done. %d HuggingFace entities upserted.", count)


if __name__ == "__main__":
    main()
