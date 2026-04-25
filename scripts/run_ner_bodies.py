#!/usr/bin/env python3
"""Section-aware GLiNER NER pass over paper bodies (M2 of full-text-apps v2).

Reads ``papers.body``, splits each body into sections via
``scix.section_parser.parse_sections``, keeps only sections whose role
classifier (``scix.section_role.classify_section_role``) returns one of
``{'method', 'result'}``, and runs the existing
``scix.extract.ner_pass.GlinerExtractor`` over the kept section text only.

Why section-aware: bibliographies and introductions dominate the lexical
noise in body NER. The S2 section-role classifier (commit 62d8740) lets
us scope inference to the parts of a paper where named entities — software,
datasets, instruments — actually live. Keeping only ``method`` and
``result`` sections drops ~60% of body tokens and removes the references
section entirely (it would flood the entity table with author surnames
mis-typed as ``location``/``organism``).

Writes go to ``staging.extractions`` with ``extraction_type='ner_body'``,
``source='ner_body'``, ``section_name`` (comma-joined list of kept sections),
``char_offset`` (first kept section's start), and a structured ``payload``
that preserves the per-section mention breakdown::

    {
      "sections": [
        {
          "name": "methods", "role": "method",
          "start": 1234, "end": 4567,
          "mentions": [{"surface": "...", "canon": "...",
                        "type": "software", "conf": 0.91}, ...]
        },
        ...
      ],
      "model": "gliner-community/gliner_large-v2.5",
      "source_version": "ner_body/v1"
    }

Always wrap heavy production runs in scix-batch (see CLAUDE.md memory rule
on systemd-oomd):

    scix-batch python scripts/run_ner_bodies.py --max-papers 1000 --dry-run

    scix-batch --mem-high 16G --mem-max 24G \\
        python scripts/run_ner_bodies.py --allow-prod

The pipeline is resumable: the loop walks bibcodes in order and
``--since-bibcode`` lets you continue from a checkpoint.

The full 100K stratified pilot is **not** executed by this script — it is
documented in ``docs/runbooks/run_ner_bodies.md`` and requires a GPU
window plus ``--allow-prod``. This script ships the pipeline and a smoke
runner so the production run is a one-liner when the operator is ready.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

# Make src/ importable when running from a worktree without an editable
# install — same pattern as scripts/run_ner_pass.py.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from scix.db import (  # noqa: E402
    DEFAULT_DSN,
    get_connection,
    is_production_dsn,
    redact_dsn,
)
from scix.extract.ner_pass import (  # noqa: E402
    DEFAULT_CONFIDENCE,
    DEFAULT_INFERENCE_BATCH,
    DEFAULT_MAX_TEXT_CHARS,
    DEFAULT_MODEL_NAME,
    GlinerExtractor,
    Mention,
    PaperInput,
    iter_paper_batches,
)
from scix.section_parser import parse_sections  # noqa: E402
from scix.section_role import (  # noqa: E402
    ROLE_METHOD,
    ROLE_RESULT,
    classify_section_role,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: extraction_type written to staging.extractions for body NER rows. The
#: MCP entity tool filters on ``source='ner_body'``; extraction_type is the
#: secondary index for promotion-script joins.
EXTRACTION_TYPE = "ner_body"

#: source stamp written to staging.extractions.source. Distinct from
#: ``ner_v1`` (abstract-level NER) so downstream consumers can tell the
#: two pipelines apart.
EXTRACTION_SOURCE = "ner_body"

#: Default extraction_version. Bump when section-filter logic, label set,
#: or model changes so the unique constraint
#: (bibcode, extraction_type, extraction_version) does not block re-runs.
DEFAULT_EXTRACTION_VERSION = "ner_body/v1"

#: Section roles we keep. Per the M2 spec: only method + result. The
#: ``observations`` keyword maps to ``method`` in section_role.py so this
#: covers astro observation sections too.
KEPT_ROLES: frozenset[str] = frozenset({ROLE_METHOD, ROLE_RESULT})

#: Confidence-tier mapping. Smaller is better; matches the convention in
#: scripts/run_claim_extractor.py so the staging table is uniform.
TIER_HIGH = 1
TIER_MEDIUM = 2
TIER_LOW = 3


# ---------------------------------------------------------------------------
# Section filtering
# ---------------------------------------------------------------------------


def select_kept_sections(
    body: str | None,
) -> list[tuple[str, str, int, int, str]]:
    """Parse ``body`` into sections and return only method/result sections.

    Returns a list of ``(name, role, start, end, text)`` tuples. Sections
    with empty text after the header are dropped.
    """
    if not body:
        return []

    kept: list[tuple[str, str, int, int, str]] = []
    for name, start, end, text in parse_sections(body):
        role = classify_section_role(name)
        if role not in KEPT_ROLES:
            continue
        stripped = text.strip()
        if not stripped:
            continue
        kept.append((name, role, start, end, stripped))
    return kept


# ---------------------------------------------------------------------------
# Tier mapping
# ---------------------------------------------------------------------------


def confidence_to_tier(conf: float) -> int:
    """Map a GLiNER confidence score to the 1..3 tier convention."""
    if conf >= 0.85:
        return TIER_HIGH
    if conf >= 0.70:
        return TIER_MEDIUM
    return TIER_LOW


# ---------------------------------------------------------------------------
# Payload + DB writer
# ---------------------------------------------------------------------------


def _mention_to_dict(m: Mention) -> dict[str, Any]:
    return {
        "surface": m.surface_text,
        "canon": m.canonical_name,
        "type": m.entity_type,
        "conf": round(m.confidence, 4),
    }


def build_payload(
    sections: list[tuple[str, str, int, int, str]],
    per_section_mentions: list[list[Mention]],
    *,
    model_name: str,
    source_version: str,
) -> dict[str, Any]:
    """Build the JSONB payload row for one paper.

    ``sections`` and ``per_section_mentions`` are aligned by index: the
    nth section matches the nth mention list. Empty sections (no kept
    mentions) are still recorded so downstream consumers know we looked
    there and found nothing — important for distinguishing "section
    skipped" (not in payload) from "section scanned, no entities".
    """
    out_sections: list[dict[str, Any]] = []
    for (name, role, start, end, _text), mentions in zip(
        sections, per_section_mentions, strict=True
    ):
        out_sections.append(
            {
                "name": name,
                "role": role,
                "start": start,
                "end": end,
                "mentions": [_mention_to_dict(m) for m in mentions],
            }
        )
    return {
        "sections": out_sections,
        "model": model_name,
        "source_version": source_version,
    }


_INSERT_SQL: str = (
    "INSERT INTO staging.extractions "
    "(bibcode, extraction_type, extraction_version, payload, "
    " source, confidence_tier, section_name, char_offset) "
    "VALUES (%s, %s, %s, %s::jsonb, %s, %s, %s, %s) "
    "ON CONFLICT (bibcode, extraction_type, extraction_version) "
    "DO UPDATE SET payload = EXCLUDED.payload, "
    "             source = EXCLUDED.source, "
    "             confidence_tier = EXCLUDED.confidence_tier, "
    "             section_name = EXCLUDED.section_name, "
    "             char_offset = EXCLUDED.char_offset, "
    "             created_at = now()"
)


def insert_paper_row(
    conn: Any,
    bibcode: str,
    sections: list[tuple[str, str, int, int, str]],
    per_section_mentions: list[list[Mention]],
    *,
    model_name: str,
    source_version: str,
    extraction_version: str,
) -> int:
    """Upsert one row per paper into ``staging.extractions``.

    Returns 1 if a row was written, 0 if no kept sections survived
    (we do not write empty rows — a paper with no method/result sections
    is just absent from the body-NER staging table).
    """
    if not sections:
        return 0

    # Aggregate provenance into the column-level fields. The full per-
    # section breakdown lives in payload['sections'].
    section_name = ",".join(name for name, _r, _s, _e, _t in sections)
    char_offset = sections[0][2]

    flat_mentions: list[Mention] = [m for ms in per_section_mentions for m in ms]
    if flat_mentions:
        best_conf = max(m.confidence for m in flat_mentions)
        tier = confidence_to_tier(best_conf)
    else:
        # Sections were scanned but produced no mentions. Still record so
        # downstream consumers can distinguish "we looked" from "we skipped".
        tier = TIER_LOW

    payload = json.dumps(
        build_payload(
            sections,
            per_section_mentions,
            model_name=model_name,
            source_version=source_version,
        )
    )

    with conn.cursor() as cur:
        cur.execute(
            _INSERT_SQL,
            (
                bibcode,
                EXTRACTION_TYPE,
                extraction_version,
                payload,
                EXTRACTION_SOURCE,
                tier,
                section_name,
                char_offset,
            ),
        )
    return 1


# ---------------------------------------------------------------------------
# Per-batch processor
# ---------------------------------------------------------------------------


def process_paper(
    extractor: GlinerExtractor,
    bibcode: str,
    body: str,
) -> tuple[list[tuple[str, str, int, int, str]], list[list[Mention]]]:
    """Run section selection + GLiNER inference for one paper.

    Returns ``(kept_sections, per_section_mentions)``. Both lists are
    aligned by index. If no sections survive the role filter, both are
    empty and the caller skips the DB write.
    """
    kept = select_kept_sections(body)
    if not kept:
        return [], []

    # Build a PaperInput per kept section so the extractor sees each
    # section as an independent document. The bibcode is suffixed with
    # the section name so the Mention.bibcode field carries provenance
    # back through extractor.predict; we strip the suffix when writing
    # to the DB (the column is the canonical bibcode).
    inputs: list[PaperInput] = [
        PaperInput(bibcode=bibcode, text=text) for _n, _r, _s, _e, text in kept
    ]
    per_section = extractor.predict(inputs)
    return kept, per_section


def run_pipeline(
    conn: Any,
    extractor: GlinerExtractor,
    *,
    target: str = "body",
    batch_size: int = 200,
    since_bibcode: str | None = None,
    max_papers: int | None = None,
    dry_run: bool = False,
    extraction_version: str = DEFAULT_EXTRACTION_VERSION,
) -> dict[str, int]:
    """Stream batches, run section-aware NER, write to staging.

    Returns a totals dict so callers (CLI + tests) can format their own
    report without scraping logger output.
    """
    totals = {
        "papers_seen": 0,
        "papers_with_kept_sections": 0,
        "sections_scanned": 0,
        "mentions_kept": 0,
        "rows_inserted": 0,
    }

    for batch in iter_paper_batches(
        conn,
        target=target,
        batch_size=batch_size,
        since_bibcode=since_bibcode,
        max_papers=max_papers,
    ):
        for paper in batch:
            totals["papers_seen"] += 1
            kept, per_section = process_paper(extractor, paper.bibcode, paper.text)
            if not kept:
                continue
            totals["papers_with_kept_sections"] += 1
            totals["sections_scanned"] += len(kept)
            totals["mentions_kept"] += sum(len(ms) for ms in per_section)

            if dry_run:
                for (name, _role, start, _end, _text), mentions in zip(
                    kept, per_section, strict=True
                ):
                    for m in mentions:
                        sys.stdout.write(
                            f"{paper.bibcode}\t{name}\t{start}\t"
                            f"{m.entity_type}\t{m.confidence:.2f}\t"
                            f"{m.canonical_name}\t{m.surface_text}\n"
                        )
            else:
                totals["rows_inserted"] += insert_paper_row(
                    conn,
                    paper.bibcode,
                    kept,
                    per_section,
                    model_name=extractor.model_name,
                    source_version=extraction_version,
                    extraction_version=extraction_version,
                )

        if not dry_run:
            conn.commit()

        logger.info(
            "progress: papers=%d kept=%d sections=%d mentions=%d inserted=%d",
            totals["papers_seen"],
            totals["papers_with_kept_sections"],
            totals["sections_scanned"],
            totals["mentions_kept"],
            totals["rows_inserted"],
        )

    return totals


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument(
        "--max-papers",
        type=int,
        default=None,
        help="Cap total papers processed (for sample / smoke runs).",
    )
    p.add_argument(
        "--since-bibcode",
        default=None,
        help="Resume watermark — only process bibcodes strictly greater than this.",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=200,
        help="Papers per cursor batch (default: 200).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Run inference but skip DB writes (sample / quality checks).",
    )
    p.add_argument(
        "--allow-prod",
        action="store_true",
        help="Required to run against the production DSN (mirrors run_ner_pass.py).",
    )
    p.add_argument(
        "--confidence",
        type=float,
        default=DEFAULT_CONFIDENCE,
        help=f"GLiNER mention confidence floor (default: {DEFAULT_CONFIDENCE}).",
    )
    p.add_argument("--model", default=DEFAULT_MODEL_NAME, help="HF model id.")
    p.add_argument(
        "--source-version",
        default=DEFAULT_EXTRACTION_VERSION,
        help=(
            "Stamp written to staging.extractions.extraction_version. Bump "
            "when section-filter logic or model changes."
        ),
    )
    p.add_argument(
        "--inference-batch",
        type=int,
        default=DEFAULT_INFERENCE_BATCH,
        help="GLiNER batch_predict batch size.",
    )
    p.add_argument(
        "--max-text-chars",
        type=int,
        default=DEFAULT_MAX_TEXT_CHARS,
        help=f"Skip section texts longer than this (default: {DEFAULT_MAX_TEXT_CHARS}).",
    )
    p.add_argument(
        "--compile",
        dest="compile_model",
        action="store_true",
        help="torch.compile the model (~60s warmup, +30-50%% steady-state).",
    )
    p.add_argument("--dsn", default=None, help="Database DSN; defaults to SCIX_DSN.")
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_argparser().parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    dsn = args.dsn or DEFAULT_DSN
    if is_production_dsn(dsn) and not args.allow_prod:
        logger.error(
            "Refusing to run against production DSN %s — pass --allow-prod to override",
            redact_dsn(dsn),
        )
        return 2

    logger.info(
        "Running body NER on %s (dry_run=%s, since=%s, max=%s, version=%s)",
        redact_dsn(dsn),
        args.dry_run,
        args.since_bibcode,
        args.max_papers,
        args.source_version,
    )

    extractor = GlinerExtractor(
        model_name=args.model,
        confidence=args.confidence,
        inference_batch=args.inference_batch,
        max_text_chars=args.max_text_chars,
        compile_model=args.compile_model,
    )

    conn = get_connection(dsn)
    try:
        totals = run_pipeline(
            conn,
            extractor,
            target="body",
            batch_size=args.batch_size,
            since_bibcode=args.since_bibcode,
            max_papers=args.max_papers,
            dry_run=args.dry_run,
            extraction_version=args.source_version,
        )
        logger.info(
            "DONE: papers=%d kept=%d sections=%d mentions=%d inserted=%d",
            totals["papers_seen"],
            totals["papers_with_kept_sections"],
            totals["sections_scanned"],
            totals["mentions_kept"],
            totals["rows_inserted"],
        )
    finally:
        conn.close()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
