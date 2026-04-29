#!/usr/bin/env python3
"""Run the GLiNER NER backfill pass over papers (dbl.3).

ALWAYS wrap in scix-batch (see CLAUDE.md memory rule on systemd-oomd):

    scix-batch --mem-high 8G --mem-max 12G \\
        python scripts/run_ner_pass.py --target abstract

Phase 1 production recipe (battle-tested 2026-04-25 — survives a long
abstract that would OOM at default settings):

    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \\
        scix-batch --mem-high 16G --mem-max 24G \\
        python scripts/run_ner_pass.py \\
            --target abstract \\
            --batch-size 1000 \\
            --inference-batch 8 \\
            --max-text-chars 3500

Phase 2 (body_sections — pre-parsed methods + introduction sections from
``papers_fulltext.sections`` for ~14.94M papers, ~3-5 days at 5090
GLiNER speeds):

    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \\
        scix-batch --mem-high 24G --mem-max 32G \\
        python scripts/run_ner_pass.py \\
            --target body_sections \\
            --batch-size 200 \\
            --inference-batch 4 \\
            --max-text-chars 8000 \\
            --source-version gliner_large-v2.5/v1_body_sections

Why those flags differ from Phase 1: methods + introduction concatenated
text is several KB per paper (vs. ~1 KB for an abstract). Smaller batch
+ larger char cap keeps VRAM comfortable while still benefiting from
GLiNER-large's longer truncation window per section paragraph. Pass a
distinct ``--source-version`` so ``entities.source_version`` records
which pass produced each row — abstract vs. body_sections.

Section role filter: ``body_sections`` keeps headings classified as
``method`` (methods/observations/data-reduction/...) or ``background``
(introduction/motivation/related-work/...) by ``scix.section_role``.
Bibliography, results, discussion, conclusion, acknowledgments, and
appendix sections are dropped — they flood the entity table with author
surnames mis-typed as location/organism, or contain prose that looks
like named entities without actually introducing software/datasets/
methods. Operators can override the role filter at the library layer
via ``run(..., section_roles=...)`` but the CLI exposes only the bead-
spec default (``method`` + ``background``).

Why those flags:
  - inference-batch 8 (not 16): activation memory scales O(seq_len^2);
    a single ~768-token abstract at batch=16 spikes ~4 GB and OOMs the
    5090 once VRAM is shared with other processes. batch=8 leaves
    enough headroom and barely costs throughput (~75 docs/s either way).
  - max-text-chars 3500: caps inputs around the GLiNER-large 768-token
    window. Anything longer just wastes attention compute on truncated
    suffix.
  - PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True: lets PyTorch
    reuse fragmented VRAM blocks instead of failing with "X reserved
    but unallocated".

Other examples:
    # Sample run: first 1000 abstracts, dry-run (no DB writes)
    python scripts/run_ner_pass.py --target abstract --max-papers 1000 --dry-run

    # Resume from a watermark bibcode (rarely needed — checkpoints handle this)
    scix-batch python scripts/run_ner_pass.py --target abstract \\
        --since-bibcode 2020ApJ...900....1S

The pipeline is resumable: every batch is checkpointed in ``ingest_log``,
so a killed run picks up at the next un-checkpointed batch on rerun.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from scix.db import get_connection
from scix.extract.ner_pass import (  # noqa: E402
    DEFAULT_CONFIDENCE,
    DEFAULT_INFERENCE_BATCH,
    DEFAULT_MAX_TEXT_CHARS,
    DEFAULT_MODEL_NAME,
    NER_SOURCE_VERSION,
    GlinerExtractor,
    run,
)

logger = logging.getLogger(__name__)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument(
        "--target",
        choices=("abstract", "body", "body_sections"),
        default="abstract",
        help=(
            "Source of input text. 'abstract' / 'body' read papers.<col>; "
            "'body_sections' reads pre-parsed methods + introduction sections "
            "from papers_fulltext.sections (default: abstract)."
        ),
    )
    p.add_argument("--batch-size", type=int, default=1000, help="Papers per checkpointed batch.")
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
        "--confidence",
        type=float,
        default=DEFAULT_CONFIDENCE,
        help=f"GLiNER mention confidence floor (default: {DEFAULT_CONFIDENCE}).",
    )
    p.add_argument("--model", default=DEFAULT_MODEL_NAME, help="HF model id.")
    p.add_argument(
        "--source-version",
        default=NER_SOURCE_VERSION,
        help="Stamp written to entities.source_version. Bump for new label set or model.",
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
        help=f"Skip texts longer than this (default: {DEFAULT_MAX_TEXT_CHARS}).",
    )
    p.add_argument(
        "--compile",
        dest="compile_model",
        action="store_true",
        help="torch.compile the model (~60s warmup, +30-50%% steady-state).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Run inference but skip DB writes (for sample / quality checks).",
    )
    p.add_argument(
        "--require-batch-scope",
        action="store_true",
        help="Refuse to run unless invoked under systemd-run scope (CLAUDE.md rule).",
    )
    p.add_argument("--dsn", default=None, help="Database DSN; defaults to SCIX_DSN.")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.require_batch_scope and "SYSTEMD_SCOPE" not in os.environ:
        sys.stderr.write(
            "ERROR: --require-batch-scope set but SYSTEMD_SCOPE not in environment.\n"
            "       Run via: scix-batch python scripts/run_ner_pass.py ...\n"
        )
        return 2

    extractor = GlinerExtractor(
        model_name=args.model,
        confidence=args.confidence,
        inference_batch=args.inference_batch,
        max_text_chars=args.max_text_chars,
        compile_model=args.compile_model,
    )

    conn = get_connection(args.dsn)
    try:
        if args.dry_run:
            logger.info("DRY RUN — DB writes suppressed.")
            from scix.extract.ner_pass import iter_paper_batches

            n_papers = 0
            n_mentions = 0
            for batch in iter_paper_batches(
                conn,
                target=args.target,
                batch_size=args.batch_size,
                since_bibcode=args.since_bibcode,
                max_papers=args.max_papers,
            ):
                per_doc = extractor.predict(batch)
                n_papers += len(batch)
                for paper, mentions in zip(batch, per_doc, strict=True):
                    for m in mentions:
                        n_mentions += 1
                        sys.stdout.write(
                            f"{paper.bibcode}\t{m.entity_type}\t{m.confidence:.2f}\t"
                            f"{m.canonical_name}\t{m.surface_text}\n"
                        )
            logger.info("dry-run done: %d papers, %d mentions", n_papers, n_mentions)
            return 0

        totals = run(
            conn,
            extractor,
            target=args.target,
            batch_size=args.batch_size,
            since_bibcode=args.since_bibcode,
            max_papers=args.max_papers,
            source_version=args.source_version,
        )
        logger.info(
            "TOTAL: papers=%d papers_with_mentions=%d mentions=%d "
            "new_entities=%d doc_entities=%d "
            "infer=%.1fs db=%.1fs",
            totals.papers_seen,
            totals.papers_with_mentions,
            totals.mentions_kept,
            totals.new_entities,
            totals.upserted_doc_entities,
            totals.elapsed_inference_s,
            totals.elapsed_db_s,
        )
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
