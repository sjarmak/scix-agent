"""Post-pass driver that runs the INDUS classifier over GLiNER mentions.

Streams ``document_entities`` rows where ``match_method='gliner'`` and
the classifier hasn't run yet (``evidence ? 'agreement'`` is false),
fetches the abstract sentence containing each mention, runs
:class:`NerClassifier`, and writes ``classifier_type``,
``classifier_score``, ``agreement`` back into ``document_entities.evidence``.

Resumable + idempotent (the ``evidence ? 'agreement'`` filter naturally
skips rows that already have a verdict). Checkpoints per batch in
``ingest_log`` under ``ner_classify_pass:{first_bibcode}`` so
out-of-process restarts pick up cleanly.

Throughput target on the 5090: ~3K mentions/sec including DB I/O,
i.e. ~30 minutes per million mentions.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import psycopg
from psycopg.rows import dict_row

from scix.extract.ner_classifier import NerClassifier, extract_sentence

logger = logging.getLogger(__name__)


@dataclass
class ClassifyStats:
    rows_seen: int = 0
    agreements: int = 0
    disagreements: int = 0
    skipped_no_text: int = 0
    elapsed_inference_s: float = 0.0
    elapsed_db_s: float = 0.0


_PENDING_QUERY = """
    SELECT
        de.bibcode,
        de.entity_id,
        de.link_type,
        de.tier,
        de.confidence,
        de.evidence,
        e.canonical_name,
        e.entity_type,
        p.abstract,
        p.title
    FROM document_entities de
    JOIN entities e ON e.id = de.entity_id
    JOIN papers p   ON p.bibcode = de.bibcode
    WHERE de.match_method = 'gliner'
      AND (de.evidence IS NULL OR NOT (de.evidence ? 'agreement'))
      AND de.bibcode > %(watermark)s
    ORDER BY de.bibcode ASC
    LIMIT %(limit)s
"""

_UPDATE_SQL = """
    UPDATE document_entities
    SET evidence = COALESCE(evidence, '{}'::jsonb)
                   || jsonb_build_object(
                          'classifier_type',  %(ctype)s::text,
                          'classifier_score', %(cscore)s::float,
                          'agreement',        %(agree)s::bool
                      )
    WHERE bibcode    = %(bibcode)s
      AND entity_id  = %(entity_id)s
      AND link_type  = %(link_type)s
      AND tier       = %(tier)s
"""


def iter_pending_batches(
    conn: psycopg.Connection,
    *,
    batch_size: int,
    since_bibcode: str | None = None,
    max_rows: int | None = None,
) -> Iterator[list[dict[str, Any]]]:
    """Stream mention rows in deterministic ``(bibcode)`` order via keyset."""
    watermark = since_bibcode or ""
    remaining = max_rows
    while True:
        limit = batch_size if remaining is None else min(batch_size, remaining)
        if limit <= 0:
            return
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(_PENDING_QUERY, {"watermark": watermark, "limit": limit})
            rows = cur.fetchall()
        if not rows:
            return
        yield rows
        watermark = rows[-1]["bibcode"]
        if remaining is not None:
            remaining -= len(rows)
            if remaining <= 0:
                return


def process_batch(
    conn: psycopg.Connection,
    classifier: NerClassifier,
    rows: list[dict[str, Any]],
) -> ClassifyStats:
    """Classify and update one batch. Caller commits."""
    stats = ClassifyStats(rows_seen=len(rows))
    items: list[tuple[str, str, str]] = []
    keep_idx: list[int] = []

    for i, row in enumerate(rows):
        text = row.get("abstract") or row.get("title") or ""
        if not text:
            stats.skipped_no_text += 1
            continue
        mention = row["canonical_name"]
        sentence = extract_sentence(text, mention)
        items.append((mention, sentence, row["entity_type"]))
        keep_idx.append(i)

    t0 = time.monotonic()
    results = classifier.classify_batch(items)
    stats.elapsed_inference_s = time.monotonic() - t0

    t0 = time.monotonic()
    with conn.cursor() as cur:
        for idx, result in zip(keep_idx, results, strict=True):
            row = rows[idx]
            cur.execute(
                _UPDATE_SQL,
                {
                    "bibcode": row["bibcode"],
                    "entity_id": row["entity_id"],
                    "link_type": row["link_type"],
                    "tier": row["tier"],
                    "ctype": result.classifier_type,
                    "cscore": result.classifier_score,
                    "agree": result.agreement,
                },
            )
            if result.agreement:
                stats.agreements += 1
            else:
                stats.disagreements += 1
    stats.elapsed_db_s = time.monotonic() - t0

    return stats


# ---------------------------------------------------------------------------
# Checkpoint helpers — same shape as ner_pass
# ---------------------------------------------------------------------------


def _checkpoint_key(first_bibcode: str) -> str:
    return f"ner_classify_pass:{first_bibcode}"


def _record_checkpoint(
    conn: psycopg.Connection,
    key: str,
    *,
    records_loaded: int,
    edges_loaded: int,
) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO ingest_log
                (filename, records_loaded, edges_loaded, status, finished_at)
            VALUES (%s, %s, %s, 'complete', now())
            ON CONFLICT (filename) DO UPDATE
                SET records_loaded = EXCLUDED.records_loaded,
                    edges_loaded   = EXCLUDED.edges_loaded,
                    status         = EXCLUDED.status,
                    finished_at    = now()
            """,
            (key, records_loaded, edges_loaded),
        )


def run(
    conn: psycopg.Connection,
    classifier: NerClassifier,
    *,
    batch_size: int = 5000,
    since_bibcode: str | None = None,
    max_rows: int | None = None,
    log_every: int = 1,
) -> ClassifyStats:
    """Top-level driver: stream batches, classify, write, checkpoint."""
    totals = ClassifyStats()
    n_batches = 0

    for batch in iter_pending_batches(
        conn,
        batch_size=batch_size,
        since_bibcode=since_bibcode,
        max_rows=max_rows,
    ):
        first_bib = batch[0]["bibcode"]
        stats = process_batch(conn, classifier, batch)
        _record_checkpoint(
            conn,
            _checkpoint_key(first_bib),
            records_loaded=stats.rows_seen,
            edges_loaded=stats.agreements + stats.disagreements,
        )
        conn.commit()

        totals.rows_seen += stats.rows_seen
        totals.agreements += stats.agreements
        totals.disagreements += stats.disagreements
        totals.skipped_no_text += stats.skipped_no_text
        totals.elapsed_inference_s += stats.elapsed_inference_s
        totals.elapsed_db_s += stats.elapsed_db_s

        n_batches += 1
        if n_batches % log_every == 0:
            n_judged = stats.agreements + stats.disagreements
            agree_pct = (stats.agreements / n_judged * 100) if n_judged else 0.0
            rate = stats.rows_seen / max(stats.elapsed_inference_s, 1e-3)
            logger.info(
                "ner_classify_pass: batch %d (%s..) rows=%d agree=%.1f%% "
                "infer=%.1fs db=%.1fs rate=%.0f rows/s",
                n_batches,
                first_bib[:18],
                stats.rows_seen,
                agree_pct,
                stats.elapsed_inference_s,
                stats.elapsed_db_s,
                rate,
            )

    return totals
