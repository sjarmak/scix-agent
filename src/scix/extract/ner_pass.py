"""Batch GLiNER NER backfill over papers.abstract / papers.body (dbl.3).

Replaces the lexical-only entity coverage in document_entities with a
zero-shot NER signal that does not depend on pre-existing dictionaries.
The lexical pipeline (keyword_exact_lower + aho_corasick_abstract) is
fundamentally astro-skewed because the dictionaries are; this module is
the architectural shift that lets the long tail (software in cs.CL,
datasets in q-bio, methods in math, organisms in biomed) become visible.

Pipeline shape::

    papers (target column non-NULL)
         ──▶ stratified-by-bibcode batch (default 1000 papers)
         ──▶ GLiNER.batch_predict @ confidence threshold (default 0.7)
         ──▶ per-doc canonicalize + dedup (lower(strip(text)))
         ──▶ upsert into entities (source='gliner', source_version)
         ──▶ upsert into document_entities (match_method='gliner', tier=4)
         ──▶ checkpoint batch in ingest_log (status='complete')

Resumable: every batch is keyed in ingest_log under
``ner_pass:{target}:{first_bibcode}``; runs that crash or are killed pick
up at the next un-checkpointed batch on rerun.

Idempotent: ``entities`` upsert is keyed on
``(canonical_name, entity_type, source)``, ``document_entities`` upsert
is keyed on ``(bibcode, entity_id, link_type, tier)`` — re-running with a
higher confidence floor only refines, never duplicates.

scix-batch wrapping: the CLI (``scripts/run_ner_pass.py``) is the
intended entry point. The module itself does not invoke scix-batch; the
CLI is wrapped instead so that ``scix-batch python scripts/run_ner_pass.py
...`` enforces the systemd-oomd budget required by CLAUDE.md.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from typing import Any

import psycopg
from psycopg.rows import dict_row

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: GLiNER model used at v1. Pinned so reruns produce reproducible mentions.
DEFAULT_MODEL_NAME = "gliner-community/gliner_large-v2.5"

#: Source-version stamp written to entities.source_version. Bump when the
#: model name or label set changes so downstream tools can filter.
NER_SOURCE_VERSION = "gliner_large-v2.5/v1"

#: Confidence floor below which mentions are dropped (bead spec).
DEFAULT_CONFIDENCE = 0.7

#: Tier written to document_entities. Existing tier ordering: 0=lexical,
#: 4=ML/NER signal. See migration 028.
NER_TIER = 4

#: Frozen v1 label set. Adding a label is a v2 change — the source_version
#: must be bumped so downstream tools can tell which schema produced a row.
NER_LABELS: tuple[str, ...] = (
    "software",
    "dataset",
    "method",
    "organism",
    "chemical",
    "gene_or_protein",
    "instrument",
    "mission",
    "location",
)

#: GLiNER label string -> entities.entity_type. Most map straight through;
#: gene_or_protein collapses into the single "gene" type already used by
#: the entity-graph schema, and instrument/mission survive as-is.
LABEL_TO_ENTITY_TYPE: dict[str, str] = {
    "software": "software",
    "dataset": "dataset",
    "method": "method",
    "organism": "organism",
    "chemical": "chemical",
    "gene_or_protein": "gene",
    "instrument": "instrument",
    "mission": "mission",
    "location": "location",
}

#: Inference batch size handed to the model. Tuned for 5090 fp16 to keep
#: VRAM under ~6 GB on average abstracts.
DEFAULT_INFERENCE_BATCH = 8

#: Hard cap on input text length (chars). Abstracts above this are dropped
#: from the batch — GLiNER truncates internally anyway (768 tokens for
#: large, 384 for medium) and the long-tail is dominated by junk like
#: pasted-in tables. The text-pass-through threshold for v1 covers ~99%
#: of real ADS abstracts.
DEFAULT_MAX_TEXT_CHARS = 5000


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PaperInput:
    """One input row for the NER pass."""

    bibcode: str
    text: str


@dataclass(frozen=True)
class Mention:
    """A single GLiNER mention after canonicalization + filtering."""

    bibcode: str
    canonical_name: str  # lower-stripped surface form
    surface_text: str  # original surface form (kept for audit)
    entity_type: str
    confidence: float


@dataclass
class BatchStats:
    papers_seen: int = 0
    papers_with_mentions: int = 0
    mentions_kept: int = 0
    mentions_dropped_low_conf: int = 0
    new_entities: int = 0
    upserted_doc_entities: int = 0
    elapsed_inference_s: float = 0.0
    elapsed_db_s: float = 0.0
    discipline_coverage: dict[str, int] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Canonicalization
# ---------------------------------------------------------------------------

# Trailing parenthetical citations / disambiguators that GLiNER often
# includes in the mention span (e.g. "PyTorch (Paszke et al.)" → "pytorch").
# Stripped before canonicalization so surface variants collapse.
_TRAILING_PAREN_RE = __import__("re").compile(r"\s*\([^)]*\)\s*$")


def canonicalize(surface: str) -> str:
    """Normalize a GLiNER surface form for entity-table dedup.

    Lower-case, strip outer whitespace, drop a trailing parenthetical, and
    collapse internal whitespace runs. We deliberately do NOT remove
    punctuation — "CRISPR-Cas9" and "p53" carry meaning in the punctuation.
    """
    s = surface.strip()
    s = _TRAILING_PAREN_RE.sub("", s).strip()
    s = " ".join(s.split())
    return s.lower()


def _dedup_within_doc(mentions: Iterable[Mention]) -> list[Mention]:
    """Per-doc dedup: keep the highest-confidence mention per (canon, type)."""
    best: dict[tuple[str, str], Mention] = {}
    for m in mentions:
        key = (m.canonical_name, m.entity_type)
        prev = best.get(key)
        if prev is None or m.confidence > prev.confidence:
            best[key] = m
    return list(best.values())


# ---------------------------------------------------------------------------
# Model wrapper
# ---------------------------------------------------------------------------


class GlinerExtractor:
    """Thin wrapper around the GLiNER model with a stable interface.

    Lazy-loads on first call so that callers (CLI, tests) can construct
    the extractor without paying the ~30s torch import + model download.
    Tests substitute a stub by passing ``model=`` explicitly.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        labels: tuple[str, ...] = NER_LABELS,
        confidence: float = DEFAULT_CONFIDENCE,
        device: str = "cuda",
        model: Any = None,  # injectable for tests
        inference_batch: int = DEFAULT_INFERENCE_BATCH,
        max_text_chars: int = DEFAULT_MAX_TEXT_CHARS,
        compile_model: bool = False,
    ) -> None:
        self.model_name = model_name
        self.labels = labels
        self.confidence = confidence
        self.device = device
        self.inference_batch = inference_batch
        self.max_text_chars = max_text_chars
        self.compile_model = compile_model
        self._model = model  # may be None; loaded lazily

    def _load(self) -> Any:
        if self._model is not None:
            return self._model
        from gliner import GLiNER  # local import — heavy

        logger.info("ner_pass: loading %s on %s", self.model_name, self.device)
        m = GLiNER.from_pretrained(self.model_name)
        m = m.to(self.device).eval()
        if self.compile_model:
            # torch.compile takes ~60-90s on first call but then improves
            # steady-state throughput by 30-50% on the 5090. Worth it for
            # multi-day runs; skip for short / sample / dev runs.
            import torch

            logger.info("ner_pass: compiling model (this takes ~60s)…")
            m = torch.compile(m, mode="reduce-overhead")
        self._model = m
        return m

    def predict(self, papers: list[PaperInput]) -> list[list[Mention]]:
        """Run inference on a list of papers; return per-paper Mention lists.

        Empty / None text yields an empty mention list for that paper so
        callers can keep papers and per-paper outputs aligned by index.
        """
        model = self._load()
        texts = [p.text or "" for p in papers]
        # GLiNER's batch entry point silently broke on empty strings in some
        # versions. Filter empties + length-clip and re-thread by index.
        active_idx = [i for i, t in enumerate(texts) if t.strip() and len(t) <= self.max_text_chars]
        active_texts = [texts[i] for i in active_idx]

        per_doc_raw: list[list[dict[str, Any]]] = [[] for _ in papers]
        if active_texts:
            results = model.batch_predict_entities(
                active_texts,
                list(self.labels),
                threshold=self.confidence,
                batch_size=self.inference_batch,
            )
            for j, ents in zip(active_idx, results, strict=True):
                per_doc_raw[j] = ents

        out: list[list[Mention]] = []
        for paper, ents in zip(papers, per_doc_raw, strict=True):
            mentions: list[Mention] = []
            for e in ents:
                surface = e.get("text") or ""
                if not surface.strip():
                    continue
                etype = LABEL_TO_ENTITY_TYPE.get(e.get("label", ""))
                if etype is None:
                    continue
                conf = float(e.get("score", 0.0))
                if conf < self.confidence:
                    continue
                mentions.append(
                    Mention(
                        bibcode=paper.bibcode,
                        canonical_name=canonicalize(surface),
                        surface_text=surface,
                        entity_type=etype,
                        confidence=conf,
                    )
                )
            out.append(_dedup_within_doc(mentions))
        return out


# ---------------------------------------------------------------------------
# DB writers
# ---------------------------------------------------------------------------


_ENTITY_UPSERT_SQL = """
    INSERT INTO entities
        (canonical_name, entity_type, source, source_version)
    VALUES (%s, %s, 'gliner', %s)
    ON CONFLICT (canonical_name, entity_type, source) DO UPDATE
        SET updated_at = now(),
            source_version = EXCLUDED.source_version
    RETURNING id, (xmax = 0) AS inserted
"""

_DOC_ENTITY_UPSERT_SQL = """
    INSERT INTO document_entities
        (bibcode, entity_id, link_type, confidence, match_method, tier)
    VALUES (%s, %s, 'mentions', %s, 'gliner', %s)
    ON CONFLICT (bibcode, entity_id, link_type, tier) DO UPDATE
        SET confidence = GREATEST(document_entities.confidence, EXCLUDED.confidence),
            match_method = 'gliner'
"""


def _upsert_entities(
    conn: psycopg.Connection,
    mentions: list[Mention],
    *,
    source_version: str,
) -> tuple[dict[tuple[str, str], int], int]:
    """Upsert distinct (canonical_name, entity_type) pairs; return id map.

    Returns ``({(canon, type): entity_id}, n_newly_inserted)``.
    """
    distinct: list[tuple[str, str]] = sorted({(m.canonical_name, m.entity_type) for m in mentions})
    id_map: dict[tuple[str, str], int] = {}
    newly = 0
    if not distinct:
        return id_map, newly
    with conn.cursor() as cur:
        for canon, etype in distinct:
            cur.execute(_ENTITY_UPSERT_SQL, (canon, etype, source_version))
            row = cur.fetchone()
            if row is None:
                continue
            entity_id, inserted = row[0], bool(row[1])
            id_map[(canon, etype)] = entity_id
            if inserted:
                newly += 1
    return id_map, newly


def _upsert_document_entities(
    conn: psycopg.Connection,
    mentions: list[Mention],
    id_map: dict[tuple[str, str], int],
) -> int:
    """Upsert (bibcode, entity_id) bridge rows; return count written."""
    if not mentions:
        return 0
    rows: list[tuple[str, int, float, int]] = []
    for m in mentions:
        eid = id_map.get((m.canonical_name, m.entity_type))
        if eid is None:
            continue
        rows.append((m.bibcode, eid, m.confidence, NER_TIER))
    if not rows:
        return 0
    with conn.cursor() as cur:
        cur.executemany(_DOC_ENTITY_UPSERT_SQL, rows)
    return len(rows)


# ---------------------------------------------------------------------------
# Batch driver
# ---------------------------------------------------------------------------


def _checkpoint_key(target: str, first_bibcode: str) -> str:
    return f"ner_pass:{target}:{first_bibcode}"


def _is_batch_done(conn: psycopg.Connection, key: str) -> bool:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT status FROM ingest_log WHERE filename = %s",
            (key,),
        )
        row = cur.fetchone()
    return bool(row and row[0] == "complete")


def _record_checkpoint(
    conn: psycopg.Connection,
    key: str,
    *,
    records_loaded: int,
    edges_loaded: int,
    status: str = "complete",
) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO ingest_log
                (filename, records_loaded, edges_loaded, status, finished_at)
            VALUES (%s, %s, %s, %s, now())
            ON CONFLICT (filename) DO UPDATE
                SET records_loaded = EXCLUDED.records_loaded,
                    edges_loaded   = EXCLUDED.edges_loaded,
                    status         = EXCLUDED.status,
                    finished_at    = now()
            """,
            (key, records_loaded, edges_loaded, status),
        )


def iter_paper_batches(
    conn: psycopg.Connection,
    *,
    target: str = "abstract",
    batch_size: int = 1000,
    since_bibcode: str | None = None,
    max_papers: int | None = None,
) -> Iterator[list[PaperInput]]:
    """Stream papers in deterministic ``bibcode`` order via keyset pagination.

    ``target`` is the column name (``abstract`` or ``body``); rows where
    the column is NULL or empty are skipped. ``since_bibcode`` sets a
    watermark for resumability and ``max_papers`` caps total yield
    (sample runs).

    Keyset pagination (``WHERE bibcode > $watermark ORDER BY bibcode
    LIMIT N``) is used instead of a server-side named cursor because the
    caller commits between batches, which would invalidate a named cursor.
    Each iteration is a fresh single-shot query that survives commits and
    resumes correctly across process restarts.
    """
    if target not in ("abstract", "body"):
        raise ValueError(f"target must be 'abstract' or 'body', got {target!r}")

    sql = (
        f"SELECT bibcode, {target} AS text FROM papers "  # noqa: S608 — column whitelisted above
        f"WHERE bibcode > %s "
        f"  AND {target} IS NOT NULL "
        f"  AND ({target}) <> '' "
        f"ORDER BY bibcode ASC "
        f"LIMIT %s "
    )

    watermark = since_bibcode or ""
    remaining = max_papers
    while True:
        limit = batch_size if remaining is None else min(batch_size, remaining)
        if limit <= 0:
            return
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(sql, (watermark, limit))
            rows = cur.fetchall()
        if not rows:
            return
        yield [PaperInput(bibcode=r["bibcode"], text=r["text"] or "") for r in rows]
        watermark = rows[-1]["bibcode"]
        if remaining is not None:
            remaining -= len(rows)
            if remaining <= 0:
                return


def process_batch(
    conn: psycopg.Connection,
    extractor: GlinerExtractor,
    batch: list[PaperInput],
    *,
    source_version: str = NER_SOURCE_VERSION,
) -> BatchStats:
    """Run inference on one batch, write entities + document_entities.

    Caller commits. The DB-write phase runs inside a single transaction so
    a failure aborts cleanly without leaving half-written batches behind.
    """
    stats = BatchStats(papers_seen=len(batch))

    t0 = time.monotonic()
    per_doc = extractor.predict(batch)
    stats.elapsed_inference_s = time.monotonic() - t0

    flat: list[Mention] = []
    for paper, mentions in zip(batch, per_doc, strict=True):
        if mentions:
            stats.papers_with_mentions += 1
        flat.extend(mentions)
    stats.mentions_kept = len(flat)

    if not flat:
        return stats

    t0 = time.monotonic()
    id_map, newly = _upsert_entities(conn, flat, source_version=source_version)
    stats.new_entities = newly
    stats.upserted_doc_entities = _upsert_document_entities(conn, flat, id_map)
    stats.elapsed_db_s = time.monotonic() - t0
    return stats


def run(
    conn: psycopg.Connection,
    extractor: GlinerExtractor,
    *,
    target: str = "abstract",
    batch_size: int = 1000,
    since_bibcode: str | None = None,
    max_papers: int | None = None,
    source_version: str = NER_SOURCE_VERSION,
    log_every: int = 1,
) -> BatchStats:
    """Top-level driver: stream batches, run inference, write, checkpoint.

    Resumes by skipping any batch whose checkpoint key already shows
    ``status='complete'`` in ``ingest_log``. The first bibcode in each
    batch is the checkpoint key — deterministic because batches are
    pulled in bibcode order.
    """
    totals = BatchStats()
    n_batches = 0

    for batch in iter_paper_batches(
        conn,
        target=target,
        batch_size=batch_size,
        since_bibcode=since_bibcode,
        max_papers=max_papers,
    ):
        key = _checkpoint_key(target, batch[0].bibcode)
        if _is_batch_done(conn, key):
            logger.info("ner_pass: skip checkpointed batch %s", key)
            continue

        stats = process_batch(conn, extractor, batch, source_version=source_version)
        _record_checkpoint(
            conn,
            key,
            records_loaded=stats.papers_with_mentions,
            edges_loaded=stats.upserted_doc_entities,
        )
        conn.commit()

        # Aggregate totals
        totals.papers_seen += stats.papers_seen
        totals.papers_with_mentions += stats.papers_with_mentions
        totals.mentions_kept += stats.mentions_kept
        totals.new_entities += stats.new_entities
        totals.upserted_doc_entities += stats.upserted_doc_entities
        totals.elapsed_inference_s += stats.elapsed_inference_s
        totals.elapsed_db_s += stats.elapsed_db_s

        n_batches += 1
        if n_batches % log_every == 0:
            rate = stats.papers_seen / max(stats.elapsed_inference_s, 1e-3)
            logger.info(
                "ner_pass: batch %d (%s..) papers=%d mentions=%d new_entities=%d "
                "infer=%.1fs db=%.1fs rate=%.1f docs/s",
                n_batches,
                batch[0].bibcode[:18],
                stats.papers_seen,
                stats.mentions_kept,
                stats.new_entities,
                stats.elapsed_inference_s,
                stats.elapsed_db_s,
                rate,
            )

    return totals
