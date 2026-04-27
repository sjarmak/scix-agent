"""Dry-run measurement of eager-load enrichment cost on `search` results.

Runs a baseline `hybrid_search` for a small set of representative queries,
then measures the per-paper added latency and added bytes of each candidate
enrichment field independently:

  - abstract_snippet  (papers.abstract, first 400 chars)
  - linked_entities   (document_entities_canonical -> entities, top-5)
  - community         (paper_metrics.community_semantic_medium -> communities)
  - intent_in         (citation_contexts.intent histogram on target_bibcode)
  - degree            (citation_edges in/out counts)
  - fulltext          (papers_fulltext: section_count + section names)

Read-only, no schema changes. Times each field both per-paper and as a single
batched call across the top-K results so we can see the win from batching.

Usage:
  python scripts/measure_eager_enrichment_dryrun.py

See bead scix_experiments-3lw4 for the eager-load proposal this script
informs, and scix_experiments-k0p8 for the papers_fulltext over-segmentation
issue this script surfaced.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any

import psycopg

from scix import search
from scix.embed import embed_batch, load_model

QUERIES = [
    "JWST exoplanet atmosphere transmission spectroscopy",
    "asteroid main belt thermal model",
    "convolutional neural network galaxy morphology",
]
TOP_K = 5
DSN = "dbname=scix"


@dataclass
class FieldTiming:
    name: str
    per_paper_ms: list[float]
    batched_ms: float
    bytes_added: int
    sample: Any


def _ms(start: float) -> float:
    return (time.perf_counter() - start) * 1000.0


def _measure_abstract(conn: psycopg.Connection, bibcodes: list[str]) -> FieldTiming:
    per_paper: list[float] = []
    samples: list[dict] = []
    for b in bibcodes:
        t = time.perf_counter()
        with conn.cursor() as cur:
            cur.execute("SELECT bibcode, LEFT(abstract, 400) FROM papers WHERE bibcode=%s", (b,))
            row = cur.fetchone()
        per_paper.append(_ms(t))
        if row:
            samples.append({"bibcode": row[0], "abstract_snippet": row[1]})

    t = time.perf_counter()
    with conn.cursor() as cur:
        cur.execute(
            "SELECT bibcode, LEFT(abstract, 400) FROM papers WHERE bibcode = ANY(%s)",
            (bibcodes,),
        )
        rows = cur.fetchall()
    batched = _ms(t)
    payload_bytes = len(json.dumps([{"bibcode": b, "abstract_snippet": a} for b, a in rows]))
    return FieldTiming("abstract_snippet", per_paper, batched, payload_bytes, samples[:1])


def _measure_linked_entities(conn: psycopg.Connection, bibcodes: list[str]) -> FieldTiming:
    sql_one = """
        SELECT e.canonical_name, e.entity_type, dec.fused_confidence
        FROM document_entities_canonical dec
        JOIN entities e ON e.id = dec.entity_id
        WHERE dec.bibcode = %s
        ORDER BY dec.fused_confidence DESC
        LIMIT 5
    """
    per_paper: list[float] = []
    samples: list[Any] = []
    for b in bibcodes:
        t = time.perf_counter()
        with conn.cursor() as cur:
            cur.execute(sql_one, (b,))
            rows = cur.fetchall()
        per_paper.append(_ms(t))
        if not samples:
            samples = [{"name": r[0], "type": r[1], "confidence": float(r[2])} for r in rows]

    sql_batch = """
        WITH ranked AS (
            SELECT dec.bibcode, e.canonical_name, e.entity_type, dec.fused_confidence,
                   ROW_NUMBER() OVER (PARTITION BY dec.bibcode ORDER BY dec.fused_confidence DESC) AS rn
            FROM document_entities_canonical dec
            JOIN entities e ON e.id = dec.entity_id
            WHERE dec.bibcode = ANY(%s)
        )
        SELECT bibcode, canonical_name, entity_type, fused_confidence
        FROM ranked WHERE rn <= 5
    """
    t = time.perf_counter()
    with conn.cursor() as cur:
        cur.execute(sql_batch, (bibcodes,))
        rows = cur.fetchall()
    batched = _ms(t)
    by_paper: dict[str, list[dict]] = {}
    for b, name, etype, conf in rows:
        by_paper.setdefault(b, []).append({"name": name, "type": etype, "confidence": float(conf)})
    payload_bytes = len(json.dumps(by_paper))
    return FieldTiming("linked_entities", per_paper, batched, payload_bytes, samples)


def _measure_community(conn: psycopg.Connection, bibcodes: list[str]) -> FieldTiming:
    sql_one = """
        SELECT pm.community_semantic_medium, c.label
        FROM paper_metrics pm
        LEFT JOIN communities c
          ON c.signal='semantic' AND c.resolution='medium' AND c.community_id = pm.community_semantic_medium
        WHERE pm.bibcode = %s
    """
    per_paper: list[float] = []
    samples: list[Any] = []
    for b in bibcodes:
        t = time.perf_counter()
        with conn.cursor() as cur:
            cur.execute(sql_one, (b,))
            row = cur.fetchone()
        per_paper.append(_ms(t))
        if row and not samples:
            samples = [{"community_id": row[0], "label": row[1]}]

    sql_batch = """
        SELECT pm.bibcode, pm.community_semantic_medium, c.label
        FROM paper_metrics pm
        LEFT JOIN communities c
          ON c.signal='semantic' AND c.resolution='medium' AND c.community_id = pm.community_semantic_medium
        WHERE pm.bibcode = ANY(%s)
    """
    t = time.perf_counter()
    with conn.cursor() as cur:
        cur.execute(sql_batch, (bibcodes,))
        rows = cur.fetchall()
    batched = _ms(t)
    payload = {b: {"community_id": cid, "label": lab} for b, cid, lab in rows}
    payload_bytes = len(json.dumps(payload))
    return FieldTiming("community", per_paper, batched, payload_bytes, samples)


def _measure_intent_in(conn: psycopg.Connection, bibcodes: list[str]) -> FieldTiming:
    sql_one = """
        SELECT intent, count(*)
        FROM citation_contexts
        WHERE target_bibcode = %s AND intent IS NOT NULL
        GROUP BY intent
    """
    per_paper: list[float] = []
    samples: list[Any] = []
    for b in bibcodes:
        t = time.perf_counter()
        with conn.cursor() as cur:
            cur.execute(sql_one, (b,))
            rows = cur.fetchall()
        per_paper.append(_ms(t))
        if rows and not samples:
            samples = [{r[0]: int(r[1]) for r in rows}]

    sql_batch = """
        SELECT target_bibcode, intent, count(*)
        FROM citation_contexts
        WHERE target_bibcode = ANY(%s) AND intent IS NOT NULL
        GROUP BY target_bibcode, intent
    """
    t = time.perf_counter()
    with conn.cursor() as cur:
        cur.execute(sql_batch, (bibcodes,))
        rows = cur.fetchall()
    batched = _ms(t)
    by_paper: dict[str, dict[str, int]] = {b: {} for b in bibcodes}
    for b, intent, n in rows:
        by_paper[b][intent] = int(n)
    payload_bytes = len(json.dumps(by_paper))
    return FieldTiming("intent_in", per_paper, batched, payload_bytes, samples)


def _measure_degree(conn: psycopg.Connection, bibcodes: list[str]) -> FieldTiming:
    sql_one_in = "SELECT count(*) FROM citation_edges WHERE target_bibcode = %s"
    sql_one_out = "SELECT count(*) FROM citation_edges WHERE source_bibcode = %s"
    per_paper: list[float] = []
    samples: list[Any] = []
    for b in bibcodes:
        t = time.perf_counter()
        with conn.cursor() as cur:
            cur.execute(sql_one_in, (b,))
            din = cur.fetchone()[0]
            cur.execute(sql_one_out, (b,))
            dout = cur.fetchone()[0]
        per_paper.append(_ms(t))
        if not samples:
            samples = [{"in": din, "out": dout}]

    sql_batch = """
        SELECT bibcode,
               (SELECT count(*) FROM citation_edges WHERE target_bibcode = b.bibcode) AS in_deg,
               (SELECT count(*) FROM citation_edges WHERE source_bibcode = b.bibcode) AS out_deg
        FROM (SELECT unnest(%s::text[]) AS bibcode) b
    """
    t = time.perf_counter()
    with conn.cursor() as cur:
        cur.execute(sql_batch, (bibcodes,))
        rows = cur.fetchall()
    batched = _ms(t)
    by_paper = {b: {"in": int(i), "out": int(o)} for b, i, o in rows}
    payload_bytes = len(json.dumps(by_paper))
    return FieldTiming("degree", per_paper, batched, payload_bytes, samples)


def _measure_fulltext(conn: psycopg.Connection, bibcodes: list[str]) -> FieldTiming:
    sql_one = """
        SELECT jsonb_array_length(sections) AS n,
               (SELECT array_agg(s->>'heading') FROM jsonb_array_elements(sections) s) AS headings
        FROM papers_fulltext WHERE bibcode = %s
    """
    per_paper: list[float] = []
    samples: list[Any] = []
    for b in bibcodes:
        t = time.perf_counter()
        with conn.cursor() as cur:
            cur.execute(sql_one, (b,))
            row = cur.fetchone()
        per_paper.append(_ms(t))
        if row and not samples:
            samples = [{"section_count": row[0], "sections": row[1]}]

    sql_batch = """
        SELECT bibcode,
               jsonb_array_length(sections) AS n,
               (SELECT array_agg(s->>'heading') FROM jsonb_array_elements(sections) s) AS headings
        FROM papers_fulltext WHERE bibcode = ANY(%s)
    """
    t = time.perf_counter()
    with conn.cursor() as cur:
        cur.execute(sql_batch, (bibcodes,))
        rows = cur.fetchall()
    batched = _ms(t)
    by_paper = {b: {"section_count": n, "sections": h} for b, n, h in rows}
    payload_bytes = len(json.dumps(by_paper, default=str))
    return FieldTiming("fulltext", per_paper, batched, payload_bytes, samples)


def _baseline_search(
    conn: psycopg.Connection, query: str
) -> tuple[search.SearchResult, float, int]:
    model, tokenizer = load_model("indus", device="cpu")
    qvec = embed_batch(model, tokenizer, [query], batch_size=1)[0]
    t = time.perf_counter()
    res = search.hybrid_search(
        conn, query, query_embedding=qvec, model_name="indus", top_n=TOP_K, reranker=None
    )
    elapsed = _ms(t)
    payload_bytes = len(json.dumps(res.papers, default=str))
    return res, elapsed, payload_bytes


def main() -> None:
    print(f"=== Eager-enrichment dry-run | top_k={TOP_K} | queries={len(QUERIES)} ===\n")
    with psycopg.connect(DSN) as conn:
        for q in QUERIES:
            print(f"--- query: {q!r}")
            res, base_ms, base_bytes = _baseline_search(conn, q)
            bibs = [p["bibcode"] for p in res.papers[:TOP_K]]
            print(f"baseline: {base_ms:.0f}ms, {base_bytes}B, {len(bibs)} papers")
            print(f"top-K: {bibs}\n")

            measurements = [
                _measure_abstract(conn, bibs),
                _measure_linked_entities(conn, bibs),
                _measure_community(conn, bibs),
                _measure_intent_in(conn, bibs),
                _measure_degree(conn, bibs),
                _measure_fulltext(conn, bibs),
            ]

            print(f"{'field':<20} {'per-paper avg ms':>18} {'batched ms':>12} {'+bytes':>10}")
            print("-" * 65)
            total_batched = 0.0
            total_bytes = 0
            for m in measurements:
                avg = sum(m.per_paper_ms) / max(len(m.per_paper_ms), 1)
                print(f"{m.name:<20} {avg:>18.2f} {m.batched_ms:>12.2f} {m.bytes_added:>10}")
                total_batched += m.batched_ms
                total_bytes += m.bytes_added
            print(f"{'TOTAL (batched)':<20} {'':>18} {total_batched:>12.2f} {total_bytes:>10}")
            print(
                f"\nbaseline {base_ms:.0f}ms -> enriched {base_ms + total_batched:.0f}ms "
                f"(+{total_batched / max(base_ms, 1) * 100:.1f}%)"
            )
            print(
                f"baseline {base_bytes}B -> enriched {base_bytes + total_bytes}B "
                f"(+{total_bytes / max(base_bytes, 1) * 100:.1f}%)\n"
            )

            print("sample top-1 enrichment payload:")
            print(json.dumps({m.name: m.sample for m in measurements}, indent=2, default=str)[:1500])
            print("\n" + "=" * 65 + "\n")


if __name__ == "__main__":
    main()
