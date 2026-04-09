#!/usr/bin/env python3
"""Filtered HNSW retest with iterative_scan=relaxed (post papers-index restore).

Runs 5+ representative filtered vector queries and captures latency + result quality.
Uses a real paper embedding as the query vector (no model loading needed).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from scix.db import get_connection
from scix.search import SearchFilters, vector_search, configure_iterative_scan


def get_sample_embedding(conn, model_name: str = "specter2") -> tuple[str, list[float]]:
    """Get a sample embedding from the database."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT bibcode, embedding::text FROM paper_embeddings WHERE model_name = %s LIMIT 1",
            (model_name,),
        )
        row = cur.fetchone()
    bibcode = row[0]
    vec = [float(x) for x in row[1].strip("[]").split(",")]
    return bibcode, vec


def run_test(
    conn, label: str, query_embedding: list[float], filters: SearchFilters, limit: int = 10
) -> dict:
    """Run a single filtered vector search and return results."""
    t0 = time.perf_counter()
    result = vector_search(
        conn,
        query_embedding,
        model_name="specter2",
        filters=filters,
        limit=limit,
        iterative_scan="relaxed_order",
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000

    return {
        "label": label,
        "filters": str(filters),
        "latency_ms": round(elapsed_ms, 1),
        "result_count": result.total,
        "iterative_scan": result.metadata.get("iterative_scan", False),
        "top_3": [
            {"bibcode": p["bibcode"], "score": round(p.get("score", 0), 4)}
            for p in result.papers[:3]
        ],
    }


def main() -> None:
    conn = get_connection()
    conn.autocommit = True

    bibcode, query_embedding = get_sample_embedding(conn, "specter2")
    print(f"Query bibcode: {bibcode}")
    print(f"Embedding dim: {len(query_embedding)}")
    print()

    tests = [
        ("year_filter", SearchFilters(year_min=2020, year_max=2024)),
        ("doctype_filter", SearchFilters(doctype="article")),
        ("arxiv_class_filter", SearchFilters(arxiv_class="astro-ph")),
        ("first_author_filter", SearchFilters(first_author="Einstein")),
        ("year+doctype", SearchFilters(year_min=2020, year_max=2024, doctype="article")),
        ("year+arxiv", SearchFilters(year_min=2015, year_max=2025, arxiv_class="astro-ph")),
        ("no_filter_baseline", SearchFilters()),
    ]

    results = []
    print(f"{'Test':<25} {'Latency':<12} {'Results':<10} {'IterScan':<10} {'Top bibcode'}")
    print("-" * 90)

    for label, filters in tests:
        r = run_test(conn, label, query_embedding, filters)
        results.append(r)
        top_bib = r["top_3"][0]["bibcode"] if r["top_3"] else "—"
        print(
            f"{label:<25} {r['latency_ms']:>8.1f} ms  {r['result_count']:<10} "
            f"{'yes' if r['iterative_scan'] else 'no':<10} {top_bib}"
        )

    print("-" * 90)
    print(f"All {len(tests)} tests completed.")

    # Write results
    out_path = Path(__file__).resolve().parent.parent / "results" / "filtered_hnsw_retest.json"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"query_bibcode": bibcode, "tests": results}, f, indent=2)
    print(f"\nResults written to {out_path}")

    conn.close()


if __name__ == "__main__":
    main()
