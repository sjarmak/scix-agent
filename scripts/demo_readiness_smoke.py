#!/usr/bin/env python3
"""Demo-readiness smoke test for the workshop talk (v2 — corrected args).

Each call uses a FRESH connection to avoid InFailedSqlTransaction cascades
when one tool errors. Args match the actual tool input schemas in
src/scix/mcp_server.py (not the v1 consolidation PRD shapes).
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("QDRANT_URL", "http://127.0.0.1:6333")

from scix.db import get_connection
from scix.mcp_server import _dispatch_tool


DEMO_QUERIES = [
    ("Galaxies", "astro"),
    ("p53 tumor suppressor", "biomed"),
    ("machine learning", "cs"),
]

# Anchor bibcodes per discipline (chosen for likely citation/contexts coverage).
ANCHOR_BIBCODES = {
    "astro": "1987gady.book.....B",
    "biomed": None,
    "cs": None,
}

# Demo claim text (used by claim_blame which is claim-keyed not bibcode-keyed).
DEMO_CLAIMS = {
    "Galaxies": "dark matter is the dominant mass component of galaxies",
    "p53 tumor suppressor": "p53 mutations are common in human cancers",
    "machine learning": "transformer architectures outperform recurrent networks",
}

QUERY_TOOLS = [
    ("entity", lambda q, _b: {"action": "search", "query": q, "entity_type": "instruments", "limit": 5}),
    ("concept_search", lambda q, _b: {"query": q, "limit": 5}),
    ("search", lambda q, _b: {"query": q, "limit": 5}),
    ("temporal_evolution", lambda q, _b: {"bibcode_or_query": q}),
    ("facet_counts", lambda q, _b: {"field": "arxiv_class", "filters": {"query": q}, "limit": 10}),
    ("section_retrieval", lambda q, _b: {"query": q, "limit": 5}),
    ("chunk_search", lambda q, _b: {"query": q, "limit": 5}),
    ("find_claims", lambda q, _b: {"query": q, "limit": 5}),
]

BIBCODE_TOOLS = [
    ("get_paper", lambda _q, b: {"bibcode": b}),
    ("read_paper", lambda _q, b: {"bibcode": b}),
    ("citation_traverse", lambda _q, b: {"bibcode": b, "direction": "both", "limit": 5}),
    ("citation_similarity", lambda _q, b: {"bibcode": b, "method": "co_citation", "limit": 5}),
    ("graph_context", lambda _q, b: {"bibcode": b, "include_community": True, "limit": 5}),
    ("find_replications", lambda _q, b: {"target_bibcode": b}),
    ("read_paper_claims", lambda _q, b: {"bibcode": b}),
]

CLAIM_TOOLS = [
    ("claim_blame", lambda c: {"claim_text": c}),
]


def call_tool(tool: str, args: dict) -> tuple[str, float, str]:
    """Each call gets its own connection so SQL errors don't cascade."""
    t0 = time.monotonic()
    conn = None
    try:
        conn = get_connection()
        raw = _dispatch_tool(conn, tool, args)
        elapsed = (time.monotonic() - t0) * 1000
        try:
            parsed = json.loads(raw)
        except (ValueError, TypeError):
            return "OK", elapsed, f"non-json: {str(raw)[:60]}"

        if isinstance(parsed, dict) and parsed.get("error"):
            return "ERROR", elapsed, str(parsed.get("error"))[:90]

        for key in ("results", "papers", "items", "edges", "entities", "claims",
                    "blame", "replications", "matches", "sections", "chunks",
                    "facets", "citations", "references", "neighbors", "gaps",
                    "trajectory", "buckets", "directions"):
            if isinstance(parsed, dict) and key in parsed:
                v = parsed[key]
                n = len(v) if isinstance(v, list) else (1 if v else 0)
                if n == 0:
                    return "EMPTY", elapsed, f"{key}=0"
                return "OK", elapsed, f"{key}={n}"

        if isinstance(parsed, list):
            n = len(parsed)
            return ("OK" if n > 0 else "EMPTY"), elapsed, f"list n={n}"
        if isinstance(parsed, dict) and parsed:
            return "OK", elapsed, f"keys={list(parsed.keys())[:5]}"
        return "EMPTY", elapsed, "empty"
    except Exception as exc:
        elapsed = (time.monotonic() - t0) * 1000
        return "FAIL", elapsed, f"{type(exc).__name__}: {str(exc)[:90]}"
    finally:
        if conn:
            conn.close()


def first_bibcode(tool: str, args: dict) -> str | None:
    try:
        conn = get_connection()
        try:
            raw = _dispatch_tool(conn, tool, args)
        finally:
            conn.close()
        parsed = json.loads(raw)
    except Exception:
        return None
    if isinstance(parsed, dict):
        for key in ("results", "papers", "items"):
            if key in parsed and isinstance(parsed[key], list) and parsed[key]:
                first = parsed[key][0]
                if isinstance(first, dict):
                    return first.get("bibcode") or first.get("source_bibcode")
    return None


def main() -> int:
    print("=" * 100)
    print("DEMO-READINESS SMOKE v2 (fresh-conn, corrected args)")
    print("=" * 100)
    print(f"QDRANT_URL={os.environ.get('QDRANT_URL', '<unset>')}")

    all_results: list[dict] = []
    for query, discipline in DEMO_QUERIES:
        print(f"\n## query={query!r}  discipline={discipline}")
        print("-" * 100)
        anchor = ANCHOR_BIBCODES.get(discipline)

        for tool, build_args in QUERY_TOOLS:
            args = build_args(query, None)
            status, ms, detail = call_tool(tool, args)
            marker = {"OK": "✓", "EMPTY": "∅", "ERROR": "!", "FAIL": "✗"}.get(status, "?")
            print(f"  {marker} {tool:<22} {ms:>8.0f}ms  [{status}]  {detail}")
            all_results.append({"query": query, "tool": tool, "status": status,
                                 "ms": ms, "detail": detail})
            if anchor is None and tool == "concept_search" and status == "OK":
                anchor = first_bibcode(tool, args)
                if anchor:
                    print(f"  → derived anchor bibcode={anchor}")

        if anchor is None:
            with get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT bibcode FROM papers WHERE title ILIKE %s "
                        "ORDER BY citation_count DESC NULLS LAST LIMIT 1",
                        (f"%{query.split()[0]}%",),
                    )
                    row = cur.fetchone()
                    anchor = row[0] if row else None
                    if anchor:
                        print(f"  → fallback anchor bibcode={anchor} (from title match)")

        if anchor:
            for tool, build_args in BIBCODE_TOOLS:
                args = build_args(query, anchor)
                status, ms, detail = call_tool(tool, args)
                marker = {"OK": "✓", "EMPTY": "∅", "ERROR": "!", "FAIL": "✗"}.get(status, "?")
                print(f"  {marker} {tool:<22} {ms:>8.0f}ms  [{status}]  {detail}")
                all_results.append({"query": query, "tool": tool, "status": status,
                                     "ms": ms, "detail": detail, "bibcode": anchor})
        else:
            print(f"  ⚠ no anchor bibcode found — skipping bibcode-keyed tools")

        # Claim-keyed tools
        claim = DEMO_CLAIMS.get(query)
        if claim:
            for tool, build_args in CLAIM_TOOLS:
                args = build_args(claim)
                status, ms, detail = call_tool(tool, args)
                marker = {"OK": "✓", "EMPTY": "∅", "ERROR": "!", "FAIL": "✗"}.get(status, "?")
                print(f"  {marker} {tool:<22} {ms:>8.0f}ms  [{status}]  {detail}  claim={claim[:40]!r}")
                all_results.append({"query": query, "tool": tool, "status": status,
                                     "ms": ms, "detail": detail, "claim": claim})

    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)
    from collections import Counter
    by_status: Counter[str] = Counter(r["status"] for r in all_results)
    print(f"  total calls: {len(all_results)}")
    for status in ("OK", "EMPTY", "ERROR", "FAIL"):
        print(f"  {status:<6}: {by_status.get(status, 0)}")

    fails = [r for r in all_results if r["status"] in ("ERROR", "FAIL")]
    empties = [r for r in all_results if r["status"] == "EMPTY"]
    if fails:
        print("\n  REAL DEMO BLOCKERS:")
        for r in fails:
            print(f"    ✗ {r['query']!r:<25} {r['tool']:<22}  {r['detail']}")
    if empties:
        print("\n  EMPTY RESULTS (call works but returns 0 — flag if demo touches these):")
        for r in empties:
            print(f"    ∅ {r['query']!r:<25} {r['tool']:<22}  {r['detail']}")

    out = Path("/tmp/demo_readiness_smoke.json")
    out.write_text(json.dumps(all_results, indent=2))
    print(f"\n  full results: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
