#!/usr/bin/env python3
"""Pre-warm the MCP tool surface for a live demo.

Run 5-10 minutes before the talk:

  scix-batch .venv/bin/python scripts/demo_prewarm.py

What it does:
  1. Loads INDUS embedder onto GPU (one-time ~10-30s warm-up).
  2. Loads BAAI/bge-reranker-large onto GPU (one-time ~3-5s warm-up).
  3. Runs each demo query 3× through concept_search to warm pgvector
     graph nodes into page cache.
  4. Runs entity(action='resolve') for each demo term to warm the
     entity resolver index.
  5. Reports the final warm latency per (tool, query) — that's what the
     audience will see live.

DEMO_QUERIES below match the workshop P0 demo (cross-discipline:
Galaxies / p53 / machine learning). Edit if your demo set changes.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from scix.db import get_connection
from scix.mcp_server import _dispatch_tool


DEMO_QUERIES = [
    "Galaxies",
    "p53 tumor suppressor",
    "machine learning",
]

# Tools to warm and their args. Order matters — model-loading tools first.
WARMUPS = [
    # First call loads INDUS model
    ("concept_search", {"query": "warmup-init", "limit": 1}),
    # Subsequent runs warm pgvector graph nodes for each demo query
    *[("concept_search", {"query": q, "limit": 5}) for q in DEMO_QUERIES],
    # Repeat each query to confirm warm latency
    *[("concept_search", {"query": q, "limit": 5}) for q in DEMO_QUERIES],
    # Entity resolver (cheap; warm the JIT cache)
    *[("entity", {"action": "resolve", "query": q}) for q in DEMO_QUERIES],
    # Citation graph queries (no vector cost; warm PG plan cache)
    ("citation_traverse", {"bibcode": "1987gady.book.....B", "direction": "both", "limit": 5}),
    ("citation_similarity", {"bibcode": "1987gady.book.....B", "method": "co_citation", "limit": 5}),
    # Fast SQL aggregations
    ("temporal_evolution", {"bibcode_or_query": "Galaxies"}),
    ("facet_counts", {"field": "arxiv_class", "limit": 10}),
    ("graph_context", {"bibcode": "1987gady.book.....B", "include_community": True, "limit": 5}),
]


def call(tool: str, args: dict) -> tuple[bool, float, str]:
    t0 = time.monotonic()
    conn = get_connection()
    try:
        try:
            raw = _dispatch_tool(conn, tool, args)
            ms = (time.monotonic() - t0) * 1000
            try:
                p = json.loads(raw)
                if isinstance(p, dict) and p.get("error"):
                    return False, ms, str(p["error"])[:60]
                return True, ms, "ok"
            except (ValueError, TypeError):
                return True, ms, "non-json"
        except Exception as e:
            ms = (time.monotonic() - t0) * 1000
            return False, ms, f"{type(e).__name__}: {str(e)[:60]}"
    finally:
        conn.close()


def main() -> int:
    print("=" * 80)
    print("DEMO PRE-WARM — run 5-10 min before talk")
    print("=" * 80)
    print(f"DEMO_QUERIES: {DEMO_QUERIES}")
    print()

    # Track per-(tool,query) latencies; final pass is the "live" estimate
    history: dict[tuple, list[float]] = {}
    for i, (tool, args) in enumerate(WARMUPS, 1):
        q_summary = args.get("query") or args.get("bibcode") or args.get("bibcode_or_query") or "-"
        ok, ms, detail = call(tool, args)
        marker = "✓" if ok else "!"
        print(f"  [{i:>2}/{len(WARMUPS)}] {marker} {tool:<22} {q_summary[:40]:<42} {ms:>7.0f}ms  {detail}")
        key = (tool, q_summary)
        history.setdefault(key, []).append(ms)

    print()
    print("=" * 80)
    print("WARM LATENCY (last call per (tool, query)) — what the audience will see")
    print("=" * 80)
    for (tool, q), runs in sorted(history.items(), key=lambda kv: -kv[1][-1]):
        last = runs[-1]
        spread = f"runs={len(runs)} cold={runs[0]:.0f}ms warm={last:.0f}ms" if len(runs) > 1 else f"single={last:.0f}ms"
        marker = "🟢" if last < 2000 else "🟡" if last < 8000 else "🔴"
        print(f"  {marker} {tool:<22} {q[:35]:<37} {spread}")

    print()
    print("RECOMMENDATIONS:")
    slow = [(t, q, runs[-1]) for (t, q), runs in history.items() if runs[-1] > 8000]
    if slow:
        print("  ⚠️  These will feel slow on stage (>8s):")
        for t, q, ms in slow:
            print(f"     {t}({q!r}) — {ms:.0f}ms")
        print("  → Consider having a backup screenshot for these queries.")
    else:
        print("  ✓ All warm latencies are <8s. Demo should feel responsive.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
