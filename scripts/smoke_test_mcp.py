#!/usr/bin/env python3
"""Smoke test all MCP tools against the live database.

Calls _dispatch_tool() directly for each of the 29 tools.
Reports pass/fail with timing for each tool.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

# Suppress torch/transformers import noise
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from scix.db import get_connection
from scix.mcp_server import _dispatch_tool, _session_state


# ---------------------------------------------------------------------------
# Test data
# ---------------------------------------------------------------------------

CITED_BIBCODE = "1973PlSoi..39..205B"  # well-cited paper
EMBED_BIBCODE = "1959Tell...11..355H"  # has INDUS embedding
EXTRACT_BIBCODE = "1996PhRvL..77.3865P"  # has method extractions
METRICS_BIBCODE = "2020ScTEn.73939864B"  # has paper_metrics
CITE_PAIR_SRC = "2018AdSpR..62.2773B"
CITE_PAIR_TGT = "1990BAICz..41..137P"
CTX_SRC = "2004RvMP...76..323Z"
CTX_TGT = "1981PhRvB..23.3159S"
ENTITY_ID = 49  # Dopplergram
DOC_CTX_BIBCODE = "1800adtd.book.....B"


# ---------------------------------------------------------------------------
# Tool test definitions
# ---------------------------------------------------------------------------

TOOL_TESTS: list[tuple[str, dict]] = [
    # -- Search tools --
    ("keyword_search", {"terms": "dark matter halo", "limit": 5}),
    # semantic_search requires model loading — test separately if GPU is free
    # ("semantic_search", {"query": "dark matter halo", "limit": 5}),

    # -- Paper lookup --
    ("get_paper", {"bibcode": CITED_BIBCODE}),

    # -- Citation graph --
    ("get_citations", {"bibcode": CITED_BIBCODE, "limit": 5}),
    ("get_references", {"bibcode": CITED_BIBCODE, "limit": 5}),
    ("co_citation_analysis", {"bibcode": CITED_BIBCODE, "min_overlap": 2, "limit": 5}),
    ("bibliographic_coupling", {"bibcode": CITED_BIBCODE, "min_overlap": 2, "limit": 5}),
    ("citation_chain", {"source_bibcode": CITE_PAIR_SRC, "target_bibcode": CITE_PAIR_TGT, "max_depth": 3}),

    # -- Author & facet --
    ("get_author_papers", {"author_name": "Einstein, A.", "year_min": 1905, "year_max": 1920}),
    ("facet_counts", {"field": "year", "limit": 10}),

    # -- Temporal --
    ("temporal_evolution", {"bibcode_or_query": CITED_BIBCODE}),

    # -- Graph metrics --
    ("get_paper_metrics", {"bibcode": METRICS_BIBCODE}),
    ("explore_community", {"bibcode": METRICS_BIBCODE, "resolution": "coarse", "limit": 5}),

    # -- Entity/extraction tools --
    ("entity_search", {"entity_type": "methods", "entity_name": "Generalized gradient approximation", "limit": 5}),
    ("entity_profile", {"bibcode": EXTRACT_BIBCODE}),
    ("concept_search", {"query": "Astronomy", "include_subtopics": False, "limit": 5}),

    # -- Citation context --
    ("get_citation_context", {"source_bibcode": CTX_SRC, "target_bibcode": CTX_TGT}),

    # -- Full-text tools (may return empty if no paper_bodies) --
    ("read_paper_section", {"bibcode": CITED_BIBCODE, "section": "full", "limit": 500}),
    ("search_within_paper", {"bibcode": CITED_BIBCODE, "query": "solar"}),

    # -- OpenAlex --
    ("get_openalex_topics", {"bibcode": CITED_BIBCODE}),

    # -- Document context (matview) --
    ("document_context", {"bibcode": DOC_CTX_BIBCODE}),

    # -- Entity graph tools --
    ("entity_context", {"entity_id": ENTITY_ID}),
    ("resolve_entity", {"query": "Dopplergram"}),

    # -- Working set tools (session-scoped) --
    ("add_to_working_set", {"bibcodes": [CITED_BIBCODE], "source_tool": "smoke_test", "source_context": "testing"}),
    ("get_working_set", {}),
    ("get_session_summary", {}),
    ("find_gaps", {"resolution": "coarse", "limit": 5}),
    ("clear_working_set", {}),

    # -- Health --
    ("health_check", {}),
]


def main() -> None:
    conn = get_connection()
    conn.autocommit = True
    results: list[dict] = []
    passed = 0
    failed = 0

    print(f"{'Tool':<30} {'Status':<8} {'Time (ms)':<12} {'Detail'}")
    print("-" * 90)

    for tool_name, args in TOOL_TESTS:
        t0 = time.monotonic()
        try:
            result_json = _dispatch_tool(conn, tool_name, args)
            elapsed_ms = (time.monotonic() - t0) * 1000
            result = json.loads(result_json)

            if "error" in result:
                status = "FAIL"
                detail = result["error"][:60]
                failed += 1
            else:
                status = "PASS"
                # Summarize result
                if "papers" in result:
                    detail = f"{result.get('total', len(result['papers']))} papers"
                elif "entries" in result:
                    detail = f"{len(result['entries'])} entries"
                elif "path" in result:
                    detail = f"path length {len(result['path'])}" if result["path"] else "no path found"
                elif "years" in result:
                    detail = f"{len(result['years'])} years"
                elif "facets" in result:
                    detail = f"{len(result['facets'])} facets"
                elif "db" in result:
                    detail = f"db={result['db']}"
                elif "candidates" in result:
                    detail = f"{result['total']} candidates"
                elif "contexts" in result:
                    detail = f"{len(result['contexts'])} contexts"
                elif "extractions" in result:
                    detail = f"{result['total']} extractions"
                elif "removed" in result:
                    detail = f"removed {result['removed']}"
                elif "added" in result:
                    detail = f"added {result['added']}"
                else:
                    detail = str(result)[:60]
                passed += 1

        except Exception as e:
            elapsed_ms = (time.monotonic() - t0) * 1000
            status = "ERROR"
            detail = f"{type(e).__name__}: {str(e)[:50]}"
            failed += 1

        results.append({
            "tool": tool_name,
            "status": status,
            "elapsed_ms": round(elapsed_ms, 1),
            "detail": detail,
        })
        print(f"{tool_name:<30} {status:<8} {elapsed_ms:>8.1f} ms  {detail}")

    print("-" * 90)
    print(f"Total: {len(TOOL_TESTS)} | Passed: {passed} | Failed: {failed}")

    # Write results
    out_path = Path(__file__).resolve().parent.parent / "results" / "mcp_smoke_test.json"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"passed": passed, "failed": failed, "total": len(TOOL_TESTS), "results": results}, f, indent=2)
    print(f"\nResults written to {out_path}")

    conn.close()
    sys.exit(1 if failed > 0 else 0)


if __name__ == "__main__":
    main()
