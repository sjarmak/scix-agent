"""Record an agent-trace replay snapshot for the website scix-viz bundle.

Runs the two demo scenarios (single hybrid search + step-by-step lit review)
against the live corpus and captures the exact trace events each publishes —
the same JSON the SSE stream would push to the live page. Also resolves a
title for every touched bibcode so the static replay can fill the panel
without the ``/viz/api/paper`` backend.

Output: ``<out_dir>/trace_replay.json``
  {
    "scenarios": [ {key, label, query, narration, events:[...]} ],
    "titles": { bibcode: title }
  }

Run from the repo root:  .venv/bin/python scripts/viz/build_trace_snapshot.py <out_dir>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from scix.db import get_connection
from scix.viz import trace_stream
from scix.viz.api import DemoSearchRequest, demo_lit_review, demo_search

SCENARIOS = [
    {
        "key": "search",
        "label": "Search",
        "query": "JWST asteroid",
        "narration": (
            "Single hybrid search — INDUS dense embedding + BM25 lexical, fused "
            "via RRF. One trace event with the top-N result set."
        ),
        "fn": demo_search,
    },
    {
        "key": "lit_review",
        "label": "Lit Review",
        "query": "galaxy clusters dark matter",
        "narration": (
            "Step-by-step research-copilot motion. Step 1: hybrid search for seed "
            "papers. Steps 2-4: citation_traverse(backward) for the top seeds — "
            "1-hop ancestors. Steps 5-7: citation_traverse(forward) — 1-hop "
            "descendants. Step 8: graph_context on the top seed. Steps 9-10: "
            "get_paper on the top seeds for synthesis material."
        ),
        "fn": demo_lit_review,
    },
]


def _read_history_lines(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def _resolve_titles(bibcodes: list[str]) -> dict[str, str]:
    if not bibcodes:
        return {}
    conn = get_connection()
    with conn.cursor() as cur:
        cur.execute(
            "SELECT bibcode, title FROM papers WHERE bibcode = ANY(%s)",
            (bibcodes,),
        )
        return {b: (t or "") for b, t in cur.fetchall()}


def main() -> int:
    out_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "web/viz")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Ensure the on-disk history file/path is installed so publish() persists.
    trace_stream.init_history()
    hist_path = trace_stream._default_history_path()

    captured: list[dict] = []
    all_bibs: set[str] = set()
    for scn in SCENARIOS:
        before = len(_read_history_lines(hist_path))
        result = scn["fn"](DemoSearchRequest(query=scn["query"], top_n=5))
        after = _read_history_lines(hist_path)
        events = after[before:]
        for ev in events:
            for b in ev.get("bibcodes") or []:
                all_bibs.add(b)
        steps = (result or {}).get("steps") if isinstance(result, dict) else None
        print(f"{scn['key']}: {len(events)} events, {sum(len(e.get('bibcodes') or []) for e in events)} bibcode refs (steps={steps})")
        captured.append(
            {
                "key": scn["key"],
                "label": scn["label"],
                "query": scn["query"],
                "narration": scn["narration"],
                "events": events,
            }
        )

    titles = _resolve_titles(sorted(all_bibs))
    payload = {"scenarios": captured, "titles": titles}
    out_file = out_dir / "trace_replay.json"
    out_file.write_text(json.dumps(payload), encoding="utf-8")
    print(f"\nwrote {out_file}: {len(captured)} scenarios, {len(titles)} titles resolved")
    return 0 if any(s["events"] for s in captured) else 1


if __name__ == "__main__":
    raise SystemExit(main())
