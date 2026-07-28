"""Generate static ego-network snapshots for the website scix-viz bundle.

Picks a handful of high-PageRank hub papers from distinct coarse communities,
exports each one's citation ego network (the same payload the live
``/viz/api/ego/{bibcode}`` endpoint returns), and writes an ``index.json``
manifest the static page uses to build its paper picker.

Run from the repo root:  .venv/bin/python scripts/viz/build_ego_snapshot.py <out_dir>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from scix.db import get_connection
from scix.viz.api import _fetch_ego_network

# Probe the global PageRank hubs (index-backed) and keep the first few that
# form a visually-rich neighborhood. Global hubs beat per-community tops here:
# their 1-hop neighbors are themselves well-cited, so the 2-hop layer fills in
# instead of collapsing to leaves.
CANDIDATE_POOL = 400
MAX_EXPORTS = 5
MIN_REFS, MIN_CITES, MIN_SECOND_HOP = 12, 12, 15
MAX_REFS, MAX_CITES, MAX_SECOND_HOP = 100, 100, 200
# One export per distinct coarse community so the picker spans the map.
ONE_PER_COMMUNITY = True


def _top_hub_candidates() -> list[str]:
    """Top-PageRank papers globally (index-backed via idx_pm_pagerank)."""
    conn = get_connection()
    with conn.cursor() as cur:
        cur.execute(
            "SELECT bibcode FROM paper_metrics "
            "ORDER BY pagerank DESC NULLS LAST LIMIT %s",
            (CANDIDATE_POOL,),
        )
        return [r[0] for r in cur.fetchall()]


def main() -> int:
    out_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "web/viz/ego")
    ego_dir = out_dir
    ego_dir.mkdir(parents=True, exist_ok=True)

    manifest: list[dict] = []
    seen_communities: set[int] = set()
    for bib in _top_hub_candidates():
        if len(manifest) >= MAX_EXPORTS:
            break
        payload = _fetch_ego_network(bib, MAX_REFS, MAX_CITES, MAX_SECOND_HOP)
        if payload is None:
            print(f"skip {bib}: not in papers")
            continue
        refs = len(payload.get("direct_refs") or [])
        cites = len(payload.get("direct_cites") or [])
        hop = len(payload.get("second_hop_sample") or [])
        # Require a graph worth looking at: real refs, cites, and 2-hop fan-out.
        if refs < MIN_REFS or cites < MIN_CITES or hop < MIN_SECOND_HOP:
            print(f"skip {bib}: thin graph ({refs} refs / {cites} cites / {hop} 2-hop)")
            continue
        center = payload.get("center") or {}
        cid = center.get("community_id")
        # Dedupe only on real community ids; None is "unknown", not a bucket.
        if ONE_PER_COMMUNITY and cid is not None and cid in seen_communities:
            print(f"skip {bib}: community {cid} already covered")
            continue
        if cid is not None:
            seen_communities.add(cid)
        safe = bib.replace("/", "_")
        (ego_dir / f"{safe}.json").write_text(json.dumps(payload), encoding="utf-8")
        manifest.append(
            {
                "bibcode": bib,
                "file": f"{safe}.json",
                "title": center.get("title") or "",
                "community_id": center.get("community_id"),
                "counts": {"direct_refs": refs, "direct_cites": cites, "second_hop": hop},
            }
        )
        print(f"export {bib}: {refs} refs / {cites} cites / {hop} 2-hop")

    (ego_dir / "index.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nwrote {len(manifest)} ego networks + index.json to {ego_dir}")
    return 0 if manifest else 1


if __name__ == "__main__":
    raise SystemExit(main())
