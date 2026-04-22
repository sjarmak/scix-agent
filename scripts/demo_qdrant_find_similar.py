"""Demo: find_similar_by_examples using Qdrant's recommendation API.

Sets QDRANT_URL, picks a few seed papers from the pilot collection, and shows
three queries:

  1. Positive-only   — "more like this landmark paper"
  2. Positive + negative — "in this direction, not that one"
  3. Positive + payload filter — "more like these, but only in astro-ph.EP"

Run:

    QDRANT_URL=http://127.0.0.1:6333 \\
        python scripts/demo_qdrant_find_similar.py
"""
from __future__ import annotations

import os
import sys
import textwrap

os.environ.setdefault("QDRANT_URL", "http://127.0.0.1:6333")

from scix import qdrant_tools as qt  # noqa: E402


def pick_seeds(n_per_community: int = 2, communities: int = 3) -> list[str]:
    """Return bibcodes from a few different communities, as seed examples."""
    from qdrant_client import QdrantClient
    client = QdrantClient(os.environ["QDRANT_URL"], timeout=10)
    scrolled, _ = client.scroll(
        collection_name=qt.COLLECTION,
        limit=2000,
        with_payload=True,
    )
    by_comm: dict[int, list[str]] = {}
    for p in scrolled:
        payload = p.payload or {}
        comm = payload.get("community_semantic_coarse")
        bibcode = payload.get("bibcode")
        if comm is None or bibcode is None:
            continue
        by_comm.setdefault(int(comm), []).append(bibcode)
    # Pick the N biggest communities, take n_per_community bibcodes from each.
    top_communities = sorted(by_comm.items(), key=lambda x: -len(x[1]))[:communities]
    picked: list[str] = []
    for _comm, codes in top_communities:
        picked.extend(codes[:n_per_community])
    return picked


def print_results(hits, header: str):
    print()
    print(header)
    print("-" * len(header))
    for i, h in enumerate(hits, 1):
        ax = ",".join(h.arxiv_class[:2]) if h.arxiv_class else "-"
        title = (h.title or "?")[:90]
        print(f"  {i:>2}. {h.bibcode:<19}  {h.year!s:<4}  c={h.community_semantic!s:<5}  "
              f"ax={ax:<18}  s={h.score:.3f}")
        print(f"      {textwrap.fill(title, 92, subsequent_indent='      ')}")


def main() -> int:
    info = qt.collection_info()
    print(f"Qdrant collection: {info}")
    if info.get("status") == "unavailable":
        print("collection not ready", file=sys.stderr)
        return 1

    seeds = pick_seeds(n_per_community=2, communities=3)
    if len(seeds) < 3:
        print("not enough seeds found", file=sys.stderr)
        return 1

    print(f"\nSeed bibcodes (sampled across communities): {seeds}")

    # Query 1: positive-only
    hits1 = qt.find_similar_by_examples(
        positive_bibcodes=seeds[:2],
        limit=8,
    )
    print_results(hits1, f"(1) positive-only: more like {seeds[:2]}")

    # Query 2: positive + negative
    hits2 = qt.find_similar_by_examples(
        positive_bibcodes=seeds[:2],
        negative_bibcodes=seeds[2:3],
        limit=8,
    )
    print_results(hits2, f"(2) positive {seeds[:2]}  NEGATIVE {seeds[2:3]}")

    # Query 3: positive + filter (restrict to most common community in seeds)
    from qdrant_client import QdrantClient
    client = QdrantClient(os.environ["QDRANT_URL"], timeout=5)
    ids = [qt.bibcode_to_point_id(b) for b in seeds[:2]]
    pts = client.retrieve(qt.COLLECTION, ids=ids, with_payload=True)
    common_comm = pts[0].payload.get("community_semantic_coarse") if pts else None
    if common_comm is not None:
        hits3 = qt.find_similar_by_examples(
            positive_bibcodes=seeds[:2],
            limit=8,
            community_semantic=int(common_comm),
        )
        print_results(hits3, f"(3) positive + filter community_semantic={common_comm}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
