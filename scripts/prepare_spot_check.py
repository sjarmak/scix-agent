#!/usr/bin/env python3
"""Prepare a 30-row operator-labeling CSV for bead xz4.1.32 anchor B.

Takes the abstract-only calibration draft, picks:

  - all 15 rows where the judge self-flagged ``needs_human_review=true``
  - 15 additional rows sampled uniformly across the 6 lanes, excluding the
    already-flagged rows

Fetches the **full** snippet from the ``papers`` table (title + abstract +
first 500 chars of body — same substrate the judge saw) so the operator has
the same evidence in front of them when labeling. Writes
``data/eval/calibration_spot_check.csv`` with:

    query_id, lane, query, bibcode, title, draft_score, snippet, human_score

``human_score`` starts blank. After the operator labels, the companion runner
``scripts/calibrate_judge.py`` computes the quadratic-weighted kappa.

Usage::

    python scripts/prepare_spot_check.py

    # smaller / larger samples
    python scripts/prepare_spot_check.py --n-random 20

    # skip DB lookup (dev mode — just copy snippet_preview)
    python scripts/prepare_spot_check.py --no-db
"""

from __future__ import annotations

import argparse
import csv
import logging
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from scix.eval.persona_judge import build_snippet  # noqa: E402

logger = logging.getLogger(__name__)

DEFAULT_SOURCE_CSV = REPO_ROOT / "data" / "eval" / "calibration_seed_draft_abstract_only.csv"
DEFAULT_OUTPUT_CSV = REPO_ROOT / "data" / "eval" / "calibration_spot_check.csv"


def fetch_full_snippets(bibcodes: list[str], *, dsn: str | None) -> dict[str, str]:
    """Fetch (title + abstract + 500-char body) from papers table."""
    if not bibcodes:
        return {}

    from scix.db import get_connection

    out: dict[str, str] = {}
    with get_connection(dsn=dsn) as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT bibcode, title, abstract, body FROM papers WHERE bibcode = ANY(%s)",
            (bibcodes,),
        )
        for bibcode, title, abstract, body in cur.fetchall():
            if not title:
                continue
            out[bibcode] = build_snippet(title=title, abstract=abstract, body=body)
    return out


def select_rows(rows: list[dict], *, n_random: int, seed: int) -> list[dict]:
    """Pick flagged + stratified random; dedupe by (query_id, bibcode)."""
    rng = random.Random(seed)

    flagged = [r for r in rows if r["needs_human_review"] == "true"]
    unflagged = [r for r in rows if r["needs_human_review"] != "true"]

    # Stratify the random pick across lanes (proportional to lane size).
    from collections import defaultdict

    lanes: dict[str, list[dict]] = defaultdict(list)
    for r in unflagged:
        lanes[r["lane"]].append(r)

    lane_names = sorted(lanes.keys())
    per_lane = max(1, n_random // len(lane_names))
    extra = n_random - per_lane * len(lane_names)

    random_picks: list[dict] = []
    for lane in lane_names:
        take = per_lane + (1 if extra > 0 else 0)
        extra = max(0, extra - 1)
        pool = lanes[lane][:]
        rng.shuffle(pool)
        random_picks.extend(pool[:take])

    picked = flagged + random_picks
    # Dedupe defensively.
    seen: set[tuple[str, str]] = set()
    unique: list[dict] = []
    for r in picked:
        key = (r["query_id"], r["bibcode"])
        if key in seen:
            continue
        seen.add(key)
        unique.append(r)

    rng.shuffle(unique)
    return unique


def write_output(
    path: Path, rows: list[dict], snippets: dict[str, str]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = [
        "query_id",
        "lane",
        "query",
        "bibcode",
        "title",
        "draft_score",
        "needs_human_review",
        "snippet",
        "human_score",  # operator fills this
    ]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, quoting=csv.QUOTE_MINIMAL)
        w.writeheader()
        for r in rows:
            snippet = snippets.get(r["bibcode"], "")
            if not snippet:
                # Fall back to the draft preview if DB miss. Mark it.
                snippet = f"[NO DB LOOKUP] {r.get('snippet_preview', '')}"
            w.writerow(
                {
                    "query_id": r["query_id"],
                    "lane": r["lane"],
                    "query": r["query"],
                    "bibcode": r["bibcode"],
                    "title": r["title"],
                    "draft_score": r["draft_score"],
                    "needs_human_review": r["needs_human_review"],
                    "snippet": snippet,
                    "human_score": "",  # blank for operator
                }
            )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", type=Path, default=DEFAULT_SOURCE_CSV)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_CSV)
    p.add_argument("--n-random", type=int, default=15)
    p.add_argument("--seed", type=int, default=20260422)
    p.add_argument("--dsn", default=None)
    p.add_argument(
        "--no-db", action="store_true", help="Skip DB lookup; copy snippet_preview as fallback."
    )
    p.add_argument("--log-level", default="INFO")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    with args.source.open() as f:
        rows = list(csv.DictReader(f))
    logger.info("%d source rows", len(rows))

    picked = select_rows(rows, n_random=args.n_random, seed=args.seed)
    logger.info("%d rows selected (flagged + random)", len(picked))

    if args.no_db:
        snippets: dict[str, str] = {}
    else:
        bibcodes = [r["bibcode"] for r in picked]
        logger.info("fetching full snippets for %d bibcodes...", len(bibcodes))
        snippets = fetch_full_snippets(bibcodes, dsn=args.dsn)
        missing = [b for b in bibcodes if b not in snippets]
        if missing:
            logger.warning("DB missing for %d bibcodes: %s", len(missing), missing[:5])

    write_output(args.output, picked, snippets)
    logger.info("wrote %s", args.output)

    # Report lane / flag distribution for sanity.
    from collections import Counter

    lane_c = Counter(r["lane"] for r in picked)
    flag_c = Counter(r["needs_human_review"] for r in picked)
    print("\nSpot-check selection distribution")
    print(f"  total     : {len(picked)}")
    print(f"  by lane   : {dict(sorted(lane_c.items()))}")
    print(f"  flagged   : {dict(flag_c)}")
    print(f"\n  output CSV: {args.output}")
    print("\nNext step: open the CSV, fill the 'human_score' column (0-3) for each row.")
    print("Then run: scripts/calibrate_judge.py --seed data/eval/calibration_spot_check.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
