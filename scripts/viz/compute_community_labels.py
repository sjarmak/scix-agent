#!/usr/bin/env python3
"""Extract top-N terms per community from paper titles/abstracts.

Samples up to `samples_per_community` papers per community, tokenises their
title (and optionally abstract), removes stopwords + common astronomy filler,
and emits the top-N highest-frequency terms per community.

Output schema:
    {
        "resolution": "coarse",
        "communities": [
            {"community_id": 0, "n_sampled": 1523, "terms": ["stars", "..."]},
            ...
        ]
    }

Usage:
    .venv/bin/python scripts/viz/compute_community_labels.py \
        --resolution coarse --samples-per-community 2000 \
        --output data/viz/community_labels.json
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import psycopg

from scix.db import DEFAULT_DSN, is_production_dsn, redact_dsn

logger = logging.getLogger("compute_community_labels")

COMMUNITY_COLUMNS = {
    "coarse": "community_semantic_coarse",
    "medium": "community_semantic_medium",
    "fine": "community_semantic_fine",
}

# English + astronomy-generic stopwords. Tuned for title/abstract terms where
# filler words dominate (e.g. "study", "analysis", "observations").
STOPWORDS = frozenset(
    """
    a an the of in on at to for and or but with from by as is are was were be
    been being have has had do does did will would should could may might can
    that this these those it its their them they we our you your i me my
    not no all any some many much more most each every other another such
    which what who whose where when why how than then so also both either
    neither one two three four five six seven eight nine ten first second
    new novel recent present presented here study studies analysis analyse
    analyzed results show shows shown find found finds observation
    observations observed observing method methods used using use uses
    models model modeled modelling modeling simulation simulations
    approach approaches paper papers report reports reported based
    suggest suggests suggested demonstrate demonstrates data way ways
    between over under above below about into within without among
    via onto upon through during after before across toward towards
    though although because since yet while whereas therefore thus
    hence however moreover furthermore additionally also still even
    only just mainly primarily essentially effectively generally
    significantly substantially notably particularly especially
    very well good high higher low lower large larger small smaller
    same similar different various several multiple single individual
    sub sup loc post inf infty alpha beta gamma delta pi tau sigma phi psi
    mml mmlmath mo mi mn mtext mfenced mrow xmlns math href span class
    proposed showed show showing shows find found finds finding findings
    mathrm mathit mathbf mathsf mathtt operatorname text hbox vbox overline
    underline frac sqrt left right cdot times approx equiv propto pm mp
    rightarrow leftarrow leftrightarrow longrightarrow longleftarrow
    order value values function functions average mean medium strong weak
    fig figure table section chapter abstract summary conclusion conclusions
    effect effects type types kind kinds level levels range ranges
    region regions area areas system systems case cases example examples
    role role-based rate rates ratio ratios factor factors number numbers
    non near far low-density high-density full-scale
    """.split()
)


@dataclass(frozen=True)
class Config:
    dsn: str
    resolution: str
    samples_per_community: int
    top_terms: int
    use_abstract: bool
    output: Path


def parse_args(argv: list[str] | None = None) -> Config:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dsn", default=DEFAULT_DSN)
    parser.add_argument("--resolution", choices=list(COMMUNITY_COLUMNS), default="coarse")
    parser.add_argument("--samples-per-community", type=int, default=2000)
    parser.add_argument("--top-terms", type=int, default=6)
    parser.add_argument(
        "--use-abstract",
        action="store_true",
        help="Include abstracts in the term pool (slower, more thematic).",
    )
    parser.add_argument("--output", type=Path, default=Path("data/viz/community_labels.json"))
    ns = parser.parse_args(argv)
    return Config(
        dsn=ns.dsn,
        resolution=ns.resolution,
        samples_per_community=ns.samples_per_community,
        top_terms=ns.top_terms,
        use_abstract=ns.use_abstract,
        output=ns.output,
    )


_WORD = re.compile(r"[A-Za-z][A-Za-z0-9\-]{2,30}")


def _tokenise(text: str) -> list[str]:
    return [w.lower() for w in _WORD.findall(text)]


def _top_terms(rows: list[tuple[str, str | None]], k: int) -> tuple[list[str], int]:
    counts: Counter[str] = Counter()
    for title, abstract in rows:
        words = _tokenise(title or "")
        if abstract:
            words.extend(_tokenise(abstract))
        for w in words:
            if w in STOPWORDS:
                continue
            counts[w] += 1
    return [w for w, _ in counts.most_common(k)], len(rows)


def run(config: Config) -> dict:
    column = COMMUNITY_COLUMNS[config.resolution]
    logger.info(
        "compute_community_labels dsn=%s resolution=%s column=%s samples_per_community=%d",
        redact_dsn(config.dsn),
        config.resolution,
        column,
        config.samples_per_community,
    )
    if is_production_dsn(config.dsn):
        logger.info("DSN points at production (read-only query, proceeding)")

    # One big TABLESAMPLE pass + Python-side bucketing is ~1000x faster than
    # per-community ORDER BY random() over the 32M papers table. Pick a sample
    # percentage that's expected to yield >= samples_per_community per bucket.
    # With 32M rows and 20 coarse communities, 0.25% yields ~80K rows total
    # ~~ 4000/community, plenty of headroom.
    sample_pct = 0.25 if config.resolution == "coarse" else (1.0 if config.resolution == "medium" else 3.0)
    select_cols = "p.title, p.abstract" if config.use_abstract else "p.title, ''"
    buckets: dict[int, list[tuple[str, str | None]]] = {}

    with psycopg.connect(config.dsn) as conn:
        conn.set_read_only(True)
        logger.info("sampling ~%.2f%% of papers in a single pass", sample_pct)
        sql = (
            f"SELECT pm.{column} AS cid, {select_cols} "
            f"FROM papers p TABLESAMPLE SYSTEM (%s) "
            f"JOIN paper_metrics pm USING (bibcode) "
            f"WHERE pm.{column} IS NOT NULL AND p.title IS NOT NULL"
        )
        with conn.cursor(name="labels_stream") as cur:
            cur.itersize = 10_000
            cur.execute(sql, (sample_pct,))
            n_seen = 0
            for cid, title, abstract in cur:
                bucket = buckets.setdefault(int(cid), [])
                if len(bucket) < config.samples_per_community:
                    bucket.append((title, abstract))
                n_seen += 1
                if n_seen % 20_000 == 0:
                    logger.info("sampled %d rows; buckets=%d", n_seen, len(buckets))
        logger.info("sample pass complete: %d rows across %d communities", n_seen, len(buckets))

    out_communities: list[dict] = []
    for cid in sorted(buckets.keys()):
        rows = buckets[cid]
        terms, n = _top_terms(rows, config.top_terms)
        logger.info("community=%d n=%d terms=%s", cid, n, terms)
        out_communities.append({"community_id": cid, "n_sampled": n, "terms": terms})
    return {"resolution": config.resolution, "communities": out_communities}


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    config = parse_args(argv)
    config.output.parent.mkdir(parents=True, exist_ok=True)
    result = run(config)
    config.output.write_text(json.dumps(result, indent=2))
    logger.info("wrote %s", config.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
