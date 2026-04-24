#!/usr/bin/env python3
"""Community-label generation (PRD M4).

For each ``(signal, resolution, community_id)`` triple across the three
community signals (citation Leiden, semantic k-means, taxonomic arXiv
class), compute:

- Top 5 arXiv-class codes by paper count
- Top 10 ``keyword_norm`` tokens by TF-IDF, where each "document" is a
  community and "term frequency" counts papers-in-community that carry
  the keyword.

Writes labels + top_keywords into the ``communities`` table (migration
052 schema), then emits a 20-row random spot-check markdown to
``results/community_labels_spotcheck.md`` with a copy at
``docs/prd/artifacts/community_labels_spotcheck.sample.md`` (since
``results/`` is gitignored).

Safety: writes are blocked against production DSNs unless
``--allow-prod`` is passed. Default DSN resolves from ``SCIX_TEST_DSN``
so forgetting to set the test env variable does not silently mutate
production.

Usage::

    SCIX_TEST_DSN=dbname=scix_test \\
    python scripts/generate_community_labels.py \\
        --signal all --seed 42

The script is deterministic given the same corpus and the same
``--seed`` (label strings are derived from sorted top-arXiv +
sorted top-keywords; the spot-check shuffle is seeded too).
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import re
import sys
import zlib
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Optional, Sequence

import psycopg

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from scix.db import is_production_dsn, redact_dsn  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("generate_community_labels")


RESULTS_PATH = REPO_ROOT / "results" / "community_labels_spotcheck.md"
SAMPLE_PATH = (
    REPO_ROOT / "docs" / "prd" / "artifacts" / "community_labels_spotcheck.sample.md"
)

# Per-signal resolution lists. Taxonomic has a single resolution because
# paper_metrics.community_taxonomic is a scalar TEXT, not per-level.
RESOLUTIONS_BY_SIGNAL: dict[str, tuple[str, ...]] = {
    "citation": ("coarse", "medium", "fine"),
    "semantic": ("coarse", "medium", "fine"),
    "taxonomic": ("coarse",),
}

VALID_SIGNALS: frozenset[str] = frozenset(RESOLUTIONS_BY_SIGNAL.keys())

TOP_ARXIV_N = 5
TOP_KEYWORDS_N = 10
LABEL_ARXIV_N = 2  # embedded in the label string
LABEL_KEYWORDS_N = 3


# ---------------------------------------------------------------------------
# Title-token fallback (when keyword_norm is empty)
#
# As of 2026-04-24 ``papers.keyword_norm`` is NULL for all 32.4M rows — the
# normalization step planned in the original pipeline never ran. Without a
# fallback, every community gets an empty ``top_keywords`` and a useless
# label. The viz tool ``scripts/viz/compute_community_labels.py`` proved
# title-text tokenization yields legible terms for the coarse semantic
# partition; we reuse its regex and stopword list here so production labels
# have the same quality as the viz labels.
# ---------------------------------------------------------------------------

_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9\-]{2,30}")

STOPWORDS: frozenset[str] = frozenset(
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


def _tokenize_title(title: Optional[str]) -> list[str]:
    """Lower-case title tokens minus stopwords, deduplicated per paper.

    Dedup-per-paper keeps a single paper from stuffing its title's repeated
    word into the community's document-frequency count; the TF-IDF already
    treats the community as the document, so "paper carries token" is the
    natural unit.
    """
    if not title:
        return []
    seen: set[str] = set()
    out: list[str] = []
    for w in _WORD_RE.findall(title):
        wl = w.lower()
        if wl in STOPWORDS or wl in seen:
            continue
        seen.add(wl)
        out.append(wl)
    return out


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CommunityLabel:
    """A row to UPSERT into communities."""

    signal: str
    resolution: str
    community_id: int
    label: str
    paper_count: int
    top_keywords: list[str]


# ---------------------------------------------------------------------------
# DSN + safety gate
# ---------------------------------------------------------------------------


def _resolve_dsn(cli_dsn: Optional[str]) -> str:
    """Resolve the effective DSN.

    Precedence: ``--dsn`` CLI flag → ``SCIX_TEST_DSN`` env var. We
    deliberately do NOT fall back to ``SCIX_DSN`` (which often points at
    production) — callers must pass ``--allow-prod`` and provide the DSN
    explicitly to target production.
    """
    if cli_dsn:
        return cli_dsn
    test_dsn = os.environ.get("SCIX_TEST_DSN")
    if test_dsn:
        return test_dsn
    raise SystemExit(
        "No DSN resolved: pass --dsn or set SCIX_TEST_DSN. "
        "SCIX_DSN fallback is intentionally disabled to prevent accidental "
        "production writes — pass --allow-prod + --dsn explicitly for prod."
    )


# ---------------------------------------------------------------------------
# Source column + type resolution
# ---------------------------------------------------------------------------


def _community_column(signal: str, resolution: str) -> str:
    """Return the ``paper_metrics`` column that carries the community id
    for the given (signal, resolution) tuple.
    """
    if signal == "citation":
        return f"community_id_{resolution}"
    if signal == "semantic":
        return f"community_semantic_{resolution}"
    if signal == "taxonomic":
        return "community_taxonomic"
    raise ValueError(f"unknown signal: {signal!r}")


def _taxonomic_id(text_label: str) -> int:
    """Map a taxonomic string (e.g. 'astro-ph.GA') to a stable non-negative INT
    suitable for ``communities.community_id``.

    We use adler32 (stable across Python versions, fast) masked to 31 bits so
    the result fits in a regular signed INT4. Collisions are astronomically
    unlikely for the ~200 arXiv classes; the risk we take is worth the
    round-tripable determinism.
    """
    return zlib.adler32(text_label.encode("utf-8")) & 0x7FFFFFFF


# ---------------------------------------------------------------------------
# Per-(signal, resolution) aggregation
# ---------------------------------------------------------------------------


def _iter_rows(
    conn: psycopg.Connection,
    signal: str,
    resolution: str,
    cursor_name: str,
) -> Iterator[
    tuple[object, str, Optional[list[str]], Optional[list[str]], Optional[str]]
]:
    """Stream (community_id_raw, bibcode, arxiv_class, keyword_norm, title) rows.

    ``community_id_raw`` is INT for citation/semantic, TEXT for taxonomic.
    Callers normalise it via ``_taxonomic_id`` as needed.

    ``title`` is used as a fallback term source when ``keyword_norm`` is
    empty (see ``_tokenize_title``).
    """
    column = _community_column(signal, resolution)
    sql = (
        f"SELECT pm.{column}, p.bibcode, p.arxiv_class, p.keyword_norm, p.title "
        f"  FROM paper_metrics pm "
        f"  JOIN papers p ON p.bibcode = pm.bibcode "
        f" WHERE pm.{column} IS NOT NULL "
        f" ORDER BY pm.{column}, p.bibcode"
    )
    with conn.cursor(name=cursor_name) as cur:
        cur.itersize = 5_000
        cur.execute(sql)
        yield from cur


@dataclass
class _Bucket:
    """Per-community accumulator."""

    bibcodes: list[str]
    arxiv_counter: Counter
    kw_counter: Counter  # papers-in-community-with-this-keyword


def _aggregate(
    conn: psycopg.Connection,
    signal: str,
    resolution: str,
) -> tuple[dict[int, _Bucket], dict[int, str]]:
    """Aggregate paper rows into per-community buckets.

    Returns ``(by_community, taxonomic_text_by_id)``. The second dict is
    populated only for ``signal='taxonomic'`` and records the original
    TEXT label for each hashed id — used later to keep the label
    human-readable.
    """
    by_community: dict[int, _Bucket] = {}
    taxonomic_text_by_id: dict[int, str] = {}

    with conn.transaction():
        for raw_cid, bibcode, arxiv_class, keyword_norm, title in _iter_rows(
            conn, signal, resolution, f"gcl_{signal}_{resolution}"
        ):
            if raw_cid is None:
                continue
            if signal == "taxonomic":
                text_label = str(raw_cid)
                cid = _taxonomic_id(text_label)
                taxonomic_text_by_id[cid] = text_label
            else:
                cid = int(raw_cid)

            bucket = by_community.get(cid)
            if bucket is None:
                bucket = _Bucket(
                    bibcodes=[], arxiv_counter=Counter(), kw_counter=Counter()
                )
                by_community[cid] = bucket
            bucket.bibcodes.append(bibcode)

            if arxiv_class:
                # Count each class once per paper (not per-token duplicated).
                bucket.arxiv_counter.update(set(arxiv_class))

            # Prefer curated keyword_norm when present; fall back to title
            # tokens so the 100% of papers that lack keyword_norm today still
            # yield legible labels.
            normed = (
                {kw for kw in keyword_norm if kw and kw.strip()}
                if keyword_norm
                else set()
            )
            if normed:
                bucket.kw_counter.update(normed)
            else:
                tokens = _tokenize_title(title)
                if tokens:
                    bucket.kw_counter.update(tokens)

    return by_community, taxonomic_text_by_id


# ---------------------------------------------------------------------------
# TF-IDF
# ---------------------------------------------------------------------------


def compute_tfidf(
    per_community_kw_counts: dict[int, Counter],
    top_n: int = TOP_KEYWORDS_N,
) -> dict[int, list[tuple[str, float]]]:
    """Compute per-community TF-IDF over keyword_norm terms.

    - TF(term, c) = papers in community c with that keyword.
    - IDF(term)   = log(N / (1 + df(term))), N = #communities, df(term) =
      #communities with TF > 0. +1 smoothing keeps IDF > 0 even for
      terms that appear everywhere.
    - Score       = TF * IDF.

    Returns a dict mapping community_id -> top_n [(term, score)] sorted
    by score DESC, term ASC for deterministic ties.
    """
    n_communities = len(per_community_kw_counts)
    if n_communities == 0:
        return {}

    # Document frequency: #communities that contain the term at least once.
    df: Counter = Counter()
    for counts in per_community_kw_counts.values():
        for term in counts.keys():
            df[term] += 1

    # Precompute idf
    idf: dict[str, float] = {
        term: math.log(n_communities / (1 + df_val))
        for term, df_val in df.items()
    }

    out: dict[int, list[tuple[str, float]]] = {}
    for cid, counts in per_community_kw_counts.items():
        scored: list[tuple[str, float]] = [
            (term, tf * idf[term]) for term, tf in counts.items()
        ]
        # Sort by score DESC, then term ASC for determinism.
        scored.sort(key=lambda pair: (-pair[1], pair[0]))
        out[cid] = scored[:top_n]
    return out


# ---------------------------------------------------------------------------
# Label format
# ---------------------------------------------------------------------------


def make_label(
    top_arxiv: Sequence[str],
    top_keywords: Sequence[str],
    *,
    arxiv_n: int = LABEL_ARXIV_N,
    keywords_n: int = LABEL_KEYWORDS_N,
) -> str:
    """Produce a short, deterministic, human-readable label.

    Format:
        "<a1> + <a2> · <kw1> / <kw2> / <kw3>"

    Empty halves are dropped. Fully unlabeled communities return
    ``"unlabeled"``.
    """
    ax = [a for a in top_arxiv[:arxiv_n] if a]
    kw = [k for k in top_keywords[:keywords_n] if k]
    ax_part = " + ".join(ax)
    kw_part = " / ".join(kw)
    if ax_part and kw_part:
        return f"{ax_part} · {kw_part}"
    if ax_part:
        return ax_part
    if kw_part:
        return kw_part
    return "unlabeled"


# ---------------------------------------------------------------------------
# UPSERT
# ---------------------------------------------------------------------------


_UPSERT_SQL = """
INSERT INTO communities
    (signal, resolution, community_id, label, paper_count, top_keywords, updated_at)
VALUES (%s, %s, %s, %s, %s, %s, now())
ON CONFLICT (signal, resolution, community_id)
DO UPDATE SET label        = EXCLUDED.label,
              paper_count  = EXCLUDED.paper_count,
              top_keywords = EXCLUDED.top_keywords,
              updated_at   = now()
"""


def _upsert_labels(
    conn: psycopg.Connection,
    rows: Iterable[CommunityLabel],
) -> int:
    """Bulk-upsert a batch of labels. Returns row count."""
    payload = [
        (
            row.signal,
            row.resolution,
            row.community_id,
            row.label,
            row.paper_count,
            row.top_keywords,
        )
        for row in rows
    ]
    if not payload:
        return 0
    with conn.cursor() as cur:
        cur.executemany(_UPSERT_SQL, payload)
    return len(payload)


# ---------------------------------------------------------------------------
# One (signal, resolution) pass
# ---------------------------------------------------------------------------


def _label_one(
    conn: psycopg.Connection,
    signal: str,
    resolution: str,
) -> int:
    """Compute labels for all communities at (signal, resolution), upsert."""
    by_community, taxonomic_text_by_id = _aggregate(conn, signal, resolution)
    if not by_community:
        logger.info(
            "signal=%s resolution=%s: no communities found, skipping",
            signal,
            resolution,
        )
        return 0

    per_community_kw_counts: dict[int, Counter] = {
        cid: bucket.kw_counter for cid, bucket in by_community.items()
    }
    tfidf = compute_tfidf(per_community_kw_counts, top_n=TOP_KEYWORDS_N)

    rows: list[CommunityLabel] = []
    for cid, bucket in by_community.items():
        top_arxiv_pairs = bucket.arxiv_counter.most_common(TOP_ARXIV_N)
        # Stable order across ties: most_common is already count-desc, we
        # append a secondary alpha sort via sorted(... key=(-count, cls)).
        top_arxiv_sorted = sorted(
            top_arxiv_pairs, key=lambda pair: (-pair[1], pair[0])
        )
        top_arxiv = [a for a, _ in top_arxiv_sorted]

        top_kw_pairs = tfidf.get(cid, [])
        top_kw = [term for term, _ in top_kw_pairs]

        label = make_label(top_arxiv, top_kw)

        # For taxonomic, prepend the source arXiv class so the label stays
        # readable even when the bucket has diverse arxiv_class arrays
        # inside it (unlikely, but defensive).
        if signal == "taxonomic":
            source_text = taxonomic_text_by_id.get(cid, "")
            if source_text and source_text not in label:
                label = f"{source_text} · {label}" if label != "unlabeled" else source_text

        rows.append(
            CommunityLabel(
                signal=signal,
                resolution=resolution,
                community_id=cid,
                label=label,
                paper_count=len(bucket.bibcodes),
                top_keywords=top_kw,
            )
        )

    # Deterministic UPSERT order.
    rows.sort(key=lambda r: (r.signal, r.resolution, r.community_id))

    with conn.transaction():
        n = _upsert_labels(conn, rows)
    logger.info(
        "signal=%s resolution=%s: upserted %d community labels",
        signal,
        resolution,
        n,
    )
    return n


# ---------------------------------------------------------------------------
# Spot-check
# ---------------------------------------------------------------------------


def _fetch_top_members(
    conn: psycopg.Connection,
    signal: str,
    resolution: str,
    community_id: int,
    taxonomic_text_by_id: dict[int, str],
    limit: int = 3,
) -> list[tuple[str, Optional[str]]]:
    """Return up to ``limit`` (bibcode, title) rows for a community.

    Ordered by citation_count DESC NULLS LAST, then bibcode ASC for
    deterministic output.
    """
    column = _community_column(signal, resolution)
    if signal == "taxonomic":
        filter_val: object = taxonomic_text_by_id.get(community_id, "")
        if not filter_val:
            return []
    else:
        filter_val = community_id

    sql = (
        f"SELECT p.bibcode, p.title "
        f"  FROM paper_metrics pm "
        f"  JOIN papers p ON p.bibcode = pm.bibcode "
        f" WHERE pm.{column} = %s "
        f" ORDER BY p.citation_count DESC NULLS LAST, p.bibcode ASC "
        f" LIMIT %s"
    )
    with conn.cursor() as cur:
        cur.execute(sql, (filter_val, limit))
        return [(row[0], row[1]) for row in cur.fetchall()]


def _spotcheck_markdown(
    conn: psycopg.Connection,
    taxonomic_text_by_id: dict[int, str],
    n: int,
    seed: int,
) -> str:
    """Return a markdown string with ``n`` random community spot-checks."""
    # Deterministic shuffle: hash (signal, resolution, community_id, seed)
    # with md5 and ORDER BY the hash.
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT signal, resolution, community_id, label, top_keywords, paper_count
              FROM communities
             WHERE label IS NOT NULL
             ORDER BY md5(signal || '|' || resolution || '|' || community_id::text
                          || '|' || %s::text)
             LIMIT %s
            """,
            (seed, n),
        )
        picks = cur.fetchall()

    if not picks:
        return (
            "# community_labels spot-check\n\n"
            "_No labeled communities found. "
            "Run `generate_community_labels.py` on a database with populated "
            "`paper_metrics.community_*` columns first._\n"
        )

    lines: list[str] = []
    lines.append("# community_labels spot-check\n")
    lines.append(
        f"Random sample of {len(picks)} communities (seed={seed}) across signals. "
        "Each section shows the generated label, top keywords, and the top 3 "
        "member papers by `citation_count`.\n"
    )

    for i, (signal, resolution, community_id, label, top_kw, paper_count) in enumerate(picks, 1):
        lines.append(
            f"## {i}. signal=`{signal}` resolution=`{resolution}` "
            f"community_id=`{community_id}`"
        )
        lines.append("")
        lines.append(f"- **Label**: {label}")
        lines.append(f"- **Paper count**: {paper_count}")
        kw_str = ", ".join(f"`{k}`" for k in (top_kw or [])) or "_none_"
        lines.append(f"- **Top keywords**: {kw_str}")

        members = _fetch_top_members(
            conn, signal, resolution, community_id, taxonomic_text_by_id
        )
        if members:
            lines.append("- **Top 3 member papers**:")
            for bib, title in members:
                safe_title = (title or "").replace("|", "\\|").strip() or "_(no title)_"
                lines.append(f"  - `{bib}` — {safe_title}")
        else:
            lines.append("- **Top 3 member papers**: _(no members found)_")
        lines.append("")

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--signal",
        choices=["citation", "semantic", "taxonomic", "all"],
        default="all",
        help="Which signal to generate labels for. Default: all.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--dsn",
        default=None,
        help="PostgreSQL DSN. Defaults to SCIX_TEST_DSN env var.",
    )
    parser.add_argument(
        "--allow-prod",
        action="store_true",
        help="Required to write to a production DSN (see is_production_dsn).",
    )
    parser.add_argument("--spotcheck-n", type=int, default=20)
    parser.add_argument(
        "--spotcheck-path",
        default=str(RESULTS_PATH),
        help="Path to the 20-row spot-check markdown (lives in results/, gitignored).",
    )
    parser.add_argument(
        "--spotcheck-sample-path",
        default=str(SAMPLE_PATH),
        help="Path to the committed sample copy of the spot-check markdown.",
    )
    return parser.parse_args(argv)


def _iter_targets(signal_arg: str) -> Iterator[tuple[str, str]]:
    """Yield (signal, resolution) pairs to process."""
    if signal_arg == "all":
        signals: tuple[str, ...] = ("citation", "semantic", "taxonomic")
    else:
        signals = (signal_arg,)
    for signal in signals:
        for resolution in RESOLUTIONS_BY_SIGNAL[signal]:
            yield signal, resolution


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)

    dsn = _resolve_dsn(args.dsn)
    if is_production_dsn(dsn) and not args.allow_prod:
        logger.error(
            "Refusing to write to production DSN %s — pass --allow-prod to override",
            redact_dsn(dsn),
        )
        return 2

    logger.info(
        "generate_community_labels: dsn=%s signal=%s seed=%d spotcheck_n=%d",
        redact_dsn(dsn),
        args.signal,
        args.seed,
        args.spotcheck_n,
    )

    # Collect taxonomic text-id mapping across the run so the spot-check
    # can resolve bibcodes for taxonomic rows.
    taxonomic_text_by_id: dict[int, str] = {}

    total_upserted = 0
    # One connection per (signal, resolution) triple. Under autocommit=False
    # a single outer psycopg.connect wraps everything in ONE implicit
    # transaction — `with conn.transaction()` inside _label_one /
    # _aggregate would only create savepoints, so no write would be
    # durable until script exit, and any accidental kill mid-run would
    # roll back all 9 triples. A fresh connection per triple commits
    # cleanly on close and resets per-triple aggregation state.
    for signal, resolution in _iter_targets(args.signal):
        with psycopg.connect(dsn) as conn:
            conn.autocommit = False
            # _label_one is atomic per resolution (one transaction).
            if signal == "taxonomic":
                # Need the id->text mapping out of the aggregation pass —
                # _label_one throws it away internally, so we re-materialize
                # here. Cheap: taxonomic cardinality is ~200 classes.
                by_community, tax_map = _aggregate(conn, signal, resolution)
                taxonomic_text_by_id.update(tax_map)
                # Fall through to the real labelling pass (which re-aggregates).
                _ = by_community  # explicit discard

            n = _label_one(conn, signal, resolution)
            total_upserted += n
            conn.commit()

    # Spot-check reads from the committed communities table; use a
    # fresh short-lived connection so we observe the just-written data
    # and don't pin any tx state.
    with psycopg.connect(dsn) as conn:
        conn.autocommit = False
        spotcheck_md = _spotcheck_markdown(
            conn, taxonomic_text_by_id, args.spotcheck_n, args.seed
        )

    spotcheck_path = Path(args.spotcheck_path)
    sample_path = Path(args.spotcheck_sample_path)
    spotcheck_path.parent.mkdir(parents=True, exist_ok=True)
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    spotcheck_path.write_text(spotcheck_md, encoding="utf-8")
    sample_path.write_text(spotcheck_md, encoding="utf-8")

    logger.info(
        "Done. Upserted %d community labels. Spot-check at %s (sample at %s).",
        total_upserted,
        spotcheck_path,
        sample_path,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
