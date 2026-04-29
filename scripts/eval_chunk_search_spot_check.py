#!/usr/bin/env python3
"""Spot-check eval for ``chunk_search`` vs paper-level ``search()``.

Implements PRD ``chunk-embeddings-build`` Phase 5 §P0-7.

Runs a curated workload of 30 queries spanning 5 SMD disciplines
(astrophysics, heliophysics, planetary, earth science, biology) against the
two retrieval surfaces side-by-side and writes a Markdown report to
``docs/eval/chunk_search_v1_eval.md``.

Each query is annotated with ``query_class`` ∈ {method, dataset, software,
broad}. The PRD acceptance bar (P0-7) is that ``chunk_search`` beats abstract
``search()`` on **≥ 70% of method/dataset/software pairs** by hand-judged
top-3 relevance. Broad queries are expected to lose; that is documented but
does not gate the bar.

This script does **not** make hand judgments — those are filled in by a human
reviewer after the report is generated. It produces:

* The two top-K result sets per query (bibcode + title + score), side by side.
* A scoring table with one row per query and an empty ``human_top3_winner``
  column (values: ``chunk``, ``search``, ``tie``).
* A summary block reading the filled-in column to compute the
  method/dataset/software win-rate when the human review is complete.

Re-running with the same input replaces the existing report.

Output schema
-------------

The Markdown report contains:

1. Header — git SHA, ran_at, total queries, chunk_search availability.
2. Summary table (top of file) — discipline × win-rate. Empty until column
   ``human_top3_winner`` is filled in.
3. Per-query sections — query text, class, discipline, then two columns of
   top-3 hits with bibcode/title/score.
4. Companion JSON ``docs/eval/chunk_search_v1_eval.json`` — machine-readable
   raw results for downstream automated analysis.

Usage
-----

    QDRANT_URL=http://127.0.0.1:6333 python scripts/eval_chunk_search_spot_check.py
    python scripts/eval_chunk_search_spot_check.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Add src/ to path for direct script execution
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Spot-check workload
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class SpotQuery:
    """A single spot-check query."""

    qid: str
    query: str
    discipline: str  # astrophysics, heliophysics, planetary, earth, biology
    query_class: str  # method, dataset, software, broad


SPOT_QUERIES: tuple[SpotQuery, ...] = (
    # Astrophysics: 6 (2 broad, 4 specific)
    SpotQuery("ap-1", "exoplanet atmospheres spectroscopy", "astrophysics", "broad"),
    SpotQuery("ap-2", "papers using MCMC on cosmological parameters", "astrophysics", "method"),
    SpotQuery("ap-3", "papers using SDSS DR16 photometry", "astrophysics", "dataset"),
    SpotQuery(
        "ap-4", "Bayesian hierarchical modeling of exoplanet transits", "astrophysics", "method"
    ),
    SpotQuery(
        "ap-5", "papers using emcee affine-invariant ensemble sampler", "astrophysics", "software"
    ),
    SpotQuery("ap-6", "dark matter direct detection", "astrophysics", "broad"),
    # Heliophysics: 6
    SpotQuery("hp-1", "solar coronal mass ejection forecasting", "heliophysics", "broad"),
    SpotQuery(
        "hp-2",
        "papers using Parker Solar Probe FIELDS magnetometer data",
        "heliophysics",
        "dataset",
    ),
    SpotQuery("hp-3", "WSA-ENLIL coupled solar wind model simulations", "heliophysics", "method"),
    SpotQuery(
        "hp-4", "papers using SDO/AIA EUV image data for flare detection", "heliophysics", "dataset"
    ),
    SpotQuery("hp-5", "papers using SWMF magnetohydrodynamic code", "heliophysics", "software"),
    SpotQuery("hp-6", "geomagnetic storm Dst index prediction", "heliophysics", "broad"),
    # Planetary: 6
    SpotQuery("pl-1", "Mars surface mineralogy", "planetary", "broad"),
    SpotQuery("pl-2", "papers using Curiosity ChemCam LIBS spectra", "planetary", "dataset"),
    SpotQuery(
        "pl-3", "transit timing variation method for exomoon detection", "planetary", "method"
    ),
    SpotQuery("pl-4", "asteroid Bond albedo radiative thermal modeling", "planetary", "method"),
    SpotQuery(
        "pl-5", "papers using SPICE toolkit for spacecraft trajectory", "planetary", "software"
    ),
    SpotQuery("pl-6", "Saturn ring particle size distribution", "planetary", "broad"),
    # Earth science: 6
    SpotQuery("es-1", "ENSO teleconnection precipitation", "earth", "broad"),
    SpotQuery("es-2", "papers using MODIS Collection 6 aerosol optical depth", "earth", "dataset"),
    SpotQuery("es-3", "Sentinel-1 InSAR persistent scatterer interferometry", "earth", "method"),
    SpotQuery("es-4", "papers using GPM IMERG precipitation product", "earth", "dataset"),
    SpotQuery("es-5", "papers using GEOS-Chem chemical transport model", "earth", "software"),
    SpotQuery("es-6", "Arctic sea ice extent decline", "earth", "broad"),
    # Biology / bioastronomy: 6
    SpotQuery("bi-1", "extremophile microbial community in Atacama Desert", "biology", "broad"),
    SpotQuery("bi-2", "papers using QIIME2 16S rRNA amplicon analysis", "biology", "software"),
    SpotQuery("bi-3", "16S rRNA shotgun metagenomics extremophile", "biology", "method"),
    SpotQuery("bi-4", "biosignature methane detection in Mars atmosphere", "biology", "broad"),
    SpotQuery(
        "bi-5", "papers using NASA GeneLab spaceflight transcriptomics", "biology", "dataset"
    ),
    SpotQuery("bi-6", "tardigrade ionizing radiation tolerance mechanism", "biology", "broad"),
)


# --------------------------------------------------------------------------
# Result dataclasses
# --------------------------------------------------------------------------


@dataclass
class TopHit:
    """A single top-K hit from either retrieval surface."""

    rank: int
    bibcode: str
    title: str | None
    score: float
    section_heading: str | None = None  # only chunk_search


@dataclass
class QueryResult:
    """Side-by-side results for one spot-check query."""

    qid: str
    query: str
    discipline: str
    query_class: str
    chunk_hits: list[TopHit]
    search_hits: list[TopHit]
    chunk_latency_ms: float
    search_latency_ms: float
    chunk_error: str | None = None
    search_error: str | None = None


# --------------------------------------------------------------------------
# Backend wiring
# --------------------------------------------------------------------------


class _StubBackends:
    """Dry-run stand-in producing deterministic stub hits.

    Lets the script run end-to-end without Postgres or Qdrant for plumbing
    validation. Real runs use ``_RealBackends``.
    """

    def chunk_top3(self, sq: SpotQuery) -> tuple[list[TopHit], float]:
        return (
            [
                TopHit(
                    rank=i + 1,
                    bibcode=f"2024STUB.{sq.qid}.{i}",
                    title=f"[stub chunk] {sq.query} — variant {i}",
                    score=0.95 - i * 0.04,
                    section_heading="Methods" if i == 0 else "Results",
                )
                for i in range(3)
            ],
            7.5,
        )

    def search_top3(self, sq: SpotQuery) -> tuple[list[TopHit], float]:
        return (
            [
                TopHit(
                    rank=i + 1,
                    bibcode=f"2023STUB.{sq.qid}.S{i}",
                    title=f"[stub abstract] {sq.query} — variant {i}",
                    score=0.90 - i * 0.05,
                )
                for i in range(3)
            ],
            12.4,
        )

    def close(self) -> None:
        pass


class _RealBackends:
    """Wires up real chunk_search (Qdrant + INDUS) and abstract hybrid_search (PG)."""

    def __init__(self) -> None:
        from scix import embed as _embed
        from scix import qdrant_tools as _qt
        from scix.db import get_connection

        self._embed = _embed
        self._qdrant_tools = _qt
        self._model, self._tokenizer = _embed.load_model("indus", device="cpu")
        self._conn = get_connection()

    def _encode(self, text: str) -> list[float]:
        vecs = self._embed.embed_batch(
            self._model, self._tokenizer, [text], batch_size=1, pooling="mean"
        )
        if not vecs:
            raise RuntimeError("encoder returned no vectors")
        return vecs[0]

    def _titles_for(self, bibcodes: list[str]) -> dict[str, str]:
        if not bibcodes:
            return {}
        with self._conn.cursor() as cur:
            cur.execute(
                "SELECT bibcode, title FROM papers WHERE bibcode = ANY(%s)",
                (bibcodes,),
            )
            return {b: t for b, t in cur.fetchall()}

    def chunk_top3(self, sq: SpotQuery) -> tuple[list[TopHit], float]:
        vec = self._encode(sq.query)
        t0 = time.perf_counter()
        hits = self._qdrant_tools.chunk_search_by_text(vec, limit=3)
        latency = (time.perf_counter() - t0) * 1000.0
        titles = self._titles_for([h.bibcode for h in hits])
        return (
            [
                TopHit(
                    rank=i + 1,
                    bibcode=h.bibcode,
                    title=titles.get(h.bibcode),
                    score=h.score,
                    section_heading=h.section_heading or h.section_heading_norm,
                )
                for i, h in enumerate(hits)
            ],
            latency,
        )

    def search_top3(self, sq: SpotQuery) -> tuple[list[TopHit], float]:
        from scix.search import hybrid_search

        vec = self._encode(sq.query)
        t0 = time.perf_counter()
        result = hybrid_search(self._conn, sq.query, query_embedding=vec, top_n=3)
        latency = (time.perf_counter() - t0) * 1000.0
        return (
            [
                TopHit(
                    rank=i + 1,
                    bibcode=p.get("bibcode", ""),
                    title=p.get("title"),
                    score=float(p.get("score", 0.0)),
                )
                for i, p in enumerate(result.papers[:3])
            ],
            latency,
        )

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:  # noqa: BLE001
            pass


def _resolve_backend(*, dry_run: bool) -> Any:
    if dry_run:
        return _StubBackends()
    if not os.environ.get("QDRANT_URL"):
        raise RuntimeError(
            "QDRANT_URL is not set. Either start the eval with "
            "QDRANT_URL=http://... or pass --dry-run."
        )
    return _RealBackends()


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parent.parent,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def run_spot_check(*, dry_run: bool) -> list[QueryResult]:
    backend = _resolve_backend(dry_run=dry_run)
    results: list[QueryResult] = []
    try:
        for sq in SPOT_QUERIES:
            chunk_err: str | None = None
            search_err: str | None = None
            try:
                chunk_hits, chunk_ms = backend.chunk_top3(sq)
            except Exception as exc:  # noqa: BLE001 — boundary
                logger.exception("chunk_search failed for %s", sq.qid)
                chunk_hits, chunk_ms = [], 0.0
                chunk_err = str(exc)
            try:
                search_hits, search_ms = backend.search_top3(sq)
            except Exception as exc:  # noqa: BLE001 — boundary
                logger.exception("search failed for %s", sq.qid)
                search_hits, search_ms = [], 0.0
                search_err = str(exc)
            results.append(
                QueryResult(
                    qid=sq.qid,
                    query=sq.query,
                    discipline=sq.discipline,
                    query_class=sq.query_class,
                    chunk_hits=chunk_hits,
                    search_hits=search_hits,
                    chunk_latency_ms=round(chunk_ms, 2),
                    search_latency_ms=round(search_ms, 2),
                    chunk_error=chunk_err,
                    search_error=search_err,
                )
            )
    finally:
        backend.close()
    return results


# --------------------------------------------------------------------------
# Markdown rendering
# --------------------------------------------------------------------------


def _format_hit(h: TopHit) -> str:
    title = (h.title or "(title unavailable)").replace("|", "\\|")
    if len(title) > 90:
        title = title[:87] + "..."
    section = f" §{h.section_heading}" if h.section_heading else ""
    return f"`{h.bibcode}`{section} — {title} (score={h.score:.3f})"


def render_markdown(results: list[QueryResult], *, dry_run: bool) -> str:
    lines: list[str] = []
    lines.append("# `chunk_search` v1 — Spot-check Eval")
    lines.append("")
    lines.append(f"- **Generated:** {_utc_now()}")
    lines.append(f"- **Git SHA:** `{_git_sha()}`")
    lines.append(f"- **Workload:** {len(results)} queries across 5 disciplines")
    lines.append(f"- **Mode:** {'dry-run (stub backends)' if dry_run else 'production'}")
    lines.append(
        "- **PRD acceptance:** P0-7 — `chunk_search` beats abstract `search()` "
        "on ≥ 70% of method/dataset/software pairs by hand-judged top-3 relevance."
    )
    lines.append("")
    lines.append("## Reviewer instructions")
    lines.append("")
    lines.append(
        "For each query below, judge which surface returned the more relevant "
        "top-3 set and fill in the `human_top3_winner` column in the table "
        "(values: `chunk`, `search`, `tie`). The summary block at the bottom "
        "is regenerated by re-running this script with `--score-from <path>` "
        "after the column is populated."
    )
    lines.append("")
    lines.append("## Pair-by-pair scoring table")
    lines.append("")
    lines.append("| qid | discipline | class | query | human_top3_winner |")
    lines.append("|---|---|---|---|---|")
    for r in results:
        q = r.query.replace("|", "\\|")
        lines.append(f"| {r.qid} | {r.discipline} | {r.query_class} | {q} |  |")
    lines.append("")
    lines.append("## Per-query results")
    lines.append("")
    for r in results:
        lines.append(f"### {r.qid} — {r.query}")
        lines.append("")
        lines.append(f"- discipline: `{r.discipline}` · class: `{r.query_class}`")
        lines.append(
            f"- latency: chunk_search={r.chunk_latency_ms} ms · " f"search={r.search_latency_ms} ms"
        )
        if r.chunk_error:
            lines.append(f"- ⚠ chunk_search error: `{r.chunk_error}`")
        if r.search_error:
            lines.append(f"- ⚠ search error: `{r.search_error}`")
        lines.append("")
        lines.append("**chunk_search top-3:**")
        if r.chunk_hits:
            for h in r.chunk_hits:
                lines.append(f"{h.rank}. {_format_hit(h)}")
        else:
            lines.append("- _(no results)_")
        lines.append("")
        lines.append("**search() top-3:**")
        if r.search_hits:
            for h in r.search_hits:
                lines.append(f"{h.rank}. {_format_hit(h)}")
        else:
            lines.append("- _(no results)_")
        lines.append("")
    lines.append("## Summary (auto-regenerated when scoring column is filled)")
    lines.append("")
    lines.append(
        "_Re-run with `--score-from <this-file>` after filling the "
        "`human_top3_winner` column. Until then this section is empty._"
    )
    lines.append("")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------
# Win-rate scorer (run after a human fills in the column)
# --------------------------------------------------------------------------


def _parse_winners(scored_md: Path) -> dict[str, str]:
    """Extract qid -> winner from the scoring table column."""
    text = scored_md.read_text(encoding="utf-8")
    winners: dict[str, str] = {}
    in_table = False
    for line in text.splitlines():
        if line.startswith("| qid "):
            in_table = True
            continue
        if in_table:
            if not line.startswith("|"):
                break
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if len(cells) < 5:
                continue
            if cells[0] == "---":
                continue
            qid, winner = cells[0], cells[-1].lower()
            if winner in {"chunk", "search", "tie"}:
                winners[qid] = winner
    return winners


def compute_win_rate(scored_md: Path) -> dict[str, Any]:
    """Compute discipline/class win-rates from a scored markdown file."""
    winners = _parse_winners(scored_md)
    by_qid = {sq.qid: sq for sq in SPOT_QUERIES}
    targeted = {"method", "dataset", "software"}

    classes: dict[str, dict[str, int]] = {}
    disciplines: dict[str, dict[str, int]] = {}
    for qid, winner in winners.items():
        sq = by_qid.get(qid)
        if sq is None:
            continue
        c = classes.setdefault(sq.query_class, {"chunk": 0, "search": 0, "tie": 0, "total": 0})
        d = disciplines.setdefault(sq.discipline, {"chunk": 0, "search": 0, "tie": 0, "total": 0})
        c[winner] += 1
        c["total"] += 1
        d[winner] += 1
        d["total"] += 1

    targeted_total = sum(c["total"] for k, c in classes.items() if k in targeted)
    targeted_chunk_wins = sum(c["chunk"] for k, c in classes.items() if k in targeted)
    targeted_win_rate = (targeted_chunk_wins / targeted_total) if targeted_total else 0.0

    return {
        "n_judged": sum(c["total"] for c in classes.values()),
        "by_class": classes,
        "by_discipline": disciplines,
        "targeted_chunk_win_rate": round(targeted_win_rate, 3),
        "p0_7_pass": targeted_win_rate >= 0.70,
    }


def render_summary_md(stats: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("## Summary (auto-regenerated when scoring column is filled)")
    lines.append("")
    lines.append(f"- **Judged pairs:** {stats['n_judged']} / {len(SPOT_QUERIES)}")
    lines.append(
        f"- **Method+dataset+software win-rate (chunk vs search):** "
        f"{stats['targeted_chunk_win_rate']:.1%}"
    )
    lines.append(f"- **PRD P0-7 pass (≥ 70%):** {'✅ pass' if stats['p0_7_pass'] else '❌ fail'}")
    lines.append("")
    lines.append("### By query class")
    lines.append("")
    lines.append("| class | chunk wins | search wins | tie | total | win rate |")
    lines.append("|---|---|---|---|---|---|")
    for cls, c in sorted(stats["by_class"].items()):
        rate = (c["chunk"] / c["total"]) if c["total"] else 0.0
        lines.append(
            f"| {cls} | {c['chunk']} | {c['search']} | {c['tie']} | {c['total']} | " f"{rate:.1%} |"
        )
    lines.append("")
    lines.append("### By discipline")
    lines.append("")
    lines.append("| discipline | chunk wins | search wins | tie | total | win rate |")
    lines.append("|---|---|---|---|---|---|")
    for disc, d in sorted(stats["by_discipline"].items()):
        rate = (d["chunk"] / d["total"]) if d["total"] else 0.0
        lines.append(
            f"| {disc} | {d['chunk']} | {d['search']} | {d['tie']} | {d['total']} | "
            f"{rate:.1%} |"
        )
    lines.append("")
    return "\n".join(lines) + "\n"


def apply_summary(report_path: Path, stats: dict[str, Any]) -> None:
    """Replace the existing summary block in ``report_path`` with computed stats."""
    text = report_path.read_text(encoding="utf-8")
    marker = "## Summary (auto-regenerated when scoring column is filled)"
    idx = text.find(marker)
    if idx == -1:
        raise ValueError(f"summary marker not found in {report_path}")
    new_summary = render_summary_md(stats)
    report_path.write_text(text[:idx] + new_summary, encoding="utf-8")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("docs/eval/chunk_search_v1_eval.md"),
        help="Markdown report path. Companion JSON sits next to it.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Use stub backends — no Postgres, no Qdrant, no INDUS load.",
    )
    parser.add_argument(
        "--score-from",
        type=Path,
        default=None,
        help=(
            "Path to a previously generated, human-scored Markdown report. "
            "When passed, the script regenerates the summary block in-place "
            "without re-running the workload."
        ),
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    if args.score_from is not None:
        stats = compute_win_rate(args.score_from)
        apply_summary(args.score_from, stats)
        logger.info(
            "Updated summary in %s (targeted win-rate %.1f%%, P0-7 %s)",
            args.score_from,
            stats["targeted_chunk_win_rate"] * 100,
            "PASS" if stats["p0_7_pass"] else "FAIL",
        )
        return 0

    results = run_spot_check(dry_run=args.dry_run)

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(render_markdown(results, dry_run=args.dry_run), encoding="utf-8")

    json_path = args.report.with_suffix(".json")
    payload = {
        "schema_version": 1,
        "tool": "chunk_search vs search()",
        "ran_at": _utc_now(),
        "git_sha": _git_sha(),
        "dry_run": args.dry_run,
        "queries": [asdict(r) for r in results],
    }
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    logger.info("Wrote %s and %s", args.report, json_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
