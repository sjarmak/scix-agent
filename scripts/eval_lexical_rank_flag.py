#!/usr/bin/env python3
"""A/B eval: ts_rank_cd normalization flag vs broad-query lexical-search quality.

Bead scix_experiments-q9k5. On broad single-token terms the uncapped
``lexical_search`` top-20 is dominated by short non-article docs (press
releases, proposals, theses) because ``ts_rank_cd`` rewards term *density* and
short text has high density. Seminal high-citation articles rank below or
outside the top-20.

The current normalization flag (32 = ``rank/(rank+1)``) is a monotonic squash
into [0,1) that does NOT length-normalize — a short dense doc still outranks a
long one on the same term (verified, q9k5 comment gc-351512). The length-aware
bits OR onto 32: 33 = 32|1 (``/(1+log(length))``), 48 = 32|16
(``/(1+log(unique words))``). This script A/Bs flags {32, 33, 48} over the
broad-term stress set and asks whether a length-aware flag lifts the
article-fraction and the seminal (high-citation) content of the top-20 WITHOUT
pulling in long, low-citation junk.

What it does
------------
For each flag it sets ``SCIX_LEXICAL_RANK_FLAG`` and runs every query through
the real :func:`scix.search.lexical_search` (DRY — same code path the MCP
serves), with the candidate pool held fixed across flags (default INF = the
full match set) so the candidate set is identical and complete and only the
ranking changes. For each top-20 it measures, against the PROD corpus:

* ``article_fraction`` — share of top-20 with ``doctype='article'`` (↑ better).
* ``short_doc_fraction`` — share in the short non-article doctypes the bead
  calls out: press release / proposal / thesis (↓ better).
* ``n_seminal`` — count with ``citation_count >= 500`` (↑ better: seminal
  articles climbing into the top-20).
* ``median_citation`` — guards against over-correction: a length-aware flag
  that merely swaps short junk for long *low-citation* junk shows no median
  lift (this is the precision proxy — there is no relevance gold for these
  queries; the existing gold_bibcodes are the short-doc ceiling, NOT a
  relevance judgment, so they are deliberately NOT used for scoring).

NOT a recall-ceiling eval (that is eval_lexical_recall_pool.py, orthogonal:
that varies the pool *cap*; this varies the rank *flag*). No gold_bibcodes
scoring here.

Decision rule (closes the bead either way)
------------------------------------------
A candidate flag is recommended iff, vs flag 32, it raises article_fraction AND
does not regress median_citation or n_seminal (seminal articles climb without a
citation-quality drop). The best such flag is adopted in
``scix.search`` via ``SCIX_LEXICAL_RANK_FLAG`` default. If no candidate clears
the bar, the negative result stands: broad-concept recall is the vector lane's
job in RRF and the lexical flag stays 32 — recorded in the eval doc/ADR.

Operational notes
-----------------
* Read-only (SELECT only); must run on prod ``scix`` (the real corpus —
  ``scix_test`` is schema-only).
* Run under ``scix-batch``. A prod-health preflight REFUSES on red (postgres
  down, low MemAvailable, or high cgroup memory pressure) — prod recovered
  from repeated OOM, so the eval bounds itself: ``work_mem=256MB``,
  ``max_parallel_workers_per_gather=0``, bounded ``statement_timeout``.

Usage
-----
::

    scix-batch python scripts/eval_lexical_rank_flag.py
    python scripts/eval_lexical_rank_flag.py --flags 32,33,48 --quiet
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from statistics import median
from typing import Any, Sequence

# Make ``src`` and the sibling ``scripts/`` dir importable from a checkout root.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

# Reuse the gold-set loader so query parsing stays identical to the canonical
# harnesses (DRY — one EvalQuery definition).
from eval_retrieval_50q import EvalQuery, load_queries  # noqa: E402

logger = logging.getLogger("eval_lexical_rank_flag")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_QUERIES: str = "eval/lexical_stress_16q.jsonl"
DEFAULT_OUTPUT: str = "docs/eval/lexical_rank_flag_2026-06.json"
# ts_rank_cd flags to sweep. 32 is the prod default (baseline); 33 = 32|1 and
# 48 = 32|16 add log-length damping. See module docstring.
DEFAULT_FLAGS: tuple[int, ...] = (32, 33, 48)
BASELINE_FLAG: int = 32
# Candidate pool. Default INF (uncapped) so ts_rank_cd ranks the FULL match set
# under every flag — the candidate set is then identical and complete across
# flags, so the rank flag is the only variable. This deliberately isolates the
# bead's question (how ranking scores short vs long docs) from the candidate
# cap (eval_lexical_recall_pool.py's orthogonal axis): at the prod cap of 30000
# the TID-ordered slice may not even contain the seminal articles, so a null
# result there cannot distinguish "flag does not help" from "seminal docs were
# capped out before ranking". INF removes that confound.
DEFAULT_POOL: str = "INF"
RETRIEVE_LIMIT: int = 20
# A result is "seminal" at or above this citation count. Matches the bead's
# cited examples (cit=633, cit=716; it names cit>500 broadly).
SEMINAL_CITATION: int = 500
ARTICLE_DOCTYPE: str = "article"
# The short non-article doctypes the bead names (pres / nsf-prop / PhDT) plus
# masters theses — the docs ts_rank_cd over-ranks on broad terms.
SHORT_DOCTYPES: frozenset[str] = frozenset(
    {"pressrelease", "proposal", "phdthesis", "mastersthesis"}
)

# Prod-health preflight gates (the eval refuses to run when prod is red).
# PSI (cgroup memory.pressure full avg10) is the PRIMARY signal: it is what
# trips user@1000's oomd at 50% and cascades into the gascity supervisor
# (CLAUDE.md #1). The MemAvailable floor is a coarse starvation guard, not the
# main gate — on this host postgres' large shared segment keeps MemAvailable
# structurally well below 20GiB during normal operation, so a 20GiB floor would
# refuse forever. The eval's incremental demand is <1GiB (client ~165MiB,
# postgres backend bounded by work_mem=256MB single-threaded; measured on the
# ~850k-row "spectroscopy" match set), so a 6GiB floor leaves >6x headroom
# while still refusing a genuinely starved host. (q9k5: gc-364392 — "RAM/PSI
# are the real signals"; the swap-saturation gate is dropped for the same.)
MIN_MEM_AVAILABLE_GIB: float = 6.0
MAX_PSI_FULL_AVG10: float = 10.0
PG_UNIT: str = "postgresql@16-main"
_PSI_PATH = Path("/sys/fs/cgroup/user.slice/user-1000.slice/user@1000.service/memory.pressure")

# Bound the eval connection. Uncapped ts_rank_cd over the largest match set
# (~850k rows for "spectroscopy", single-threaded at work_mem=256MB) runs
# ~100s; 180s leaves headroom without unbounding a post-OOM prod.
DEFAULT_STATEMENT_TIMEOUT_MS: int = 180_000
# Exit code distinct from a clean run / decision so the caller can tell a
# refused preflight from a completed eval.
EXIT_PREFLIGHT_REFUSED: int = 3


# ---------------------------------------------------------------------------
# Prod-health preflight
# ---------------------------------------------------------------------------


def _mem_available_gib() -> float | None:
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / (1024 * 1024)  # kB -> GiB
    except OSError:
        return None
    return None


def _psi_full_avg10() -> float | None:
    """cgroup memory pressure (the oomd-cascade signal). None if unreadable."""
    try:
        for line in _PSI_PATH.read_text().splitlines():
            if line.startswith("full"):
                for field in line.split():
                    if field.startswith("avg10="):
                        return float(field.split("=", 1)[1])
    except OSError:
        return None
    return None


def _pg_active() -> bool:
    try:
        out = subprocess.run(
            ["systemctl", "is-active", PG_UNIT],
            capture_output=True,
            text=True,
            timeout=15,
        )
        return out.stdout.strip() == "active"
    except (OSError, subprocess.SubprocessError):
        return False


def preflight(min_mem_gib: float) -> list[str]:
    """Return a list of failing-gate reasons; empty list means prod is green.

    Hard gates: postgres must be active and MemAvailable above the floor.
    PSI is checked only when readable (a missing cgroup file is not a failure —
    the path differs across hosts/scopes).
    """
    reasons: list[str] = []
    if not _pg_active():
        reasons.append(f"{PG_UNIT} is not active")
    mem = _mem_available_gib()
    if mem is None:
        reasons.append("could not read MemAvailable from /proc/meminfo")
    elif mem < min_mem_gib:
        reasons.append(f"MemAvailable {mem:.1f}GiB < {min_mem_gib:.0f}GiB floor")
    psi = _psi_full_avg10()
    if psi is not None and psi > MAX_PSI_FULL_AVG10:
        reasons.append(f"cgroup memory.pressure full avg10 {psi:.1f} > {MAX_PSI_FULL_AVG10:.0f}")
    return reasons


# ---------------------------------------------------------------------------
# Per-(query, flag) result
# ---------------------------------------------------------------------------


class FlagQueryResult:
    """Top-20 quality of one (query, flag) pair on the prod corpus."""

    __slots__ = (
        "query",
        "n_hits",
        "article_fraction",
        "short_doc_fraction",
        "n_seminal",
        "median_citation",
        "mean_citation",
        "top1_citation",
        "latency_ms",
        "error",
    )

    def __init__(
        self,
        *,
        query: str,
        n_hits: int,
        article_fraction: float | None,
        short_doc_fraction: float | None,
        n_seminal: int,
        median_citation: float | None,
        mean_citation: float | None,
        top1_citation: int | None,
        latency_ms: float,
        error: str | None,
    ) -> None:
        self.query = query
        self.n_hits = n_hits
        self.article_fraction = article_fraction
        self.short_doc_fraction = short_doc_fraction
        self.n_seminal = n_seminal
        self.median_citation = median_citation
        self.mean_citation = mean_citation
        self.top1_citation = top1_citation
        self.latency_ms = latency_ms
        self.error = error

    @property
    def scored(self) -> bool:
        return self.error is None and self.n_hits > 0


# ---------------------------------------------------------------------------
# Retrieval + scoring
# ---------------------------------------------------------------------------


def _fetch_doctypes(conn: Any, bibcodes: Sequence[str]) -> dict[str, str | None]:
    """Look up doctype for the returned bibcodes (PK lookup, cheap)."""
    if not bibcodes:
        return {}
    with conn.cursor() as cur:
        cur.execute(
            "SELECT bibcode, doctype FROM papers WHERE bibcode = ANY(%s)",
            (list(bibcodes),),
        )
        return {row[0]: row[1] for row in cur.fetchall()}


def _score_top20(
    query: str,
    papers: Sequence[dict[str, Any]],
    doctypes: dict[str, str | None],
    latency_ms: float,
) -> FlagQueryResult:
    """Compute the top-20 quality metrics for one (query, flag) result."""
    n = len(papers)
    n_article = 0
    n_short = 0
    n_seminal = 0
    citations: list[int] = []
    top1_citation: int | None = None
    for i, p in enumerate(papers):
        dt = doctypes.get(p["bibcode"])
        if dt == ARTICLE_DOCTYPE:
            n_article += 1
        if dt in SHORT_DOCTYPES:
            n_short += 1
        cit = p.get("citation_count")
        if cit is not None:
            citations.append(int(cit))
            if int(cit) >= SEMINAL_CITATION:
                n_seminal += 1
            if i == 0:
                top1_citation = int(cit)
    return FlagQueryResult(
        query=query,
        n_hits=n,
        article_fraction=(n_article / n) if n else None,
        short_doc_fraction=(n_short / n) if n else None,
        n_seminal=n_seminal,
        median_citation=float(median(citations)) if citations else None,
        mean_citation=(sum(citations) / len(citations)) if citations else None,
        top1_citation=top1_citation,
        latency_ms=latency_ms,
        error=None,
    )


def run_flag(
    conn: Any,
    flag: int,
    queries: Sequence[EvalQuery],
    pool: str,
) -> list[FlagQueryResult]:
    """Run every query through ``lexical_search`` at one rank flag.

    Sets ``SCIX_LEXICAL_RANK_FLAG`` (read per call by ``_resolve_lexical_rank_flag``)
    and pins ``SCIX_LEXICAL_POOL`` to ``pool`` for the duration so the candidate
    set is held fixed across flags.
    """
    from scix.search import lexical_search

    prev_flag = os.environ.get("SCIX_LEXICAL_RANK_FLAG")
    prev_pool = os.environ.get("SCIX_LEXICAL_POOL")
    os.environ["SCIX_LEXICAL_RANK_FLAG"] = str(flag)
    os.environ["SCIX_LEXICAL_POOL"] = pool
    results: list[FlagQueryResult] = []
    try:
        for q in queries:
            t0 = time.perf_counter()
            try:
                sr = lexical_search(conn, q.query, limit=RETRIEVE_LIMIT)
                latency_ms = (time.perf_counter() - t0) * 1000.0
                doctypes = _fetch_doctypes(conn, [p["bibcode"] for p in sr.papers])
                results.append(_score_top20(q.query, sr.papers, doctypes, latency_ms))
            except Exception as exc:  # noqa: BLE001 — record, don't score
                try:
                    conn.rollback()
                except Exception:
                    pass
                latency_ms = (time.perf_counter() - t0) * 1000.0
                logger.warning("flag=%d query=%r failed: %s", flag, q.query, exc)
                results.append(
                    FlagQueryResult(
                        query=q.query,
                        n_hits=0,
                        article_fraction=None,
                        short_doc_fraction=None,
                        n_seminal=0,
                        median_citation=None,
                        mean_citation=None,
                        top1_citation=None,
                        latency_ms=latency_ms,
                        error=f"{type(exc).__name__}: {exc}",
                    )
                )
    finally:
        for var, prev in (
            ("SCIX_LEXICAL_RANK_FLAG", prev_flag),
            ("SCIX_LEXICAL_POOL", prev_pool),
        ):
            if prev is None:
                os.environ.pop(var, None)
            else:
                os.environ[var] = prev
    return results


# ---------------------------------------------------------------------------
# Aggregation + decision
# ---------------------------------------------------------------------------


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def aggregate(results: Sequence[FlagQueryResult]) -> dict[str, Any]:
    scored = [r for r in results if r.scored]
    return {
        "n_queries": len(results),
        "n_scored": len(scored),
        "n_errored": sum(1 for r in results if r.error is not None),
        "article_fraction": _mean([r.article_fraction for r in scored]),  # type: ignore[misc]
        "short_doc_fraction": _mean([r.short_doc_fraction for r in scored]),  # type: ignore[misc]
        "n_seminal_mean": _mean([float(r.n_seminal) for r in scored]),
        "median_citation": _mean(
            [r.median_citation for r in scored if r.median_citation is not None]
        ),
        "mean_citation": _mean([r.mean_citation for r in scored if r.mean_citation is not None]),
        "latency_ms_max": max((r.latency_ms for r in results), default=0.0),
    }


def decide(by_flag: dict[int, dict[str, Any]]) -> dict[str, Any]:
    """Pick the best length-aware flag, or keep 32 (negative result).

    A candidate is eligible iff vs the baseline it raises article_fraction and
    regresses neither median_citation nor n_seminal_mean. Among eligible
    candidates the one with the largest article_fraction gain wins.
    """
    base = by_flag[BASELINE_FLAG]
    candidates: list[dict[str, Any]] = []
    deltas: dict[int, dict[str, Any]] = {}
    for flag, agg in by_flag.items():
        if flag == BASELINE_FLAG:
            continue
        d = {
            "article_fraction_delta": _delta(agg["article_fraction"], base["article_fraction"]),
            "short_doc_fraction_delta": _delta(
                agg["short_doc_fraction"], base["short_doc_fraction"]
            ),
            "n_seminal_delta": _delta(agg["n_seminal_mean"], base["n_seminal_mean"]),
            "median_citation_delta": _delta(agg["median_citation"], base["median_citation"]),
        }
        deltas[flag] = d
        af = d["article_fraction_delta"]
        sem = d["n_seminal_delta"]
        med = d["median_citation_delta"]
        eligible = (
            af is not None
            and af > 0.0
            and (sem is None or sem >= 0.0)
            and (med is None or med >= 0.0)
        )
        if eligible:
            candidates.append({"flag": flag, "article_fraction_delta": af})

    if candidates:
        winner = max(candidates, key=lambda c: c["article_fraction_delta"])
        recommended = winner["flag"]
        rationale = (
            f"flag {recommended} raises article_fraction by "
            f"{winner['article_fraction_delta']:+.3f} vs baseline {BASELINE_FLAG} "
            f"without regressing median_citation or n_seminal — adopt it as the "
            f"SCIX_LEXICAL_RANK_FLAG default."
        )
    else:
        recommended = None
        rationale = (
            f"no length-aware flag cleared the bar (article_fraction up AND no "
            f"median_citation/n_seminal regression) — keep flag {BASELINE_FLAG}; "
            f"broad-concept recall stays the vector lane's job in RRF."
        )
    return {
        "baseline_flag": BASELINE_FLAG,
        "deltas_vs_baseline": deltas,
        "recommended_flag": recommended,
        "rationale": rationale,
    }


def _delta(a: float | None, b: float | None) -> float | None:
    if a is None or b is None:
        return None
    return a - b


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _fmt(value: float | None, places: int = 3) -> str:
    return "   n/a" if value is None else f"{value:.{places}f}"


def render_report(payload: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("=" * 78)
    lines.append("ts_rank_cd normalization flag A/B — broad-query lexical quality (q9k5)")
    lines.append(f"gold set: {payload['queries_path']}  ({payload['n_queries']} queries)")
    lines.append(
        f"pool pinned at {payload['fixed_pool']}; "
        f"seminal = citation_count >= {SEMINAL_CITATION}"
    )
    lines.append("=" * 78)
    lines.append("")
    header = (
        f"{'flag':>5} | {'article_frac':>12} | {'short_frac':>10} | "
        f"{'seminal/q':>9} | {'med_cit':>8} | {'max ms':>8} | {'err':>3}"
    )
    lines.append(header)
    lines.append("-" * len(header))
    for flag in payload["flags_order"]:
        a = payload["by_flag"][str(flag)]
        tag = "  (base)" if flag == BASELINE_FLAG else ""
        lines.append(
            f"{flag:>5} | {_fmt(a['article_fraction']):>12} | "
            f"{_fmt(a['short_doc_fraction']):>10} | {_fmt(a['n_seminal_mean'], 2):>9} | "
            f"{_fmt(a['median_citation'], 1):>8} | {a['latency_ms_max']:>8.0f} | "
            f"{a['n_errored']:>3}{tag}"
        )
    lines.append("")
    dec = payload["decision"]
    lines.append(
        f"Deltas vs baseline flag {dec['baseline_flag']} "
        "(article_frac↑ good, short_frac↓ good, seminal↑ good, med_cit↑ good):"
    )
    for flag_str, d in sorted(dec["deltas_vs_baseline"].items(), key=lambda kv: int(kv[0])):
        lines.append(
            f"  flag {flag_str}: article {_fmt_pp(d['article_fraction_delta'])}  "
            f"short {_fmt_pp(d['short_doc_fraction_delta'])}  "
            f"seminal/q {_fmt_pp(d['n_seminal_delta'])}  "
            f"med_cit {_fmt_pp(d['median_citation_delta'], 1)}"
        )
    lines.append("")
    rec = dec["recommended_flag"]
    verdict = (
        f"ADOPT flag {rec}" if rec is not None else f"KEEP flag {BASELINE_FLAG} (negative result)"
    )
    lines.append(f"DECISION [{verdict}]: {dec['rationale']}")
    lines.append("=" * 78)
    return "\n".join(lines)


def _fmt_pp(value: float | None, places: int = 3) -> str:
    return "  n/a" if value is None else f"{value:+.{places}f}"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_flags(raw: str) -> list[int]:
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if not parts:
        raise argparse.ArgumentTypeError("--flags requires at least one value")
    seen: set[int] = set()
    out: list[int] = []
    for p in parts:
        try:
            value = int(p)
        except ValueError:
            raise argparse.ArgumentTypeError(f"flag {p!r} must be an integer") from None
        if value < 0 or value & ~63:
            raise argparse.ArgumentTypeError(
                f"flag {value} has bits outside the 0..63 ts_rank_cd mask"
            )
        if value not in seen:
            seen.add(value)
            out.append(value)
    return out


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="ts_rank_cd normalization flag A/B for broad-query lexical quality",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--queries", type=Path, default=Path(DEFAULT_QUERIES), help="Path to the JSONL query set"
    )
    p.add_argument(
        "--output", type=Path, default=Path(DEFAULT_OUTPUT), help="Path to write the JSON results"
    )
    p.add_argument(
        "--flags",
        type=_parse_flags,
        default=list(DEFAULT_FLAGS),
        help="Comma-separated ts_rank_cd flags to sweep (must include 32)",
    )
    p.add_argument(
        "--pool",
        type=str,
        default=DEFAULT_POOL,
        help="SCIX_LEXICAL_POOL held fixed across flags "
        "(INF = rank full match set, isolates ranking from the cap)",
    )
    p.add_argument(
        "--statement-timeout-ms",
        type=int,
        default=DEFAULT_STATEMENT_TIMEOUT_MS,
        help="statement_timeout for the eval connection",
    )
    p.add_argument(
        "--min-mem-gib",
        type=float,
        default=MIN_MEM_AVAILABLE_GIB,
        help="Preflight MemAvailable floor; eval refuses below this",
    )
    p.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress the human-readable report (still writes JSON)",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = _build_parser().parse_args(argv)

    if not args.queries.exists():
        logger.error("queries file %s not found", args.queries)
        return 2
    flags = list(args.flags)
    if BASELINE_FLAG not in flags:
        logger.error("--flags must include the baseline %d", BASELINE_FLAG)
        return 2

    reasons = preflight(args.min_mem_gib)
    if reasons:
        logger.error("PREFLIGHT REFUSED — prod is red:\n  - %s", "\n  - ".join(reasons))
        return EXIT_PREFLIGHT_REFUSED
    logger.info("preflight green; running flag A/B")

    queries = load_queries(args.queries)
    logger.info("loaded %d queries from %s", len(queries), args.queries)

    try:
        from scix.db import get_connection
    except ImportError:
        logger.exception("scix.db is unavailable; cannot run the eval")
        return 1

    conn = None
    by_flag_results: dict[int, list[FlagQueryResult]] = {}
    try:
        conn = get_connection()
        with conn.cursor() as cur:
            cur.execute(
                "SELECT set_config('statement_timeout', %s, false)",
                (str(int(args.statement_timeout_ms)),),
            )
            # Bound prod-side memory (pg_workmem_parallel_oom): scix-batch's
            # cgroup does NOT cap the postmaster.
            cur.execute("SET work_mem = '256MB'")
            cur.execute("SET max_parallel_workers_per_gather = 0")
        conn.commit()
        for flag in flags:
            logger.info(
                "running flag=%d over %d queries (pool=%s)",
                flag,
                len(queries),
                args.pool,
            )
            by_flag_results[flag] = run_flag(conn, flag, queries, args.pool)
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass

    by_flag_agg = {flag: aggregate(rs) for flag, rs in by_flag_results.items()}
    decision = decide(by_flag_agg)

    payload: dict[str, Any] = {
        "bead": "scix_experiments-q9k5",
        "queries_path": str(args.queries),
        "n_queries": len(queries),
        "retrieve_limit": RETRIEVE_LIMIT,
        "fixed_pool": args.pool,
        "seminal_citation_threshold": SEMINAL_CITATION,
        "short_doctypes": sorted(SHORT_DOCTYPES),
        "statement_timeout_ms": int(args.statement_timeout_ms),
        "flags_order": flags,
        "by_flag": {str(flag): agg for flag, agg in by_flag_agg.items()},
        "per_query": {
            str(flag): [
                {
                    "query": r.query,
                    "n_hits": r.n_hits,
                    "article_fraction": r.article_fraction,
                    "short_doc_fraction": r.short_doc_fraction,
                    "n_seminal": r.n_seminal,
                    "median_citation": r.median_citation,
                    "top1_citation": r.top1_citation,
                    "error": r.error,
                }
                for r in rs
            ]
            for flag, rs in by_flag_results.items()
        },
        "decision": decision,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    logger.info("results written to %s", args.output)

    if not args.quiet:
        print(render_report(payload))
    return 0


if __name__ == "__main__":
    sys.exit(main())
