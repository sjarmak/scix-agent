#!/usr/bin/env python3
"""M9 stage 2: apply calibrated tier weights and refresh the canonical MV.

Reads the empirical per-tier precision from ``entity_link_audits`` (filtered
to a single ``--annotator``), generates a new ``tier_weight()`` SQL function
body, writes a calibration row to ``tier_weight_calibration_log``, and
optionally refreshes ``document_entities_canonical``.

Defaults are conservative — the weight assigned to a tier is its
LOWER Wilson 95% CI bound on observed precision (a pessimistic estimate
that under-weights when the sample is thin). Tiers with too few decisive
labels fall back to the previous calibration row's weight rather than
a placeholder.

Usage::

    # Dry run — print SQL only, no DB writes
    python scripts/m9_apply_calibration.py \
        --db-url "dbname=scix" \
        --annotator claude_oauth_judge_v1 \
        --version m9_llm_judge_only_2026-04-29 \
        --dry-run

    # Apply (function + log row), but NOT refresh MV
    python scripts/m9_apply_calibration.py \
        --db-url "dbname=scix" \
        --annotator claude_oauth_judge_v1 \
        --version m9_llm_judge_only_2026-04-29 \
        --apply

    # Apply AND refresh MV (long-running, gated on explicit flag)
    python scripts/m9_apply_calibration.py \
        --db-url "dbname=scix" \
        --annotator claude_oauth_judge_v1 \
        --version m9_llm_judge_only_2026-04-29 \
        --apply --refresh-mv
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pathlib
import sys
from dataclasses import dataclass

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

import psycopg  # noqa: E402

from scix.eval.wilson import wilson_95_ci  # noqa: E402

logger = logging.getLogger("m9_apply_calibration")

CANONICAL_TIERS = (1, 2, 3, 4, 5)
DEFAULT_FALLBACK_WEIGHT = 0.50
MIN_DECISIVE_FOR_CALIBRATION = 10


@dataclass(frozen=True)
class TierEmpirical:
    tier: int
    n_total: int
    n_correct: int
    n_incorrect: int
    n_ambiguous: int

    @property
    def n_decisive(self) -> int:
        return self.n_correct + self.n_incorrect

    @property
    def point_precision(self) -> float:
        return (self.n_correct / self.n_decisive) if self.n_decisive else 0.0

    def wilson_lower(self) -> float:
        lo, _ = wilson_95_ci(self.n_correct, self.n_decisive)
        return lo


def _load_empirical(conn: psycopg.Connection, annotator: str) -> list[TierEmpirical]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT tier, label, count(*)
            FROM entity_link_audits
            WHERE annotator = %s
            GROUP BY tier, label
            ORDER BY tier, label
            """,
            (annotator,),
        )
        rows = cur.fetchall()
    by_tier: dict[int, dict[str, int]] = {}
    for tier, label, n in rows:
        d = by_tier.setdefault(int(tier), {"correct": 0, "incorrect": 0, "ambiguous": 0})
        d[str(label)] = int(n)
    out: list[TierEmpirical] = []
    for tier, counts in sorted(by_tier.items()):
        out.append(
            TierEmpirical(
                tier=int(tier),
                n_total=sum(counts.values()),
                n_correct=counts.get("correct", 0),
                n_incorrect=counts.get("incorrect", 0),
                n_ambiguous=counts.get("ambiguous", 0),
            )
        )
    return out


def _load_previous_weights(conn: psycopg.Connection) -> dict[int, float]:
    """Load the most recent calibration row's weights as fallback."""
    with conn.cursor() as cur:
        cur.execute("SELECT weights FROM tier_weight_calibration_log " "ORDER BY id DESC LIMIT 1")
        row = cur.fetchone()
    if row is None:
        return {}
    weights = row[0]
    if not isinstance(weights, dict):
        return {}
    out: dict[int, float] = {}
    for k, v in weights.items():
        try:
            out[int(k)] = float(v)
        except (TypeError, ValueError):
            continue
    return out


def _resolve_new_weights(
    empirical: list[TierEmpirical], fallback: dict[int, float]
) -> tuple[dict[int, float], dict[int, str]]:
    """Return (weights, sources) where sources annotates how each weight was set."""
    by_tier = {e.tier: e for e in empirical}
    weights: dict[int, float] = {}
    sources: dict[int, str] = {}
    for tier in CANONICAL_TIERS:
        e = by_tier.get(tier)
        if e is not None and e.n_decisive >= MIN_DECISIVE_FOR_CALIBRATION:
            w = max(0.0, min(0.9999, e.wilson_lower()))
            weights[tier] = round(w, 4)
            sources[tier] = f"wilson_lower(correct={e.n_correct}, decisive={e.n_decisive})"
        elif tier in fallback:
            weights[tier] = round(float(fallback[tier]), 4)
            sources[tier] = (
                f"fallback (n_decisive={e.n_decisive if e else 0} < {MIN_DECISIVE_FOR_CALIBRATION})"
            )
        else:
            weights[tier] = DEFAULT_FALLBACK_WEIGHT
            sources[tier] = "default 0.50 (no fallback)"
    return weights, sources


def _format_function_sql(weights: dict[int, float]) -> str:
    """Render CREATE OR REPLACE FUNCTION tier_weight(...) SQL."""
    cases = "\n".join(
        f"        WHEN {t}::SMALLINT THEN {weights[t]:.4f}::float8" for t in CANONICAL_TIERS
    )
    return f"""CREATE OR REPLACE FUNCTION tier_weight(tier SMALLINT)
RETURNS DOUBLE PRECISION
LANGUAGE sql
IMMUTABLE
LEAKPROOF
PARALLEL SAFE
AS $$
    SELECT CASE tier
{cases}
        ELSE 0.50::float8
    END
$$;
"""


def _insert_calibration_row(
    conn: psycopg.Connection,
    *,
    version: str,
    weights: dict[int, float],
    notes: str,
) -> int:
    payload = {str(t): weights[t] for t in sorted(weights)}
    payload["default"] = 0.50
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO tier_weight_calibration_log (version, weights, notes)
            VALUES (%s, %s::jsonb, %s)
            ON CONFLICT (version) DO UPDATE
                SET weights = EXCLUDED.weights,
                    notes   = EXCLUDED.notes
            RETURNING id
            """,
            (version, json.dumps(payload), notes),
        )
        new_id = cur.fetchone()[0]
    conn.commit()
    return int(new_id)


def _refresh_mv(conn: psycopg.Connection) -> None:
    """REFRESH MATERIALIZED VIEW CONCURRENTLY document_entities_canonical."""
    logger.info("REFRESH MATERIALIZED VIEW CONCURRENTLY (this may take a long time)...")
    # CONCURRENTLY cannot be inside a transaction block. We commit/abort first.
    conn.commit()
    conn.autocommit = True
    try:
        with conn.cursor() as cur:
            cur.execute("REFRESH MATERIALIZED VIEW CONCURRENTLY document_entities_canonical")
            cur.execute(
                "UPDATE fusion_mv_state SET dirty = false, last_refresh_at = now() " "WHERE id = 1"
            )
    finally:
        conn.autocommit = False
    logger.info("MV refresh complete")


def _resolve_dsn(cli_dsn: str | None) -> str:
    if cli_dsn:
        return cli_dsn
    dsn = os.environ.get("SCIX_DSN") or os.environ.get("SCIX_TEST_DSN")
    if dsn:
        return dsn
    return "dbname=scix"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db-url", default=None)
    parser.add_argument("--annotator", default="claude_oauth_judge_v1")
    parser.add_argument(
        "--version",
        required=True,
        help="Unique calibration row version, e.g. m9_llm_judge_only_2026-04-29",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply: CREATE OR REPLACE tier_weight() and INSERT calibration row.",
    )
    parser.add_argument(
        "--refresh-mv",
        action="store_true",
        help="REFRESH MATERIALIZED VIEW CONCURRENTLY (long-running). " "Implies --apply.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print SQL only; no DB writes.")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.refresh_mv and not args.apply:
        logger.info("--refresh-mv implies --apply; enabling --apply")
        args.apply = True

    dsn = _resolve_dsn(args.db_url)
    logger.info("connecting to %s", dsn)
    conn = psycopg.connect(dsn)
    try:
        empirical = _load_empirical(conn, args.annotator)
        if not empirical:
            logger.error(
                "no rows in entity_link_audits with annotator=%s — run "
                "scripts/m9_audit_judge.py first",
                args.annotator,
            )
            return 2

        logger.info("empirical per-tier counts:")
        for e in empirical:
            logger.info(
                "  tier %d: total=%d correct=%d incorrect=%d ambiguous=%d "
                "decisive=%d point_p=%.3f wilson_lo=%.3f",
                e.tier,
                e.n_total,
                e.n_correct,
                e.n_incorrect,
                e.n_ambiguous,
                e.n_decisive,
                e.point_precision,
                e.wilson_lower(),
            )

        prev = _load_previous_weights(conn)
        new_weights, sources = _resolve_new_weights(empirical, prev)

        logger.info("resolved new weights:")
        for t in sorted(new_weights):
            logger.info(
                "  tier %d -> w=%.4f  (%s; previous=%s)",
                t,
                new_weights[t],
                sources[t],
                f"{prev.get(t):.4f}" if t in prev else "<none>",
            )

        sql = _format_function_sql(new_weights)
        print("\n----- proposed tier_weight() function -----")
        print(sql)
        print("----- end -----\n")

        notes_lines = [
            f"M9 calibration from annotator={args.annotator}.",
            "Source: OAuth Claude subagent only (no human ground truth in this pass).",
            "kappa gate (>=0.6 vs human) NOT satisfied — see eval_report.md.",
            "Per-tier weight = Wilson 95% lower bound on (correct / decisive); "
            "ambiguous excluded from denominator.",
        ]
        for t in sorted(new_weights):
            notes_lines.append(f"  tier {t}: w={new_weights[t]:.4f}  ({sources[t]})")
        notes = "\n".join(notes_lines)

        if args.dry_run or not args.apply:
            logger.info("--dry-run / no --apply: skipping DB writes")
            return 0

        with conn.cursor() as cur:
            cur.execute(sql)
        conn.commit()
        logger.info("applied tier_weight() function")

        new_id = _insert_calibration_row(
            conn, version=args.version, weights=new_weights, notes=notes
        )
        logger.info("wrote tier_weight_calibration_log id=%d version=%s", new_id, args.version)

        # Mark MV dirty regardless of refresh flag.
        with conn.cursor() as cur:
            cur.execute("UPDATE fusion_mv_state SET dirty = true WHERE id = 1")
        conn.commit()

        if args.refresh_mv:
            _refresh_mv(conn)
        else:
            logger.info(
                "skipping MV refresh; set fusion_mv_state.dirty=true so the "
                "background refresher (or a later --refresh-mv pass) picks it up"
            )

        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(main())
