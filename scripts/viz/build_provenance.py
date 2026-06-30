#!/usr/bin/env python3
"""Build data for the V12 Provenance viz — entity source + precision profile.

The complement to the V8 entity-coverage dashboard: V8 says *what we have*, V12
says *how trustworthy each piece is*. It surfaces where SciX entities come from
(GLiNER open-weight NER vs curated authorities — SSODNet, VizieR, ASCL, …) and how
precision varies across sources, reusing the dbl.3 quality profile
(:mod:`scix.extract.ner_quality_profile`) so the page mirrors the
``precision_band`` an agent sees on the MCP entity tool.

Three aggregates:

1. **entities by (source, entity_type)** — exact ``GROUP BY`` over ``entities``.
2. **ambiguity_class distribution** — exact ``GROUP BY`` over ``entities``.
3. **per-source precision-band profile** — the band histogram each source's links
   fall into. Non-GLiNER (lexical/curated) sources have a *constant* precision
   (``precision_estimate`` returns ``LEXICAL_PRECISION_DEFAULT`` for them), so
   their profile is assigned analytically — no sampling needed. GLiNER precision
   depends on (entity_type, era(year), classifier agreement), so its profile is
   estimated from a representative sample of ``document_entities`` links, with the
   agreement aggregated per (bibcode, entity_id) exactly as
   ``_attach_precision_to_linked_entities`` does. A handful of sampled links are
   emitted as ``spot_checks`` so the acceptance criterion (band mirrors the MCP
   output for spot-checked entity_ids) can be verified directly. NOTE: the
   ``precision_band`` computation mirrors the MCP attach exactly, but get_paper
   additionally drops denylisted generic-word (name, type) pairs before display
   (``scix.extract.ner_denylist``); this provenance view profiles the *full*
   entity graph, so a denylisted entity carries a band here but is hidden on the
   tool.

EVAL/VIZ ONLY: read-only aggregates, no writes. The GROUP BY over ~19M entities
and the link sample are heavy — run under scix-batch::

    scix-batch .venv/bin/python scripts/viz/build_provenance.py \
        --output data/viz/provenance.json

Offline smoke-test (no DB), for render/probe checks::

    .venv/bin/python scripts/viz/build_provenance.py --synthetic \
        --output data/viz/provenance.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from scix.db import DEFAULT_DSN, redact_dsn  # noqa: E402
from scix.extract.ner_quality_profile import (  # noqa: E402
    LEXICAL_PRECISION_DEFAULT,
    precision_band,
    precision_estimate,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("build_provenance")

DEFAULT_SAMPLE_SIZE = 200_000
# Page-sample fraction of document_entities handed to TABLESAMPLE before the
# LIMIT. Small — the table is ~95M links; even 0.5% is plenty for a band histogram.
DEFAULT_SAMPLE_PCT = 0.5
BANDS = ("high", "medium", "low", "noisy")
GLINER_SOURCE = "gliner"


@dataclass(frozen=True)
class Config:
    dsn: str
    output: Path
    sample_size: int
    sample_pct: float
    synthetic: bool


def _parse_agreement(raw: str | None) -> bool | None:
    """Map an ``evidence->>'agreement'`` text value to a tristate bool."""
    if raw == "true":
        return True
    if raw == "false":
        return False
    return None


def _union_positive(prev: bool | None, cur: bool | None) -> bool | None:
    """Aggregate two agreement verdicts the way the MCP precision attach does:
    True wins over False wins over None."""
    if prev is True or cur is True:
        return True
    if prev is False or cur is False:
        return False
    return None


# --- DB loaders ---------------------------------------------------------------


def load_entity_counts(dsn: str) -> tuple[list[dict], list[dict]]:
    """Exact (source, entity_type) and ambiguity_class counts over ``entities``."""
    import psycopg

    by_source: dict[str, dict] = {}
    ambiguity: dict[str | None, int] = {}
    with psycopg.connect(dsn) as conn, conn.cursor() as cur:
        cur.execute("SET statement_timeout = '20min'")
        cur.execute(
            "SELECT source, entity_type, COUNT(*) " "FROM entities GROUP BY source, entity_type"
        )
        for source, etype, n in cur.fetchall():
            rec = by_source.setdefault(
                source, {"source": source, "n_entities": 0, "entity_types": []}
            )
            rec["n_entities"] += n
            rec["entity_types"].append({"entity_type": etype, "n": n})
        cur.execute("SELECT ambiguity_class::text, COUNT(*) FROM entities GROUP BY ambiguity_class")
        for cls, n in cur.fetchall():
            ambiguity[cls] = n

    sources = sorted(by_source.values(), key=lambda r: r["n_entities"], reverse=True)
    for rec in sources:
        rec["entity_types"].sort(key=lambda t: t["n"], reverse=True)
    amb = sorted(
        ({"ambiguity_class": k, "n": v} for k, v in ambiguity.items()),
        key=lambda r: r["n"],
        reverse=True,
    )
    return sources, amb


def load_link_estimate(dsn: str) -> int:
    """Fast row estimate for document_entities (planner stats, not a full count)."""
    import psycopg

    with psycopg.connect(dsn) as conn, conn.cursor() as cur:
        cur.execute("SELECT reltuples::bigint FROM pg_class WHERE relname = 'document_entities'")
        row = cur.fetchone()
    return int(row[0]) if row and row[0] is not None else 0


def load_gliner_precision_sample(dsn: str, sample_pct: float, sample_size: int) -> list[dict]:
    """Sample GLiNER links with their agreement + paper year for band estimation.

    One row per sampled (bibcode, entity_id, link) — aggregation to the entity
    level happens in Python so it matches the MCP attach exactly.
    """
    import psycopg

    sql = (
        "SELECT de.bibcode, de.entity_id, e.entity_type, "
        "       de.evidence->>'agreement' AS agreement, p.year "
        "FROM document_entities de TABLESAMPLE SYSTEM (%s) "
        "JOIN entities e ON e.id = de.entity_id AND e.source = %s "
        "LEFT JOIN papers p ON p.bibcode = de.bibcode "
        "LIMIT %s"
    )
    rows: list[dict] = []
    with psycopg.connect(dsn) as conn, conn.cursor() as cur:
        cur.execute("SET statement_timeout = '20min'")
        cur.execute(sql, (sample_pct, GLINER_SOURCE, sample_size))
        for bibcode, entity_id, etype, agreement, year in cur:
            rows.append(
                {
                    "bibcode": bibcode,
                    "entity_id": entity_id,
                    "entity_type": etype,
                    "agreement": _parse_agreement(agreement),
                    "year": int(year) if year is not None else None,
                }
            )
    return rows


# --- precision aggregation (shared by DB + synthetic paths) -------------------


def gliner_profile_from_sample(rows: list[dict]) -> tuple[dict[str, float], list[dict], int]:
    """Aggregate sampled GLiNER links into a band-fraction profile + spot checks.

    Returns (band_fractions, spot_checks, n_entity_links). Agreement is reduced
    per (bibcode, entity_id) union-positive, then precision_estimate/precision_band
    are applied — identical to ``_attach_precision_to_linked_entities``.
    """
    agg: dict[tuple[str, int], dict] = {}
    for r in rows:
        key = (r["bibcode"], r["entity_id"])
        cur = agg.get(key)
        if cur is None:
            agg[key] = {
                "entity_id": r["entity_id"],
                "bibcode": r["bibcode"],
                "entity_type": r["entity_type"],
                "year": r["year"],
                "agreement": r["agreement"],
            }
        else:
            cur["agreement"] = _union_positive(cur["agreement"], r["agreement"])

    counts = {b: 0 for b in BANDS}
    # Collect spot checks spread across distinct (entity_type, agreement, era)
    # combos — TABLESAMPLE returns contiguous pages, so first-N would cluster on
    # one journal/year and verify little. Diverse combos exercise more of the
    # precision table.
    spot_by_combo: dict[tuple, dict] = {}
    spot_extra: list[dict] = []
    for rec in agg.values():
        p = precision_estimate(
            entity_type=rec["entity_type"],
            source=GLINER_SOURCE,
            agreement=rec["agreement"],
            year=rec["year"],
        )
        band = precision_band(p)
        counts[band] = counts.get(band, 0) + 1
        era = "modern" if (rec["year"] is not None and rec["year"] >= 2010) else "pre_1990"
        check = {
            "entity_id": rec["entity_id"],
            "bibcode": rec["bibcode"],
            "source": GLINER_SOURCE,
            "entity_type": rec["entity_type"],
            "year": rec["year"],
            "agreement": rec["agreement"],
            "precision_estimate": round(p, 2),
            "precision_band": band,
        }
        combo = (rec["entity_type"], rec["agreement"], era)
        if combo not in spot_by_combo:
            spot_by_combo[combo] = check
        elif len(spot_extra) < 12:
            spot_extra.append(check)
    spot_checks = (list(spot_by_combo.values()) + spot_extra)[:12]
    if len(spot_by_combo) > 12:
        logger.debug("found %d distinct spot-check combos; emitting first 12", len(spot_by_combo))

    total = sum(counts.values())
    fractions = {b: (counts[b] / total if total else 0.0) for b in BANDS}
    return fractions, spot_checks, total


def lexical_profile() -> dict[str, float]:
    """Band profile for non-GLiNER sources: a single constant precision.

    ``precision_estimate`` returns ``LEXICAL_PRECISION_DEFAULT`` for every
    non-GLiNER source regardless of type/agreement/year, so the whole source
    lands in one band — assigned here without sampling.
    """
    band = precision_band(LEXICAL_PRECISION_DEFAULT)
    return {b: (1.0 if b == band else 0.0) for b in BANDS}


def attach_precision_profiles(
    sources: list[dict], gliner_fractions: dict[str, float], gliner_n: int
) -> None:
    """Attach a ``precision_profile`` to each source record in place."""
    lex = lexical_profile()
    for rec in sources:
        if rec["source"] == GLINER_SOURCE:
            rec["precision_profile"] = gliner_fractions
            rec["precision_basis"] = "sampled"
            rec["n_precision_sample"] = gliner_n
        else:
            rec["precision_profile"] = lex
            rec["precision_basis"] = "lexical_constant"
            rec["n_precision_sample"] = 0


# --- synthetic mode -----------------------------------------------------------


def synthesize() -> tuple[list[dict], list[dict], int, list[dict]]:
    """Deterministic offline data shaped like the real aggregates (no DB)."""
    import random

    rng = random.Random(42)
    type_pool = {
        "gliner": [
            "method",
            "chemical",
            "instrument",
            "dataset",
            "software",
            "organism",
            "location",
            "gene",
            "mission",
        ],
        "ssodnet": ["target", "taxon"],
        "vizier": ["dataset"],
        "ascl": ["software"],
        "bio.tools": ["software"],
        "gcmd": ["instrument"],
    }
    base_n = {
        "gliner": 17_795_149,
        "ssodnet": 1_487_350,
        "vizier": 62_388,
        "ascl": 3_958,
        "bio.tools": 1_204,
        "gcmd": 980,
    }
    sources: list[dict] = []
    for src, types in type_pool.items():
        weights = [rng.random() + 0.2 for _ in types]
        wsum = sum(weights)
        ets = [
            {"entity_type": t, "n": max(1, int(base_n[src] * w / wsum))}
            for t, w in zip(types, weights)
        ]
        ets.sort(key=lambda x: x["n"], reverse=True)
        sources.append({"source": src, "n_entities": sum(e["n"] for e in ets), "entity_types": ets})
    sources.sort(key=lambda r: r["n_entities"], reverse=True)

    # Synthetic GLiNER sample: vary entity_type / year / agreement.
    rows: list[dict] = []
    etypes = type_pool["gliner"]
    for i in range(4000):
        rows.append(
            {
                "bibcode": f"synth{i:06d}",
                "entity_id": i,
                "entity_type": rng.choice(etypes),
                "agreement": rng.choice([True, False, None, None]),
                "year": rng.choice([1985, 2005, 2015, 2021, None]),
            }
        )
    fractions, spot_checks, n = gliner_profile_from_sample(rows)
    attach_precision_profiles(sources, fractions, n)

    ambiguity = [
        {"ambiguity_class": None, "n": 19_315_055},
        {"ambiguity_class": "domain_safe", "n": 78_336},
        {"ambiguity_class": "banned", "n": 9_066},
        {"ambiguity_class": "unique", "n": 1_487},
        {"ambiguity_class": "homograph", "n": 1_425},
    ]
    link_estimate = 94_636_846
    return sources, ambiguity, link_estimate, spot_checks


# --- payload + io -------------------------------------------------------------


def build_payload(
    sources: list[dict], ambiguity: list[dict], link_estimate: int, spot_checks: list[dict]
) -> dict:
    n_entities = sum(r["n_entities"] for r in sources)
    return {
        "totals": {
            "n_entities": n_entities,
            "n_links_estimate": link_estimate,
            "n_sources": len(sources),
            "n_entity_types": len({t["entity_type"] for r in sources for t in r["entity_types"]}),
        },
        "by_source": sources,
        "ambiguity": ambiguity,
        "spot_checks": spot_checks,
        "precision_bands": list(BANDS),
    }


def serialize(payload: dict, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload), encoding="utf-8")


def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dsn", default=DEFAULT_DSN)
    p.add_argument("--output", default="data/viz/provenance.json")
    p.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE)
    p.add_argument("--sample-pct", type=float, default=DEFAULT_SAMPLE_PCT)
    p.add_argument("--synthetic", action="store_true")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    config = Config(
        dsn=args.dsn,
        output=Path(args.output),
        sample_size=args.sample_size,
        sample_pct=args.sample_pct,
        synthetic=args.synthetic,
    )

    if config.synthetic:
        logger.info("synthetic mode (no DB)")
        sources, ambiguity, link_estimate, spot_checks = synthesize()
    else:
        logger.info("loading from DB dsn=%s", redact_dsn(config.dsn))
        sources, ambiguity = load_entity_counts(config.dsn)
        link_estimate = load_link_estimate(config.dsn)
        sample = load_gliner_precision_sample(config.dsn, config.sample_pct, config.sample_size)
        logger.info("gliner precision sample: %d links", len(sample))
        fractions, spot_checks, n = gliner_profile_from_sample(sample)
        attach_precision_profiles(sources, fractions, n)

    payload = build_payload(sources, ambiguity, link_estimate, spot_checks)
    serialize(payload, config.output)
    logger.info(
        "wrote %s (%d sources, %d entities, %d ambiguity classes)",
        config.output,
        len(sources),
        payload["totals"]["n_entities"],
        len(ambiguity),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
