#!/usr/bin/env python3
"""Pilot embedding comparison: evaluate 3 models on a 10K paper sample.

Models compared:
  1. nasa-impact/nasa-smd-ibm-st-v2  (INDUS — trained on ADS papers)
  2. allenai/specter2_base            (SPECTER2 — scientific domain)
  3. nomic-ai/nomic-embed-text-v1.5   (Nomic — long context, Matryoshka)

Evaluation:
  - Embeds a stratified 10K sample (by year decade + citation count)
  - Stores all embeddings in paper_embeddings (model_name differentiates)
  - Runs retrieval quality tests: given a seed paper, find neighbors,
    measure overlap with known citation/reference network

Usage:
    .venv/bin/python scripts/pilot_embed_compare.py [--sample-size 10000] [--seed-papers 50]
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import psycopg
from psycopg.rows import dict_row

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from scix.db import get_connection

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("pilot_embed")

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

MODELS = {
    "indus": {
        "hf_name": "nasa-impact/nasa-smd-ibm-st-v2",
        "type": "sentence-transformers",
        "dims": 768,
        "max_tokens": 512,
        "prefix": "",  # no prefix needed
    },
    "specter2": {
        "hf_name": "allenai/specter2_base",
        "type": "transformers-cls",
        "dims": 768,
        "max_tokens": 512,
        "prefix": "",
    },
    "nomic": {
        "hf_name": "nomic-ai/nomic-embed-text-v1.5",
        "type": "sentence-transformers",
        "dims": 768,
        "max_tokens": 8192,
        "prefix": "search_document: ",  # nomic requires task prefix
    },
}


@dataclass(frozen=True)
class EmbedResult:
    model_name: str
    papers_embedded: int
    elapsed_s: float
    rate_per_s: float
    dims: int


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


def create_pilot_sample(conn: psycopg.Connection, sample_size: int) -> int:
    """Create a stratified sample of papers for the pilot.

    Stratifies by decade and citation tier to get a representative mix.
    Returns actual number of papers sampled.
    """
    logger.info("Creating stratified pilot sample of %d papers...", sample_size)

    with conn.cursor() as cur:
        # Drop old sample if exists
        cur.execute("DROP TABLE IF EXISTS _pilot_sample")

        # Stratified sample: mix of decades and citation tiers
        # Use TABLESAMPLE for speed, then enrich with targeted picks
        cur.execute(
            """
            CREATE TABLE _pilot_sample AS
            WITH decade_strata AS (
                -- Sample across decades, weighted toward recent (more papers)
                SELECT bibcode, title, abstract, year, citation_count,
                       ROW_NUMBER() OVER (
                           PARTITION BY (year / 10) * 10
                           ORDER BY random()
                       ) as rn,
                       COUNT(*) OVER (PARTITION BY (year / 10) * 10) as decade_total
                FROM papers
                WHERE title IS NOT NULL
                  AND year >= 1950
            ),
            sampled AS (
                SELECT bibcode, title, abstract, year, citation_count
                FROM decade_strata
                WHERE rn <= GREATEST(
                    -- At least 100 per decade, proportional to decade size
                    100,
                    (%s::float / (SELECT count(DISTINCT (year/10)*10) FROM papers WHERE year >= 1950))
                )
                LIMIT %s
            )
            SELECT * FROM sampled
        """,
            [sample_size, sample_size],
        )

        cur.execute("SELECT count(*) FROM _pilot_sample")
        actual = cur.fetchone()[0]

        # Add indexes for joins
        cur.execute("CREATE INDEX ON _pilot_sample (bibcode)")

        conn.commit()

    logger.info("Pilot sample: %d papers", actual)

    # Log distribution
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute("""
            SELECT (year / 10) * 10 as decade, count(*) as n,
                   round(avg(citation_count)) as avg_cites
            FROM _pilot_sample
            GROUP BY (year / 10) * 10
            ORDER BY decade
        """)
        for row in cur:
            logger.info(
                "  %ds: %d papers, avg %s citations",
                row["decade"],
                row["n"],
                row["avg_cites"],
            )

    return actual


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------


def _prepare_text(title: str, abstract: str | None, prefix: str) -> str:
    """Format paper text for embedding."""
    if abstract:
        text = f"{title} [SEP] {abstract}"
    else:
        text = title
    return f"{prefix}{text}" if prefix else text


def _source_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _vec_to_pgvector(vec: list[float]) -> str:
    return "[" + ",".join(str(v) for v in vec) + "]"


def embed_with_specter2(papers: list[dict], batch_size: int = 64) -> list[list[float]]:
    """Embed using SPECTER2 (transformers CLS pooling)."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    cfg = MODELS["specter2"]
    tokenizer = AutoTokenizer.from_pretrained(cfg["hf_name"])
    model = AutoModel.from_pretrained(cfg["hf_name"]).cuda().eval()

    all_vecs: list[list[float]] = []
    texts = [_prepare_text(p["title"], p["abstract"], cfg["prefix"]) for p in papers]

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=cfg["max_tokens"],
            return_tensors="pt",
        ).to("cuda")
        with torch.no_grad():
            outputs = model(**inputs)
        vecs = outputs.last_hidden_state[:, 0, :].cpu().tolist()
        all_vecs.extend(vecs)

    return all_vecs


def embed_with_sentence_transformer(
    model_key: str,
    papers: list[dict],
    batch_size: int = 64,
) -> list[list[float]]:
    """Embed using a sentence-transformers model."""
    from sentence_transformers import SentenceTransformer

    cfg = MODELS[model_key]
    model = SentenceTransformer(cfg["hf_name"], trust_remote_code=True)
    model = model.to("cuda")

    texts = [_prepare_text(p["title"], p["abstract"], cfg["prefix"]) for p in papers]

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    return embeddings.tolist()


def embed_model(model_key: str, papers: list[dict], batch_size: int = 64) -> list[list[float]]:
    """Route to the right embedding function based on model type."""
    cfg = MODELS[model_key]
    if cfg["type"] == "transformers-cls":
        return embed_with_specter2(papers, batch_size)
    else:
        return embed_with_sentence_transformer(model_key, papers, batch_size)


def store_pilot_embeddings(
    conn: psycopg.Connection,
    papers: list[dict],
    vectors: list[list[float]],
    model_key: str,
) -> int:
    """Store embeddings in paper_embeddings table."""
    cfg = MODELS[model_key]
    model_name = model_key  # use short key as model_name

    with conn.cursor() as cur:
        # Batch insert with ON CONFLICT
        for paper, vec in zip(papers, vectors):
            text = _prepare_text(paper["title"], paper["abstract"], cfg["prefix"])
            sh = _source_hash(text)
            input_type = "title_abstract" if paper["abstract"] else "title_only"

            cur.execute(
                """
                INSERT INTO paper_embeddings (bibcode, model_name, embedding, input_type, source_hash)
                VALUES (%s, %s, %s::vector, %s, %s)
                ON CONFLICT (bibcode, model_name) DO UPDATE SET
                    embedding = EXCLUDED.embedding,
                    input_type = EXCLUDED.input_type,
                    source_hash = EXCLUDED.source_hash
            """,
                (paper["bibcode"], model_name, _vec_to_pgvector(vec), input_type, sh),
            )

    conn.commit()
    return len(vectors)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate_retrieval(
    conn: psycopg.Connection,
    model_key: str,
    seed_bibcodes: list[str],
    k: int = 20,
) -> dict:
    """Evaluate retrieval quality for a model.

    For each seed paper, find k nearest neighbors via cosine similarity,
    then measure overlap with the paper's actual citation network
    (references + cited-by).

    Returns metrics dict with recall@k against citation network.
    """
    results: list[dict] = []

    with conn.cursor(row_factory=dict_row) as cur:
        for bibcode in seed_bibcodes:
            # Get seed embedding
            cur.execute(
                "SELECT embedding FROM paper_embeddings WHERE bibcode = %s AND model_name = %s",
                [bibcode, model_key],
            )
            row = cur.fetchone()
            if row is None or row["embedding"] is None:
                continue

            # Find k nearest neighbors (sequential scan — fine for pilot)
            cur.execute(
                """
                SELECT pe.bibcode,
                       1 - (pe.embedding <=> %s::vector) as similarity
                FROM paper_embeddings pe
                WHERE pe.model_name = %s
                  AND pe.bibcode != %s
                ORDER BY pe.embedding <=> %s::vector
                LIMIT %s
            """,
                [row["embedding"], model_key, bibcode, row["embedding"], k],
            )
            neighbors = {r["bibcode"] for r in cur.fetchall()}

            # Get citation network (references + cited-by) within pilot sample
            cur.execute(
                """
                SELECT target_bibcode as bibcode FROM citation_edges
                WHERE source_bibcode = %s
                  AND target_bibcode IN (SELECT bibcode FROM _pilot_sample)
                UNION
                SELECT source_bibcode as bibcode FROM citation_edges
                WHERE target_bibcode = %s
                  AND source_bibcode IN (SELECT bibcode FROM _pilot_sample)
            """,
                [bibcode, bibcode],
            )
            citation_net = {r["bibcode"] for r in cur.fetchall()}

            if not citation_net:
                continue

            overlap = neighbors & citation_net
            recall = len(overlap) / len(citation_net) if citation_net else 0

            results.append(
                {
                    "bibcode": bibcode,
                    "neighbors": len(neighbors),
                    "citation_net": len(citation_net),
                    "overlap": len(overlap),
                    "recall": recall,
                }
            )

    if not results:
        return {"model": model_key, "seed_papers": 0, "mean_recall": 0}

    mean_recall = sum(r["recall"] for r in results) / len(results)
    mean_overlap = sum(r["overlap"] for r in results) / len(results)
    mean_citation_net = sum(r["citation_net"] for r in results) / len(results)

    return {
        "model": model_key,
        "seed_papers": len(results),
        "mean_recall_at_k": round(mean_recall, 4),
        "mean_overlap": round(mean_overlap, 1),
        "mean_citation_net_size": round(mean_citation_net, 1),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Pilot embedding comparison")
    parser.add_argument("--sample-size", type=int, default=10_000)
    parser.add_argument(
        "--seed-papers", type=int, default=50, help="Number of seed papers for retrieval evaluation"
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--k", type=int, default=20, help="Top-k for retrieval eval")
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(MODELS.keys()),
        choices=list(MODELS.keys()),
        help="Models to evaluate (default: all)",
    )
    parser.add_argument(
        "--reuse-sample",
        action="store_true",
        help="Reuse existing _pilot_sample table instead of creating a new one",
    )
    args = parser.parse_args()

    conn = get_connection()

    # Step 1: Create or reuse sample
    if args.reuse_sample:
        with conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM _pilot_sample")
            actual_size = cur.fetchone()[0]
        logger.info("Reusing existing pilot sample: %d papers", actual_size)
    else:
        actual_size = create_pilot_sample(conn, args.sample_size)

    # Load sample papers
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute("SELECT bibcode, title, abstract FROM _pilot_sample")
        papers = cur.fetchall()

    logger.info("Loaded %d papers for embedding", len(papers))

    # Step 2: Embed with each model
    embed_results: list[EmbedResult] = []

    for model_key in args.models:
        cfg = MODELS[model_key]
        logger.info("=" * 60)
        logger.info("Embedding with %s (%s, %d dims)", model_key, cfg["hf_name"], cfg["dims"])

        t0 = time.monotonic()
        vectors = embed_model(model_key, papers, batch_size=args.batch_size)
        elapsed = time.monotonic() - t0
        rate = len(vectors) / elapsed if elapsed > 0 else 0

        logger.info("Embedded %d papers in %.1fs (%.0f/s)", len(vectors), elapsed, rate)

        # Store
        stored = store_pilot_embeddings(conn, papers, vectors, model_key)
        logger.info("Stored %d embeddings for %s", stored, model_key)

        embed_results.append(
            EmbedResult(
                model_name=model_key,
                papers_embedded=stored,
                elapsed_s=round(elapsed, 1),
                rate_per_s=round(rate, 1),
                dims=cfg["dims"],
            )
        )

        # Free GPU memory between models
        import torch

        torch.cuda.empty_cache()
        import gc

        gc.collect()

    # Step 3: Select seed papers for evaluation
    # Pick papers with decent citation networks (at least 5 citations in sample)
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT ps.bibcode, ps.citation_count
            FROM _pilot_sample ps
            WHERE ps.citation_count >= 10
              AND ps.abstract IS NOT NULL
            ORDER BY random()
            LIMIT %s
        """,
            [args.seed_papers],
        )
        seed_bibcodes = [r[0] for r in cur.fetchall()]

    logger.info("Selected %d seed papers for retrieval evaluation", len(seed_bibcodes))

    # Step 4: Evaluate retrieval quality
    eval_results = []
    for model_key in args.models:
        logger.info("Evaluating retrieval for %s...", model_key)
        result = evaluate_retrieval(conn, model_key, seed_bibcodes, k=args.k)
        eval_results.append(result)
        logger.info("  %s", result)

    # Step 5: Print summary
    print("\n" + "=" * 70)
    print("EMBEDDING PILOT RESULTS")
    print("=" * 70)

    print("\n## Throughput")
    print(f"{'Model':<12} {'Papers':>8} {'Time':>8} {'Rate':>10} {'Dims':>6}")
    print("-" * 50)
    for r in embed_results:
        print(
            f"{r.model_name:<12} {r.papers_embedded:>8} {r.elapsed_s:>7.1f}s {r.rate_per_s:>9.1f}/s {r.dims:>6}"
        )

    print("\n## Retrieval Quality (citation network recall)")
    print(f"{'Model':<12} {'Seeds':>7} {'Recall@k':>10} {'Overlap':>9} {'Net Size':>10}")
    print("-" * 52)
    for r in eval_results:
        print(
            f"{r['model']:<12} {r['seed_papers']:>7} "
            f"{r['mean_recall_at_k']:>10.4f} {r['mean_overlap']:>9.1f} "
            f"{r['mean_citation_net_size']:>10.1f}"
        )

    print("\n" + "=" * 70)

    conn.close()


if __name__ == "__main__":
    main()
