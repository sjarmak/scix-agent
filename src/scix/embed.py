"""Embedding pipeline: generate SPECTER2 embeddings and store in paper_embeddings.

Uses HuggingFace transformers + torch directly (no sentence-transformers wrapper).
Writes embeddings via psycopg COPY for bulk throughput.
"""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass
from typing import Any

import psycopg

from scix.db import get_connection

logger = logging.getLogger(__name__)

# Stub columns for the SELECT when fetching papers to embed
_PAPER_COLS = "bibcode, title, abstract"


@dataclass(frozen=True)
class EmbeddingInput:
    """Prepared input for the embedding model."""

    bibcode: str
    text: str
    input_type: str  # "title_abstract" or "title_only"
    source_hash: str  # SHA-256 of the input text


def prepare_input(bibcode: str, title: str | None, abstract: str | None) -> EmbeddingInput | None:
    """Prepare embedding input text from paper metadata.

    Returns None if the paper has no title (cannot embed without at least a title).
    Uses "title [SEP] abstract" format per SPECTER2 convention.
    """
    if not title:
        return None

    if abstract:
        text = f"{title} [SEP] {abstract}"
        input_type = "title_abstract"
    else:
        text = title
        input_type = "title_only"

    source_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return EmbeddingInput(
        bibcode=bibcode,
        text=text,
        input_type=input_type,
        source_hash=source_hash,
    )


def load_model(
    model_name: str = "specter2", device: str = "cpu"
) -> tuple[Any, Any]:
    """Load SPECTER2 model and tokenizer via transformers.

    Returns (model, tokenizer) tuple. The model is moved to the specified device.
    """
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except ImportError:
        raise ImportError(
            "transformers and torch are required for embedding. "
            "Install with: pip install transformers torch"
        )

    if model_name == "specter2":
        hf_name = "allenai/specter2_base"
    else:
        raise ValueError(f"Unknown model: {model_name}. Supported: specter2")

    logger.info("Loading model %s on %s", hf_name, device)
    tokenizer = AutoTokenizer.from_pretrained(hf_name)
    model = AutoModel.from_pretrained(hf_name)

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    logger.info("Model loaded on %s", device)
    return model, tokenizer


def embed_batch(
    model: Any, tokenizer: Any, texts: list[str], batch_size: int = 32
) -> list[list[float]]:
    """Generate embeddings for a batch of texts using CLS token pooling.

    Returns list of float vectors (one per input text).
    """
    import torch

    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    )

    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    # SPECTER2 uses CLS token embedding (first token of last_hidden_state)
    embeddings = outputs.last_hidden_state[:, 0, :].cpu().tolist()
    return embeddings


def _vec_to_pgvector(vec: list[float]) -> str:
    """Format a float list as a pgvector literal."""
    return "[" + ",".join(str(v) for v in vec) + "]"


def store_embeddings_copy(
    conn: psycopg.Connection,
    inputs: list[EmbeddingInput],
    vectors: list[list[float]],
    model_name: str,
) -> int:
    """Store embeddings via COPY for bulk throughput.

    Uses a staging table + INSERT ON CONFLICT for idempotent upsert.
    Returns number of rows written.
    """
    if not inputs:
        return 0

    with conn.cursor() as cur:
        cur.execute(
            "CREATE TEMP TABLE IF NOT EXISTS _embed_staging ("
            "  bibcode TEXT, model_name TEXT, embedding vector(768),"
            "  input_type TEXT, source_hash TEXT"
            ") ON COMMIT DELETE ROWS"
        )

        with cur.copy(
            "COPY _embed_staging (bibcode, model_name, embedding, input_type, source_hash) "
            "FROM STDIN"
        ) as copy:
            for inp, vec in zip(inputs, vectors):
                copy.write_row(
                    (inp.bibcode, model_name, _vec_to_pgvector(vec), inp.input_type, inp.source_hash)
                )

        cur.execute(
            "INSERT INTO paper_embeddings "
            "  (bibcode, model_name, embedding, input_type, source_hash) "
            "SELECT bibcode, model_name, embedding, input_type, source_hash "
            "FROM _embed_staging "
            "ON CONFLICT (bibcode, model_name) DO UPDATE SET "
            "  embedding = EXCLUDED.embedding, "
            "  input_type = EXCLUDED.input_type, "
            "  source_hash = EXCLUDED.source_hash"
        )
        count = cur.rowcount

    conn.commit()
    return count


def store_embeddings(
    conn: psycopg.Connection,
    inputs: list[EmbeddingInput],
    vectors: list[list[float]],
    model_name: str,
) -> int:
    """Store embeddings via individual INSERT (fallback for small batches).

    Returns number of rows written.
    """
    if not inputs:
        return 0

    with conn.cursor() as cur:
        for inp, vec in zip(inputs, vectors):
            cur.execute(
                """
                INSERT INTO paper_embeddings (bibcode, model_name, embedding, input_type, source_hash)
                VALUES (%s, %s, %s::vector, %s, %s)
                ON CONFLICT (bibcode, model_name) DO UPDATE SET
                    embedding = EXCLUDED.embedding,
                    input_type = EXCLUDED.input_type,
                    source_hash = EXCLUDED.source_hash
                """,
                (inp.bibcode, model_name, _vec_to_pgvector(vec), inp.input_type, inp.source_hash),
            )
    conn.commit()
    return len(inputs)


def fetch_unembedded_bibcodes(
    conn: psycopg.Connection,
    model_name: str,
    limit: int | None = None,
) -> list[tuple[str, str | None, str | None]]:
    """Fetch papers that don't yet have embeddings for the given model.

    Returns list of (bibcode, title, abstract) tuples.
    """
    query = """
        SELECT p.bibcode, p.title, p.abstract
        FROM papers p
        LEFT JOIN paper_embeddings pe
            ON p.bibcode = pe.bibcode AND pe.model_name = %s
        WHERE pe.bibcode IS NULL
          AND p.title IS NOT NULL
        ORDER BY p.bibcode
    """
    params: list[Any] = [model_name]
    if limit is not None:
        query += " LIMIT %s"
        params.append(limit)

    with conn.cursor() as cur:
        cur.execute(query, params)
        return cur.fetchall()


def run_embedding_pipeline(
    dsn: str | None = None,
    model_name: str = "specter2",
    batch_size: int = 32,
    device: str = "cpu",
    limit: int | None = None,
    use_copy: bool = True,
) -> int:
    """Full embedding pipeline: fetch unembedded papers, embed, store.

    Args:
        dsn: Database connection string.
        model_name: Model name (currently only 'specter2' supported).
        batch_size: GPU inference batch size.
        device: Torch device ('cpu', 'cuda', 'auto').
        limit: Max papers to embed (None = all).
        use_copy: Use COPY-based bulk writes (faster) vs individual INSERTs.

    Returns total number of papers embedded.
    """
    conn = get_connection(dsn)
    try:
        # Fetch papers needing embeddings
        papers = fetch_unembedded_bibcodes(conn, model_name, limit=limit)
        if not papers:
            logger.info("No papers need embedding for model %s", model_name)
            return 0

        logger.info("Found %d papers to embed with %s", len(papers), model_name)

        # Prepare inputs
        inputs: list[EmbeddingInput] = []
        for bibcode, title, abstract in papers:
            inp = prepare_input(bibcode, title, abstract)
            if inp is not None:
                inputs.append(inp)

        if not inputs:
            logger.info("No valid inputs after preparation")
            return 0

        logger.info(
            "%d inputs prepared (%d title_abstract, %d title_only)",
            len(inputs),
            sum(1 for i in inputs if i.input_type == "title_abstract"),
            sum(1 for i in inputs if i.input_type == "title_only"),
        )

        # Load model
        model, tokenizer = load_model(model_name, device=device)

        # Pick storage function
        store_fn = store_embeddings_copy if use_copy else store_embeddings

        # Embed in batches
        total_embedded = 0
        t_start = time.monotonic()

        for batch_start in range(0, len(inputs), batch_size):
            batch_inputs = inputs[batch_start : batch_start + batch_size]
            batch_texts = [inp.text for inp in batch_inputs]

            vectors = embed_batch(model, tokenizer, batch_texts, batch_size=batch_size)
            stored = store_fn(conn, batch_inputs, vectors, model_name)
            total_embedded += stored

            elapsed = time.monotonic() - t_start
            rate = total_embedded / elapsed if elapsed > 0 else 0
            logger.info(
                "Embedded %d/%d (%.0f rec/s)",
                total_embedded,
                len(inputs),
                rate,
            )

        elapsed = time.monotonic() - t_start
        logger.info(
            "Embedding complete: %d papers in %.1fs (%.0f rec/s)",
            total_embedded,
            elapsed,
            total_embedded / elapsed if elapsed > 0 else 0,
        )
        return total_embedded
    finally:
        conn.close()
