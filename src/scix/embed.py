"""Embedding pipeline: SPECTER2 and OpenAI embeddings, stored in paper_embeddings.

Uses HuggingFace transformers + torch directly (no sentence-transformers wrapper)
for SPECTER2, and the OpenAI SDK for text-embedding-3-large.
Writes embeddings via psycopg COPY for bulk throughput.
"""

from __future__ import annotations

import functools
import hashlib
import logging
import os
import time
from dataclasses import dataclass
from typing import Any

import psycopg

from scix.db import get_connection

logger = logging.getLogger(__name__)

# Module-level model cache: (model_name, device) -> (model, tokenizer)
_model_cache: dict[tuple[str, str], tuple[Any, Any]] = {}


def clear_model_cache() -> None:
    """Clear the cached models, freeing memory."""
    _model_cache.clear()
    logger.info("Model cache cleared")


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


def load_model(model_name: str = "specter2", device: str = "cpu") -> tuple[Any, Any]:
    """Load SPECTER2 model and tokenizer via transformers.

    Returns (model, tokenizer) tuple. The model is moved to the specified device.
    Results are cached in _model_cache so subsequent calls with the same
    (model_name, device) return instantly.
    """
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except ImportError:
        raise ImportError(
            "transformers and torch are required for embedding. "
            "Install with: pip install transformers torch"
        )

    # Resolve "auto" before forming cache key so that auto→cpu and cpu
    # share the same cache entry on non-GPU machines.
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    cache_key = (model_name, device)
    if cache_key in _model_cache:
        logger.debug("Model cache hit for %s on %s", model_name, device)
        return _model_cache[cache_key]

    if model_name == "specter2":
        hf_name = "allenai/specter2_base"
    else:
        raise ValueError(f"Unknown model: {model_name}. Supported: specter2")

    logger.info("Loading model %s on %s", hf_name, device)
    tokenizer = AutoTokenizer.from_pretrained(hf_name)
    model = AutoModel.from_pretrained(hf_name)
    model = model.to(device)
    model.eval()
    logger.info("Model loaded on %s", device)

    result = (model, tokenizer)
    _model_cache[cache_key] = result
    return result


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
                    (
                        inp.bibcode,
                        model_name,
                        _vec_to_pgvector(vec),
                        inp.input_type,
                        inp.source_hash,
                    )
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


_UNEMBEDDED_WHERE = """
    FROM papers p
    LEFT JOIN paper_embeddings pe
        ON p.bibcode = pe.bibcode AND pe.model_name = %s
    WHERE pe.bibcode IS NULL AND p.title IS NOT NULL
"""


def run_embedding_pipeline(
    dsn: str | None = None,
    model_name: str = "specter2",
    batch_size: int = 32,
    device: str = "cpu",
    limit: int | None = None,
    use_copy: bool = True,
    write_buffer: int = 2000,
) -> int:
    """Full embedding pipeline: stream unembedded papers, embed, store.

    Uses a server-side cursor to avoid loading all 5M papers into memory.
    Accumulates embeddings in a write buffer before flushing to DB via COPY.

    Args:
        dsn: Database connection string.
        model_name: Model name (currently only 'specter2' supported).
        batch_size: GPU inference batch size.
        device: Torch device ('cpu', 'cuda', 'auto').
        limit: Max papers to embed (None = all).
        use_copy: Use COPY-based bulk writes (faster) vs individual INSERTs.
        write_buffer: Number of embeddings to accumulate before DB flush.

    Returns total number of papers embedded.
    """
    read_conn = get_connection(dsn)
    write_conn = get_connection(dsn)
    try:
        # Count papers needing embeddings
        with read_conn.cursor() as cur:
            cur.execute("SELECT count(*) " + _UNEMBEDDED_WHERE, [model_name])
            total_to_embed = cur.fetchone()[0]

        if total_to_embed == 0:
            logger.info("No papers need embedding for model %s", model_name)
            return 0

        actual_total = min(total_to_embed, limit) if limit else total_to_embed
        logger.info("Found %d papers to embed with %s", actual_total, model_name)

        # Load model
        model, tokenizer = load_model(model_name, device=device)

        # Pick storage function
        store_fn = store_embeddings_copy if use_copy else store_embeddings

        # Stream papers via server-side cursor on read_conn,
        # write embeddings via write_conn (separate connection to avoid
        # COPY commits closing the server-side cursor).
        total_embedded = 0
        title_abstract_count = 0
        title_only_count = 0
        t_start = time.monotonic()

        buffer_inputs: list[EmbeddingInput] = []
        buffer_vectors: list[list[float]] = []

        fetch_sql = (
            "SELECT p.bibcode, p.title, p.abstract " + _UNEMBEDDED_WHERE + " ORDER BY p.bibcode"
        )
        fetch_params: list[Any] = [model_name]
        if limit is not None:
            fetch_sql += " LIMIT %s"
            fetch_params.append(limit)

        with read_conn.cursor(name="embed_cursor") as cur:
            cur.itersize = batch_size * 4
            cur.execute(fetch_sql, fetch_params)

            gpu_batch_texts: list[str] = []
            gpu_batch_inputs: list[EmbeddingInput] = []

            for bibcode, title, abstract in cur:
                inp = prepare_input(bibcode, title, abstract)
                if inp is None:
                    continue

                gpu_batch_texts.append(inp.text)
                gpu_batch_inputs.append(inp)

                if inp.input_type == "title_abstract":
                    title_abstract_count += 1
                else:
                    title_only_count += 1

                # When GPU batch is full, run inference
                if len(gpu_batch_texts) >= batch_size:
                    vectors = embed_batch(model, tokenizer, gpu_batch_texts, batch_size=batch_size)
                    buffer_inputs.extend(gpu_batch_inputs)
                    buffer_vectors.extend(vectors)
                    gpu_batch_texts.clear()
                    gpu_batch_inputs.clear()

                    # Flush write buffer when large enough
                    if len(buffer_inputs) >= write_buffer:
                        stored = store_fn(write_conn, buffer_inputs, buffer_vectors, model_name)
                        total_embedded += stored
                        buffer_inputs.clear()
                        buffer_vectors.clear()

                        elapsed = time.monotonic() - t_start
                        rate = total_embedded / elapsed if elapsed > 0 else 0
                        logger.info(
                            "Embedded %d/%d (%.0f rec/s)",
                            total_embedded,
                            actual_total,
                            rate,
                        )

            # Process remaining GPU batch
            if gpu_batch_texts:
                vectors = embed_batch(model, tokenizer, gpu_batch_texts, batch_size=batch_size)
                buffer_inputs.extend(gpu_batch_inputs)
                buffer_vectors.extend(vectors)

            # Flush remaining buffer
            if buffer_inputs:
                stored = store_fn(write_conn, buffer_inputs, buffer_vectors, model_name)
                total_embedded += stored

        elapsed = time.monotonic() - t_start
        logger.info(
            "Embedding complete: %d papers in %.1fs (%.0f rec/s) "
            "(%d title_abstract, %d title_only)",
            total_embedded,
            elapsed,
            total_embedded / elapsed if elapsed > 0 else 0,
            title_abstract_count,
            title_only_count,
        )
        return total_embedded
    finally:
        read_conn.close()
        write_conn.close()


# ---------------------------------------------------------------------------
# OpenAI text-embedding-3-large support
# ---------------------------------------------------------------------------

_OPENAI_MODEL = "text-embedding-3-large"
_OPENAI_DEFAULT_DIM = 1024


def _get_openai_client() -> Any:
    """Build an OpenAI client, reading OPENAI_API_KEY from environment.

    Raises ValueError if the key is not set.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY environment variable is not set. "
            "Set it before calling OpenAI embedding functions."
        )
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError(
            "openai package is required for OpenAI embeddings. " "Install with: pip install openai"
        )
    return OpenAI(api_key=api_key)


def embed_openai(texts: list[str], dimensions: int = _OPENAI_DEFAULT_DIM) -> list[list[float]]:
    """Generate embeddings via OpenAI text-embedding-3-large.

    Uses Matryoshka dimensionality reduction to *dimensions* (default 1024).

    Args:
        texts: List of input strings to embed.
        dimensions: Desired embedding dimension (Matryoshka truncation).

    Returns:
        List of float vectors, one per input text.

    Raises:
        ValueError: If OPENAI_API_KEY is missing or returned dimensions mismatch.
    """
    if not texts:
        return []

    client = _get_openai_client()
    response = client.embeddings.create(
        model=_OPENAI_MODEL,
        input=texts,
        dimensions=dimensions,
    )

    embeddings: list[list[float]] = [item.embedding for item in response.data]

    # Validate returned dimensions
    for idx, vec in enumerate(embeddings):
        if len(vec) != dimensions:
            raise ValueError(f"Embedding {idx} has {len(vec)} dimensions, expected {dimensions}")

    logger.debug(
        "OpenAI embedded %d texts (%d dims, model=%s)",
        len(texts),
        dimensions,
        _OPENAI_MODEL,
    )
    return embeddings


# Cache persists for process lifetime (MCP server is long-running).
# 512 entries × 1024 floats × 8 bytes ≈ 4MB — acceptable.
# Call embed_query_openai.cache_clear() to evict if needed.
@functools.lru_cache(maxsize=512)
def embed_query_openai(text: str, dimensions: int = _OPENAI_DEFAULT_DIM) -> list[float]:
    """Embed a single query string via OpenAI, with LRU caching.

    Args:
        text: Query string to embed.
        dimensions: Desired embedding dimension (Matryoshka truncation).

    Returns:
        List of floats representing the embedding vector.

    Raises:
        ValueError: If OPENAI_API_KEY is missing or returned dimensions mismatch.
    """
    vectors = embed_openai([text], dimensions=dimensions)
    return vectors[0]
