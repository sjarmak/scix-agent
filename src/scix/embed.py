"""Embedding pipeline: SPECTER2, INDUS, and OpenAI embeddings, stored in paper_embeddings.

Uses HuggingFace transformers + torch directly (no sentence-transformers wrapper)
for SPECTER2 and INDUS, and the OpenAI SDK for text-embedding-3-large.
Writes embeddings via psycopg COPY for bulk throughput.

Pooling strategies:
  - SPECTER2 (allenai/specter2_base): CLS token (first token of last_hidden_state)
  - INDUS (nasa-impact/nasa-smd-ibm-st-v2): Mean pooling over non-padding tokens
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

# Halfvec cutover gate. Mirrors src/scix/search.py — migrations 053/054 add
# paper_embeddings.embedding_hv + idx_embed_hnsw_indus_hv but were not applied
# to prod scix as of 2026-04-22 (bead scix_experiments-d0a). When False
# (default), INDUS writes go to the legacy vector(768) `embedding` column so
# the daily cron pipeline does not crash on missing `embedding_hv`. Flip to
# "1" only after migrations + scripts/backfill_halfvec.py finish.
_HALFVEC_ENABLED = os.environ.get("SCIX_USE_HALFVEC", "0") == "1"

# Module-level model cache: (model_name, device) -> (model, tokenizer)
_model_cache: dict[tuple[str, str], tuple[Any, Any]] = {}

# Pooling strategy per model: "cls" (first token) or "mean" (attention-masked average)
MODEL_POOLING: dict[str, str] = {
    "specter2": "cls",
    "indus": "mean",
}


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


def load_model(model_name: str = "indus", device: str = "cpu") -> tuple[Any, Any]:
    """Load embedding model and tokenizer via transformers.

    Supports SPECTER2 (CLS pooling) and INDUS (mean pooling).
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

    MODEL_REGISTRY: dict[str, str] = {
        "specter2": "allenai/specter2_base",
        "indus": "nasa-impact/nasa-smd-ibm-st-v2",
    }
    hf_name = MODEL_REGISTRY.get(model_name)
    if hf_name is None:
        raise ValueError(f"Unknown model: {model_name}. Supported: {', '.join(MODEL_REGISTRY)}")

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
    model: Any,
    tokenizer: Any,
    texts: list[str],
    batch_size: int = 32,
    pooling: str = "cls",
) -> list[list[float]]:
    """Generate embeddings for a batch of texts.

    Args:
        pooling: "cls" for CLS token (SPECTER2) or "mean" for mean pooling (INDUS).

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

    if pooling == "mean":
        # Mean pooling: average over non-padding tokens
        attention_mask = inputs["attention_mask"].unsqueeze(-1)  # (B, T, 1)
        masked = outputs.last_hidden_state * attention_mask
        summed = masked.sum(dim=1)  # (B, D)
        counts = attention_mask.sum(dim=1).clamp(min=1)  # (B, 1)
        embeddings = (summed / counts).cpu().tolist()
    else:
        # CLS pooling: first token of last_hidden_state (SPECTER2 default)
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

    # INDUS writes land in the halfvec(768) shadow column populated by
    # the storage/halfvec-migration cutover (bead scix_experiments-0vy),
    # but only after migrations 053/054 are applied to prod scix. Until
    # then, _HALFVEC_ENABLED stays False and INDUS writes the same
    # legacy vector(768) column as the pilot models — see bead
    # scix_experiments-d0a for context.
    use_halfvec = _HALFVEC_ENABLED and model_name == "indus"

    with conn.cursor() as cur:
        cur.execute(
            "CREATE TEMP TABLE IF NOT EXISTS _embed_staging ("
            "  bibcode TEXT, model_name TEXT,"
            "  embedding    vector(768),"
            "  embedding_hv halfvec(768),"
            "  input_type TEXT, source_hash TEXT"
            ") ON COMMIT DELETE ROWS"
        )

        with cur.copy(
            "COPY _embed_staging "
            "  (bibcode, model_name, embedding, embedding_hv, input_type, source_hash) "
            "FROM STDIN"
        ) as copy:
            for inp, vec in zip(inputs, vectors):
                literal = _vec_to_pgvector(vec)
                copy.write_row(
                    (
                        inp.bibcode,
                        model_name,
                        None if use_halfvec else literal,
                        literal if use_halfvec else None,
                        inp.input_type,
                        inp.source_hash,
                    )
                )

        if use_halfvec:
            cur.execute(
                "INSERT INTO paper_embeddings "
                "  (bibcode, model_name, embedding, embedding_hv, input_type, source_hash) "
                "SELECT bibcode, model_name, embedding, embedding_hv, input_type, source_hash "
                "FROM _embed_staging "
                "ON CONFLICT (bibcode, model_name) DO UPDATE SET "
                "  embedding    = EXCLUDED.embedding, "
                "  embedding_hv = EXCLUDED.embedding_hv, "
                "  input_type   = EXCLUDED.input_type, "
                "  source_hash  = EXCLUDED.source_hash"
            )
        else:
            cur.execute(
                "INSERT INTO paper_embeddings "
                "  (bibcode, model_name, embedding, input_type, source_hash) "
                "SELECT bibcode, model_name, embedding, input_type, source_hash "
                "FROM _embed_staging "
                "ON CONFLICT (bibcode, model_name) DO UPDATE SET "
                "  embedding   = EXCLUDED.embedding, "
                "  input_type  = EXCLUDED.input_type, "
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
        use_halfvec = _HALFVEC_ENABLED and model_name == "indus"
        for inp, vec in zip(inputs, vectors):
            literal = _vec_to_pgvector(vec)
            if use_halfvec:
                cur.execute(
                    """
                    INSERT INTO paper_embeddings
                        (bibcode, model_name, embedding_hv, input_type, source_hash)
                    VALUES (%s, %s, %s::halfvec(768), %s, %s)
                    ON CONFLICT (bibcode, model_name) DO UPDATE SET
                        embedding_hv = EXCLUDED.embedding_hv,
                        input_type   = EXCLUDED.input_type,
                        source_hash  = EXCLUDED.source_hash
                    """,
                    (inp.bibcode, model_name, literal, inp.input_type, inp.source_hash),
                )
                continue
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
    model_name: str = "indus",
    batch_size: int = 32,
    device: str = "cpu",
    limit: int | None = None,
    use_copy: bool = True,
    write_buffer: int = 2000,
) -> int:
    """Full embedding pipeline: stream unembedded papers, embed, store.

    Uses a 3-stage producer/consumer pipeline with threading so that
    DB reads, GPU inference, and DB writes overlap:

      Reader thread → [read_queue] → GPU thread → [write_queue] → Writer thread

    This keeps the GPU busy instead of waiting for DB I/O.

    Args:
        dsn: Database connection string.
        model_name: Model name ('indus' or 'specter2').
        batch_size: GPU inference batch size.
        device: Torch device ('cpu', 'cuda', 'auto').
        limit: Max papers to embed (None = all).
        use_copy: Use COPY-based bulk writes (faster) vs individual INSERTs.
        write_buffer: Number of embeddings to accumulate before DB flush.

    Returns total number of papers embedded.
    """
    import queue
    import threading

    _SENTINEL = None  # signals end-of-stream

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

        # Queues for the pipeline stages
        # read_queue: batches of (texts, inputs) ready for GPU
        # write_queue: batches of (inputs, vectors) ready for DB
        read_queue: queue.Queue[list[EmbeddingInput] | None] = queue.Queue(maxsize=8)
        write_queue: queue.Queue[tuple[list[EmbeddingInput], list[list[float]]] | None] = (
            queue.Queue(maxsize=4)
        )

        # Shared counters (protected by the GIL for simple int ops)
        stats = {"embedded": 0, "title_abstract": 0, "title_only": 0}
        reader_error: list[Exception] = []
        writer_error: list[Exception] = []
        t_start = time.monotonic()

        # --- Reader thread: stream from DB cursor into read_queue ---
        def reader() -> None:
            try:
                fetch_sql = (
                    "SELECT p.bibcode, p.title, p.abstract "
                    + _UNEMBEDDED_WHERE
                    + " ORDER BY p.bibcode"
                )
                fetch_params: list[Any] = [model_name]
                if limit is not None:
                    fetch_sql += " LIMIT %s"
                    fetch_params.append(limit)

                with read_conn.cursor(name="embed_cursor") as cur:
                    cur.itersize = batch_size * 8
                    cur.execute(fetch_sql, fetch_params)

                    batch: list[EmbeddingInput] = []
                    for bibcode, title, abstract in cur:
                        inp = prepare_input(bibcode, title, abstract)
                        if inp is None:
                            continue
                        batch.append(inp)
                        if len(batch) >= batch_size:
                            read_queue.put(batch)
                            batch = []

                    # Send remaining partial batch
                    if batch:
                        read_queue.put(batch)
            except Exception as e:
                reader_error.append(e)
                logger.error("Reader thread error: %s", e)
            finally:
                read_queue.put(_SENTINEL)

        # --- Writer thread: drain write_queue into DB ---
        def writer() -> None:
            try:
                buf_inputs: list[EmbeddingInput] = []
                buf_vectors: list[list[float]] = []

                while True:
                    item = write_queue.get()
                    if item is _SENTINEL:
                        # Flush remaining
                        if buf_inputs:
                            stored = store_fn(write_conn, buf_inputs, buf_vectors, model_name)
                            stats["embedded"] += stored
                        break

                    inputs_chunk, vectors_chunk = item
                    buf_inputs.extend(inputs_chunk)
                    buf_vectors.extend(vectors_chunk)

                    if len(buf_inputs) >= write_buffer:
                        stored = store_fn(write_conn, buf_inputs, buf_vectors, model_name)
                        stats["embedded"] += stored
                        buf_inputs = []
                        buf_vectors = []

                        elapsed = time.monotonic() - t_start
                        rate = stats["embedded"] / elapsed if elapsed > 0 else 0
                        logger.info(
                            "Embedded %d/%d (%.0f rec/s)",
                            stats["embedded"],
                            actual_total,
                            rate,
                        )
            except Exception as e:
                writer_error.append(e)
                logger.error("Writer thread error: %s", e)

        # Start reader and writer threads
        reader_thread = threading.Thread(target=reader, name="embed-reader", daemon=True)
        writer_thread = threading.Thread(target=writer, name="embed-writer", daemon=True)
        reader_thread.start()
        writer_thread.start()

        # --- Main thread: GPU inference ---
        while True:
            batch_inputs = read_queue.get()
            if batch_inputs is _SENTINEL:
                # Signal writer to flush and exit
                write_queue.put(_SENTINEL)
                break

            texts = [inp.text for inp in batch_inputs]
            for inp in batch_inputs:
                if inp.input_type == "title_abstract":
                    stats["title_abstract"] += 1
                else:
                    stats["title_only"] += 1

            pooling = MODEL_POOLING.get(model_name, "cls")
            vectors = embed_batch(model, tokenizer, texts, batch_size=batch_size, pooling=pooling)
            write_queue.put((batch_inputs, vectors))

        # Wait for threads to complete
        reader_thread.join(timeout=300)
        writer_thread.join(timeout=300)

        # Check for errors
        if reader_error:
            raise reader_error[0]
        if writer_error:
            raise writer_error[0]

        elapsed = time.monotonic() - t_start
        logger.info(
            "Embedding complete: %d papers in %.1fs (%.0f rec/s) "
            "(%d title_abstract, %d title_only)",
            stats["embedded"],
            elapsed,
            stats["embedded"] / elapsed if elapsed > 0 else 0,
            stats["title_abstract"],
            stats["title_only"],
        )
        return stats["embedded"]
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
