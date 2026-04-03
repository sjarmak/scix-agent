#!/usr/bin/env python3
"""Optimized GPU embedding pipeline.

Key optimizations over embed_fast.py:
1. Tokenization happens in reader threads (CPU-bound), not GPU thread
2. Multiple reader threads prefetch and tokenize in parallel
3. Batch size 512 for better GPU utilization
4. Deep queues so GPU never stalls
5. Direct torch tensor passing to avoid repeated tokenization

Requires: _to_embed table (see embed_fast.py header).
"""

from __future__ import annotations

import io
import logging
import queue
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch

from scix.db import get_connection
from scix.embed import EmbeddingInput, load_model, prepare_input, _vec_to_pgvector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("embed_opt")

MODEL_NAME = "specter2"
BATCH_SIZE = 512
WRITE_BUFFER = 20_000
NUM_TOKENIZER_THREADS = 3
GPU_QUEUE_DEPTH = 8
WRITE_QUEUE_DEPTH = 4


def main() -> None:
    _SENTINEL = None

    read_conn = get_connection()
    write_conn = get_connection()

    with read_conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM _to_embed")
        total = cur.fetchone()[0]

    if total == 0:
        logger.info("Nothing to embed")
        return

    logger.info("Papers to embed: %d", total)

    model, tokenizer = load_model(MODEL_NAME, device="cuda")
    device = next(model.parameters()).device
    logger.info("Model on %s", device)

    # Stage 1: raw_queue — reader puts lists of EmbeddingInput
    # Stage 2: gpu_queue — tokenizer threads put (inputs, token_tensors) ready for GPU
    # Stage 3: write_queue — GPU thread puts (inputs, vectors) for DB write
    raw_queue: queue.Queue = queue.Queue(maxsize=GPU_QUEUE_DEPTH * 2)
    gpu_queue: queue.Queue = queue.Queue(maxsize=GPU_QUEUE_DEPTH)
    write_queue: queue.Queue = queue.Queue(maxsize=WRITE_QUEUE_DEPTH)

    stats = {"embedded": 0}
    errors: list[Exception] = []
    t_start = time.monotonic()

    # --- Reader: sequential scan from _to_embed ---
    def reader() -> None:
        try:
            with read_conn.cursor(name="opt_embed_cur") as cur:
                cur.itersize = BATCH_SIZE * 16
                cur.execute("SELECT bibcode, title, abstract FROM _to_embed")
                batch: list[EmbeddingInput] = []
                for bibcode, title, abstract in cur:
                    inp = prepare_input(bibcode, title, abstract)
                    if inp is None:
                        continue
                    batch.append(inp)
                    if len(batch) >= BATCH_SIZE:
                        raw_queue.put(batch)
                        batch = []
                if batch:
                    raw_queue.put(batch)
        except Exception as e:
            errors.append(e)
            logger.error("Reader error: %s", e)
        finally:
            for _ in range(NUM_TOKENIZER_THREADS):
                raw_queue.put(_SENTINEL)

    # --- Tokenizer threads: CPU-bound tokenization ---
    tokenizer_done_count = [0]
    tokenizer_lock = threading.Lock()

    def tokenizer_worker() -> None:
        try:
            while True:
                batch = raw_queue.get()
                if batch is _SENTINEL:
                    with tokenizer_lock:
                        tokenizer_done_count[0] += 1
                        if tokenizer_done_count[0] == NUM_TOKENIZER_THREADS:
                            gpu_queue.put(_SENTINEL)
                    return

                texts = [inp.text for inp in batch]
                tokens = tokenizer(
                    texts,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                )
                # Move to GPU here so transfer overlaps with next tokenization
                tokens = {k: v.to(device) for k, v in tokens.items()}
                gpu_queue.put((batch, tokens))
        except Exception as e:
            errors.append(e)
            logger.error("Tokenizer error: %s", e)

    # --- GPU thread: pure inference, no tokenization ---
    def gpu_worker() -> None:
        try:
            while True:
                item = gpu_queue.get()
                if item is _SENTINEL:
                    write_queue.put(_SENTINEL)
                    return

                batch_inputs, tokens = item
                with torch.no_grad():
                    outputs = model(**tokens)
                embeddings = outputs.last_hidden_state[:, 0, :].cpu().tolist()
                write_queue.put((batch_inputs, embeddings))
        except Exception as e:
            errors.append(e)
            logger.error("GPU error: %s", e)

    # --- Writer: COPY to PostgreSQL ---
    def writer() -> None:
        try:
            buf_inputs: list[EmbeddingInput] = []
            buf_vectors: list[list[float]] = []

            while True:
                item = write_queue.get()
                if item is _SENTINEL:
                    if buf_inputs:
                        _flush(write_conn, buf_inputs, buf_vectors)
                        stats["embedded"] += len(buf_inputs)
                    break

                inputs, vectors = item
                buf_inputs.extend(inputs)
                buf_vectors.extend(vectors)

                if len(buf_inputs) >= WRITE_BUFFER:
                    _flush(write_conn, buf_inputs, buf_vectors)
                    stats["embedded"] += len(buf_inputs)
                    buf_inputs = []
                    buf_vectors = []

                    elapsed = time.monotonic() - t_start
                    rate = stats["embedded"] / elapsed if elapsed > 0 else 0
                    logger.info(
                        "Embedded %d/%d (%.0f rec/s)",
                        stats["embedded"],
                        total,
                        rate,
                    )
        except Exception as e:
            errors.append(e)
            logger.error("Writer error: %s", e)

    # Start all threads
    threads = []
    t = threading.Thread(target=reader, name="reader", daemon=True)
    t.start()
    threads.append(t)

    for i in range(NUM_TOKENIZER_THREADS):
        t = threading.Thread(target=tokenizer_worker, name=f"tok-{i}", daemon=True)
        t.start()
        threads.append(t)

    t = threading.Thread(target=gpu_worker, name="gpu", daemon=True)
    t.start()
    threads.append(t)

    # Writer runs on main thread
    writer()

    for t in threads:
        t.join(timeout=300)

    if errors:
        raise errors[0]

    elapsed = time.monotonic() - t_start
    rate = stats["embedded"] / elapsed if elapsed > 0 else 0
    logger.info("Complete: %d papers in %.1fs (%.0f rec/s)", stats["embedded"], elapsed, rate)

    # Cleanup temp table
    read_conn.close()
    write_conn.close()


def _flush(
    conn,
    inputs: list[EmbeddingInput],
    vectors: list[list[float]],
) -> None:
    """Bulk write via COPY + staging table."""
    model_name = MODEL_NAME
    with conn.cursor() as cur:
        cur.execute(
            "CREATE TEMP TABLE IF NOT EXISTS _embed_staging ("
            "bibcode TEXT, model_name TEXT, embedding vector, "
            "input_type TEXT, source_hash TEXT) ON COMMIT DELETE ROWS"
        )

    copy_sql = (
        "COPY _embed_staging (bibcode, model_name, embedding, input_type, source_hash) FROM STDIN"
    )
    buf = io.StringIO()
    for inp, vec in zip(inputs, vectors):
        pgvec = _vec_to_pgvector(vec)
        buf.write(f"{inp.bibcode}\t{model_name}\t{pgvec}\t{inp.input_type}\t{inp.source_hash}\n")
    buf.seek(0)

    with conn.transaction():
        with conn.cursor() as cur:
            cur.execute(
                "CREATE TEMP TABLE IF NOT EXISTS _embed_staging ("
                "bibcode TEXT, model_name TEXT, embedding vector, "
                "input_type TEXT, source_hash TEXT) ON COMMIT DELETE ROWS"
            )
            with cur.copy(copy_sql) as copy:
                while chunk := buf.read(65536):
                    copy.write(chunk.encode("utf-8"))
            cur.execute("""
                INSERT INTO paper_embeddings (bibcode, model_name, embedding, input_type, source_hash)
                SELECT bibcode, model_name, embedding, input_type, source_hash
                FROM _embed_staging
                ON CONFLICT (bibcode, model_name) DO UPDATE SET
                    embedding = EXCLUDED.embedding,
                    input_type = EXCLUDED.input_type,
                    source_hash = EXCLUDED.source_hash
                """)


if __name__ == "__main__":
    main()
