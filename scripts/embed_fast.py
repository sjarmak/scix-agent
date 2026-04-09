#!/usr/bin/env python3
"""Fast GPU embedding pipeline using materialized unembedded table.

Pre-requisite: run this SQL first to create the source table:
    CREATE TABLE _to_embed AS
    SELECT p.bibcode, p.title, p.abstract
    FROM papers p
    LEFT JOIN paper_embeddings pe ON p.bibcode = pe.bibcode AND pe.model_name = 'indus'
    WHERE pe.bibcode IS NULL AND p.title IS NOT NULL
    ORDER BY p.bibcode;

This avoids the slow Merge Anti Join on every cursor fetch.
"""

from __future__ import annotations

import logging
import queue
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from scix.db import get_connection
from scix.embed import (
    MODEL_POOLING,
    EmbeddingInput,
    embed_batch,
    load_model,
    prepare_input,
    store_embeddings_copy,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("embed_fast")

MODEL_NAME = "indus"
BATCH_SIZE = 256
WRITE_BUFFER = 10_000
DEVICE = "cuda"


def main() -> None:
    _SENTINEL = None

    read_conn = get_connection()
    write_conn = get_connection()

    # Count
    with read_conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM _to_embed")
        total = cur.fetchone()[0]

    if total == 0:
        logger.info("Nothing to embed")
        return

    logger.info("Papers to embed: %d", total)

    # Load model
    model, tokenizer = load_model(MODEL_NAME, device=DEVICE)
    logger.info("Model loaded on %s", DEVICE)

    # Queues
    read_q: queue.Queue = queue.Queue(maxsize=16)
    write_q: queue.Queue = queue.Queue(maxsize=8)

    stats = {"embedded": 0, "ta": 0, "to": 0}
    errors: list[Exception] = []
    t_start = time.monotonic()

    def reader() -> None:
        """Read from _to_embed via sequential scan — no anti-join needed."""
        try:
            with read_conn.cursor(name="fast_embed_cur") as cur:
                cur.itersize = BATCH_SIZE * 16  # prefetch aggressively
                cur.execute("SELECT bibcode, title, abstract FROM _to_embed")

                batch: list[EmbeddingInput] = []
                for bibcode, title, abstract in cur:
                    inp = prepare_input(bibcode, title, abstract)
                    if inp is None:
                        continue
                    batch.append(inp)
                    if len(batch) >= BATCH_SIZE:
                        read_q.put(batch)
                        batch = []
                if batch:
                    read_q.put(batch)
        except Exception as e:
            errors.append(e)
            logger.error("Reader error: %s", e)
        finally:
            read_q.put(_SENTINEL)

    def writer() -> None:
        """Write embeddings to paper_embeddings via COPY."""
        try:
            buf_inputs: list[EmbeddingInput] = []
            buf_vectors: list[list[float]] = []

            while True:
                item = write_q.get()
                if item is _SENTINEL:
                    if buf_inputs:
                        stored = store_embeddings_copy(
                            write_conn, buf_inputs, buf_vectors, MODEL_NAME
                        )
                        stats["embedded"] += stored
                    break

                inputs, vectors = item
                buf_inputs.extend(inputs)
                buf_vectors.extend(vectors)

                if len(buf_inputs) >= WRITE_BUFFER:
                    stored = store_embeddings_copy(write_conn, buf_inputs, buf_vectors, MODEL_NAME)
                    stats["embedded"] += stored
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

    # Start threads
    reader_t = threading.Thread(target=reader, name="reader", daemon=True)
    writer_t = threading.Thread(target=writer, name="writer", daemon=True)
    reader_t.start()
    writer_t.start()

    # Main thread: GPU inference
    while True:
        batch_inputs = read_q.get()
        if batch_inputs is _SENTINEL:
            write_q.put(_SENTINEL)
            break

        texts = [inp.text for inp in batch_inputs]
        for inp in batch_inputs:
            if inp.input_type == "title_abstract":
                stats["ta"] += 1
            else:
                stats["to"] += 1

        pooling = MODEL_POOLING.get(MODEL_NAME, "cls")
        vectors = embed_batch(model, tokenizer, texts, batch_size=BATCH_SIZE, pooling=pooling)
        write_q.put((batch_inputs, vectors))

    reader_t.join(timeout=300)
    writer_t.join(timeout=300)

    if errors:
        raise errors[0]

    elapsed = time.monotonic() - t_start
    rate = stats["embedded"] / elapsed if elapsed > 0 else 0
    logger.info(
        "Complete: %d papers in %.1fs (%.0f rec/s) (%d title_abstract, %d title_only)",
        stats["embedded"],
        elapsed,
        rate,
        stats["ta"],
        stats["to"],
    )

    # Cleanup
    read_conn.close()
    write_conn.close()


if __name__ == "__main__":
    main()
