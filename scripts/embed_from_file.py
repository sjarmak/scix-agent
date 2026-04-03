#!/usr/bin/env python3
"""GPU embedding pipeline reading from local TSV file.

Architecture: multiprocessing to avoid GIL contention.
  Main process: file reader → tokenizer → GPU inference
  Writer process: receives numpy arrays via pipe, does binary COPY to PostgreSQL

DB COPY can do 36K rec/s. GPU can do 7.5K rec/s. Tokenizer can do 100K rec/s.
The previous threading approach hit 523 rec/s due to GIL contention.
This approach should hit 2-5K rec/s.

Usage:
    psql -d scix -c "\\COPY (SELECT bibcode, title, abstract FROM _to_embed) TO '/tmp/to_embed.tsv'"
    python3 scripts/embed_from_file.py --input /tmp/to_embed.tsv
"""

from __future__ import annotations

import argparse
import io
import logging
import multiprocessing as mp
import struct
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import torch

from scix.embed import EmbeddingInput, load_model, prepare_input

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("embed_file")

MODEL_NAME = "specter2"
BATCH_SIZE = 512
WRITE_BUFFER = 20_000


def writer_process(write_queue: mp.Queue, total: int, dsn: str | None) -> None:
    """Separate process for DB writes — has its own GIL."""
    import psycopg
    from scix.db import get_connection

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    log = logging.getLogger("writer")
    conn = get_connection(dsn)

    vec_header = struct.pack(">HH", 768, 0)
    vec_field_len = struct.pack(">i", 4 + 768 * 4)
    mn_bytes = MODEL_NAME.encode("utf-8")
    mn_header = struct.pack(">i", len(mn_bytes))
    field_count = struct.pack(">h", 5)

    embedded = 0
    t_start = time.monotonic()

    while True:
        item = write_queue.get()
        if item is None:
            break

        bibcodes, input_types, source_hashes, vecs_bytes = item
        n = len(bibcodes)

        # Reconstruct numpy array from shared bytes
        vecs = np.frombuffer(vecs_bytes, dtype=">f4").reshape(n, 768)

        # Build binary COPY buffer
        buf = io.BytesIO()
        buf.write(b"PGCOPY\n\xff\r\n\0")
        buf.write(struct.pack(">II", 0, 0))

        for i in range(n):
            buf.write(field_count)
            bib = bibcodes[i].encode("utf-8")
            buf.write(struct.pack(">i", len(bib)))
            buf.write(bib)
            buf.write(mn_header)
            buf.write(mn_bytes)
            buf.write(vec_field_len)
            buf.write(vec_header)
            buf.write(vecs[i].tobytes())
            it = input_types[i].encode("utf-8")
            buf.write(struct.pack(">i", len(it)))
            buf.write(it)
            sh = source_hashes[i].encode("utf-8")
            buf.write(struct.pack(">i", len(sh)))
            buf.write(sh)

        buf.write(struct.pack(">h", -1))
        buf.seek(0)
        data = buf.read()

        with conn.cursor() as cur:
            with cur.copy(
                "COPY paper_embeddings (bibcode, model_name, embedding, input_type, source_hash) "
                "FROM STDIN WITH (FORMAT binary)"
            ) as copy:
                copy.write(data)
        conn.commit()

        embedded += n
        elapsed = time.monotonic() - t_start
        rate = embedded / elapsed if elapsed > 0 else 0
        log.info("Written %d/%d (%.0f rec/s)", embedded, total, rate)

    conn.close()
    log.info("Writer done: %d rows", embedded)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("/tmp/to_embed.tsv"))
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dsn", default=None)
    args = parser.parse_args()

    input_path = args.input
    batch_size = args.batch_size

    logger.info("Counting lines in %s...", input_path)
    total = sum(1 for _ in open(input_path, "r", encoding="utf-8"))
    logger.info("Total: %d papers", total)

    model, tokenizer = load_model(MODEL_NAME, device=args.device)
    device = next(model.parameters()).device
    logger.info("Model on %s, batch_size=%d", device, batch_size)

    # Start writer process
    write_queue: mp.Queue = mp.Queue(maxsize=8)
    wp = mp.Process(target=writer_process, args=(write_queue, total, args.dsn), daemon=True)
    wp.start()

    # Main process: read file → tokenize → GPU → send to writer
    embedded = 0
    t_start = time.monotonic()

    # Accumulate for write buffer
    buf_bibcodes: list[str] = []
    buf_input_types: list[str] = []
    buf_source_hashes: list[str] = []
    buf_vecs: list[np.ndarray] = []

    def flush_to_writer() -> None:
        nonlocal buf_bibcodes, buf_input_types, buf_source_hashes, buf_vecs
        if not buf_bibcodes:
            return
        vecs_arr = np.concatenate(buf_vecs, axis=0).astype(">f4")
        write_queue.put(
            (
                buf_bibcodes,
                buf_input_types,
                buf_source_hashes,
                vecs_arr.tobytes(),
            )
        )
        buf_bibcodes = []
        buf_input_types = []
        buf_source_hashes = []
        buf_vecs = []

    # Read and process
    batch_inputs: list[EmbeddingInput] = []
    batch_texts: list[str] = []

    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t", 2)
            if len(parts) < 2:
                continue
            bibcode = parts[0]
            title = parts[1] if parts[1] != "\\N" else None
            abstract = parts[2] if len(parts) > 2 and parts[2] != "\\N" else None

            inp = prepare_input(bibcode, title, abstract)
            if inp is None:
                continue

            batch_inputs.append(inp)
            batch_texts.append(inp.text)

            if len(batch_texts) >= batch_size:
                # Tokenize + GPU
                tokens = tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                )
                tokens = {k: v.to(device) for k, v in tokens.items()}

                with torch.no_grad():
                    outputs = model(**tokens)
                vecs = outputs.last_hidden_state[:, 0, :].cpu().numpy().astype(np.float32)

                # Accumulate
                for inp_item in batch_inputs:
                    buf_bibcodes.append(inp_item.bibcode)
                    buf_input_types.append(inp_item.input_type)
                    buf_source_hashes.append(inp_item.source_hash)
                buf_vecs.append(vecs)
                embedded += len(batch_inputs)

                batch_inputs = []
                batch_texts = []

                # Flush when buffer is full
                if len(buf_bibcodes) >= WRITE_BUFFER:
                    flush_to_writer()
                    elapsed = time.monotonic() - t_start
                    rate = embedded / elapsed if elapsed > 0 else 0
                    logger.info("Embedded %d/%d (%.0f rec/s)", embedded, total, rate)

    # Process remaining
    if batch_texts:
        tokens = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        tokens = {k: v.to(device) for k, v in tokens.items()}
        with torch.no_grad():
            outputs = model(**tokens)
        vecs = outputs.last_hidden_state[:, 0, :].cpu().numpy().astype(np.float32)
        for inp_item in batch_inputs:
            buf_bibcodes.append(inp_item.bibcode)
            buf_input_types.append(inp_item.input_type)
            buf_source_hashes.append(inp_item.source_hash)
        buf_vecs.append(vecs)
        embedded += len(batch_inputs)

    flush_to_writer()
    write_queue.put(None)  # Signal writer to exit

    wp.join(timeout=600)

    elapsed = time.monotonic() - t_start
    rate = embedded / elapsed if elapsed > 0 else 0
    logger.info("Complete: %d papers in %.1fs (%.0f rec/s)", embedded, elapsed, rate)


if __name__ == "__main__":
    main()
