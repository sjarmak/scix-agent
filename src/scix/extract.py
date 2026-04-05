"""Entity extraction pipeline: batch extraction via Anthropic Messages Batches API.

Extracts methods, datasets, instruments, and materials from paper abstracts
using Claude with tool-use schema. Results are checkpointed to local JSONL
before loading into the extractions table.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import psycopg

from scix.db import get_connection

logger = logging.getLogger(__name__)


class BudgetExceededError(Exception):
    """Raised when cumulative extraction cost reaches the budget threshold."""

    pass


EXTRACTION_VERSION = "v1"
EXTRACTION_TYPES = ("methods", "datasets", "instruments", "materials")

# Cost per million tokens (USD) — Batches API pricing (50% of standard)
# Updated 2026-04. See https://docs.anthropic.com/en/docs/about-claude/models
_MODEL_COSTS: dict[str, tuple[float, float]] = {
    # (input_per_M, output_per_M)
    "claude-haiku-4-5-20251001": (0.40, 2.00),
    "claude-sonnet-4-20250514": (1.50, 7.50),
    "claude-opus-4-20250514": (7.50, 37.50),
}
_DEFAULT_BUDGET_USD = 10.0

_SAFE_ID_RE = re.compile(r"[^a-zA-Z0-9_-]")


def _encode_bibcode(bibcode: str) -> str:
    """Encode bibcode to Batches API custom_id (alphanumeric + _ -)."""
    return _SAFE_ID_RE.sub("_", bibcode)[:64]


def _decode_bibcode(custom_id: str) -> str:
    """Best-effort decode — stored in JSONL alongside raw bibcode for safety."""
    return custom_id


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------


def _checkpoint_path(output_dir: str, version: str) -> Path:
    """Return the path to the checkpoint file for a given version."""
    return Path(output_dir) / f".checkpoint_{version}.json"


def _load_checkpoint(output_dir: str, version: str) -> dict[str, Any]:
    """Load checkpoint state from disk.

    Returns a dict with 'processed_bibcodes' (set) and 'cumulative_cost_usd' (float).
    If no checkpoint exists, returns empty defaults.
    """
    path = _checkpoint_path(output_dir, version)
    if not path.exists():
        return {"version": version, "processed_bibcodes": set(), "cumulative_cost_usd": 0.0}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {
        "version": data.get("version", version),
        "processed_bibcodes": set(data.get("processed_bibcodes", [])),
        "cumulative_cost_usd": float(data.get("cumulative_cost_usd", 0.0)),
    }


def _save_checkpoint(
    output_dir: str,
    version: str,
    processed_bibcodes: set[str],
    cumulative_cost_usd: float,
) -> Path:
    """Persist checkpoint state to disk.

    Writes a JSON file with the current set of processed bibcodes and
    cumulative cost. Creates parent directories if needed.
    """
    path = _checkpoint_path(output_dir, version)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "version": version,
        "processed_bibcodes": sorted(processed_bibcodes),
        "cumulative_cost_usd": cumulative_cost_usd,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    logger.debug(
        "Checkpoint saved: %d bibcodes, $%.4f cumulative",
        len(processed_bibcodes),
        cumulative_cost_usd,
    )
    return path


# ---------------------------------------------------------------------------
# DB idempotency helpers
# ---------------------------------------------------------------------------


def _get_existing_bibcodes(
    conn: psycopg.Connection,
    extraction_version: str,
) -> set[str]:
    """Return bibcodes that already have extractions for the given version."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT DISTINCT bibcode FROM extractions WHERE extraction_version = %s",
            (extraction_version,),
        )
        return {row[0] for row in cur.fetchall()}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExtractionRequest:
    """A paper queued for entity extraction."""

    bibcode: str
    title: str
    abstract: str


@dataclass(frozen=True)
class ExtractionRow:
    """A single extraction result ready for DB insertion."""

    bibcode: str
    extraction_type: str
    extraction_version: str
    payload: dict[str, Any]


# ---------------------------------------------------------------------------
# Cohort selection
# ---------------------------------------------------------------------------


def select_pilot_cohort(
    conn: psycopg.Connection,
    limit: int = 10_000,
) -> list[ExtractionRequest]:
    """Select top papers by citation_count with abstract length > 100.

    Returns papers ordered by citation_count DESC, filtered to those
    with a non-null abstract longer than 100 characters.
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT bibcode, title, abstract
            FROM papers
            WHERE abstract IS NOT NULL
              AND length(abstract) > 100
            ORDER BY citation_count DESC NULLS LAST
            LIMIT %s
            """,
            (limit,),
        )
        rows = cur.fetchall()

    results = [ExtractionRequest(bibcode=r[0], title=r[1] or "", abstract=r[2]) for r in rows]
    logger.info("Selected %d papers for extraction (requested %d)", len(results), limit)
    return results


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a scientific entity extraction assistant specializing in astronomy \
and astrophysics literature. Your task is to extract structured entities from \
paper abstracts.

For each paper, identify and extract:
- **methods**: Statistical methods, computational techniques, algorithms, \
analysis approaches (e.g., "Markov chain Monte Carlo", "spectral energy \
distribution fitting", "principal component analysis")
- **datasets**: Named surveys, catalogs, data releases, or observational \
datasets (e.g., "Sloan Digital Sky Survey DR16", "Gaia DR3", "2MASS")
- **instruments**: Telescopes, detectors, spectrographs, or space missions \
(e.g., "Hubble Space Telescope", "ALMA", "James Webb Space Telescope")
- **materials**: Physical materials, chemical compounds, or sample types \
studied (e.g., "carbonaceous chondrites", "iron meteorites", "silicate dust")

Extract ONLY entities explicitly mentioned in the abstract. Do not infer \
entities not present in the text. Return empty lists for categories with \
no matches."""

_FEW_SHOT_EXAMPLES: list[dict[str, Any]] = [
    {
        "role": "user",
        "content": (
            "Title: Constraining Cosmological Parameters with CMB Lensing\n\n"
            "Abstract: We present new constraints on cosmological parameters "
            "using cosmic microwave background lensing data from the Planck "
            "2018 release combined with baryon acoustic oscillation "
            "measurements from the Sloan Digital Sky Survey DR16. Our analysis "
            "employs Markov chain Monte Carlo sampling to explore the parameter "
            "space, finding tight constraints on the matter density and Hubble "
            "constant. We use the CAMB Boltzmann solver for theoretical "
            "predictions."
        ),
    },
    {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "type": "function",
                "function": {
                    "name": "extract_entities",
                    "arguments": json.dumps(
                        {
                            "methods": [
                                "Markov chain Monte Carlo sampling",
                                "CAMB Boltzmann solver",
                                "baryon acoustic oscillation measurements",
                            ],
                            "datasets": [
                                "Planck 2018 release",
                                "Sloan Digital Sky Survey DR16",
                            ],
                            "instruments": [],
                            "materials": [],
                        }
                    ),
                },
            }
        ],
    },
    {
        "role": "user",
        "content": (
            "Title: Characterizing Exoplanet Atmospheres with JWST\n\n"
            "Abstract: We report transmission spectroscopy observations of the "
            "hot Jupiter WASP-39b obtained with the James Webb Space Telescope "
            "NIRSpec instrument. Using nested sampling with the MultiNest "
            "algorithm and the POSEIDON retrieval framework, we detect carbon "
            "dioxide, water vapor, and sulfur dioxide in the planetary "
            "atmosphere. Our results are compared with Hubble Space Telescope "
            "WFC3 archival data from the Mikulski Archive for Space Telescopes."
        ),
    },
    {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "type": "function",
                "function": {
                    "name": "extract_entities",
                    "arguments": json.dumps(
                        {
                            "methods": [
                                "transmission spectroscopy",
                                "nested sampling",
                                "MultiNest algorithm",
                                "POSEIDON retrieval framework",
                            ],
                            "datasets": [
                                "Mikulski Archive for Space Telescopes",
                            ],
                            "instruments": [
                                "James Webb Space Telescope",
                                "NIRSpec",
                                "Hubble Space Telescope",
                                "WFC3",
                            ],
                            "materials": [
                                "carbon dioxide",
                                "water vapor",
                                "sulfur dioxide",
                            ],
                        }
                    ),
                },
            }
        ],
    },
    {
        "role": "user",
        "content": (
            "Title: Solar Wind Interaction with Martian Atmosphere\n\n"
            "Abstract: Using magnetometer data from the Mars Atmosphere and "
            "Volatile EvolutioN (MAVEN) mission, we study the interaction of "
            "solar wind with the Martian ionosphere. We apply hybrid particle-"
            "in-cell simulations calibrated against in-situ measurements from "
            "the Solar Wind Ion Analyzer. Analysis of ion escape rates reveals "
            "enhanced oxygen and carbon ion loss during solar energetic particle "
            "events, with implications for long-term atmospheric erosion of "
            "silicate-rich planetary surfaces."
        ),
    },
    {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "type": "function",
                "function": {
                    "name": "extract_entities",
                    "arguments": json.dumps(
                        {
                            "methods": [
                                "hybrid particle-in-cell simulations",
                            ],
                            "datasets": [],
                            "instruments": [
                                "MAVEN",
                                "Solar Wind Ion Analyzer",
                            ],
                            "materials": [
                                "silicate-rich planetary surfaces",
                            ],
                        }
                    ),
                },
            }
        ],
    },
]

_TOOL_SCHEMA: dict[str, Any] = {
    "name": "extract_entities",
    "description": (
        "Extract scientific entities from a paper abstract. "
        "Returns lists of methods, datasets, instruments, and materials "
        "mentioned in the text."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "methods": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Statistical methods, computational techniques, "
                    "algorithms, or analysis approaches."
                ),
            },
            "datasets": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Named surveys, catalogs, data releases, " "or observational datasets."
                ),
            },
            "instruments": {
                "type": "array",
                "items": {"type": "string"},
                "description": ("Telescopes, detectors, spectrographs, " "or space missions."),
            },
            "materials": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Physical materials, chemical compounds, " "or sample types studied."
                ),
            },
        },
        "required": ["methods", "datasets", "instruments", "materials"],
    },
}


def estimate_cost(
    num_requests: int,
    model: str,
    avg_input_tokens: int = 1200,
    avg_output_tokens: int = 250,
) -> float:
    """Estimate batch cost in USD.

    Args:
        num_requests: Number of papers to process.
        model: Model ID string.
        avg_input_tokens: Estimated input tokens per request (system + few-shot + abstract).
        avg_output_tokens: Estimated output tokens per response (tool_use JSON ~200-300 tokens).

    Returns estimated cost in USD.
    """
    costs = _MODEL_COSTS.get(model)
    if costs is None:
        logger.warning("No cost data for model %s — cannot estimate", model)
        return float("inf")
    input_cost, output_cost = costs
    total_input = num_requests * avg_input_tokens / 1_000_000
    total_output = num_requests * avg_output_tokens / 1_000_000
    return total_input * input_cost + total_output * output_cost


def build_extraction_prompt(
    title: str,
    abstract: str,
) -> dict[str, Any]:
    """Build an Anthropic Messages API request body for entity extraction.

    Returns a dict with 'system', 'messages', and 'tools' keys suitable
    for passing to client.messages.create() or batch request construction.
    """
    user_message = f"Title: {title}\n\nAbstract: {abstract}"

    # Convert few-shot examples to Anthropic message format.
    # Must avoid consecutive same-role turns — merge text into prior
    # tool_result user turns when needed.
    messages: list[dict[str, Any]] = []
    example_idx = 0
    for example in _FEW_SHOT_EXAMPLES:
        if example["role"] == "user":
            text_block = {"type": "text", "text": example["content"]}
            # Merge into previous user turn (tool_result) if it exists
            if (
                messages
                and messages[-1]["role"] == "user"
                and isinstance(messages[-1]["content"], list)
            ):
                messages[-1]["content"].append(text_block)
            else:
                messages.append({"role": "user", "content": [text_block]})
        elif example["role"] == "assistant":
            tool_call = example["tool_calls"][0]
            tool_id = f"toolu_example_{example_idx}"
            example_idx += 1
            messages.append(
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": tool_id,
                            "name": tool_call["function"]["name"],
                            "input": json.loads(tool_call["function"]["arguments"]),
                        }
                    ],
                }
            )
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": tool_id,
                            "content": "Entities extracted successfully.",
                        }
                    ],
                }
            )

    # Merge actual request into the last user turn (tool_result)
    last_user_block = {"type": "text", "text": user_message}
    if messages and messages[-1]["role"] == "user" and isinstance(messages[-1]["content"], list):
        messages[-1]["content"].append(last_user_block)
    else:
        messages.append({"role": "user", "content": [last_user_block]})

    return {
        "system": _SYSTEM_PROMPT,
        "messages": messages,
        "tools": [_TOOL_SCHEMA],
        "tool_choice": {"type": "tool", "name": "extract_entities"},
    }


# ---------------------------------------------------------------------------
# Batch submission and polling
# ---------------------------------------------------------------------------


def submit_batch(
    client: Any,
    requests: list[ExtractionRequest],
    model: str = "claude-sonnet-4-20250514",
    max_tokens: int = 1024,
) -> tuple[str, dict[str, str]]:
    """Submit a batch of extraction requests via Anthropic Messages Batches API.

    Args:
        client: An anthropic.Anthropic client instance.
        requests: Papers to extract entities from.
        model: Anthropic model ID.
        max_tokens: Max tokens per response.

    Returns the batch ID string.
    """
    # Build a mapping from encoded custom_id back to original bibcode,
    # since the encoding is lossy (special chars like & become _).
    id_to_bibcode: dict[str, str] = {}
    batch_requests = []
    for req in requests:
        encoded_id = _encode_bibcode(req.bibcode)
        id_to_bibcode[encoded_id] = req.bibcode
        prompt = build_extraction_prompt(req.title, req.abstract)
        batch_requests.append(
            {
                "custom_id": encoded_id,
                "params": {
                    "model": model,
                    "max_tokens": max_tokens,
                    "system": prompt["system"],
                    "messages": prompt["messages"],
                    "tools": prompt["tools"],
                    "tool_choice": prompt["tool_choice"],
                },
            }
        )

    batch = client.messages.batches.create(requests=batch_requests)
    logger.info("Submitted batch %s with %d requests", batch.id, len(requests))
    return batch.id, id_to_bibcode


def poll_batch(
    client: Any,
    batch_id: str,
    interval: int = 60,
    max_wait: int = 86400,
) -> Any:
    """Poll a batch until it reaches a terminal state.

    Args:
        client: An anthropic.Anthropic client instance.
        batch_id: The batch ID to poll.
        interval: Seconds between polls.
        max_wait: Maximum seconds to wait before raising TimeoutError.

    Returns the final batch object.
    """
    elapsed = 0
    while elapsed < max_wait:
        batch = client.messages.batches.retrieve(batch_id)
        status = batch.processing_status
        logger.info(
            "Batch %s status: %s (elapsed %ds)",
            batch_id,
            status,
            elapsed,
        )
        if status == "ended":
            return batch
        time.sleep(interval)
        elapsed += interval

    raise TimeoutError(f"Batch {batch_id} did not complete within {max_wait}s")


# ---------------------------------------------------------------------------
# JSONL checkpoint
# ---------------------------------------------------------------------------


def save_results_jsonl(
    client: Any,
    batch_id: str,
    output_path: Path,
    id_to_bibcode: dict[str, str] | None = None,
) -> Path:
    """Save batch results to a local JSONL checkpoint file.

    Streams results from the Anthropic API and writes each as a JSON line.
    This checkpoint ensures results are persisted locally before any DB writes.

    Args:
        client: An anthropic.Anthropic client instance.
        batch_id: The completed batch ID.
        output_path: Path to write the JSONL file.
        id_to_bibcode: Mapping from encoded custom_id back to original bibcode.

    Returns the output path.
    """
    id_to_bibcode = id_to_bibcode or {}
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for result in client.messages.batches.results(batch_id):
            # Restore original bibcode from the mapping
            bibcode = id_to_bibcode.get(result.custom_id, result.custom_id)
            line = {
                "custom_id": bibcode,
                "result": (
                    result.result.model_dump()
                    if hasattr(result.result, "model_dump")
                    else result.result
                ),
            }
            f.write(json.dumps(line) + "\n")
            count += 1

    logger.info(
        "Saved %d results to %s",
        count,
        output_path,
    )
    return output_path


# ---------------------------------------------------------------------------
# DB loading
# ---------------------------------------------------------------------------


def _parse_extraction_rows(
    line: dict[str, Any],
    extraction_version: str,
) -> list[ExtractionRow]:
    """Parse a single JSONL result line into ExtractionRow objects."""
    bibcode = line["custom_id"]
    result = line.get("result", {})

    # Handle different result structures
    result_type = result.get("type", "")
    if result_type == "errored" or result_type == "expired":
        logger.warning("Skipping %s result for %s", result_type, bibcode)
        return []

    # Extract tool_use content from the message
    message = result.get("message", {})
    content_blocks = message.get("content", [])

    rows: list[ExtractionRow] = []
    for block in content_blocks:
        if block.get("type") != "tool_use":
            continue
        if block.get("name") != "extract_entities":
            continue

        tool_input = block.get("input", {})
        for etype in EXTRACTION_TYPES:
            entities = tool_input.get(etype, [])
            if entities:
                rows.append(
                    ExtractionRow(
                        bibcode=bibcode,
                        extraction_type=etype,
                        extraction_version=extraction_version,
                        payload={"entities": entities},
                    )
                )
    return rows


def load_results_to_db(
    conn: psycopg.Connection,
    jsonl_path: Path,
    extraction_version: str = EXTRACTION_VERSION,
    chunk_size: int = 500,
) -> int:
    """Load extraction results from JSONL checkpoint into the extractions table.

    Reads the JSONL file, parses tool_use results, and writes in chunks
    of `chunk_size` rows with COMMIT between chunks. Uses ON CONFLICT
    for idempotent upserts.

    Args:
        conn: Database connection (should NOT be in autocommit mode).
        jsonl_path: Path to the JSONL checkpoint file.
        extraction_version: Version tag for these extractions.
        chunk_size: Number of rows per chunk (COMMIT between chunks).

    Returns total number of rows written/updated.
    """
    jsonl_path = Path(jsonl_path)
    all_rows: list[ExtractionRow] = []

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line_str in f:
            line_str = line_str.strip()
            if not line_str:
                continue
            line = json.loads(line_str)
            all_rows.extend(_parse_extraction_rows(line, extraction_version))

    logger.info(
        "Parsed %d extraction rows from %s",
        len(all_rows),
        jsonl_path,
    )

    total_written = 0
    for i in range(0, len(all_rows), chunk_size):
        chunk = all_rows[i : i + chunk_size]
        with conn.cursor() as cur:
            for row in chunk:
                cur.execute(
                    """
                    INSERT INTO extractions
                        (bibcode, extraction_type, extraction_version, payload)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (bibcode, extraction_type, extraction_version)
                    DO UPDATE SET
                        payload = EXCLUDED.payload,
                        created_at = NOW()
                    """,
                    (
                        row.bibcode,
                        row.extraction_type,
                        row.extraction_version,
                        json.dumps(row.payload),
                    ),
                )
        conn.commit()
        total_written += len(chunk)
        logger.info(
            "Loaded chunk %d-%d (%d/%d rows)",
            i,
            i + len(chunk),
            total_written,
            len(all_rows),
        )

    logger.info("Total rows loaded: %d", total_written)
    return total_written


# ---------------------------------------------------------------------------
# Full pipeline orchestration
# ---------------------------------------------------------------------------


def run_extraction_pipeline(
    dsn: str | None = None,
    pilot_size: int = 10_000,
    model: str = "claude-sonnet-4-20250514",
    output_dir: str = "data/extractions",
    extraction_version: str = EXTRACTION_VERSION,
    batch_size: int = 10_000,
    poll_interval: int = 60,
    budget_usd: float = _DEFAULT_BUDGET_USD,
) -> int:
    """Run the full extraction pipeline: select -> submit -> poll -> save -> load.

    Args:
        dsn: Database connection string.
        pilot_size: Number of papers to extract.
        model: Anthropic model ID.
        output_dir: Directory for JSONL checkpoint files.
        extraction_version: Version tag.
        batch_size: Papers per batch submission.
        poll_interval: Seconds between batch status polls.
        budget_usd: Maximum spend allowed. Pipeline aborts before submission
            if the estimated cost exceeds this. Default $10.

    Returns total extraction rows loaded to DB.
    """
    try:
        import anthropic
    except ImportError:
        raise ImportError(
            "anthropic is required for extraction. " "Install with: pip install anthropic"
        )

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError(
            "ANTHROPIC_API_KEY environment variable is not set. "
            "Set it before running the extraction pipeline."
        )
    client = anthropic.Anthropic(api_key=api_key)
    conn = get_connection(dsn)

    try:
        # 1. Select cohort
        cohort = select_pilot_cohort(conn, limit=pilot_size)
        if not cohort:
            logger.info("No papers found for extraction")
            return 0

        # 2. Load checkpoint and filter already-processed bibcodes
        checkpoint = _load_checkpoint(output_dir, extraction_version)
        processed_bibcodes: set[str] = checkpoint["processed_bibcodes"]
        cumulative_cost_usd: float = checkpoint["cumulative_cost_usd"]

        # Query DB for bibcodes that already have extractions (idempotency)
        existing_in_db = _get_existing_bibcodes(conn, extraction_version)
        skip_bibcodes = processed_bibcodes | existing_in_db

        original_count = len(cohort)
        cohort = [r for r in cohort if r.bibcode not in skip_bibcodes]
        skipped = original_count - len(cohort)
        if skipped > 0:
            logger.info(
                "Skipped %d already-processed bibcodes (%d from checkpoint, %d from DB)",
                skipped,
                len(processed_bibcodes),
                len(existing_in_db),
            )

        if not cohort:
            logger.info("All papers already processed — nothing to do")
            return 0

        # 3. Budget check on remaining cohort
        est = estimate_cost(len(cohort), model)
        logger.info(
            "Estimated cost: $%.2f for %d papers with %s (budget: $%.2f, cumulative: $%.2f)",
            est,
            len(cohort),
            model,
            budget_usd,
            cumulative_cost_usd,
        )
        if est + cumulative_cost_usd > budget_usd:
            raise ValueError(
                f"Estimated cost ${est:.2f} + cumulative ${cumulative_cost_usd:.2f} "
                f"exceeds budget ${budget_usd:.2f}. "
                f"Reduce pilot_size, switch to a cheaper model, or raise budget_usd."
            )

        total_loaded = 0
        output_path = Path(output_dir)
        budget_threshold = 0.8 * budget_usd

        # 4. Process in batches
        for batch_start in range(0, len(cohort), batch_size):
            batch_reqs = cohort[batch_start : batch_start + batch_size]
            batch_num = batch_start // batch_size + 1

            # Check cumulative cost against 80% budget threshold before submitting
            if cumulative_cost_usd >= budget_threshold:
                msg = (
                    f"Cumulative cost ${cumulative_cost_usd:.2f} has reached 80% of "
                    f"budget ${budget_usd:.2f}. Halting extraction pipeline."
                )
                logger.warning(msg)
                _save_checkpoint(
                    output_dir, extraction_version, processed_bibcodes, cumulative_cost_usd
                )
                raise BudgetExceededError(msg)

            logger.info(
                "Processing batch %d (%d papers, cumulative cost: $%.4f)",
                batch_num,
                len(batch_reqs),
                cumulative_cost_usd,
            )

            # Submit
            batch_id, id_to_bibcode = submit_batch(client, batch_reqs, model=model)

            # Poll
            poll_batch(client, batch_id, interval=poll_interval)

            # Save JSONL checkpoint
            safe_id = batch_id.replace("/", "_").replace("..", "__")
            jsonl_file = output_path / f"batch_{safe_id}.jsonl"
            save_results_jsonl(client, batch_id, jsonl_file, id_to_bibcode)

            # Load to DB
            loaded = load_results_to_db(
                conn,
                jsonl_file,
                extraction_version=extraction_version,
            )
            total_loaded += loaded

            # Update checkpoint: mark batch bibcodes as processed, accumulate cost
            batch_bibcodes = {r.bibcode for r in batch_reqs}
            processed_bibcodes |= batch_bibcodes
            batch_cost = estimate_cost(len(batch_reqs), model)
            cumulative_cost_usd += batch_cost
            _save_checkpoint(
                output_dir, extraction_version, processed_bibcodes, cumulative_cost_usd
            )

            logger.info(
                "Batch %d complete: %d rows loaded, cumulative cost: $%.4f",
                batch_num,
                loaded,
                cumulative_cost_usd,
            )

            # Post-batch budget check
            if cumulative_cost_usd >= budget_threshold:
                msg = (
                    f"Cumulative cost ${cumulative_cost_usd:.2f} has reached 80% of "
                    f"budget ${budget_usd:.2f}. Halting extraction pipeline."
                )
                logger.warning(msg)
                _save_checkpoint(
                    output_dir, extraction_version, processed_bibcodes, cumulative_cost_usd
                )
                raise BudgetExceededError(msg)

        logger.info(
            "Extraction pipeline complete: %d total rows loaded",
            total_loaded,
        )
        return total_loaded
    finally:
        conn.close()
