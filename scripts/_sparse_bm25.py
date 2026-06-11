"""Shared FastEmbed BM25 tokenizer configuration for the ADR-014 sparse-lexical
pilot (``qdrant_sparse_pilot.py`` builds the collection; ``eval_sparse_hybrid_
pilot.py`` embeds the queries against it).

The index-time and query-time tokenizers **must** be identical: BM25 matches
term ids, so a query embedded with a different stemmer / token-length policy
than the collection silently scores against the wrong vocabulary and the eval
numbers become fiction. This module is the single source of that config so the
two scripts cannot drift, and it records the config used to a sidecar so the
eval defaults to whatever the build actually used.

Tokenizer levers (FastEmbed 0.8 ``Bm25``):

* ``disable_stemmer`` — turn off Snowball stemming. The Postgres ``scix_english``
  lane is ``simple_nostem`` (no stemming), so ``True`` mirrors it and is the
  primary lever for the ``title_matchable`` regression on scientific tokens
  (e.g. ``X-ray``, ``Lyman-alpha``, instrument designations).
* ``token_max_length`` — drop tokens longer than this (default 40). Raise to
  keep long chemical / catalog identifiers.
* ``k``/``b``/``avg_len`` — Okapi parameters; defaults match the pilot.

Local-only: ``Qdrant/bm25`` is a ~10 MB CPU tokenizer, no neural net, no paid
API (``feedback_no_paid_apis``).
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

from fastembed import SparseTextEmbedding

log = logging.getLogger("sparse_bm25")

BM25_MODEL = "Qdrant/bm25"
_REPO_ROOT = Path(__file__).resolve().parent.parent
# One sidecar keyed by collection name, so the pilot and full-corpus collections
# can record different tokenizer configs without clobbering each other.
TOKENIZER_SIDECAR = _REPO_ROOT / "results" / "sparse_bm25_tokenizer.json"


@dataclass(frozen=True)
class Bm25TokenizerConfig:
    """Index/query tokenizer config. Defaults reproduce the i5oa pilot exactly,
    so existing ``results/sparse_hybrid_pilot_eval*`` numbers stay comparable."""

    disable_stemmer: bool = False
    token_max_length: int = 40
    language: str = "english"
    k: float = 1.2
    b: float = 0.75
    avg_len: float = 256.0

    def signature(self) -> str:
        stem = "nostem" if self.disable_stemmer else f"stem-{self.language}"
        return (
            f"bm25[{stem},maxtok={self.token_max_length},"
            f"k={self.k},b={self.b},avglen={self.avg_len}]"
        )


def add_bm25_tokenizer_args(parser: argparse.ArgumentParser) -> None:
    """Register the tokenizer flags on a build or eval parser."""
    g = parser.add_argument_group("BM25 tokenizer (index/query must match)")
    g.add_argument(
        "--disable-stemmer",
        action="store_true",
        help="Disable Snowball stemming to mirror scix_english simple_nostem "
        "(title_matchable recovery lever).",
    )
    g.add_argument("--token-max-length", type=int, default=40, help="Drop tokens longer than this.")
    g.add_argument(
        "--bm25-language", default="english", help="Stemmer language (ignored if stemmer disabled)."
    )
    g.add_argument("--bm25-k", type=float, default=1.2, help="Okapi BM25 k1.")
    g.add_argument(
        "--bm25-b", type=float, default=0.75, help="Okapi BM25 b (length normalization)."
    )
    g.add_argument(
        "--bm25-avg-len", type=float, default=256.0, help="Okapi BM25 average doc length."
    )


def config_from_args(args: argparse.Namespace) -> Bm25TokenizerConfig:
    return Bm25TokenizerConfig(
        disable_stemmer=args.disable_stemmer,
        token_max_length=args.token_max_length,
        language=args.bm25_language,
        k=args.bm25_k,
        b=args.bm25_b,
        avg_len=args.bm25_avg_len,
    )


def build_model(config: Bm25TokenizerConfig) -> SparseTextEmbedding:
    """Construct the FastEmbed BM25 embedder for this tokenizer config."""
    return SparseTextEmbedding(
        model_name=BM25_MODEL,
        k=config.k,
        b=config.b,
        avg_len=config.avg_len,
        language=config.language,
        token_max_length=config.token_max_length,
        disable_stemmer=config.disable_stemmer,
    )


def record_config(collection: str, config: Bm25TokenizerConfig) -> None:
    """Persist the tokenizer config the build used, keyed by collection name."""
    data: dict[str, dict] = {}
    if TOKENIZER_SIDECAR.exists():
        data = json.loads(TOKENIZER_SIDECAR.read_text())
    data[collection] = asdict(config)
    TOKENIZER_SIDECAR.parent.mkdir(parents=True, exist_ok=True)
    TOKENIZER_SIDECAR.write_text(json.dumps(data, indent=2))
    log.info("recorded tokenizer config for %s -> %s", collection, config.signature())


def load_recorded_config(collection: str) -> Bm25TokenizerConfig | None:
    """Return the tokenizer config recorded for ``collection`` at build time, or
    ``None`` if the build predates the sidecar. The eval uses this so its query
    tokenizer defaults to whatever built the collection."""
    if not TOKENIZER_SIDECAR.exists():
        return None
    data = json.loads(TOKENIZER_SIDECAR.read_text())
    raw = data.get(collection)
    return Bm25TokenizerConfig(**raw) if raw else None
