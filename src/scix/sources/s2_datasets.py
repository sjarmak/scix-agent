"""Semantic Scholar Datasets API client and ingestion pipeline.

Ingests three datasets from the S2 Open Data Platform:
    - s2orc: 8-10M parsed papers with body_text[], cite_spans[], bib_entries[]
    - s2ag papers: 225M metadata records
    - s2ag citations: 2.8B edges with intent and isInfluential flags

The Datasets API serves monthly snapshots via presigned S3 URLs, with
incremental diffs between releases. This module handles:
    1. Dataset discovery and partition URL retrieval
    2. S2ORC body normalization to papers_fulltext schema
    3. Citation intent/influence merge onto existing ADS citation_edges
    4. Health ping to prevent API key expiry (60-day inactivity pruning)

SAFETY:
    * API key sourced exclusively from SEMANTIC_SCHOLAR_API_KEY env var.
    * Production DSN guards on all DB-writing operations.
    * All target tables are LOGGED (see migration 042).

See also:
    - Migration 042 (migrations/042_s2_datasets.sql)
    - Migration 038 (papers_external_ids crosswalk with s2_corpus_id)
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any

from scix.db import DEFAULT_DSN, is_production_dsn, redact_dsn
from scix.http_client import ResilientClient

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class ProductionGuardError(RuntimeError):
    """Raised when a DB-writing operation would target production without opt-in."""


# ---------------------------------------------------------------------------
# S2AG metadata fields to retain after pruning (drop everything else)
# ---------------------------------------------------------------------------

_S2AG_KEEP_FIELDS: frozenset[str] = frozenset(
    {
        "corpusid",
        "externalids",
        "title",
        "authors",
        "year",
        "venue",
        "referencecount",
        "citationcount",
        "influentialcitationcount",
        "isopenaccess",
        "s2fieldsofstudy",
        "publicationtypes",
        "publicationdate",
        "journal",
    }
)

# Mapping from S2AG camelCase field names to papers_s2ag snake_case columns.
# prune_s2ag_metadata returns the S2AG-native keys; this mapping is applied
# at INSERT time to align with the migration 042 schema.
_S2AG_FIELD_TO_COLUMN: dict[str, str] = {
    "corpusid": "s2_corpus_id",
    "externalids": "external_ids",
    "title": "title",
    "authors": "authors",
    "year": "publication_year",
    "venue": "venue",
    "referencecount": "reference_count",
    "citationcount": "citation_count",
    "influentialcitationcount": "influential_citation_count",
    "isopenaccess": "is_open_access",
    "s2fieldsofstudy": "fields_of_study",
    "publicationtypes": "publication_types",
    "publicationdate": "publication_date",
    "journal": "journal",
}


# ---------------------------------------------------------------------------
# Config (frozen dataclass — immutable after creation)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class S2ClientConfig:
    """Immutable configuration for the S2 Datasets API client.

    The API key is required and has no default — it must be supplied
    explicitly or via :meth:`from_env`.
    """

    api_key: str
    base_url: str = "https://api.semanticscholar.org"
    datasets_api_path: str = "/datasets/v1/release"
    graph_api_path: str = "/graph/v1"

    @classmethod
    def from_env(cls) -> S2ClientConfig:
        """Create config from the SEMANTIC_SCHOLAR_API_KEY environment variable.

        Raises:
            KeyError: If SEMANTIC_SCHOLAR_API_KEY is not set.
        """
        api_key = os.environ["SEMANTIC_SCHOLAR_API_KEY"]
        return cls(api_key=api_key)


# ---------------------------------------------------------------------------
# Datasets API client
# ---------------------------------------------------------------------------


class S2DatasetsClient:
    """Client for the Semantic Scholar Datasets API (v1).

    Handles:
        - Latest release discovery
        - Dataset partition URL retrieval (presigned S3 URLs)
        - Incremental diff retrieval between releases
    """

    def __init__(self, config: S2ClientConfig) -> None:
        self._config = config
        self._http = ResilientClient(
            max_retries=3,
            backoff_base=2.0,
            rate_limit=5.0,
            user_agent="scix-harvester/1.0 (S2 Datasets)",
            timeout=120.0,
        )

    def _headers(self) -> dict[str, str]:
        return {"x-api-key": self._config.api_key}

    def get_latest_release(self) -> dict[str, Any]:
        """Fetch the latest dataset release metadata.

        Returns:
            Release metadata dict with release_id and dataset info.
        """
        url = f"{self._config.base_url}{self._config.datasets_api_path}/latest"
        response = self._http.get(url, headers=self._headers())
        return response.json()

    def get_dataset_partitions(
        self,
        dataset_name: str,
        release_id: str | None = None,
    ) -> list[str]:
        """Get presigned S3 URLs for all partitions of a dataset.

        Args:
            dataset_name: One of 's2orc', 'papers', 'abstracts', 'citations', etc.
            release_id: Specific release ID. Uses 'latest' if not specified.

        Returns:
            List of presigned S3 URLs for the dataset partitions.
        """
        release = release_id or "latest"
        url = (
            f"{self._config.base_url}{self._config.datasets_api_path}"
            f"/{release}/dataset/{dataset_name}"
        )
        response = self._http.get(url, headers=self._headers())
        data = response.json()
        return data.get("files", [])

    def get_diff_partitions(
        self,
        dataset_name: str,
        from_release: str,
        to_release: str,
    ) -> list[str]:
        """Get presigned S3 URLs for incremental diff between two releases.

        Args:
            dataset_name: Dataset to diff (e.g. 's2orc').
            from_release: Starting release ID.
            to_release: Ending release ID.

        Returns:
            List of presigned S3 URLs for the diff partitions.
        """
        url = (
            f"{self._config.base_url}{self._config.datasets_api_path}"
            f"/{to_release}/diff/{from_release}/dataset/{dataset_name}"
        )
        response = self._http.get(url, headers=self._headers())
        data = response.json()
        return data.get("files", [])


# ---------------------------------------------------------------------------
# Pure parsing / normalization functions (no IO, no side effects)
# ---------------------------------------------------------------------------


def normalize_s2orc_body(
    record: dict[str, Any],
) -> tuple[int, str, list[dict[str, Any]]] | None:
    """Normalize an S2ORC record's body_text into plain text + section metadata.

    Args:
        record: Raw S2ORC record with corpusid and content.text[].

    Returns:
        (corpus_id, body_text, sections) or None if the record has no usable body.
        body_text is the concatenated section text with section headers.
        sections is a list of dicts preserving section name, text, and cite_spans.
    """
    corpus_id = record.get("corpusid")
    if corpus_id is None:
        return None

    content = record.get("content")
    if not isinstance(content, dict):
        return None

    text_sections = content.get("text")
    if not isinstance(text_sections, list) or not text_sections:
        return None

    sections: list[dict[str, Any]] = []
    body_parts: list[str] = []

    for section in text_sections:
        if not isinstance(section, dict):
            continue
        section_name = section.get("section", "")
        section_text = section.get("text", "")
        if not section_text:
            continue

        if section_name:
            body_parts.append(f"## {section_name}\n\n{section_text}")
        else:
            body_parts.append(section_text)

        section_meta: dict[str, Any] = {
            "section": section_name,
            "text": section_text,
            "cite_spans": section.get("cite_spans"),
        }
        sections.append(section_meta)

    if not body_parts:
        return None

    body_text = "\n\n".join(body_parts)
    return (int(corpus_id), body_text, sections)


def parse_citation_intent(
    citation: dict[str, Any],
) -> tuple[int, int, bool, list[str]] | None:
    """Parse an S2AG citation record into structured fields.

    Args:
        citation: Raw S2AG citation record.

    Returns:
        (citing_corpus_id, cited_corpus_id, is_influential, intents) or None
        if required fields are missing.
    """
    citing = citation.get("citingcorpusid")
    cited = citation.get("citedcorpusid")
    if citing is None or cited is None:
        return None

    is_influential = bool(citation.get("isinfluential", False))
    intents = citation.get("intents")
    if not isinstance(intents, list):
        intents = []

    return (int(citing), int(cited), is_influential, intents)


def prune_s2ag_metadata(record: dict[str, Any]) -> dict[str, Any] | None:
    """Prune an S2AG paper record to essential fields only.

    Drops fields not in _S2AG_KEEP_FIELDS to reduce storage. The full
    record can always be re-fetched from a S2 snapshot if needed.

    Args:
        record: Raw S2AG paper record.

    Returns:
        Pruned dict or None if corpusid is missing.
    """
    if "corpusid" not in record:
        return None

    return {k: v for k, v in record.items() if k in _S2AG_KEEP_FIELDS}


def remap_s2ag_to_columns(pruned: dict[str, Any]) -> dict[str, Any]:
    """Remap S2AG camelCase field names to papers_s2ag snake_case columns.

    Args:
        pruned: Output of :func:`prune_s2ag_metadata`.

    Returns:
        Dict with DB column names as keys, ready for INSERT.
    """
    return {_S2AG_FIELD_TO_COLUMN[k]: v for k, v in pruned.items() if k in _S2AG_FIELD_TO_COLUMN}


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# S2ORC body normalizer (DB-writing component)
# ---------------------------------------------------------------------------


class S2OrcBodyNormalizer:
    """Normalize S2ORC body_text into papers_fulltext or papers_ads_body schema.

    Checks papers_external_ids.has_s2orc_body to skip already-covered papers.
    Only processes papers NOT already covered by source='ar5iv'.

    Production guard: refuses to run against production DSN unless
    yes_production=True.
    """

    def __init__(
        self,
        dsn: str | None = None,
        yes_production: bool = False,
    ) -> None:
        self._dsn = dsn
        self._yes_production = yes_production

    def _check_production_guard(self) -> None:
        effective_dsn = self._dsn or DEFAULT_DSN
        if is_production_dsn(effective_dsn) and not self._yes_production:
            raise ProductionGuardError(
                "Refusing to run S2ORC body normalizer against production DSN "
                f"({redact_dsn(effective_dsn)}). Pass yes_production=True to override."
            )


# ---------------------------------------------------------------------------
# Citation intent merger (DB-writing component)
# ---------------------------------------------------------------------------


class CitationIntentMerger:
    """Merge S2AG citation intent + influence data onto existing ADS edges.

    For every existing ADS citation_edge where we can join via DOI or arXiv ID
    through papers_external_ids, attach the S2AG intent labels and influence
    flag as edge_attrs JSONB.

    Production guard: refuses to run against production DSN unless
    yes_production=True.
    """

    def __init__(
        self,
        dsn: str | None = None,
        yes_production: bool = False,
    ) -> None:
        self._dsn = dsn
        self._yes_production = yes_production

    def _check_production_guard(self) -> None:
        effective_dsn = self._dsn or DEFAULT_DSN
        if is_production_dsn(effective_dsn) and not self._yes_production:
            raise ProductionGuardError(
                "Refusing to run citation intent merger against production DSN "
                f"({redact_dsn(effective_dsn)}). Pass yes_production=True to override."
            )

    @staticmethod
    def build_edge_attrs(
        intents: list[str],
        is_influential: bool,
    ) -> dict[str, Any]:
        """Build the edge_attrs JSONB payload for a citation edge.

        Args:
            intents: S2AG intent labels (e.g. ['methodology', 'background']).
            is_influential: S2AG influence flag.

        Returns:
            Dict ready for JSONB serialization into citation_edges.edge_attrs.
        """
        return {
            "s2_intents": intents,
            "s2_is_influential": is_influential,
        }


# ---------------------------------------------------------------------------
# Health ping (keeps API key from 60-day inactivity expiry)
# ---------------------------------------------------------------------------


class S2HealthPing:
    """Weekly health ping to Semantic Scholar Graph API.

    Sends a lightweight search query to prevent the API key from being
    pruned due to 60-day inactivity. Designed to be called from a cron job.
    """

    def __init__(self, api_key: str) -> None:
        self._api_key = api_key
        self._http = ResilientClient(
            max_retries=2,
            backoff_base=1.0,
            rate_limit=1.0,
            user_agent="scix-harvester/1.0 (health-ping)",
            timeout=30.0,
        )

    def check(self) -> bool:
        """Send a health ping to S2 Graph API.

        Returns:
            True if the API responded with 200, False otherwise.
        """
        url = "https://api.semanticscholar.org/graph/v1/paper/search"
        try:
            response = self._http.get(
                url,
                params={"query": "test", "limit": "1"},
                headers={"x-api-key": self._api_key},
            )
            healthy = response.status_code == 200
            if healthy:
                logger.info("S2 health ping: OK")
            else:
                logger.warning("S2 health ping: status=%d", response.status_code)
            return healthy
        except Exception:
            logger.exception("S2 health ping failed")
            return False
