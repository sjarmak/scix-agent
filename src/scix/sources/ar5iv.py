"""ar5iv HTML fetcher and LaTeXML parser for papers_fulltext.

Fetches rendered HTML from ``https://arxiv.org/html/{arxiv_id}``, parses the
LaTeXML-generated HTML5 into structured sections/citations/figures/tables/
equations, and writes the result into ``papers_fulltext`` with
``source='ar5iv'``.

The fetcher uses a content-addressed gzip cache under ``raw_html/`` so the
same paper is never downloaded twice. Rate limiting (default 5 req/s) keeps
us polite to the ar5iv service.

LICENSING: ar5iv text is LaTeX-derived and subject to ADR-006 internal-use-only
restrictions. This module tags records with ``source='ar5iv'`` so downstream
emission enforcement (bead wqr.5.1) can identify them. The canonical_url is
stored as metadata for attribution.

SAFETY: Refuses to target a production DSN without explicit opt-in, following
the same guard pattern as ``src/scix/ads_body.py``.
"""

from __future__ import annotations

import dataclasses
import gzip
import json
import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bs4 import BeautifulSoup, Tag

from scix.db import DEFAULT_DSN, is_production_dsn, redact_dsn
from scix.sources.licensing import SnippetPayload, enforce_snippet_budget

logger = logging.getLogger(__name__)

AR5IV_BASE_URL = "https://arxiv.org/html"
SOURCE_TAG = "ar5iv"

# Sources whose body text is LaTeX-derived and subject to ADR-006
# internal-use-only restrictions (snippet budget enforcement).
LATEX_DERIVED_SOURCES: frozenset[str] = frozenset({"ar5iv", "arxiv_local"})

# Valid arXiv ID formats:
#   New-style: 2301.12345, 2301.12345v2
#   Old-style: astro-ph/9901001, hep-th/0001234v1
_ARXIV_ID_RE = re.compile(r"^(?:[a-z][\w.-]*/\d{7}(?:v\d+)?|\d{4}\.\d{4,5}(?:v\d+)?)$")


# ---------------------------------------------------------------------------
# Data classes (all frozen / immutable)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Section:
    """A parsed document section."""

    heading: str
    level: int
    text: str
    offset: int


@dataclass(frozen=True)
class InlineCite:
    """A parsed inline citation reference."""

    offset: int
    bib_ref: str
    target_bibcode_or_null: str | None


@dataclass(frozen=True)
class Figure:
    """A parsed figure element."""

    id: str
    caption: str


@dataclass(frozen=True)
class Table:
    """A parsed table element."""

    id: str
    caption: str


@dataclass(frozen=True)
class Equation:
    """A parsed equation element."""

    id: str
    latex: str


@dataclass(frozen=True)
class ParsedFulltext:
    """Immutable result of parsing an ar5iv HTML page."""

    sections: list[Section]
    inline_cites: list[InlineCite]
    figures: list[Figure]
    tables: list[Table]
    equations: list[Equation]
    parser_version: str

    def sections_json(self) -> str:
        """Serialize sections to JSON for DB insertion."""
        return json.dumps(
            [dataclasses.asdict(s) for s in self.sections],
            ensure_ascii=False,
        )

    def inline_cites_json(self) -> str:
        """Serialize inline citations to JSON for DB insertion."""
        return json.dumps(
            [dataclasses.asdict(c) for c in self.inline_cites],
            ensure_ascii=False,
        )

    def figures_json(self) -> str:
        """Serialize figures to JSON for DB insertion."""
        return json.dumps(
            [dataclasses.asdict(f) for f in self.figures],
            ensure_ascii=False,
        )

    def tables_json(self) -> str:
        """Serialize tables to JSON for DB insertion."""
        return json.dumps(
            [dataclasses.asdict(t) for t in self.tables],
            ensure_ascii=False,
        )

    def equations_json(self) -> str:
        """Serialize equations to JSON for DB insertion."""
        return json.dumps(
            [dataclasses.asdict(e) for e in self.equations],
            ensure_ascii=False,
        )


class ProductionGuardError(RuntimeError):
    """Raised when the loader would target a production DSN without explicit opt-in."""


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Ar5ivConfig:
    """Immutable configuration for the ar5iv loader."""

    dsn: str | None
    cache_dir: Path
    rate_limit: float = 5.0
    batch_size: int = 1000
    dry_run: bool = False
    yes_production: bool = False


# ---------------------------------------------------------------------------
# Canonical URL
# ---------------------------------------------------------------------------


def _build_canonical_url(arxiv_id: str) -> str:
    """Build the canonical arXiv abstract URL for attribution."""
    return f"https://arxiv.org/abs/{arxiv_id}"


def get_body_snippet(
    parsed: "ParsedFulltext",
    arxiv_id: str,
    *,
    budget: int | None = None,
) -> SnippetPayload:
    """Return a snippet-budget-enforced body payload for user-facing emission.

    Reconstructs the body text by joining section texts from the parsed
    fulltext, then applies :func:`enforce_snippet_budget` with the canonical
    arXiv URL. This is the single user-facing body emission point for ar5iv
    parsed content (ADR-006).

    Parameters
    ----------
    parsed : ParsedFulltext
        The parsed fulltext (from :class:`Ar5ivParser`).
    arxiv_id : str
        The arXiv identifier (e.g. ``"2301.12345"``). Used to build the
        canonical URL.
    budget : int or None
        Optional explicit snippet budget override. Defaults to the value
        from :data:`~scix.sources.licensing.DEFAULT_SNIPPET_BUDGET` or the
        ``SCIX_LATEX_SNIPPET_BUDGET`` env var.

    Returns
    -------
    SnippetPayload
        Frozen payload with the trimmed snippet, canonical URL, truncation
        flag, original length, and resolved budget.
    """
    body = "\n\n".join(s.text for s in parsed.sections if s.text)
    canonical_url = _build_canonical_url(arxiv_id)
    return enforce_snippet_budget(body, canonical_url, budget=budget)


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

# Mapping from LaTeXML CSS class to section nesting level.
_SECTION_LEVEL_MAP: dict[str, int] = {
    "ltx_section": 1,
    "ltx_subsection": 2,
    "ltx_subsubsection": 3,
    "ltx_paragraph": 4,
}


class Ar5ivParser:
    """Parse LaTeXML-generated HTML5 into structured fulltext."""

    def parse(self, html: str) -> ParsedFulltext:
        """Parse an ar5iv HTML page into a ParsedFulltext."""
        soup = BeautifulSoup(html, "html.parser")

        sections = self._extract_sections(soup)
        inline_cites = self._extract_inline_cites(soup)
        figures = self._extract_figures(soup)
        tables = self._extract_tables(soup)
        equations = self._extract_equations(soup)
        parser_version = self._extract_parser_version(soup)

        return ParsedFulltext(
            sections=sections,
            inline_cites=inline_cites,
            figures=figures,
            tables=tables,
            equations=equations,
            parser_version=parser_version,
        )

    def _extract_sections(self, soup: BeautifulSoup) -> list[Section]:
        """Extract sections from LaTeXML section elements."""
        sections: list[Section] = []
        running_offset = 0

        for class_name, level in _SECTION_LEVEL_MAP.items():
            for elem in soup.find_all(class_=class_name):
                if not isinstance(elem, Tag):
                    continue

                # Extract heading from the title element
                title_elem = elem.find(
                    class_=f"ltx_title ltx_title_{class_name.removeprefix('ltx_')}"
                )
                heading = title_elem.get_text(strip=True) if title_elem else ""

                # Extract body text from paragraphs within this section
                # (excluding nested subsections to avoid duplication)
                text_parts: list[str] = []
                for para in elem.find_all(class_="ltx_para", recursive=False):
                    text_parts.append(para.get_text(separator=" ", strip=True))

                text = "\n".join(text_parts)
                if not text and not heading:
                    continue

                sections.append(
                    Section(
                        heading=heading,
                        level=level,
                        text=text,
                        offset=running_offset,
                    )
                )
                running_offset += len(text)

        # Sort by offset to maintain document order
        sections.sort(key=lambda s: s.offset)
        return sections

    def _extract_inline_cites(self, soup: BeautifulSoup) -> list[InlineCite]:
        """Extract inline citation references from ltx_cite elements."""
        cites: list[InlineCite] = []
        seen_refs: set[str] = set()

        for cite_elem in soup.find_all(class_="ltx_cite"):
            if not isinstance(cite_elem, Tag):
                continue

            for link in cite_elem.find_all("a", class_="ltx_ref"):
                href = link.get("href", "")
                if not isinstance(href, str) or not href.startswith("#"):
                    continue

                bib_ref = href.lstrip("#")
                if not bib_ref or bib_ref in seen_refs:
                    continue
                seen_refs.add(bib_ref)

                # Offset is approximate — position in the source string
                source_pos = (
                    cite_elem.sourcepos
                    if hasattr(cite_elem, "sourcepos") and cite_elem.sourcepos
                    else 0
                )

                cites.append(
                    InlineCite(
                        offset=source_pos,
                        bib_ref=bib_ref,
                        target_bibcode_or_null=None,
                    )
                )

        return cites

    def _extract_figures(self, soup: BeautifulSoup) -> list[Figure]:
        """Extract figure elements."""
        figures: list[Figure] = []
        for fig in soup.find_all(class_="ltx_figure"):
            if not isinstance(fig, Tag):
                continue
            fig_id = fig.get("id", "")
            caption_elem = fig.find(class_="ltx_caption")
            caption = caption_elem.get_text(strip=True) if caption_elem else ""
            figures.append(Figure(id=str(fig_id), caption=caption))
        return figures

    def _extract_tables(self, soup: BeautifulSoup) -> list[Table]:
        """Extract table elements (ltx_table wraps tabular data in ar5iv)."""
        tables: list[Table] = []
        for tbl in soup.find_all(class_="ltx_table"):
            if not isinstance(tbl, Tag):
                continue
            tbl_id = tbl.get("id", "")
            caption_elem = tbl.find(class_="ltx_caption")
            caption = caption_elem.get_text(strip=True) if caption_elem else ""
            tables.append(Table(id=str(tbl_id), caption=caption))
        return tables

    def _extract_equations(self, soup: BeautifulSoup) -> list[Equation]:
        """Extract equation elements."""
        equations: list[Equation] = []
        for eq in soup.find_all(class_="ltx_equation"):
            if not isinstance(eq, Tag):
                continue
            eq_id = eq.get("id", "")

            # Try to get LaTeX from alttext attribute on math element
            math_elem = eq.find("math")
            if math_elem and isinstance(math_elem, Tag):
                latex = math_elem.get("alttext", "")
                if isinstance(latex, list):
                    latex = " ".join(latex)
            else:
                latex = eq.get_text(strip=True)

            equations.append(Equation(id=str(eq_id), latex=str(latex)))
        return equations

    def _extract_parser_version(self, soup: BeautifulSoup) -> str:
        """Extract LaTeXML version from meta generator tag."""
        meta = soup.find("meta", attrs={"name": "generator"})
        if meta and isinstance(meta, Tag):
            content = meta.get("content", "")
            if content:
                return str(content)
        return "ar5iv-unknown"


# ---------------------------------------------------------------------------
# Fetcher (HTTP + cache)
# ---------------------------------------------------------------------------


class Ar5ivFetcher:
    """Fetch ar5iv HTML with rate limiting and content-addressed caching."""

    def __init__(
        self,
        *,
        cache_dir: Path,
        rate_limit: float = 5.0,
    ) -> None:
        self._cache_dir = cache_dir
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._min_interval = 1.0 / rate_limit if rate_limit > 0 else 0.0
        self._last_request_time: float = 0.0
        # ResilientClient tracks per-host state (circuit breaker), so we
        # import lazily to avoid hard dependency at module level and reuse
        # the same instance across fetches.
        self._client: Any = None

    @staticmethod
    def _validate_arxiv_id(arxiv_id: str) -> None:
        """Validate arxiv_id format to prevent path traversal and URL injection.

        Raises ValueError for IDs that don't match known arXiv ID patterns.
        """
        if not _ARXIV_ID_RE.match(arxiv_id):
            raise ValueError(f"Rejected invalid arxiv_id: {arxiv_id!r}")

    def _cache_path(self, arxiv_id: str) -> Path:
        """Return the cache file path for an arxiv_id.

        Old-style IDs with slashes (e.g. astro-ph/9901001) get the slash
        replaced to avoid filesystem path issues. The resolved path is
        verified to stay within the cache directory.
        """
        self._validate_arxiv_id(arxiv_id)
        safe_id = arxiv_id.replace("/", "_")
        result = (self._cache_dir / f"{safe_id}.html.gz").resolve()
        if not str(result).startswith(str(self._cache_dir.resolve())):
            raise ValueError(f"Cache path escape attempt: {arxiv_id!r}")
        return result

    def _read_cache(self, arxiv_id: str) -> str | None:
        """Read cached HTML for an arxiv_id. Returns None on cache miss."""
        cache_file = self._cache_path(arxiv_id)
        if not cache_file.exists():
            return None
        try:
            with gzip.open(cache_file, "rt", encoding="utf-8") as f:
                return f.read()
        except (OSError, gzip.BadGzipFile) as exc:
            logger.warning("Failed to read cache for %s: %s", arxiv_id, exc)
            return None

    def _write_cache(self, arxiv_id: str, html: str) -> None:
        """Write HTML to the content-addressed cache."""
        cache_file = self._cache_path(arxiv_id)
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        try:
            with gzip.open(cache_file, "wt", encoding="utf-8") as f:
                f.write(html)
        except OSError as exc:
            logger.warning("Failed to write cache for %s: %s", arxiv_id, exc)

    def _apply_rate_limit(self) -> None:
        """Sleep if needed to enforce the per-host rate limit."""
        if self._min_interval <= 0:
            return
        now = time.monotonic()
        elapsed = now - self._last_request_time
        if elapsed < self._min_interval:
            sleep_time = self._min_interval - elapsed
            time.sleep(sleep_time)
        self._last_request_time = time.monotonic()

    def _get_client(self) -> Any:
        """Lazily create and cache the HTTP client (ResilientClient)."""
        if self._client is None:
            from scix.http_client import ResilientClient

            self._client = ResilientClient(
                rate_limit=0.0,  # we handle rate limiting ourselves
                max_retries=3,
                backoff_base=2.0,
                timeout=60.0,
                user_agent="scix-ar5iv-fetcher/1.0",
            )
        return self._client

    def fetch(self, arxiv_id: str) -> str | None:
        """Fetch ar5iv HTML for an arxiv_id, using cache if available.

        Returns the HTML string, or None if the page is not available
        (404 or persistent error).

        Raises ValueError if arxiv_id doesn't match known arXiv ID formats.
        """
        self._validate_arxiv_id(arxiv_id)

        # Check cache first
        cached = self._read_cache(arxiv_id)
        if cached is not None:
            logger.debug("Cache hit for %s", arxiv_id)
            return cached

        # Rate limit before network request
        self._apply_rate_limit()

        url = f"{AR5IV_BASE_URL}/{arxiv_id}"
        try:
            client = self._get_client()
            response = client.get(url)

            if hasattr(response, "status_code") and response.status_code == 404:
                logger.debug("ar5iv page not found for %s", arxiv_id)
                return None

            if hasattr(response, "status_code") and response.status_code >= 400:
                logger.warning("ar5iv fetch failed for %s: HTTP %d", arxiv_id, response.status_code)
                return None

            html = response.text
            self._write_cache(arxiv_id, html)
            return html

        except Exception as exc:
            logger.warning("ar5iv fetch failed for %s: %s", arxiv_id, exc)
            return None


# ---------------------------------------------------------------------------
# Loader (DB writer)
# ---------------------------------------------------------------------------


class Ar5ivLoader:
    """Load parsed ar5iv fulltext into papers_fulltext.

    Follows the same pattern as AdsBodyLoader: production guard, staging
    table + COPY, ON CONFLICT upsert.
    """

    def __init__(self, config: Ar5ivConfig) -> None:
        self._cfg = config
        self._parser = Ar5ivParser()
        self._fetcher = Ar5ivFetcher(
            cache_dir=config.cache_dir,
            rate_limit=config.rate_limit,
        )

    def _check_production_guard(self) -> None:
        """Refuse to run against production unless explicitly authorized."""
        effective_dsn = self._cfg.dsn or DEFAULT_DSN
        if is_production_dsn(effective_dsn) and not self._cfg.yes_production:
            raise ProductionGuardError(
                "Refusing to run ar5iv loader against production DSN "
                f"({redact_dsn(effective_dsn)}). Pass yes_production=True "
                "(or --yes-production on the CLI) to override. See "
                "CLAUDE.md Testing -- Database Safety."
            )
