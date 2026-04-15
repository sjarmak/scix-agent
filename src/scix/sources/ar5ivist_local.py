"""Local ar5ivist / LaTeXML runner for papers without ar5iv HTML.

Converts arXiv LaTeX source tarballs to HTML5 using the ar5ivist Docker image
(LaTeXML), then reuses the ``Ar5ivParser`` from ``ar5iv.py`` to normalize
the output into ``papers_fulltext`` rows with ``source='arxiv_local'``.

Routing: triggered only for papers where ``has_arxiv_source=True AND
has_ar5iv_html=False`` — i.e., the ~25% of arXiv papers without a clean
hosted HTML rendering.

SAFETY:
    * Docker image pinned by digest (no mutable tags).
    * arXiv IDs validated before use in Docker commands (command injection
      prevention).
    * Tarball mount is read-only inside the container.
    * Production DSN guard on all DB-writing operations.

See also:
    - ``src/scix/sources/arxiv_s3.py`` (populates the raw_latex cache)
    - ``src/scix/sources/ar5iv.py`` (parser reused here)
    - ``docs/runbooks/arxiv_s3_ingest.md`` (operational guide)
"""

from __future__ import annotations

import logging
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from scix.db import DEFAULT_DSN, is_production_dsn, redact_dsn
from scix.sources.ar5iv import Ar5ivParser, _ARXIV_ID_RE

logger = logging.getLogger(__name__)

# Docker image pinned by digest for reproducibility. Replace the placeholder
# digest with a real one before first use:
#   docker pull ghcr.io/ar5iv/ar5ivist:latest
#   docker inspect --format='{{index .RepoDigests 0}}' ghcr.io/ar5iv/ar5ivist:latest
_AR5IVIST_DIGEST = "sha256:0000000000000000000000000000000000000000000000000000000000000000"
AR5IVIST_IMAGE = f"ghcr.io/ar5iv/ar5ivist@{_AR5IVIST_DIGEST}"

_PLACEHOLDER_DIGEST = "sha256:" + "0" * 64


def _check_image_digest() -> None:
    """Fail loudly if the Docker digest has not been configured."""
    if _AR5IVIST_DIGEST == _PLACEHOLDER_DIGEST:
        raise NotImplementedError(  # stub-ok: digest requires pulling the real Docker image
            "AR5IVIST_IMAGE has a placeholder digest. Pin a real digest in "
            "ar5ivist_local.py before running conversions. See the comment "
            "above AR5IVIST_IMAGE for instructions."
        )


# Source tag for papers_fulltext rows written by this module.
SOURCE_TAG = "arxiv_local"


# ---------------------------------------------------------------------------
# Data classes (all frozen / immutable)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ArxivLocalConfig:
    """Immutable configuration for the local LaTeXML converter."""

    cache_dir: Path
    dsn: str | None = None
    workers: int = 4
    timeout_seconds: int = 120
    batch_size: int = 1000
    yes_production: bool = False


@dataclass(frozen=True)
class ConversionResult:
    """Immutable result of a single paper conversion."""

    arxiv_id: str
    success: bool
    html: str | None
    error: str | None


class ProductionGuardError(RuntimeError):
    """Raised when a DB-writing operation would target production without opt-in."""


# ---------------------------------------------------------------------------
# Routing logic
# ---------------------------------------------------------------------------


def needs_local_conversion(*, has_arxiv_source: bool, has_ar5iv_html: bool) -> bool:
    """Determine if a paper needs local LaTeXML conversion.

    Returns True only when the paper has arXiv LaTeX source available but
    no ar5iv HTML rendering.
    """
    return has_arxiv_source and not has_ar5iv_html


# ---------------------------------------------------------------------------
# Converter
# ---------------------------------------------------------------------------


class ArxivLocalConverter:
    """Convert arXiv LaTeX sources to HTML5 via Docker-packaged LaTeXML.

    Reuses ``Ar5ivParser`` from ``ar5iv.py`` for HTML normalization.
    """

    def __init__(self, config: ArxivLocalConfig) -> None:
        self._cfg = config
        self._parser = Ar5ivParser()

    def _check_production_guard(self) -> None:
        """Refuse to run against production unless explicitly authorized."""
        effective_dsn = self._cfg.dsn or DEFAULT_DSN
        if is_production_dsn(effective_dsn) and not self._cfg.yes_production:
            raise ProductionGuardError(
                "Refusing to run arxiv_local converter against production DSN "
                f"({redact_dsn(effective_dsn)}). Pass yes_production=True to override."
            )

    @staticmethod
    def _validate_arxiv_id(arxiv_id: str) -> bool:
        """Validate arxiv_id format to prevent command injection."""
        return bool(_ARXIV_ID_RE.match(arxiv_id))

    def _build_docker_command(self, arxiv_id: str, tarball_path: Path) -> list[str]:
        """Build the Docker run command for LaTeXML conversion.

        The tarball is mounted read-only. Output is written to stdout.
        """
        return [
            "docker",
            "run",
            "--rm",
            "--network=none",
            "-v",
            f"{tarball_path.resolve()}:/input/source.tar.gz:ro",
            AR5IVIST_IMAGE,
            "/input/source.tar.gz",
        ]

    def convert_paper(self, arxiv_id: str, tarball_path: Path) -> ConversionResult:
        """Convert a single paper's LaTeX source to HTML5 via Docker.

        Returns a ConversionResult with success/failure and the HTML or error.
        """
        _check_image_digest()

        if not self._validate_arxiv_id(arxiv_id):
            return ConversionResult(
                arxiv_id=arxiv_id,
                success=False,
                html=None,
                error=f"Invalid arXiv ID: {arxiv_id!r}",
            )

        if not tarball_path.exists():
            return ConversionResult(
                arxiv_id=arxiv_id,
                success=False,
                html=None,
                error=f"Tarball not found: {tarball_path}",
            )

        cmd = self._build_docker_command(arxiv_id, tarball_path)

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self._cfg.timeout_seconds,
            )
        except subprocess.TimeoutExpired:
            return ConversionResult(
                arxiv_id=arxiv_id,
                success=False,
                html=None,
                error=f"Docker timeout after {self._cfg.timeout_seconds}s for {arxiv_id}",
            )

        if result.returncode != 0:
            return ConversionResult(
                arxiv_id=arxiv_id,
                success=False,
                html=None,
                error=f"LaTeXML failed (exit {result.returncode}): {result.stderr[:500]}",
            )

        return ConversionResult(
            arxiv_id=arxiv_id,
            success=True,
            html=result.stdout,
            error=None,
        )

    def _resolve_tarball_path(self, arxiv_id: str) -> Path:
        """Resolve the cache path for a paper's LaTeX tarball.

        Raises ValueError if the resolved path escapes the cache directory
        (defense-in-depth against crafted arxiv_id values).
        """
        safe_name = arxiv_id.replace("/", "_") + ".tar.gz"
        result = (self._cfg.cache_dir / "raw_latex" / safe_name).resolve()
        cache_root = self._cfg.cache_dir.resolve()
        if not result.is_relative_to(cache_root):
            raise ValueError(f"Tarball path escapes cache_dir: {result} is not under {cache_root}")
        return result

    def batch_convert(
        self,
        arxiv_ids: list[str],
        *,
        workers: int | None = None,
    ) -> list[ConversionResult]:
        """Convert multiple papers in parallel.

        Papers without cached tarballs are reported as failures.
        """
        effective_workers = workers if workers is not None else self._cfg.workers
        results: list[ConversionResult] = []

        def _convert_one(aid: str) -> ConversionResult:
            tarball = self._resolve_tarball_path(aid)
            return self.convert_paper(aid, tarball)

        with ThreadPoolExecutor(max_workers=effective_workers) as pool:
            futures = {pool.submit(_convert_one, aid): aid for aid in arxiv_ids}
            for future in as_completed(futures):
                aid = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as exc:
                    logger.error("Unexpected error converting %s: %s", aid, exc)
                    results.append(
                        ConversionResult(
                            arxiv_id=aid,
                            success=False,
                            html=None,
                            error=str(exc),
                        )
                    )

        succeeded = sum(1 for r in results if r.success)
        logger.info("Batch conversion: %d/%d succeeded", succeeded, len(results))
        return results

    def ingest_to_fulltext(
        self,
        html: str,
        arxiv_id: str,
        bibcode: str,
        dsn: str | None = None,
    ) -> bool:
        """Parse HTML and write a papers_fulltext row with source='arxiv_local'.

        Reuses the Ar5ivParser from ar5iv.py for normalization.
        Returns True on success, False on failure.
        """
        self._check_production_guard()

        import psycopg

        parsed = self._parser.parse(html)
        effective_dsn = dsn or self._cfg.dsn or DEFAULT_DSN

        try:
            with psycopg.connect(effective_dsn) as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO papers_fulltext
                            (bibcode, source, sections, inline_cites,
                             figures, tables, equations, parser_version)
                        VALUES (%s, %s, %s::jsonb, %s::jsonb,
                                %s::jsonb, %s::jsonb, %s::jsonb, %s)
                        ON CONFLICT (bibcode) DO UPDATE SET
                            source = EXCLUDED.source,
                            sections = EXCLUDED.sections,
                            inline_cites = EXCLUDED.inline_cites,
                            figures = EXCLUDED.figures,
                            tables = EXCLUDED.tables,
                            equations = EXCLUDED.equations,
                            parser_version = EXCLUDED.parser_version,
                            parsed_at = now()
                        """,
                        (
                            bibcode,
                            SOURCE_TAG,
                            parsed.sections_json(),
                            parsed.inline_cites_json(),
                            parsed.figures_json(),
                            parsed.tables_json(),
                            parsed.equations_json(),
                            parsed.parser_version,
                        ),
                    )
                conn.commit()
            return True
        except Exception as exc:
            logger.error("Failed to ingest fulltext for %s: %s", arxiv_id, exc)
            return False
