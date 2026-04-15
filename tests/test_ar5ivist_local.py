"""Unit tests for src/scix/sources/ar5ivist_local.py — local LaTeXML conversion.

Covers:
- ArxivLocalConfig immutability
- Docker image digest pinning
- Single-paper conversion (mocked Docker)
- Batch conversion with workers
- Routing logic: only process papers where ar5iv HTML is missing
- Parser reuse from ar5iv.py
- Ingest to papers_fulltext with source='arxiv_local'
- Production DSN guard
- Docker command injection prevention

No actual Docker or DB access. All external deps are mocked.
"""

from __future__ import annotations

import subprocess
from dataclasses import FrozenInstanceError
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from scix.sources.ar5ivist_local import (
    AR5IVIST_IMAGE,
    ArxivLocalConfig,
    ArxivLocalConverter,
    ConversionResult,
    ProductionGuardError,
    _AR5IVIST_DIGEST,
    _PLACEHOLDER_DIGEST,
    _check_image_digest,
    needs_local_conversion,
)

# ---------------------------------------------------------------------------
# Config immutability
# ---------------------------------------------------------------------------


class TestArxivLocalConfig:
    def test_is_frozen(self) -> None:
        cfg = ArxivLocalConfig(
            cache_dir=Path("/tmp/test"),
            dsn="dbname=scix_test",
        )
        with pytest.raises(FrozenInstanceError):
            cfg.dsn = "mutated"  # type: ignore[misc]

    def test_defaults(self) -> None:
        cfg = ArxivLocalConfig(
            cache_dir=Path("/tmp/test"),
            dsn="dbname=scix_test",
        )
        assert cfg.workers == 4
        assert cfg.timeout_seconds == 120
        assert cfg.yes_production is False

    def test_docker_image_pinned(self) -> None:
        """The Docker image must include a digest, not just a tag."""
        assert "@sha256:" in AR5IVIST_IMAGE

    def test_placeholder_digest_detected(self) -> None:
        """_check_image_digest raises when digest is the all-zeros placeholder."""
        assert _AR5IVIST_DIGEST == _PLACEHOLDER_DIGEST  # currently a placeholder
        with pytest.raises(NotImplementedError, match="placeholder digest"):
            _check_image_digest()


# ---------------------------------------------------------------------------
# ConversionResult immutability
# ---------------------------------------------------------------------------


class TestConversionResult:
    def test_is_frozen(self) -> None:
        result = ConversionResult(
            arxiv_id="2301.00001",
            success=True,
            html="<html></html>",
            error=None,
        )
        with pytest.raises(FrozenInstanceError):
            result.success = False  # type: ignore[misc]

    def test_success_result(self) -> None:
        result = ConversionResult(
            arxiv_id="2301.00001",
            success=True,
            html="<html>content</html>",
            error=None,
        )
        assert result.html is not None
        assert result.error is None

    def test_failure_result(self) -> None:
        result = ConversionResult(
            arxiv_id="2301.00001",
            success=False,
            html=None,
            error="LaTeXML conversion failed",
        )
        assert result.html is None
        assert result.error is not None


# ---------------------------------------------------------------------------
# Production DSN guard
# ---------------------------------------------------------------------------


class TestProductionGuard:
    def test_blocks_production_dsn(self) -> None:
        cfg = ArxivLocalConfig(
            cache_dir=Path("/tmp/test"),
            dsn="dbname=scix",
        )
        converter = ArxivLocalConverter(cfg)
        with pytest.raises(ProductionGuardError, match="production"):
            converter._check_production_guard()

    def test_allows_test_dsn(self) -> None:
        cfg = ArxivLocalConfig(
            cache_dir=Path("/tmp/test"),
            dsn="dbname=scix_test",
        )
        converter = ArxivLocalConverter(cfg)
        converter._check_production_guard()  # should not raise

    def test_yes_production_overrides(self) -> None:
        cfg = ArxivLocalConfig(
            cache_dir=Path("/tmp/test"),
            dsn="dbname=scix",
            yes_production=True,
        )
        converter = ArxivLocalConverter(cfg)
        converter._check_production_guard()  # should not raise

    def test_uri_production_dsn_blocked(self) -> None:
        cfg = ArxivLocalConfig(
            cache_dir=Path("/tmp/test"),
            dsn="postgresql://user:pw@host:5432/scix",
        )
        converter = ArxivLocalConverter(cfg)
        with pytest.raises(ProductionGuardError):
            converter._check_production_guard()


# ---------------------------------------------------------------------------
# Single paper conversion (mocked Docker)
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _bypass_digest_check() -> None:
    """Bypass the placeholder digest check for all convert tests."""
    with patch("scix.sources.ar5ivist_local._check_image_digest"):
        yield  # type: ignore[func-returns-value]


class TestConvertPaper:
    def test_successful_conversion(self, tmp_path: Path) -> None:
        cfg = ArxivLocalConfig(cache_dir=tmp_path, dsn="dbname=scix_test")
        converter = ArxivLocalConverter(cfg)

        # Create a fake latex tarball
        latex_dir = tmp_path / "raw_latex"
        latex_dir.mkdir()
        tarball = latex_dir / "2301.00001.tar.gz"
        tarball.write_bytes(b"fake tarball")

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "<html><body>Converted paper</body></html>"

        with patch("subprocess.run", return_value=mock_result) as mock_run:
            result = converter.convert_paper("2301.00001", tarball)

        assert result.success is True
        assert result.html is not None
        assert "Converted paper" in result.html
        mock_run.assert_called_once()

    def test_docker_failure(self, tmp_path: Path) -> None:
        cfg = ArxivLocalConfig(cache_dir=tmp_path, dsn="dbname=scix_test")
        converter = ArxivLocalConverter(cfg)

        tarball = tmp_path / "2301.00001.tar.gz"
        tarball.write_bytes(b"fake tarball")

        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stderr = "LaTeXML conversion error"
        mock_result.stdout = ""

        with patch("subprocess.run", return_value=mock_result):
            result = converter.convert_paper("2301.00001", tarball)

        assert result.success is False
        assert result.error is not None

    def test_docker_timeout(self, tmp_path: Path) -> None:
        cfg = ArxivLocalConfig(
            cache_dir=tmp_path,
            dsn="dbname=scix_test",
            timeout_seconds=10,
        )
        converter = ArxivLocalConverter(cfg)

        tarball = tmp_path / "2301.00001.tar.gz"
        tarball.write_bytes(b"fake tarball")

        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired("docker", 10)):
            result = converter.convert_paper("2301.00001", tarball)

        assert result.success is False
        assert "timeout" in result.error.lower()

    def test_rejects_invalid_arxiv_id(self, tmp_path: Path) -> None:
        """Prevent command injection via arxiv_id."""
        cfg = ArxivLocalConfig(cache_dir=tmp_path, dsn="dbname=scix_test")
        converter = ArxivLocalConverter(cfg)

        tarball = tmp_path / "evil.tar.gz"
        tarball.write_bytes(b"fake")

        result = converter.convert_paper("'; rm -rf /; echo '", tarball)
        assert result.success is False
        assert "invalid" in result.error.lower()

    def test_nonexistent_tarball(self, tmp_path: Path) -> None:
        cfg = ArxivLocalConfig(cache_dir=tmp_path, dsn="dbname=scix_test")
        converter = ArxivLocalConverter(cfg)

        tarball = tmp_path / "nonexistent.tar.gz"

        result = converter.convert_paper("2301.00001", tarball)
        assert result.success is False
        assert "not found" in result.error.lower() or "exist" in result.error.lower()


# ---------------------------------------------------------------------------
# Docker command construction
# ---------------------------------------------------------------------------


class TestDockerCommand:
    def test_uses_pinned_image(self, tmp_path: Path) -> None:
        cfg = ArxivLocalConfig(cache_dir=tmp_path, dsn="dbname=scix_test")
        converter = ArxivLocalConverter(cfg)

        tarball = tmp_path / "2301.00001.tar.gz"
        tarball.write_bytes(b"fake")

        cmd = converter._build_docker_command("2301.00001", tarball)
        assert AR5IVIST_IMAGE in cmd

    def test_mounts_tarball_readonly(self, tmp_path: Path) -> None:
        cfg = ArxivLocalConfig(cache_dir=tmp_path, dsn="dbname=scix_test")
        converter = ArxivLocalConverter(cfg)

        tarball = tmp_path / "2301.00001.tar.gz"
        tarball.write_bytes(b"fake")

        cmd = converter._build_docker_command("2301.00001", tarball)
        cmd_str = " ".join(cmd)
        assert ":ro" in cmd_str


# ---------------------------------------------------------------------------
# Batch conversion
# ---------------------------------------------------------------------------


class TestBatchConvert:
    def test_batch_converts_multiple(self, tmp_path: Path) -> None:
        cfg = ArxivLocalConfig(cache_dir=tmp_path, dsn="dbname=scix_test", workers=2)
        converter = ArxivLocalConverter(cfg)

        latex_dir = tmp_path / "raw_latex"
        latex_dir.mkdir()
        ids = ["2301.00001", "2301.00002", "2301.00003"]
        for aid in ids:
            (latex_dir / f"{aid}.tar.gz").write_bytes(b"fake")

        success_result = ConversionResult(
            arxiv_id="test",
            success=True,
            html="<html></html>",
            error=None,
        )

        with patch.object(converter, "convert_paper", return_value=success_result):
            results = converter.batch_convert(ids)

        assert len(results) == 3

    def test_batch_handles_missing_tarballs(self, tmp_path: Path) -> None:
        """Papers without cached tarballs should be skipped gracefully."""
        cfg = ArxivLocalConfig(cache_dir=tmp_path, dsn="dbname=scix_test", workers=1)
        converter = ArxivLocalConverter(cfg)

        # No raw_latex dir — no tarballs
        results = converter.batch_convert(["2301.00001"])
        assert len(results) == 1
        assert results[0].success is False


# ---------------------------------------------------------------------------
# Routing logic: needs_local_conversion
# ---------------------------------------------------------------------------


class TestNeedsLocalConversion:
    def test_needs_conversion_when_source_missing(self) -> None:
        """Paper has arXiv source but no ar5iv HTML → needs local conversion."""
        assert needs_local_conversion(has_arxiv_source=True, has_ar5iv_html=False) is True

    def test_no_conversion_when_ar5iv_exists(self) -> None:
        """Paper already has ar5iv HTML → skip local conversion."""
        assert needs_local_conversion(has_arxiv_source=True, has_ar5iv_html=True) is False

    def test_no_conversion_without_source(self) -> None:
        """Paper has no arXiv source → cannot convert."""
        assert needs_local_conversion(has_arxiv_source=False, has_ar5iv_html=False) is False

    def test_no_conversion_without_source_but_has_html(self) -> None:
        assert needs_local_conversion(has_arxiv_source=False, has_ar5iv_html=True) is False


# ---------------------------------------------------------------------------
# Parser reuse from ar5iv.py
# ---------------------------------------------------------------------------


class TestParserReuse:
    def test_uses_ar5iv_parser(self, tmp_path: Path) -> None:
        """The converter must reuse the Ar5ivParser from ar5iv.py."""
        cfg = ArxivLocalConfig(cache_dir=tmp_path, dsn="dbname=scix_test")
        converter = ArxivLocalConverter(cfg)

        from scix.sources.ar5iv import Ar5ivParser

        assert isinstance(converter._parser, Ar5ivParser)


# ---------------------------------------------------------------------------
# Tarball path escape prevention
# ---------------------------------------------------------------------------


class TestTarballPathEscape:
    def test_rejects_path_traversal(self, tmp_path: Path) -> None:
        """_resolve_tarball_path must reject IDs that escape cache_dir.

        The replace('/', '_') transform makes simple '../' attacks safe,
        but a symlink in the raw_latex dir could still escape. We test by
        creating a symlink that points outside cache_dir.
        """
        cfg = ArxivLocalConfig(cache_dir=tmp_path, dsn="dbname=scix_test")
        converter = ArxivLocalConverter(cfg)

        # Create raw_latex as a symlink pointing outside cache_dir
        raw_latex = tmp_path / "raw_latex"
        raw_latex.symlink_to("/tmp")

        with pytest.raises(ValueError, match="escapes cache_dir"):
            converter._resolve_tarball_path("2301.00001")

    def test_normal_id_resolves(self, tmp_path: Path) -> None:
        cfg = ArxivLocalConfig(cache_dir=tmp_path, dsn="dbname=scix_test")
        converter = ArxivLocalConverter(cfg)

        path = converter._resolve_tarball_path("2301.00001")
        assert path.name == "2301.00001.tar.gz"
        assert path.is_relative_to(tmp_path)

    def test_old_style_id_resolves(self, tmp_path: Path) -> None:
        cfg = ArxivLocalConfig(cache_dir=tmp_path, dsn="dbname=scix_test")
        converter = ArxivLocalConverter(cfg)

        path = converter._resolve_tarball_path("astro-ph/0001001")
        assert path.name == "astro-ph_0001001.tar.gz"
        assert path.is_relative_to(tmp_path)


# ---------------------------------------------------------------------------
# ingest_to_fulltext (mocked DB + parser)
# ---------------------------------------------------------------------------


class TestIngestToFulltext:
    def test_calls_parser_and_writes_db(self, tmp_path: Path) -> None:
        """ingest_to_fulltext must call parser.parse and execute an upsert."""
        cfg = ArxivLocalConfig(cache_dir=tmp_path, dsn="dbname=scix_test")
        converter = ArxivLocalConverter(cfg)

        mock_parsed = MagicMock()
        mock_parsed.sections_json.return_value = "[]"
        mock_parsed.inline_cites_json.return_value = "[]"
        mock_parsed.figures_json.return_value = "[]"
        mock_parsed.tables_json.return_value = "[]"
        mock_parsed.equations_json.return_value = "[]"
        mock_parsed.parser_version = "1.0"

        # psycopg is imported inside the function body, so patch it as a
        # builtins-level import via the module's __import__ path.
        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cur)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

        mock_psycopg = MagicMock()
        mock_psycopg.connect.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_psycopg.connect.return_value.__exit__ = MagicMock(return_value=False)

        with patch.object(converter._parser, "parse", return_value=mock_parsed) as mock_parse:
            with patch.dict("sys.modules", {"psycopg": mock_psycopg}):
                result = converter.ingest_to_fulltext(
                    html="<html>test</html>",
                    arxiv_id="2301.00001",
                    bibcode="2023Test..001A",
                )

        assert result is True
        mock_parse.assert_called_once_with("<html>test</html>")

    def test_blocks_production_dsn(self, tmp_path: Path) -> None:
        """ingest_to_fulltext must refuse to write to production."""
        cfg = ArxivLocalConfig(cache_dir=tmp_path, dsn="dbname=scix")
        converter = ArxivLocalConverter(cfg)

        with pytest.raises(ProductionGuardError):
            converter.ingest_to_fulltext(
                html="<html></html>",
                arxiv_id="2301.00001",
                bibcode="2023Test..001A",
            )
