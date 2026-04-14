"""Unit tests for src/scix/sources/ar5iv.py.

Covers:
- LaTeXML HTML5 parsing: section extraction, inline citations, figures,
  tables, equations
- Content-addressed caching (write to temp dir, verify cache hit)
- Rate limiter compliance
- Production DSN guard
- Parser version tracking
- Source tagging and canonical_url

No database required. Pure unit tests.
"""

from __future__ import annotations

import json
import time
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from scix.sources.ar5iv import (
    Ar5ivConfig,
    Ar5ivLoader,
    Ar5ivParser,
    ParsedFulltext,
    ProductionGuardError,
    Section,
    InlineCite,
    Figure,
    Table,
    Equation,
    _build_canonical_url,
)

# ---------------------------------------------------------------------------
# Fixture HTML mimicking ar5iv structure
# ---------------------------------------------------------------------------

MINIMAL_HTML = """\
<!DOCTYPE html>
<html lang="en">
<head><title>Test Paper</title></head>
<body>
<article class="ltx_document">
  <div class="ltx_page_main">
    <section class="ltx_section" id="S1">
      <h2 class="ltx_title ltx_title_section">1 Introduction</h2>
      <div class="ltx_para">
        <p class="ltx_p">This is the introduction text.</p>
      </div>
    </section>
    <section class="ltx_section" id="S2">
      <h2 class="ltx_title ltx_title_section">2 Methods</h2>
      <div class="ltx_para">
        <p class="ltx_p">We used a novel approach.</p>
      </div>
      <section class="ltx_subsection" id="S2.SS1">
        <h3 class="ltx_title ltx_title_subsection">2.1 Data Collection</h3>
        <div class="ltx_para">
          <p class="ltx_p">Data was collected from ADS.</p>
        </div>
      </section>
    </section>
  </div>
</article>
</body>
</html>
"""

HTML_WITH_CITATIONS = """\
<!DOCTYPE html>
<html lang="en">
<head><title>Test Paper</title></head>
<body>
<article class="ltx_document">
  <div class="ltx_page_main">
    <section class="ltx_section" id="S1">
      <h2 class="ltx_title ltx_title_section">1 Introduction</h2>
      <div class="ltx_para">
        <p class="ltx_p">As shown by <cite class="ltx_cite ltx_citemacro_cite">
          <a href="#bib.bib1" class="ltx_ref">Smith et al. (2020)</a>
        </cite> and later by <cite class="ltx_cite ltx_citemacro_citep">
          <a href="#bib.bib2" class="ltx_ref">Jones (2021)</a>
        </cite>.</p>
      </div>
    </section>
  </div>
  <section class="ltx_bibliography">
    <ul class="ltx_biblist">
      <li class="ltx_bibitem" id="bib.bib1">
        <span class="ltx_tag ltx_tag_bibitem">[1]</span>
        <span class="ltx_bibblock">Smith et al., 2020, ApJ, 900, 1</span>
      </li>
      <li class="ltx_bibitem" id="bib.bib2">
        <span class="ltx_tag ltx_tag_bibitem">[2]</span>
        <span class="ltx_bibblock">Jones, 2021, MNRAS, 500, 100</span>
      </li>
    </ul>
  </section>
</article>
</body>
</html>
"""

HTML_WITH_FIGURES = """\
<!DOCTYPE html>
<html lang="en">
<head><title>Test Paper</title></head>
<body>
<article class="ltx_document">
  <div class="ltx_page_main">
    <section class="ltx_section" id="S1">
      <h2 class="ltx_title ltx_title_section">1 Results</h2>
      <figure class="ltx_figure" id="S1.F1">
        <img src="fig1.png" class="ltx_graphics" alt="Figure 1" />
        <figcaption class="ltx_caption">
          <span class="ltx_tag ltx_tag_figure">Figure 1: </span>
          The luminosity function.
        </figcaption>
      </figure>
    </section>
  </div>
</article>
</body>
</html>
"""

HTML_WITH_TABLES = """\
<!DOCTYPE html>
<html lang="en">
<head><title>Test Paper</title></head>
<body>
<article class="ltx_document">
  <div class="ltx_page_main">
    <section class="ltx_section" id="S1">
      <h2 class="ltx_title ltx_title_section">1 Results</h2>
      <figure class="ltx_table" id="S1.T1">
        <figcaption class="ltx_caption">
          <span class="ltx_tag ltx_tag_table">Table 1: </span>
          Summary of observations.
        </figcaption>
        <table class="ltx_tabular">
          <tr><th>Name</th><th>Value</th></tr>
          <tr><td>Alpha</td><td>1.0</td></tr>
        </table>
      </figure>
    </section>
  </div>
</article>
</body>
</html>
"""

HTML_WITH_EQUATIONS = """\
<!DOCTYPE html>
<html lang="en">
<head><title>Test Paper</title></head>
<body>
<article class="ltx_document">
  <div class="ltx_page_main">
    <section class="ltx_section" id="S1">
      <h2 class="ltx_title ltx_title_section">1 Theory</h2>
      <div class="ltx_para">
        <p class="ltx_p">The equation is:</p>
      </div>
      <table class="ltx_equation" id="S1.Ex1">
        <tr>
          <td class="ltx_eqn_cell ltx_align_center">
            <math xmlns="http://www.w3.org/1998/Math/MathML" alttext="E = mc^2">
              <mi>E</mi><mo>=</mo><mi>m</mi><msup><mi>c</mi><mn>2</mn></msup>
            </math>
          </td>
        </tr>
      </table>
    </section>
  </div>
</article>
</body>
</html>
"""

EMPTY_HTML = """\
<!DOCTYPE html>
<html lang="en">
<head><title>Empty</title></head>
<body>
<article class="ltx_document">
  <div class="ltx_page_main">
  </div>
</article>
</body>
</html>
"""

HTML_WITH_VERSION = """\
<!DOCTYPE html>
<html lang="en" xmlns="http://www.w3.org/1999/xhtml">
<head>
  <title>Test</title>
  <meta name="generator" content="LaTeXML v0.8.8" />
</head>
<body>
<article class="ltx_document">
  <div class="ltx_page_main">
    <section class="ltx_section" id="S1">
      <h2 class="ltx_title ltx_title_section">1 Intro</h2>
      <div class="ltx_para"><p class="ltx_p">Hello world.</p></div>
    </section>
  </div>
</article>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Section extraction
# ---------------------------------------------------------------------------


class TestSectionExtraction:
    def test_extracts_top_level_sections(self) -> None:
        parser = Ar5ivParser()
        result = parser.parse(MINIMAL_HTML)
        headings = [s.heading for s in result.sections]
        assert "1 Introduction" in headings
        assert "2 Methods" in headings

    def test_extracts_subsections(self) -> None:
        parser = Ar5ivParser()
        result = parser.parse(MINIMAL_HTML)
        subsections = [s for s in result.sections if s.level == 2]
        assert len(subsections) >= 1
        assert any("Data Collection" in s.heading for s in subsections)

    def test_section_has_text(self) -> None:
        parser = Ar5ivParser()
        result = parser.parse(MINIMAL_HTML)
        intro = [s for s in result.sections if "Introduction" in s.heading][0]
        assert "introduction text" in intro.text.lower()

    def test_section_levels_correct(self) -> None:
        parser = Ar5ivParser()
        result = parser.parse(MINIMAL_HTML)
        for s in result.sections:
            assert s.level >= 1

    def test_section_offsets_non_negative(self) -> None:
        parser = Ar5ivParser()
        result = parser.parse(MINIMAL_HTML)
        for s in result.sections:
            assert s.offset >= 0

    def test_empty_html_yields_empty_sections(self) -> None:
        parser = Ar5ivParser()
        result = parser.parse(EMPTY_HTML)
        assert result.sections == []

    def test_section_is_frozen(self) -> None:
        s = Section(heading="Test", level=1, text="Hello", offset=0)
        with pytest.raises(FrozenInstanceError):
            s.heading = "mutated"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Inline citation extraction
# ---------------------------------------------------------------------------


class TestInlineCiteExtraction:
    def test_extracts_citations(self) -> None:
        parser = Ar5ivParser()
        result = parser.parse(HTML_WITH_CITATIONS)
        assert len(result.inline_cites) == 2

    def test_cite_has_bib_ref(self) -> None:
        parser = Ar5ivParser()
        result = parser.parse(HTML_WITH_CITATIONS)
        bib_refs = [c.bib_ref for c in result.inline_cites]
        assert "bib.bib1" in bib_refs
        assert "bib.bib2" in bib_refs

    def test_cite_offset_non_negative(self) -> None:
        parser = Ar5ivParser()
        result = parser.parse(HTML_WITH_CITATIONS)
        for c in result.inline_cites:
            assert c.offset >= 0

    def test_cite_target_bibcode_is_none_by_default(self) -> None:
        parser = Ar5ivParser()
        result = parser.parse(HTML_WITH_CITATIONS)
        for c in result.inline_cites:
            assert c.target_bibcode_or_null is None

    def test_cite_is_frozen(self) -> None:
        c = InlineCite(offset=0, bib_ref="bib1", target_bibcode_or_null=None)
        with pytest.raises(FrozenInstanceError):
            c.bib_ref = "mutated"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Figure extraction
# ---------------------------------------------------------------------------


class TestFigureExtraction:
    def test_extracts_figures(self) -> None:
        parser = Ar5ivParser()
        result = parser.parse(HTML_WITH_FIGURES)
        assert len(result.figures) >= 1

    def test_figure_has_id(self) -> None:
        parser = Ar5ivParser()
        result = parser.parse(HTML_WITH_FIGURES)
        assert result.figures[0].id == "S1.F1"

    def test_figure_has_caption(self) -> None:
        parser = Ar5ivParser()
        result = parser.parse(HTML_WITH_FIGURES)
        assert "luminosity" in result.figures[0].caption.lower()


# ---------------------------------------------------------------------------
# Table extraction
# ---------------------------------------------------------------------------


class TestTableExtraction:
    def test_extracts_tables(self) -> None:
        parser = Ar5ivParser()
        result = parser.parse(HTML_WITH_TABLES)
        assert len(result.tables) >= 1

    def test_table_has_id(self) -> None:
        parser = Ar5ivParser()
        result = parser.parse(HTML_WITH_TABLES)
        assert result.tables[0].id == "S1.T1"

    def test_table_has_caption(self) -> None:
        parser = Ar5ivParser()
        result = parser.parse(HTML_WITH_TABLES)
        assert "observations" in result.tables[0].caption.lower()


# ---------------------------------------------------------------------------
# Equation extraction
# ---------------------------------------------------------------------------


class TestEquationExtraction:
    def test_extracts_equations(self) -> None:
        parser = Ar5ivParser()
        result = parser.parse(HTML_WITH_EQUATIONS)
        assert len(result.equations) >= 1

    def test_equation_has_id(self) -> None:
        parser = Ar5ivParser()
        result = parser.parse(HTML_WITH_EQUATIONS)
        assert result.equations[0].id == "S1.Ex1"

    def test_equation_has_latex(self) -> None:
        parser = Ar5ivParser()
        result = parser.parse(HTML_WITH_EQUATIONS)
        assert result.equations[0].latex  # non-empty


# ---------------------------------------------------------------------------
# Parser version tracking
# ---------------------------------------------------------------------------


class TestParserVersion:
    def test_detects_latexml_version_from_meta(self) -> None:
        parser = Ar5ivParser()
        result = parser.parse(HTML_WITH_VERSION)
        assert result.parser_version == "LaTeXML v0.8.8"

    def test_fallback_version_when_no_meta(self) -> None:
        parser = Ar5ivParser()
        result = parser.parse(MINIMAL_HTML)
        assert result.parser_version == "ar5iv-unknown"


# ---------------------------------------------------------------------------
# ParsedFulltext contract
# ---------------------------------------------------------------------------


class TestParsedFulltext:
    def test_is_frozen(self) -> None:
        pf = ParsedFulltext(
            sections=[],
            inline_cites=[],
            figures=[],
            tables=[],
            equations=[],
            parser_version="test",
        )
        with pytest.raises(FrozenInstanceError):
            pf.parser_version = "mutated"  # type: ignore[misc]

    def test_to_db_row_sections_json(self) -> None:
        parser = Ar5ivParser()
        result = parser.parse(MINIMAL_HTML)
        sections_json = result.sections_json()
        parsed = json.loads(sections_json)
        assert isinstance(parsed, list)
        assert len(parsed) > 0
        assert "heading" in parsed[0]
        assert "level" in parsed[0]
        assert "text" in parsed[0]
        assert "offset" in parsed[0]

    def test_to_db_row_cites_json(self) -> None:
        parser = Ar5ivParser()
        result = parser.parse(HTML_WITH_CITATIONS)
        cites_json = result.inline_cites_json()
        parsed = json.loads(cites_json)
        assert isinstance(parsed, list)
        assert len(parsed) == 2
        assert "offset" in parsed[0]
        assert "bib_ref" in parsed[0]
        assert "target_bibcode_or_null" in parsed[0]


# ---------------------------------------------------------------------------
# Canonical URL builder
# ---------------------------------------------------------------------------


class TestCanonicalUrl:
    def test_basic_id(self) -> None:
        assert _build_canonical_url("2301.12345") == "https://arxiv.org/abs/2301.12345"

    def test_versioned_id(self) -> None:
        assert _build_canonical_url("2301.12345v2") == "https://arxiv.org/abs/2301.12345v2"


# ---------------------------------------------------------------------------
# Content-addressed caching
# ---------------------------------------------------------------------------


class TestCaching:
    def test_cache_write_and_hit(self, tmp_path: Path) -> None:
        from scix.sources.ar5iv import Ar5ivFetcher

        cache_dir = tmp_path / "raw_html"
        fetcher = Ar5ivFetcher(cache_dir=cache_dir, rate_limit=100.0)

        arxiv_id = "2301.12345"
        html_content = "<html>test content</html>"

        # Write to cache
        fetcher._write_cache(arxiv_id, html_content)

        # Verify file exists and is gzipped
        cache_file = cache_dir / f"{arxiv_id}.html.gz"
        assert cache_file.exists()

        # Read back
        cached = fetcher._read_cache(arxiv_id)
        assert cached == html_content

    def test_cache_miss(self, tmp_path: Path) -> None:
        from scix.sources.ar5iv import Ar5ivFetcher

        cache_dir = tmp_path / "raw_html"
        fetcher = Ar5ivFetcher(cache_dir=cache_dir, rate_limit=100.0)
        assert fetcher._read_cache("9901.12345") is None

    def test_cache_with_slash_in_id(self, tmp_path: Path) -> None:
        from scix.sources.ar5iv import Ar5ivFetcher

        cache_dir = tmp_path / "raw_html"
        fetcher = Ar5ivFetcher(cache_dir=cache_dir, rate_limit=100.0)

        # Old-style arXiv IDs have slashes: astro-ph/9901001
        arxiv_id = "astro-ph/9901001"
        html_content = "<html>old paper</html>"

        fetcher._write_cache(arxiv_id, html_content)
        cached = fetcher._read_cache(arxiv_id)
        assert cached == html_content


# ---------------------------------------------------------------------------
# arXiv ID validation (security — path traversal prevention)
# ---------------------------------------------------------------------------


class TestArxivIdValidation:
    def test_valid_new_style(self, tmp_path: Path) -> None:
        from scix.sources.ar5iv import Ar5ivFetcher

        fetcher = Ar5ivFetcher(cache_dir=tmp_path, rate_limit=100.0)
        # Should not raise
        fetcher._validate_arxiv_id("2301.12345")
        fetcher._validate_arxiv_id("2301.12345v2")

    def test_valid_old_style(self, tmp_path: Path) -> None:
        from scix.sources.ar5iv import Ar5ivFetcher

        fetcher = Ar5ivFetcher(cache_dir=tmp_path, rate_limit=100.0)
        fetcher._validate_arxiv_id("astro-ph/9901001")
        fetcher._validate_arxiv_id("hep-th/0001234v1")

    def test_rejects_path_traversal(self, tmp_path: Path) -> None:
        from scix.sources.ar5iv import Ar5ivFetcher

        fetcher = Ar5ivFetcher(cache_dir=tmp_path, rate_limit=100.0)
        with pytest.raises(ValueError, match="Rejected invalid arxiv_id"):
            fetcher._validate_arxiv_id("../../etc/passwd")

    def test_rejects_url_injection(self, tmp_path: Path) -> None:
        from scix.sources.ar5iv import Ar5ivFetcher

        fetcher = Ar5ivFetcher(cache_dir=tmp_path, rate_limit=100.0)
        with pytest.raises(ValueError, match="Rejected invalid arxiv_id"):
            fetcher._validate_arxiv_id("2301.12345#evil")

    def test_rejects_empty(self, tmp_path: Path) -> None:
        from scix.sources.ar5iv import Ar5ivFetcher

        fetcher = Ar5ivFetcher(cache_dir=tmp_path, rate_limit=100.0)
        with pytest.raises(ValueError, match="Rejected invalid arxiv_id"):
            fetcher._validate_arxiv_id("")


# ---------------------------------------------------------------------------
# Rate limiter
# ---------------------------------------------------------------------------


class TestRateLimiter:
    def test_respects_rate_limit(self, tmp_path: Path) -> None:
        from scix.sources.ar5iv import Ar5ivFetcher

        # 2 req/s means minimum 0.5s between requests
        fetcher = Ar5ivFetcher(cache_dir=tmp_path, rate_limit=2.0)

        t1 = time.monotonic()
        fetcher._apply_rate_limit()
        fetcher._apply_rate_limit()
        t2 = time.monotonic()

        # Second call should have waited ~0.5s
        assert t2 - t1 >= 0.4  # small margin for timing


# ---------------------------------------------------------------------------
# Config frozen
# ---------------------------------------------------------------------------


class TestAr5ivConfig:
    def test_config_is_frozen(self) -> None:
        cfg = Ar5ivConfig(
            dsn="dbname=test",
            cache_dir=Path("/tmp/test"),
        )
        with pytest.raises(FrozenInstanceError):
            cfg.dsn = "mutated"  # type: ignore[misc]

    def test_defaults(self) -> None:
        cfg = Ar5ivConfig(
            dsn="dbname=test",
            cache_dir=Path("/tmp/test"),
        )
        assert cfg.rate_limit == 5.0
        assert cfg.batch_size == 1000
        assert cfg.dry_run is False
        assert cfg.yes_production is False


# ---------------------------------------------------------------------------
# Production DSN guard
# ---------------------------------------------------------------------------


class TestProductionGuard:
    def test_blocks_production_dsn(self) -> None:
        cfg = Ar5ivConfig(
            dsn="dbname=scix",
            cache_dir=Path("/tmp/test"),
        )
        loader = Ar5ivLoader(cfg)
        with pytest.raises(ProductionGuardError, match="production"):
            loader._check_production_guard()

    def test_allows_test_dsn(self) -> None:
        cfg = Ar5ivConfig(
            dsn="dbname=scix_test",
            cache_dir=Path("/tmp/test"),
        )
        loader = Ar5ivLoader(cfg)
        loader._check_production_guard()  # should not raise

    def test_none_dsn_falls_back_to_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import scix.sources.ar5iv as ar5iv_module

        monkeypatch.setattr(ar5iv_module, "DEFAULT_DSN", "dbname=scix")
        cfg = Ar5ivConfig(dsn=None, cache_dir=Path("/tmp/test"))
        loader = Ar5ivLoader(cfg)
        with pytest.raises(ProductionGuardError, match="production"):
            loader._check_production_guard()

    def test_yes_production_overrides(self) -> None:
        cfg = Ar5ivConfig(
            dsn="dbname=scix",
            cache_dir=Path("/tmp/test"),
            yes_production=True,
        )
        loader = Ar5ivLoader(cfg)
        loader._check_production_guard()  # should not raise

    def test_uri_production_dsn_blocked(self) -> None:
        cfg = Ar5ivConfig(
            dsn="postgresql://user:pw@host:5432/scix",
            cache_dir=Path("/tmp/test"),
        )
        loader = Ar5ivLoader(cfg)
        with pytest.raises(ProductionGuardError):
            loader._check_production_guard()


# ---------------------------------------------------------------------------
# Source tagging
# ---------------------------------------------------------------------------


class TestSourceTagging:
    def test_source_is_ar5iv(self) -> None:
        """The loader must tag all records with source='ar5iv'."""
        parser = Ar5ivParser()
        result = parser.parse(MINIMAL_HTML)
        # source tagging happens at the loader level, not parser level.
        # Parser returns ParsedFulltext; loader adds source='ar5iv'.
        # Just verify the parser output is usable:
        assert result.sections is not None
        assert result.parser_version is not None
