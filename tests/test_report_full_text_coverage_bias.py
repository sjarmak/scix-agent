"""Unit tests for ``scripts/report_full_text_coverage_bias.py``.

Covers:

* ``kl_divergence`` numerical correctness on known inputs (identity,
  asymmetry, smoothing of zeros, length mismatch).
* ``rows_to_distributions`` projection from DistributionRow.
* ``build_facet_payload`` produces the documented JSON shape with both
  absolute counts and ``kl_divergence_vs_corpus_prior`` per facet.
* ``run_report(dry_run=True)`` writes a JSON file matching the
  acceptance schema and refreshes the agent-guidance docs section
  in-place (idempotently).

No database connection is required — synthetic distributions are used
throughout. Matplotlib is not imported.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

# Match the import convention used by tests/test_coverage_bias.py.
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "scripts"))
sys.path.insert(0, str(_REPO_ROOT / "src"))

from coverage_bias_analysis import DistributionRow  # noqa: E402

from report_full_text_coverage_bias import (  # noqa: E402
    build_facet_payload,
    build_payload,
    kl_divergence,
    render_agent_guidance,
    rows_to_distributions,
    run_report,
    synthetic_facets,
    upsert_agent_guidance_section,
)


# ---------------------------------------------------------------------------
# kl_divergence
# ---------------------------------------------------------------------------


class TestKLDivergence:
    def test_identity_is_zero(self) -> None:
        assert kl_divergence([0.5, 0.5], [0.5, 0.5]) == pytest.approx(0.0, abs=1e-9)

    def test_uniform_vs_skewed_matches_manual(self) -> None:
        # P = [0.5, 0.5], Q = [0.25, 0.75]
        # D = 0.5*ln(0.5/0.25) + 0.5*ln(0.5/0.75) = 0.5*ln(2) + 0.5*ln(2/3)
        expected = 0.5 * math.log(2.0) + 0.5 * math.log(2.0 / 3.0)
        got = kl_divergence([0.5, 0.5], [0.25, 0.75])
        assert got == pytest.approx(expected, rel=1e-6)

    def test_concentrated_vs_uniform_is_large(self) -> None:
        # P puts all mass on bucket 0, Q uniform — large divergence.
        assert kl_divergence([1.0, 0.0], [0.5, 0.5]) > 0.5

    def test_asymmetry(self) -> None:
        a = kl_divergence([0.1, 0.9], [0.9, 0.1])
        b = kl_divergence([0.9, 0.1], [0.1, 0.9])
        assert a == pytest.approx(b, rel=1e-9)  # symmetric only because mirrored
        # Genuinely-asymmetric example
        a2 = kl_divergence([0.5, 0.5], [0.9, 0.1])
        b2 = kl_divergence([0.9, 0.1], [0.5, 0.5])
        assert a2 != pytest.approx(b2, rel=1e-3)

    def test_smoothing_handles_zeros_in_p(self) -> None:
        # Zero in P contributes zero (0 * log(0/q) = 0) — must not raise.
        got = kl_divergence([0.0, 1.0], [0.5, 0.5])
        assert math.isfinite(got)
        assert got > 0.0

    def test_smoothing_handles_zeros_in_q(self) -> None:
        # Zero in Q where P has mass would diverge to +inf without smoothing;
        # eps>0 keeps it finite.
        got = kl_divergence([0.5, 0.5], [1.0, 0.0])
        assert math.isfinite(got)
        assert got > 0.0

    def test_zero_eps_diverges_when_q_is_zero(self) -> None:
        # Without smoothing, q=0 and p>0 must yield +inf.
        got = kl_divergence([0.5, 0.5], [1.0, 0.0], eps=0.0)
        assert math.isinf(got)

    def test_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError):
            kl_divergence([0.5, 0.5], [0.25, 0.25, 0.5])

    def test_empty_inputs_zero(self) -> None:
        assert kl_divergence([], []) == 0.0

    def test_unnormalised_counts_are_normalised(self) -> None:
        # Counts (not probabilities) should be normalised internally.
        norm = kl_divergence([1.0, 1.0], [1.0, 1.0])
        unnorm = kl_divergence([10.0, 10.0], [50.0, 50.0])
        assert norm == pytest.approx(unnorm, abs=1e-9)


# ---------------------------------------------------------------------------
# rows_to_distributions / build_facet_payload
# ---------------------------------------------------------------------------


class TestFacetPayload:
    @pytest.fixture()
    def rows(self) -> list[DistributionRow]:
        return [
            DistributionRow("A", total=100, with_body=90, without_body=10, pct_with_body=90.0),
            DistributionRow("B", total=100, with_body=10, without_body=90, pct_with_body=10.0),
        ]

    def test_rows_to_distributions_extracts_counts(self, rows: list[DistributionRow]) -> None:
        p, q = rows_to_distributions(rows)
        assert p == [90.0, 10.0]
        assert q == [100.0, 100.0]

    def test_build_facet_payload_has_required_keys(self, rows: list[DistributionRow]) -> None:
        payload = build_facet_payload(rows)
        assert "kl_divergence_vs_corpus_prior" in payload
        assert "row_count" in payload
        assert "rows" in payload
        assert payload["row_count"] == 2
        for r in payload["rows"]:
            assert {
                "label",
                "total",
                "with_body",
                "without_body",
                "pct_with_body",
                "p_fulltext",
                "q_corpus",
                "ratio_p_over_q",
            } <= set(r.keys())

    def test_build_facet_payload_kl_positive_when_skewed(
        self, rows: list[DistributionRow]
    ) -> None:
        # P heavily skewed to A relative to uniform Q — KL > 0.
        payload = build_facet_payload(rows)
        assert payload["kl_divergence_vs_corpus_prior"] > 0.0

    def test_build_facet_payload_kl_zero_when_aligned(self) -> None:
        # P and Q identical — KL ~ 0 (modulo eps smoothing rounding).
        rows = [
            DistributionRow("A", total=100, with_body=50, without_body=50, pct_with_body=50.0),
            DistributionRow("B", total=200, with_body=100, without_body=100, pct_with_body=50.0),
        ]
        payload = build_facet_payload(rows)
        assert payload["kl_divergence_vs_corpus_prior"] == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# build_payload schema and dry-run end-to-end
# ---------------------------------------------------------------------------


REQUIRED_TOP_LEVEL_KEYS = {
    "generated_at",
    "dsn_redacted",
    "corpus_total",
    "fulltext_total",
    "fulltext_pct",
    "kl_divergence_basis",
    "facets",
}

REQUIRED_FACET_NAMES = {"arxiv_class", "year", "citation_bucket", "bibstem"}


class TestBuildPayloadSchema:
    def test_synthetic_payload_top_level_shape(self) -> None:
        facets = synthetic_facets()
        citation_rows = facets["citation_bucket"]
        payload = build_payload(
            corpus_total=sum(r.total for r in citation_rows),
            fulltext_total=sum(r.with_body for r in citation_rows),
            facet_rows_by_name=dict(facets),
            dsn_redacted="synthetic://test",
        )
        assert REQUIRED_TOP_LEVEL_KEYS <= payload.keys()
        assert REQUIRED_FACET_NAMES <= payload["facets"].keys()
        for name, facet in payload["facets"].items():
            assert "kl_divergence_vs_corpus_prior" in facet, name
            assert "rows" in facet, name
            assert facet["rows"], f"{name} must have non-empty rows"
            for r in facet["rows"]:
                # Both absolute counts AND KL-derived numbers per row.
                assert "total" in r and "with_body" in r and "without_body" in r
                assert "p_fulltext" in r and "q_corpus" in r

    def test_skipped_facet_is_omitted(self) -> None:
        facets = synthetic_facets()
        facets["community_semantic_medium"] = None  # type: ignore[assignment]
        payload = build_payload(
            corpus_total=100,
            fulltext_total=46,
            facet_rows_by_name=dict(facets),
            dsn_redacted="synthetic://test",
        )
        assert "community_semantic_medium" not in payload["facets"]


# ---------------------------------------------------------------------------
# run_report --dry-run end-to-end (writes JSON, refreshes docs)
# ---------------------------------------------------------------------------


class TestDryRunEndToEnd:
    def test_dry_run_writes_valid_json(self, tmp_path: Path) -> None:
        json_out = tmp_path / "out.json"
        payload = run_report(json_out=json_out, docs_path=None, dry_run=True)
        assert json_out.exists()

        on_disk = json.loads(json_out.read_text())
        assert REQUIRED_TOP_LEVEL_KEYS <= on_disk.keys()
        assert REQUIRED_FACET_NAMES <= on_disk["facets"].keys()
        # Returned object equals on-disk object.
        assert on_disk["corpus_total"] == payload["corpus_total"]

    def test_dry_run_upserts_docs_section_idempotently(self, tmp_path: Path) -> None:
        docs = tmp_path / "doc.md"
        docs.write_text("# Existing Title\n\nExisting content paragraph.\n", encoding="utf-8")

        json_out = tmp_path / "out.json"
        run_report(json_out=json_out, docs_path=docs, dry_run=True)
        first = docs.read_text()

        # Second run must replace, not duplicate.
        run_report(json_out=json_out, docs_path=docs, dry_run=True)
        second = docs.read_text()

        assert first.count("<!-- agent-guidance:start -->") == 1
        assert second.count("<!-- agent-guidance:start -->") == 1
        # Existing content is preserved on both runs.
        assert "Existing content paragraph." in second

    def test_agent_guidance_has_three_safe_and_three_unsafe_examples(
        self, tmp_path: Path
    ) -> None:
        json_out = tmp_path / "out.json"
        payload = run_report(json_out=json_out, docs_path=None, dry_run=True)
        section = render_agent_guidance(payload, json_out)

        # Heading present
        assert (
            "## Agent guidance: safe vs unsafe queries on the full-text cohort"
            in section
        )
        # Locate the safe / unsafe sub-sections by their h3 markers and count
        # numbered bullets within each.
        safe_idx = section.index("### Safe queries to restrict to the full-text cohort")
        unsafe_idx = section.index("### Unsafe queries (filtering to full-text would bias")
        safe_block = section[safe_idx:unsafe_idx]
        unsafe_block = section[unsafe_idx:]

        # Each block must enumerate 1., 2., 3. bullets (at minimum).
        for bullet in ("1. ", "2. ", "3. "):
            assert bullet in safe_block, f"safe block missing {bullet!r}"
            assert bullet in unsafe_block, f"unsafe block missing {bullet!r}"

    def test_upsert_refuses_when_docs_path_missing(self, tmp_path: Path) -> None:
        missing = tmp_path / "does_not_exist.md"
        with pytest.raises(FileNotFoundError):
            upsert_agent_guidance_section(missing, "## section\n")
