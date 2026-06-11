"""Regression guards for scripts/diskann_quiet_window_rebuild.sh.

Bead: scix_experiments-12rp — reconcile the prod DiskANN rebuild-script DDL with
the benchmarked halfvec variant (scripts/build_streamingdiskann_variants.py V1)
and the halfvec rationale in CLAUDE.md.

The shell script is not unit-testable directly, so these tests assert the
content of the canonical DDL it emits — specifically that it indexes the
fixed-dimension halfvec(768) column (scannable) and never reintroduces the
(((embedding)::vector(768)) vector_cosine_ops) expression cast that built an
UNSCANNABLE index in prod (56h build, dropped 2026-06-11, attnum>0 on every
kNN scan).
"""

from __future__ import annotations

from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "diskann_quiet_window_rebuild.sh"


@pytest.fixture(scope="module")
def script_text() -> str:
    return SCRIPT.read_text()


@pytest.fixture(scope="module")
def script_code(script_text: str) -> str:
    """Script text with comment lines stripped, so DDL assertions ignore the
    explanatory comments that intentionally name the rejected forms."""
    return "\n".join(
        line for line in script_text.splitlines() if not line.lstrip().startswith("#")
    )


def test_script_exists() -> None:
    assert SCRIPT.is_file(), f"missing {SCRIPT}"


def test_canonical_ddl_uses_halfvec_column_and_opclass(script_text: str) -> None:
    # Matches the benchmarked V1 variant: USING diskann (<col> halfvec_cosine_ops).
    assert "embedding_hv halfvec_cosine_ops" in script_text
    assert 'HV_COL="embedding_hv"' in script_text


def test_no_unscannable_expression_cast(script_code: str) -> None:
    # No actual index-DDL line may use the float32 opclass or the expression
    # cast (comments and the capability-gate error message may name them to
    # explain why they're rejected; those are not DDL).
    ddl_lines = [
        line
        for line in script_code.splitlines()
        if "USING diskann" in line or "CREATE INDEX" in line
    ]
    assert ddl_lines, "expected at least one index-DDL line in the script"
    for line in ddl_lines:
        assert "vector_cosine_ops" not in line, f"DDL reintroduces float32 opclass: {line}"
        assert "(embedding)::vector(768)" not in line, f"DDL reintroduces expression cast: {line}"


def test_partial_predicate_is_indus_equality(script_text: str) -> None:
    assert "WHERE model_name='indus'" in script_text


def test_has_scannability_validation_gate(script_text: str) -> None:
    # The 50k-row scratch gate required by 12rp before any multi-hour build.
    assert "cmd_validate()" in script_text
    assert "SCANNABLE_OK" in script_text
    assert "enable_seqscan" in script_text


def test_build_paths_run_preflight_and_validate(script_text: str) -> None:
    # Both build entrypoints must gate on preflight + validate.
    build_idx = script_text.index("cmd_build()")
    coexist_idx = script_text.index("cmd_build_coexist()")
    build_body = script_text[build_idx:coexist_idx]
    coexist_body = script_text[coexist_idx:]
    for name, body in (("cmd_build", build_body), ("cmd_build_coexist", coexist_body)):
        assert "cmd_preflight" in body, f"{name} does not run cmd_preflight"
        assert "cmd_validate" in body, f"{name} does not run cmd_validate"


def test_opclass_capability_gate_present(script_text: str) -> None:
    # Refuses to build when pgvectorscale lacks a halfvec diskann opclass.
    assert "assert_diskann_halfvec_capable" in script_text
    assert "halfvec_cosine_ops" in script_text


def test_documents_halfvec_query_path_coupling(script_text: str) -> None:
    # The planner only picks the halfvec index on the SCIX_USE_HALFVEC=1 path.
    assert "SCIX_USE_HALFVEC=1" in script_text
