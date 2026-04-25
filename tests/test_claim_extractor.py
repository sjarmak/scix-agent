"""Tests for the regex-first quantitative-claim extractor (M4).

Covers:

* Per-uncertainty-form unit tests (``±``, ``+/-``, ``\\pm``, asymmetric
  ``^{+a}_{-b}``).
* Quantity-canonicalisation across surface variants.
* Per-quantity recall (>= 0.80) on the 50-snippet cosmology gold fixture
  for ``H0``, ``Omega_m``, ``sigma_8``.
* The ``llm_disambiguate`` hook raising ``NotImplementedError``.
* ``to_payload`` shape (PRD-required keys present per claim).
* Mocked psycopg insert into ``staging.extractions`` with
  ``extraction_type='quant_claim'``.

No live database is required — the ``insert_claims`` test mocks psycopg.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
SCRIPTS = REPO_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from scix.claim_extractor import (  # noqa: E402
    EXTRACTION_TYPE,
    EXTRACTION_VERSION,
    ClaimSpan,
    claim_to_dict,
    extract_claims,
    llm_disambiguate,
    to_payload,
)

# ---------------------------------------------------------------------------
# Per-uncertainty-form unit tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text,expected_value,expected_unc",
    [
        ("Omega_m = 0.315 ± 0.007", 0.315, 0.007),  # unicode ±
        ("H0 = 73.4 +/- 1.1 km/s/Mpc", 73.4, 1.1),  # ASCII +/-
        ("sigma_8 = 0.811 \\pm 0.006", 0.811, 0.006),  # LaTeX \pm
        ("\\Omega_m = 0.305 \\pm 0.012", 0.305, 0.012),  # LaTeX surface + \pm
        ("Ω_m = 0.315 ± 0.008", 0.315, 0.008),  # unicode quantity + ±
    ],
)
def test_extract_symmetric_uncertainty_forms(
    text: str, expected_value: float, expected_unc: float
) -> None:
    claims = extract_claims(text)
    assert len(claims) == 1
    c = claims[0]
    assert math.isclose(c.value, expected_value, rel_tol=1e-9)
    assert c.uncertainty is not None
    assert math.isclose(c.uncertainty, expected_unc, rel_tol=1e-9)


def test_extract_asymmetric_uncertainty() -> None:
    text = "H_0 = 67.4 ^{+1.2}_{-1.5} km/s/Mpc"
    claims = extract_claims(text)
    assert len(claims) == 1
    c = claims[0]
    assert c.quantity == "H0"
    assert math.isclose(c.value, 67.4)
    assert c.uncertainty_pos == 1.2
    assert c.uncertainty_neg == 1.5
    # Symmetric summary uncertainty is the average of pos / neg magnitudes.
    assert math.isclose(c.uncertainty, (1.2 + 1.5) / 2.0)
    assert c.unit == "km/s/Mpc"


def test_extract_value_only_no_uncertainty() -> None:
    text = "We adopt H0 = 70 km/s/Mpc throughout the paper."
    claims = extract_claims(text)
    assert len(claims) == 1
    c = claims[0]
    assert c.quantity == "H0"
    assert c.value == 70.0
    assert c.uncertainty is None
    assert c.unit == "km/s/Mpc"


def test_extract_returns_empty_for_empty_input() -> None:
    assert extract_claims("") == []
    assert extract_claims("This sentence has no measurements at all.") == []


def test_extract_handles_scientific_notation() -> None:
    text = "We find Omega_b = 4.93e-2 \\pm 6.0e-4 from the joint fit."
    claims = extract_claims(text)
    assert len(claims) == 1
    c = claims[0]
    assert c.quantity == "Omega_b"
    assert math.isclose(c.value, 0.0493, rel_tol=1e-6)
    assert c.uncertainty is not None
    assert math.isclose(c.uncertainty, 0.00060, rel_tol=1e-6)


def test_extract_multiple_claims_in_one_body() -> None:
    body = (
        "We measure H0 = 73.4 +/- 1.1 km/s/Mpc and Omega_m = 0.298 \\pm 0.022 "
        "in the joint Planck+BAO fit."
    )
    claims = extract_claims(body)
    quantities = sorted(c.quantity for c in claims)
    assert quantities == ["H0", "Omega_m"]


# ---------------------------------------------------------------------------
# Quantity canonicalisation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "surface,canonical",
    [
        ("H0", "H0"),
        ("H_0", "H0"),
        ("H_{0}", "H0"),
        ("Omega_m", "Omega_m"),
        ("\\Omega_m", "Omega_m"),
        ("\\Omega_{m}", "Omega_m"),
        ("Omega_M", "Omega_m"),
        ("Ω_m", "Omega_m"),
        ("sigma_8", "sigma_8"),
        ("sigma8", "sigma_8"),
        ("\\sigma_8", "sigma_8"),
        ("σ_8", "sigma_8"),
        ("Omega_Lambda", "Omega_Lambda"),
        ("\\Omega_\\Lambda", "Omega_Lambda"),
    ],
)
def test_quantity_canonicalisation(surface: str, canonical: str) -> None:
    text = f"{surface} = 0.5 \\pm 0.01"
    claims = extract_claims(text)
    assert any(c.quantity == canonical for c in claims), (
        f"surface {surface!r} did not canonicalise to {canonical!r}; got "
        f"{[c.quantity for c in claims]}"
    )


# ---------------------------------------------------------------------------
# Recall on the 50-snippet cosmology fixture
# ---------------------------------------------------------------------------


FIXTURE_PATH = REPO_ROOT / "tests" / "fixtures" / "quant_claims_cosmology_50.jsonl"


def _load_fixture() -> list[dict]:
    rows: list[dict] = []
    with FIXTURE_PATH.open() as fh:
        for ln in fh:
            ln = ln.strip()
            if not ln:
                continue
            rows.append(json.loads(ln))
    return rows


def test_fixture_has_exactly_50_lines() -> None:
    rows = _load_fixture()
    assert len(rows) == 50, f"fixture must have exactly 50 lines, got {len(rows)}"


def test_fixture_schema_is_consistent() -> None:
    rows = _load_fixture()
    for i, row in enumerate(rows):
        assert "text" in row, f"row {i} missing 'text'"
        assert "expected" in row, f"row {i} missing 'expected'"
        exp = row["expected"]
        for key in ("quantity", "value", "uncertainty", "unit"):
            assert key in exp, f"row {i} expected missing key {key!r}"


def _claim_matches_expected(claim: ClaimSpan, expected: dict) -> bool:
    """A claim is a recall hit if quantity matches and value is close."""
    if claim.quantity != expected["quantity"]:
        return False
    return math.isclose(claim.value, float(expected["value"]), rel_tol=1e-3, abs_tol=1e-6)


def test_recall_per_quantity_on_cosmology_fixture() -> None:
    """PRD acceptance: per-quantity recall >= 0.80 on H0, Omega_m, sigma_8."""
    rows = _load_fixture()
    # Group by canonical quantity.
    by_q: dict[str, list[dict]] = {}
    for row in rows:
        by_q.setdefault(row["expected"]["quantity"], []).append(row)

    recall_thresholds = {"H0": 0.80, "Omega_m": 0.80, "sigma_8": 0.80}
    failures: list[str] = []
    for quantity, threshold in recall_thresholds.items():
        examples = by_q.get(quantity, [])
        assert examples, f"fixture has no examples for {quantity!r}"
        hits = 0
        for ex in examples:
            claims = extract_claims(ex["text"])
            if any(_claim_matches_expected(c, ex["expected"]) for c in claims):
                hits += 1
        recall = hits / len(examples)
        if recall < threshold:
            failures.append(
                f"{quantity}: recall {recall:.2f} < {threshold:.2f} "
                f"({hits}/{len(examples)})"
            )
    assert not failures, "Per-quantity recall below PRD threshold:\n  " + "\n  ".join(
        failures
    )


# ---------------------------------------------------------------------------
# Hook + payload + dict helpers
# ---------------------------------------------------------------------------


def test_llm_disambiguate_raises_not_implemented() -> None:
    span = ClaimSpan(
        quantity="H0",
        value=70.0,
        uncertainty=None,
        unit=None,
        span=(0, 7),
    )
    with pytest.raises(NotImplementedError):
        llm_disambiguate(span)


def test_to_payload_contains_required_keys_per_claim() -> None:
    body = "H0 = 73.4 +/- 1.1 km/s/Mpc and Omega_m = 0.30 \\pm 0.02"
    claims = extract_claims(body)
    payload = to_payload(claims)
    assert payload["extraction_type"] == EXTRACTION_TYPE
    assert payload["extraction_version"] == EXTRACTION_VERSION
    assert isinstance(payload["claims"], list)
    assert len(payload["claims"]) == len(claims) == 2
    for c in payload["claims"]:
        # PRD-required keys
        for key in ("quantity", "value", "uncertainty", "unit", "span"):
            assert key in c, f"missing required payload key: {key!r}"
        # span is a JSON-serialisable list of two ints
        assert isinstance(c["span"], list)
        assert len(c["span"]) == 2
        assert all(isinstance(x, int) for x in c["span"])


def test_payload_is_json_serialisable() -> None:
    body = "H_0 = 67.4 ^{+1.2}_{-1.5} km/s/Mpc"
    payload = to_payload(extract_claims(body))
    # Must round-trip through json.dumps without raising.
    s = json.dumps(payload)
    back = json.loads(s)
    assert back["claims"][0]["quantity"] == "H0"


def test_claim_to_dict_round_trip() -> None:
    span = ClaimSpan(
        quantity="sigma_8",
        value=0.811,
        uncertainty=0.006,
        unit=None,
        span=(0, 25),
        surface="sigma_8 = 0.811 \\pm 0.006",
    )
    d = claim_to_dict(span)
    assert d["quantity"] == "sigma_8"
    assert d["span"] == [0, 25]


# ---------------------------------------------------------------------------
# Mocked psycopg insert — exercise scripts/run_claim_extractor.insert_claims
# ---------------------------------------------------------------------------


def _import_script_module():
    """Import scripts/run_claim_extractor.py as ``run_claim_extractor``."""
    import importlib.util

    if "run_claim_extractor" in sys.modules:
        return sys.modules["run_claim_extractor"]
    spec = importlib.util.spec_from_file_location(
        "run_claim_extractor",
        SCRIPTS / "run_claim_extractor.py",
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["run_claim_extractor"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_insert_claims_writes_quant_claim_extraction_type() -> None:
    """The mocked cursor must receive the extraction_type='quant_claim' param."""
    mod = _import_script_module()

    fake_cursor = MagicMock()
    fake_cursor.__enter__ = MagicMock(return_value=fake_cursor)
    fake_cursor.__exit__ = MagicMock(return_value=False)

    fake_conn = MagicMock()
    fake_conn.cursor.return_value = fake_cursor

    body = "We measure H0 = 73.4 +/- 1.1 km/s/Mpc."
    claims = extract_claims(body)
    assert claims, "precondition: extractor must find a claim in the test body"

    n_inserted = mod.insert_claims(fake_conn, "2024TestPaper.001", claims)

    assert n_inserted == 1
    fake_cursor.execute.assert_called_once()
    args = fake_cursor.execute.call_args
    # call_args.args = (sql, params)
    sql_text, params = args.args
    assert "staging.extractions" in sql_text
    assert "ON CONFLICT" in sql_text
    # Positional params: (bibcode, type, version, payload_json, source, tier)
    assert params[0] == "2024TestPaper.001"
    assert params[1] == EXTRACTION_TYPE  # 'quant_claim'
    assert params[2] == EXTRACTION_VERSION
    payload = json.loads(params[3])
    assert payload["extraction_type"] == "quant_claim"
    assert payload["claims"][0]["quantity"] == "H0"


def test_insert_claims_skips_when_no_claims() -> None:
    mod = _import_script_module()
    fake_conn = MagicMock()
    n = mod.insert_claims(fake_conn, "2024Empty.001", [])
    assert n == 0
    fake_conn.cursor.assert_not_called()


# ---------------------------------------------------------------------------
# CLI guard — production DSN refuses without --allow-prod
# ---------------------------------------------------------------------------


def test_cli_refuses_production_dsn_without_allow_prod(monkeypatch) -> None:
    """``main(['--dsn', 'dbname=scix'])`` must exit with code 2."""
    mod = _import_script_module()

    # Sanity: get_connection must not be called at all.
    called = {"n": 0}

    def fail_get_connection(*_a, **_kw):
        called["n"] += 1
        raise AssertionError("get_connection must NOT be called for prod DSN")

    monkeypatch.setattr(mod, "get_connection", fail_get_connection)

    rc = mod.main(["--dsn", "dbname=scix"])
    assert rc == 2
    assert called["n"] == 0
