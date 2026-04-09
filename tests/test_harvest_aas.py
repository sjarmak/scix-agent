"""Tests for AAS Facility Keywords harvester.

Unit tests verify HTML parsing with mock data.
Integration tests (marked @pytest.mark.integration) require a running scix
database with migration 013 applied.
"""

from __future__ import annotations

import psycopg
import pytest
from helpers import DSN, is_production_dsn

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from harvest_aas_facilities import (
    _FacilityTableParser,
    _classify_header,
    parse_aas_facilities,
)
from scix.dictionary import bulk_load, lookup

# ---------------------------------------------------------------------------
# Sample HTML mimicking the AAS facility keywords page structure
# ---------------------------------------------------------------------------

SAMPLE_HTML = """\
<html>
<body>
<p>The table below list the 500+ facility keywords.</p>
<table id="facility-table" class="tablepress">
<thead>
<tr>
<th>Full Facility Name</th>
<th>Keyword</th>
<th>Location</th>
<th>Gamma-ray (&gt; 120 keV)</th>
<th>X-ray (0.1 - 100 Angstroms)</th>
<th>Ultraviolet (100 - 3000 Angstroms)</th>
<th>Optical (3000 - 10,000 Angstroms)</th>
<th>Infrared (1 - 100 microns)</th>
<th>Millimeter (0.1 - 10 mm)</th>
<th>Radio (&lt; 30 GHz)</th>
<th>Neutrinos, particles, and gravitational waves</th>
<th>Solar Facility</th>
<th>Archive/Database</th>
<th>Computational Center</th>
</tr>
</thead>
<tbody>
<tr>
<td>NASA/European Space Agency (ESA) 2.4m Hubble Space Telescope (HST)</td>
<td>HST</td>
<td>Space</td>
<td></td>
<td></td>
<td>ultraviolet</td>
<td>optical</td>
<td>infrared</td>
<td></td>
<td></td>
<td></td>
<td></td>
<td></td>
<td></td>
</tr>
<tr>
<td>NASA Chandra X-ray Observatory (CXO) Satellite Mission</td>
<td>CXO</td>
<td>Space</td>
<td></td>
<td>x-ray</td>
<td></td>
<td></td>
<td></td>
<td></td>
<td></td>
<td></td>
<td></td>
<td></td>
<td></td>
</tr>
<tr>
<td>NASA/European Space Agency (ESA) 6.5m James Webb Space Telescope</td>
<td>JWST</td>
<td>Space</td>
<td></td>
<td></td>
<td></td>
<td>optical</td>
<td>infrared</td>
<td></td>
<td></td>
<td></td>
<td></td>
<td></td>
<td></td>
</tr>
<tr>
<td>ESO/NSF/NINS Atacama Large Millimeter Array at Llano de Chajnantor</td>
<td>ALMA</td>
<td>South America</td>
<td></td>
<td></td>
<td></td>
<td></td>
<td></td>
<td>millimeter</td>
<td></td>
<td></td>
<td></td>
<td></td>
<td></td>
</tr>
<tr>
<td>NRAO Karl G. Jansky Very Large Array (VLA) on San Agustin plains</td>
<td>VLA</td>
<td>North America &amp; Hawaii</td>
<td></td>
<td></td>
<td></td>
<td></td>
<td></td>
<td></td>
<td>radio</td>
<td></td>
<td></td>
<td></td>
<td></td>
</tr>
<tr>
<td>NASA Fermi Gamma-ray Space Telescope</td>
<td>Fermi</td>
<td>Space</td>
<td>gamma-ray</td>
<td></td>
<td></td>
<td></td>
<td></td>
<td></td>
<td></td>
<td></td>
<td></td>
<td></td>
<td></td>
</tr>
<tr>
<td>NRC Herzberg Solar Radio Flux Monitor at Dominion Radio Astrophysical Observatory (DRAO)</td>
<td>DRAO:SRFM</td>
<td>North America &amp; Hawaii</td>
<td></td>
<td></td>
<td></td>
<td></td>
<td></td>
<td></td>
<td>radio</td>
<td></td>
<td>Solar Facility</td>
<td></td>
<td></td>
</tr>
<tr>
<td>LIGO Scientific Collaboration (LSC) LIGO Detector</td>
<td>LIGO</td>
<td>North America &amp; Hawaii</td>
<td></td>
<td></td>
<td></td>
<td></td>
<td></td>
<td></td>
<td></td>
<td>gravitational waves</td>
<td></td>
<td></td>
<td></td>
</tr>
<tr>
<td>NASA/IPAC Infrared Science Archive (IRSA)</td>
<td>IRSA</td>
<td>North America &amp; Hawaii</td>
<td></td>
<td></td>
<td></td>
<td></td>
<td></td>
<td></td>
<td></td>
<td></td>
<td></td>
<td>Archive/Database</td>
<td></td>
</tr>
<tr>
<td>University of Texas TACC</td>
<td>TACC</td>
<td>North America &amp; Hawaii</td>
<td></td>
<td></td>
<td></td>
<td></td>
<td></td>
<td></td>
<td></td>
<td></td>
<td></td>
<td></td>
<td>Computational Center</td>
</tr>
</tbody>
</table>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Unit tests: HTML parsing
# ---------------------------------------------------------------------------


class TestClassifyHeader:
    """Unit: _classify_header maps header text to normalized keys."""

    def test_gamma_ray(self) -> None:
        assert _classify_header("Gamma-ray (> 120 keV)") == "gamma_ray"

    def test_x_ray(self) -> None:
        assert _classify_header("X-ray (0.1 - 100 Angstroms)") == "x_ray"

    def test_ultraviolet(self) -> None:
        assert _classify_header("Ultraviolet (100 - 3000 Angstroms)") == "ultraviolet"

    def test_optical(self) -> None:
        assert _classify_header("Optical (3000 - 10,000 Angstroms)") == "optical"

    def test_infrared(self) -> None:
        assert _classify_header("Infrared (1 - 100 microns)") == "infrared"

    def test_millimeter(self) -> None:
        assert _classify_header("Millimeter (0.1 - 10 mm)") == "millimeter"

    def test_radio(self) -> None:
        assert _classify_header("Radio (< 30 GHz)") == "radio"

    def test_neutrinos(self) -> None:
        assert (
            _classify_header("Neutrinos, particles, and gravitational waves")
            == "neutrinos_particles_gw"
        )

    def test_solar(self) -> None:
        assert _classify_header("Solar Facility") == "solar"

    def test_archive(self) -> None:
        assert _classify_header("Archive/Database") == "archive_database"

    def test_computational(self) -> None:
        assert _classify_header("Computational Center") == "computational_center"

    def test_unrecognized_returns_none(self) -> None:
        assert _classify_header("Full Facility Name") is None
        assert _classify_header("Keyword") is None
        assert _classify_header("Location") is None


class TestFacilityTableParser:
    """Unit: _FacilityTableParser extracts headers and rows from HTML."""

    def test_headers_extracted(self) -> None:
        parser = _FacilityTableParser()
        parser.feed(SAMPLE_HTML)
        assert len(parser.headers) == 14
        assert parser.headers[0] == "Full Facility Name"
        assert parser.headers[1] == "Keyword"
        assert parser.headers[2] == "Location"

    def test_row_count(self) -> None:
        parser = _FacilityTableParser()
        parser.feed(SAMPLE_HTML)
        assert len(parser.rows) == 10

    def test_first_row_content(self) -> None:
        parser = _FacilityTableParser()
        parser.feed(SAMPLE_HTML)
        row = parser.rows[0]
        assert "Hubble" in row[0]
        assert row[1] == "HST"
        assert row[2] == "Space"


class TestParseAasFacilities:
    """Unit: parse_aas_facilities returns correctly structured entries."""

    def test_returns_list(self) -> None:
        entries = parse_aas_facilities(SAMPLE_HTML)
        assert isinstance(entries, list)
        assert len(entries) > 0

    def test_entry_count(self) -> None:
        entries = parse_aas_facilities(SAMPLE_HTML)
        assert len(entries) == 10

    def test_entry_structure(self) -> None:
        entries = parse_aas_facilities(SAMPLE_HTML)
        required_keys = {"canonical_name", "entity_type", "source", "aliases", "metadata"}
        for entry in entries:
            assert required_keys.issubset(entry.keys()), f"Missing keys in {entry}"

    def test_entity_type_is_instrument(self) -> None:
        entries = parse_aas_facilities(SAMPLE_HTML)
        for entry in entries:
            assert entry["entity_type"] == "instrument"

    def test_source_is_aas(self) -> None:
        entries = parse_aas_facilities(SAMPLE_HTML)
        for entry in entries:
            assert entry["source"] == "aas"

    def test_hst_entry(self) -> None:
        entries = parse_aas_facilities(SAMPLE_HTML)
        hst = next(e for e in entries if "HST" in e["aliases"])
        assert "Hubble" in hst["canonical_name"]
        assert hst["entity_type"] == "instrument"
        assert hst["source"] == "aas"
        wl = hst["metadata"]["wavelength_regimes"]
        assert "ultraviolet" in wl
        assert "optical" in wl
        assert "infrared" in wl
        assert len(wl) == 3

    def test_chandra_entry(self) -> None:
        entries = parse_aas_facilities(SAMPLE_HTML)
        cxo = next(e for e in entries if "CXO" in e["aliases"])
        assert "Chandra" in cxo["canonical_name"]
        wl = cxo["metadata"]["wavelength_regimes"]
        assert wl == ["x_ray"]

    def test_jwst_entry(self) -> None:
        entries = parse_aas_facilities(SAMPLE_HTML)
        jwst = next(e for e in entries if "JWST" in e["aliases"])
        wl = jwst["metadata"]["wavelength_regimes"]
        assert "optical" in wl
        assert "infrared" in wl
        assert len(wl) == 2

    def test_alma_entry(self) -> None:
        entries = parse_aas_facilities(SAMPLE_HTML)
        alma = next(e for e in entries if "ALMA" in e["aliases"])
        wl = alma["metadata"]["wavelength_regimes"]
        assert wl == ["millimeter"]
        assert alma["metadata"]["location"] == "South America"

    def test_vla_entry(self) -> None:
        entries = parse_aas_facilities(SAMPLE_HTML)
        vla = next(e for e in entries if "VLA" in e["aliases"])
        wl = vla["metadata"]["wavelength_regimes"]
        assert wl == ["radio"]
        assert vla["metadata"]["location"] == "North America & Hawaii"

    def test_fermi_gamma_ray(self) -> None:
        entries = parse_aas_facilities(SAMPLE_HTML)
        fermi = next(e for e in entries if "Fermi" in e["aliases"])
        wl = fermi["metadata"]["wavelength_regimes"]
        assert wl == ["gamma_ray"]

    def test_solar_facility_flag(self) -> None:
        entries = parse_aas_facilities(SAMPLE_HTML)
        drao = next(e for e in entries if "DRAO:SRFM" in e["aliases"])
        assert "solar" in drao["metadata"]["facility_flags"]
        assert "radio" in drao["metadata"]["wavelength_regimes"]

    def test_gravitational_waves_flag(self) -> None:
        entries = parse_aas_facilities(SAMPLE_HTML)
        ligo = next(e for e in entries if "LIGO" in e["aliases"])
        assert "neutrinos_particles_gw" in ligo["metadata"]["wavelength_regimes"]

    def test_archive_flag(self) -> None:
        entries = parse_aas_facilities(SAMPLE_HTML)
        irsa = next(e for e in entries if "IRSA" in e["aliases"])
        assert "archive_database" in irsa["metadata"]["facility_flags"]

    def test_computational_center_flag(self) -> None:
        entries = parse_aas_facilities(SAMPLE_HTML)
        tacc = next(e for e in entries if "TACC" in e["aliases"])
        assert "computational_center" in tacc["metadata"]["facility_flags"]

    def test_location_in_metadata(self) -> None:
        entries = parse_aas_facilities(SAMPLE_HTML)
        hst = next(e for e in entries if "HST" in e["aliases"])
        assert hst["metadata"]["location"] == "Space"

    def test_alias_is_keyword(self) -> None:
        entries = parse_aas_facilities(SAMPLE_HTML)
        hst = next(e for e in entries if "HST" in e["aliases"])
        assert "HST" in hst["aliases"]
        assert hst["canonical_name"] != "HST"

    def test_empty_html_returns_empty_list(self) -> None:
        entries = parse_aas_facilities("<html><body></body></html>")
        assert entries == []

    def test_table_without_rows(self) -> None:
        html = """
        <table>
        <thead><tr><th>Name</th><th>Keyword</th></tr></thead>
        <tbody></tbody>
        </table>
        """
        entries = parse_aas_facilities(html)
        assert entries == []


# ---------------------------------------------------------------------------
# Integration tests: database loading
# ---------------------------------------------------------------------------


def _has_entity_dictionary(conn: psycopg.Connection) -> bool:
    """Check if entity_dictionary table exists."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT COUNT(*) FROM information_schema.tables
            WHERE table_schema = 'public'
              AND table_name = 'entity_dictionary'
        """)
        return cur.fetchone()[0] == 1


@pytest.fixture()
def db_conn():
    """Provide a database connection, skip if unavailable or table missing."""
    if is_production_dsn(DSN):
        pytest.skip("Refuses to write test data to production. Set SCIX_TEST_DSN.")
    try:
        conn = psycopg.connect(DSN)
    except psycopg.OperationalError:
        pytest.skip("Database not available")
        return

    if not _has_entity_dictionary(conn):
        conn.close()
        pytest.skip("entity_dictionary table not found (migration 013 not applied)")
        return

    yield conn

    # Clean up test data
    try:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM entity_dictionary WHERE source = 'aas-test'")
        conn.commit()
    except Exception:
        conn.rollback()
    finally:
        conn.close()


@pytest.mark.integration
class TestAasBulkLoad:
    """Integration: AAS facility entries load correctly into entity_dictionary."""

    def test_bulk_load_count(self, db_conn: psycopg.Connection) -> None:
        entries = parse_aas_facilities(SAMPLE_HTML)
        # Override source to 'aas-test' so cleanup is isolated
        for e in entries:
            e["source"] = "aas-test"

        count = bulk_load(db_conn, entries)
        assert count == 10

    def test_lookup_hst_by_alias(self, db_conn: psycopg.Connection) -> None:
        entries = parse_aas_facilities(SAMPLE_HTML)
        for e in entries:
            e["source"] = "aas-test"
        bulk_load(db_conn, entries)

        result = lookup(db_conn, "HST")
        assert result is not None
        assert "Hubble" in result["canonical_name"]
        assert result["entity_type"] == "instrument"
        wl = result["metadata"]["wavelength_regimes"]
        assert "ultraviolet" in wl
        assert "optical" in wl
        assert "infrared" in wl

    def test_lookup_by_canonical_name(self, db_conn: psycopg.Connection) -> None:
        entries = parse_aas_facilities(SAMPLE_HTML)
        for e in entries:
            e["source"] = "aas-test"
        bulk_load(db_conn, entries)

        result = lookup(
            db_conn,
            "NASA/European Space Agency (ESA) 2.4m Hubble Space Telescope (HST)",
        )
        assert result is not None
        assert result["entity_type"] == "instrument"
