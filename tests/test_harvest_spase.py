"""Tests for scripts/harvest_spase.py — SPASE heliophysics vocabulary harvester."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from harvest_spase import (
    DISCIPLINE,
    INSTRUMENT_LISTS,
    MEASUREMENT_LISTS,
    REGION_SUB_LISTS,
    REGION_TOP_LIST,
    SOURCE,
    camel_case_split,
    download_and_parse,
    parse_instrument_entries,
    parse_measurement_entries,
    parse_region_entries,
    parse_tab_file,
    _build_definition_map,
    _make_entry,
    main,
)

# ---------------------------------------------------------------------------
# Sample tab-delimited data for mocking
# ---------------------------------------------------------------------------

SAMPLE_MEMBER_TAB = """\
#Version\tSince\tList\tItem
2.7.0\t1.0.0\tMeasurementType\tActivityIndex
2.7.0\t1.3.5\tMeasurementType\tElectricField
2.7.0\t1.0.0\tMeasurementType\tMagneticField
2.7.0\t1.0.0\tMeasurementType\tEnergeticParticles
2.7.0\t1.0.0\tMeasurementType\tThermalPlasma
2.7.0\t1.0.0\tMeasurementType\tDust
2.7.0\t1.0.0\tMeasurementType\tWaves
2.7.0\t1.0.0\tFieldQuantity\tElectric
2.7.0\t1.0.0\tFieldQuantity\tMagnetic
2.7.0\t1.0.0\tFieldQuantity\tPoynting
2.7.0\t1.0.0\tParticleQuantity\tArrivalDirection
2.7.0\t1.0.0\tParticleQuantity\tCountRate
2.7.0\t1.0.0\tParticleQuantity\tEnergyDensity
2.7.0\t1.0.0\tParticleQuantity\tFlowSpeed
2.7.0\t1.0.0\tWaveQuantity\tFrequency
2.7.0\t1.0.0\tWaveQuantity\tWavelength
2.7.0\t1.0.0\tMixedQuantity\tTotalPressure
2.7.0\t1.0.0\tInstrumentType\tMagnetometer
2.7.0\t1.0.0\tInstrumentType\tSpectrometer
2.7.0\t1.0.0\tInstrumentType\tFaradayCup
2.7.0\t1.0.0\tInstrumentType\tSearchCoil
2.7.0\t1.0.0\tInstrumentType\tLangmuirProbe
2.7.0\t1.0.0\tRegion\tEarth
2.7.0\t1.0.0\tRegion\tSun
2.7.0\t1.0.0\tRegion\tHeliosphere
2.7.0\t1.0.0\tRegion\tJupiter
2.7.0\t1.0.0\tRegion\tInterstellar
2.7.0\t1.0.0\tEarth\tMagnetosphere
2.7.0\t1.0.0\tEarth\tNearSurface
2.7.0\t1.0.0\tEarth\tMoon
2.7.0\t1.0.0\tHeliosphere\tInner
2.7.0\t1.0.0\tHeliosphere\tNearEarth
2.7.0\t1.0.0\tSun\tChromosphere
2.7.0\t1.0.0\tSun\tCorona
2.7.0\t1.0.0\tSun\tPhotosphere
2.7.0\t1.0.0\tJupiter\tIo
2.7.0\t1.0.0\tJupiter\tMagnetosphere
2.7.0\t1.0.0\tAccessRights\tOpen
"""

SAMPLE_DICTIONARY_TAB = """\
#Version\tSince\tTerm\tType\tList\tElements\tAttributes\tDefinition
2.7.0\t1.0.0\tMagneticField\tItem\t\t\t\tThe field produced by a magnet or a changing electric field.
2.7.0\t1.0.0\tElectricField\tItem\t\t\t\tA region of space around a charged particle.
2.7.0\t1.0.0\tMagnetometer\tItem\t\t\t\tAn instrument that measures the strength and direction of magnetic fields.
2.7.0\t1.0.0\tEarth\tItem\t\t\t\tThe third planet from the Sun.
"""


# ---------------------------------------------------------------------------
# Tests for camel_case_split
# ---------------------------------------------------------------------------


class TestCamelCaseSplit:
    def test_simple_two_words(self) -> None:
        assert camel_case_split("MagneticField") == "Magnetic Field"

    def test_simple_two_words_2(self) -> None:
        assert camel_case_split("ElectricField") == "Electric Field"

    def test_three_words(self) -> None:
        assert camel_case_split("EnergeticParticles") == "Energetic Particles"

    def test_acronym_prefix(self) -> None:
        assert camel_case_split("ACMagneticField") == "AC Magnetic Field"

    def test_all_caps(self) -> None:
        assert camel_case_split("SPICE") == "SPICE"

    def test_single_word(self) -> None:
        assert camel_case_split("Dust") == "Dust"

    def test_near_earth(self) -> None:
        assert camel_case_split("NearEarth") == "Near Earth"

    def test_already_split(self) -> None:
        assert camel_case_split("Io") == "Io"

    def test_langmuir_probe(self) -> None:
        assert camel_case_split("LangmuirProbe") == "Langmuir Probe"

    def test_faraday_cup(self) -> None:
        assert camel_case_split("FaradayCup") == "Faraday Cup"

    def test_total_pressure(self) -> None:
        assert camel_case_split("TotalPressure") == "Total Pressure"

    def test_arrival_direction(self) -> None:
        assert camel_case_split("ArrivalDirection") == "Arrival Direction"


# ---------------------------------------------------------------------------
# Tests for parse_tab_file
# ---------------------------------------------------------------------------


class TestParseTabFile:
    def test_parse_member_tab(self) -> None:
        rows = parse_tab_file(SAMPLE_MEMBER_TAB)
        assert len(rows) > 0
        # Check that header names are used as keys
        first = rows[0]
        assert "Version" in first
        assert "Since" in first
        assert "List" in first
        assert "Item" in first

    def test_parse_by_header_name(self) -> None:
        """Verify parsing uses header names, not column indices."""
        rows = parse_tab_file(SAMPLE_MEMBER_TAB)
        measurement_rows = [r for r in rows if r["List"] == "MeasurementType"]
        assert len(measurement_rows) == 7

    def test_parse_dictionary_tab(self) -> None:
        rows = parse_tab_file(SAMPLE_DICTIONARY_TAB)
        assert len(rows) > 0
        first = rows[0]
        assert "Term" in first
        assert "Definition" in first

    def test_hash_stripped_from_header(self) -> None:
        """The '#' prefix on the header line should be stripped."""
        rows = parse_tab_file(SAMPLE_MEMBER_TAB)
        first = rows[0]
        # Should be 'Version', not '#Version'
        assert "Version" in first
        assert "#Version" not in first

    def test_empty_content(self) -> None:
        assert parse_tab_file("") == []

    def test_header_only(self) -> None:
        assert parse_tab_file("#Version\tSince\tList\tItem\n") == []


# ---------------------------------------------------------------------------
# Tests for _build_definition_map
# ---------------------------------------------------------------------------


class TestBuildDefinitionMap:
    def test_builds_map(self) -> None:
        rows = parse_tab_file(SAMPLE_DICTIONARY_TAB)
        defs = _build_definition_map(rows)
        assert "MagneticField" in defs
        assert "magnet" in defs["MagneticField"].lower()

    def test_missing_terms_absent(self) -> None:
        rows = parse_tab_file(SAMPLE_DICTIONARY_TAB)
        defs = _build_definition_map(rows)
        assert "NonExistentTerm" not in defs


# ---------------------------------------------------------------------------
# Tests for _make_entry
# ---------------------------------------------------------------------------


class TestMakeEntry:
    def test_basic_entry(self) -> None:
        entry = _make_entry("MagneticField", "observable", spase_list="MeasurementType")
        assert entry["canonical_name"] == "MagneticField"
        assert entry["entity_type"] == "observable"
        assert entry["source"] == "spase"
        assert entry["external_id"] == "spase:MeasurementType:MagneticField"
        assert "Magnetic Field" in entry["aliases"]
        assert entry["metadata"]["spase_list"] == "MeasurementType"

    def test_no_alias_for_single_word(self) -> None:
        entry = _make_entry("Dust", "observable", spase_list="MeasurementType")
        assert entry["aliases"] == []

    def test_definition_in_metadata(self) -> None:
        entry = _make_entry(
            "MagneticField",
            "observable",
            spase_list="MeasurementType",
            definition="A field produced by magnets.",
        )
        assert entry["metadata"]["description"] == "A field produced by magnets."

    def test_dotted_path_aliases(self) -> None:
        entry = _make_entry(
            "Earth.Magnetosphere",
            "observable",
            spase_list="ObservedRegion",
        )
        assert "Earth Magnetosphere" in entry["aliases"]

    def test_dotted_path_with_camelcase(self) -> None:
        entry = _make_entry(
            "Earth.NearSurface",
            "observable",
            spase_list="ObservedRegion",
        )
        aliases = entry["aliases"]
        assert "Earth NearSurface" in aliases or "Earth Near Surface" in aliases


# ---------------------------------------------------------------------------
# Tests for parse_measurement_entries
# ---------------------------------------------------------------------------


class TestParseMeasurementEntries:
    def setup_method(self) -> None:
        self.member_rows = parse_tab_file(SAMPLE_MEMBER_TAB)
        dict_rows = parse_tab_file(SAMPLE_DICTIONARY_TAB)
        self.definitions = _build_definition_map(dict_rows)

    def test_count(self) -> None:
        entries = parse_measurement_entries(self.member_rows, self.definitions)
        # 7 MeasurementType + 3 FieldQuantity + 4 ParticleQuantity
        # + 2 WaveQuantity + 1 MixedQuantity = 17
        assert len(entries) == 17

    def test_entity_type(self) -> None:
        entries = parse_measurement_entries(self.member_rows, self.definitions)
        for entry in entries:
            assert entry["entity_type"] == "observable"

    def test_source(self) -> None:
        entries = parse_measurement_entries(self.member_rows, self.definitions)
        for entry in entries:
            assert entry["source"] == SOURCE

    def test_has_definitions(self) -> None:
        entries = parse_measurement_entries(self.member_rows, self.definitions)
        mag_field = [e for e in entries if e["canonical_name"] == "MagneticField"]
        assert len(mag_field) == 1
        assert "description" in mag_field[0]["metadata"]

    def test_camelcase_aliases(self) -> None:
        entries = parse_measurement_entries(self.member_rows, self.definitions)
        mag_field = [e for e in entries if e["canonical_name"] == "MagneticField"]
        assert "Magnetic Field" in mag_field[0]["aliases"]

    def test_excludes_non_measurement_lists(self) -> None:
        entries = parse_measurement_entries(self.member_rows, self.definitions)
        names = {e["canonical_name"] for e in entries}
        assert "Magnetometer" not in names  # InstrumentType
        assert "Earth" not in names  # Region


# ---------------------------------------------------------------------------
# Tests for parse_instrument_entries
# ---------------------------------------------------------------------------


class TestParseInstrumentEntries:
    def setup_method(self) -> None:
        self.member_rows = parse_tab_file(SAMPLE_MEMBER_TAB)
        dict_rows = parse_tab_file(SAMPLE_DICTIONARY_TAB)
        self.definitions = _build_definition_map(dict_rows)

    def test_count(self) -> None:
        entries = parse_instrument_entries(self.member_rows, self.definitions)
        assert len(entries) == 5

    def test_entity_type(self) -> None:
        entries = parse_instrument_entries(self.member_rows, self.definitions)
        for entry in entries:
            assert entry["entity_type"] == "instrument"

    def test_camelcase_aliases(self) -> None:
        entries = parse_instrument_entries(self.member_rows, self.definitions)
        faraday = [e for e in entries if e["canonical_name"] == "FaradayCup"]
        assert len(faraday) == 1
        assert "Faraday Cup" in faraday[0]["aliases"]

    def test_excludes_non_instrument_lists(self) -> None:
        entries = parse_instrument_entries(self.member_rows, self.definitions)
        names = {e["canonical_name"] for e in entries}
        assert "MagneticField" not in names


# ---------------------------------------------------------------------------
# Tests for parse_region_entries
# ---------------------------------------------------------------------------


class TestParseRegionEntries:
    def setup_method(self) -> None:
        self.member_rows = parse_tab_file(SAMPLE_MEMBER_TAB)
        dict_rows = parse_tab_file(SAMPLE_DICTIONARY_TAB)
        self.definitions = _build_definition_map(dict_rows)

    def test_count(self) -> None:
        entries = parse_region_entries(self.member_rows, self.definitions)
        # 5 top-level (Earth, Sun, Heliosphere, Jupiter, Interstellar)
        # + 3 Earth sub (Magnetosphere, NearSurface, Moon)
        # + 2 Heliosphere sub (Inner, NearEarth)
        # + 3 Sun sub (Chromosphere, Corona, Photosphere)
        # + 2 Jupiter sub (Io, Magnetosphere)
        # = 15
        assert len(entries) == 15

    def test_entity_type(self) -> None:
        entries = parse_region_entries(self.member_rows, self.definitions)
        for entry in entries:
            assert entry["entity_type"] == "observable"

    def test_top_level_regions(self) -> None:
        entries = parse_region_entries(self.member_rows, self.definitions)
        names = {e["canonical_name"] for e in entries}
        assert "Earth" in names
        assert "Sun" in names
        assert "Heliosphere" in names

    def test_sub_regions_dotted_path(self) -> None:
        entries = parse_region_entries(self.member_rows, self.definitions)
        names = {e["canonical_name"] for e in entries}
        assert "Earth.Magnetosphere" in names
        assert "Sun.Corona" in names
        assert "Heliosphere.NearEarth" in names
        assert "Jupiter.Io" in names

    def test_sub_region_aliases(self) -> None:
        entries = parse_region_entries(self.member_rows, self.definitions)
        near_earth = [e for e in entries if e["canonical_name"] == "Heliosphere.NearEarth"]
        assert len(near_earth) == 1
        aliases = near_earth[0]["aliases"]
        # Should have space-separated form
        assert "Heliosphere NearEarth" in aliases or "Heliosphere Near Earth" in aliases

    def test_spase_list_is_observed_region(self) -> None:
        entries = parse_region_entries(self.member_rows, self.definitions)
        for entry in entries:
            assert entry["metadata"]["spase_list"] == "ObservedRegion"

    def test_excludes_non_region_lists(self) -> None:
        entries = parse_region_entries(self.member_rows, self.definitions)
        names = {e["canonical_name"] for e in entries}
        assert "Magnetometer" not in names
        assert "MagneticField" not in names
        # AccessRights.Open should NOT appear
        assert "AccessRights.Open" not in names


# ---------------------------------------------------------------------------
# Tests for download_and_parse (with mocked downloads)
# ---------------------------------------------------------------------------


class TestDownloadAndParse:
    @patch("harvest_spase.download_tab_file")
    def test_all_vocabularies(self, mock_download: MagicMock) -> None:
        mock_download.side_effect = [SAMPLE_MEMBER_TAB, SAMPLE_DICTIONARY_TAB]
        entries = download_and_parse(vocabulary="all")
        # Should have measurement + instrument + region entries
        types = {e["entity_type"] for e in entries}
        assert "observable" in types
        assert "instrument" in types

    @patch("harvest_spase.download_tab_file")
    def test_measurement_only(self, mock_download: MagicMock) -> None:
        mock_download.side_effect = [SAMPLE_MEMBER_TAB, SAMPLE_DICTIONARY_TAB]
        entries = download_and_parse(vocabulary="measurement")
        for entry in entries:
            assert entry["entity_type"] == "observable"
        # No instrument entries
        assert all(e["metadata"]["spase_list"] != "InstrumentType" for e in entries)

    @patch("harvest_spase.download_tab_file")
    def test_instrument_only(self, mock_download: MagicMock) -> None:
        mock_download.side_effect = [SAMPLE_MEMBER_TAB, SAMPLE_DICTIONARY_TAB]
        entries = download_and_parse(vocabulary="instrument")
        for entry in entries:
            assert entry["entity_type"] == "instrument"

    @patch("harvest_spase.download_tab_file")
    def test_region_only(self, mock_download: MagicMock) -> None:
        mock_download.side_effect = [SAMPLE_MEMBER_TAB, SAMPLE_DICTIONARY_TAB]
        entries = download_and_parse(vocabulary="region")
        for entry in entries:
            assert entry["entity_type"] == "observable"
            assert entry["metadata"]["spase_list"] == "ObservedRegion"


# ---------------------------------------------------------------------------
# Tests for main() CLI
# ---------------------------------------------------------------------------


class TestMain:
    @patch("harvest_spase.download_tab_file")
    def test_help(self, mock_download: MagicMock) -> None:
        with pytest.raises(SystemExit) as exc_info:
            main(["--help"])
        assert exc_info.value.code == 0

    @patch("harvest_spase.download_tab_file")
    def test_dry_run(self, mock_download: MagicMock) -> None:
        mock_download.side_effect = [SAMPLE_MEMBER_TAB, SAMPLE_DICTIONARY_TAB]
        result = main(["--dry-run"])
        assert result == 0

    @patch("harvest_spase.download_tab_file")
    def test_dry_run_vocabulary_filter(self, mock_download: MagicMock) -> None:
        mock_download.side_effect = [SAMPLE_MEMBER_TAB, SAMPLE_DICTIONARY_TAB]
        result = main(["--dry-run", "--vocabulary", "instrument"])
        assert result == 0


# ---------------------------------------------------------------------------
# Tests for live data counts (downloads from GitHub, integration-style)
# These use @pytest.mark.network and are skipped by default.
# Run with: pytest -m network tests/test_harvest_spase.py
# ---------------------------------------------------------------------------


@pytest.mark.network
class TestLiveDataCounts:
    """Integration tests that download real SPASE vocabulary files.

    Skipped by default. Run with: pytest -m network
    """

    @pytest.fixture(autouse=True)
    def _download(self) -> None:
        from harvest_spase import download_tab_file, MEMBER_URL, DICTIONARY_URL

        member_text = download_tab_file(MEMBER_URL)
        dictionary_text = download_tab_file(DICTIONARY_URL)
        self.member_rows = parse_tab_file(member_text)
        dict_rows = parse_tab_file(dictionary_text)
        self.definitions = _build_definition_map(dict_rows)

    def test_measurement_count_exceeds_50(self) -> None:
        entries = parse_measurement_entries(self.member_rows, self.definitions)
        assert len(entries) > 50

    def test_instrument_count_exceeds_30(self) -> None:
        entries = parse_instrument_entries(self.member_rows, self.definitions)
        assert len(entries) > 30

    def test_region_count_exceeds_40(self) -> None:
        entries = parse_region_entries(self.member_rows, self.definitions)
        assert len(entries) > 40
