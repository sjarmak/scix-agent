# Plan: harvest-spase

## Implementation Steps

1. **Create scripts/harvest_spase.py**
   - Download `member.tab` and `dictionary.tab` from spase-base-2.7.0
   - Parse tab-delimited files by header name (not column index)
   - Extract three vocabulary categories:
     - MeasurementType + FieldQuantity + ParticleQuantity + WaveQuantity + MixedQuantity -> entity_type='observable' (combined to exceed 50 entries)
     - InstrumentType -> entity_type='instrument'
     - Region + sub-region lists (Earth, Heliosphere, Sun, etc.) -> entity_type='observable'
   - CamelCase split: regex to produce space-separated aliases
   - For observed regions, build dotted paths (e.g., "Earth.Magnetosphere") as canonical names
   - All entries: source='spase', discipline='heliophysics'
   - CLI flags: --dsn, --dry-run, --vocabulary (choices: measurement, instrument, region, all), --verbose

2. **Create tests/test_harvest_spase.py**
   - Mock member.tab content with sample data
   - Test CamelCase splitting function
   - Test parsing of each vocabulary type
   - Test metadata structure (entity_type, source, discipline)
   - Test --dry-run behavior
   - Test header-based parsing robustness

3. **Commit**
