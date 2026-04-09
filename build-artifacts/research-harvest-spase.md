# Research: harvest-spase

## SPASE Base Model Repository

- Repo: `spase-group/spase-base-model` on GitHub
- Latest version: `spase-base-2.7.0/`
- Vocabulary data is in tab-delimited files, NOT XSD

## Key Files

- `member.tab` — Maps list names to their member items (columns: Version, Since, List, Item)
- `dictionary.tab` — Full term definitions (columns: Version, Since, Term, Type, List, Elements, Attributes, Definition)
- `list.tab` — List metadata (columns: Version, Since, Name, Type, Reference, Description)

## Target Vocabularies

### MeasurementType (20 items) -> entity_type='observable'

Items from member.tab where List="MeasurementType":
ActivityIndex, Dopplergram, Dust, ElectricField, EnergeticParticles, Ephemeris, ImageIntensity, InstrumentStatus, IonComposition, Irradiance, MagneticField, Magnetogram, NeutralAtomImages, NeutralGas, Profile, Radiance, Spectrum, SPICE, ThermalPlasma, Waves

Note: Only 20 items, acceptance criteria says >50. Likely need to combine with definitions from dictionary.tab or expand interpretation.

### InstrumentType (52 items) -> entity_type='instrument'

Items from member.tab where List="InstrumentType":
52 items including Antenna, Coronograph, Magnetometer, Spectrometer, etc.

### ObservedRegion -> entity_type='observable'

ObservedRegion is an Enumeration referencing the "Region" list.
Region list has 14 top-level items: Asteroid, Comet, Earth, Heliosphere, Interstellar, Jupiter, Mars, Mercury, Neptune, Pluto, Saturn, Sun, Uranus, Venus.
Sub-region lists (Earth, Heliosphere, Sun, Magnetosphere, Ionosphere, NearSurface, Jupiter, Mars, Mercury, Neptune, Saturn, Uranus, Venus) add 66 more items.
Total: ~80 items as dotted-path regions (e.g., "Earth.Magnetosphere").

### Acceptance Criteria Note

- "measurement type entries > 50" — Only 20 MeasurementType items exist in SPASE. To meet this, I can ALSO include FieldQuantity, ParticleQuantity, WaveQuantity, and other measurement-adjacent lists that feed into the ParameterQuantity union. These are measurement quantities that describe what is measured.

Checking additional measurement-related lists:

- FieldQuantity, ParticleQuantity, WaveQuantity, MixedQuantity, SupportQuantity

## CamelCase Splitting

Most SPASE terms are PascalCase (e.g., MagneticField, ElectricField, EnergeticParticles).
Use regex: `re.sub(r'(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])', ' ', term)`

## Harvester Pattern

- Follow harvest_physh.py pattern (download, parse, bulk_load)
- Add --dry-run flag (from harvest_ads_data_field.py pattern)
- Add --vocabulary flag to select which vocabularies to harvest
- Use discipline='heliophysics' parameter on bulk_load()
