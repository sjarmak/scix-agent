# Research: harvest-physh

## Data Source

PhySH (Physics Subject Headings) is a physics classification scheme by APS.

- GitHub: https://github.com/physh-org/PhySH (branch: master)
- Files: physh.json.gz (JSON-LD), physh.rdf.gz, physh.ttl, physh_skos_compat.ttl
- JSON-LD is easiest to parse: 3910 total items, gzipped

## PhySH Structure

- Types: `physh_rdf:Discipline`, `physh_rdf:Facet`, `physh_rdf:Concept`, `skos:ConceptScheme`
- Facets: Techniques (fa2a6718), Experimental Techniques (705f7ed8), Computational Techniques (1e0c099a), Theoretical Techniques (b96dac97), Theoretical & Computational Techniques (233a6cd0), plus Properties, Physical Systems, Research Areas, Professional Topics
- Concepts linked to facets via `physh_rdf:contains` on facet entries
- Hierarchy via `skos:broader` / `skos:narrower` on individual concepts
- IDs: DOI URIs like `https://doi.org/10.29172/{uuid}`

## Techniques Facet Analysis

- 4 sub-facets under Techniques
- 196 direct concepts via `physh_rdf:contains`
- 907 total concepts including all descendants (via BFS on broader/narrower)
- 234 concepts have altLabels
- Keys: skos:prefLabel, skos:altLabel, skos:broader, skos:narrower, skos:scopeNote

## Example: Monte Carlo

- "Monte Carlo methods" (eb9bd2e1) -> parent: "Numerical techniques", children: Langevin algorithm, Simulated annealing, Path-integral Monte Carlo, Entropic sampling methods, Metropolis algorithm, Quantum Monte Carlo, Heatbath algorithm, Hybrid Monte Carlo algorithm
- "Quantum Monte Carlo" (9dc2ee1a) -> parent: Monte Carlo methods, child: Diffusion quantum Monte Carlo

## Codebase Patterns

- `src/scix/dictionary.py`: `bulk_load(conn, entries)` takes list of dicts with keys: canonical_name, entity_type, source, external_id, aliases, metadata. Uses `ON CONFLICT (canonical_name, entity_type, source) DO UPDATE`.
- `src/scix/dictionary.py`: `lookup(conn, name)` searches canonical_name (case-insensitive) then aliases.
- `src/scix/uat.py`: Downloads SKOS RDF/XML, parses with xml.etree.ElementTree, loads into DB. Good pattern reference but we use JSON-LD since PhySH provides it.
- `src/scix/db.py`: `get_connection(dsn)` for DB access.
- `tests/helpers.py`: `DSN` constant for test DB connection.
- Scripts pattern: argparse CLI, sys.path manipulation, call pipeline function.
