# Plan — unit-v2-sankey-data

## Files

1. `scripts/viz/__init__.py` — minimal docstring, marks viz as a package.
2. `scripts/viz/build_temporal_sankey_data.py` — main CLI.
3. `tests/test_build_temporal_sankey_data.py` — four unit tests.

## Module structure: `build_temporal_sankey_data.py`

### Imports
`argparse`, `dataclasses`, `json`, `logging`, `os`, `random`, `sys`,
`collections.Counter`, `pathlib.Path`, `typing.NamedTuple / Iterable / Iterator`,
`psycopg` (lazy — only when not `--synthetic`), and `scix.db` (DEFAULT_DSN,
is_production_dsn, redact_dsn).

### Constants
- `RESOLUTIONS = ("coarse", "medium", "fine")`.
- `COMMUNITY_COLUMNS = { "coarse": "community_semantic_coarse", ... }`.
- `DEFAULT_OUTPUT = Path("data/viz/sankey.json")` (relative to repo root).

### Dataclasses / NamedTuples
- `class PaperRow(NamedTuple): bibcode: str; year: int; community_id: int`.
- `@dataclass(frozen=True) class Node: id: str; decade: int; community_id: int; paper_count: int`.
- `@dataclass(frozen=True) class Link: source: str; target: str; value: int`.
- `@dataclass(frozen=True) class SankeyData: nodes: tuple[Node, ...]; links: tuple[Link, ...]`.
- `@dataclass(frozen=True) class Config`: `resolution`, `top_flows`, `output`, `dsn`, `dry_run`, `synthetic`.

### Functions
1. `decade_of(year: int) -> int` — returns `year // 10 * 10`.
2. `node_id(decade: int, community_id: int) -> str` — `f"{decade}-{community_id}"`.
3. `aggregate(rows: Iterable[PaperRow], top_flows: int) -> SankeyData` — pure:
   - Count papers per `(decade, community_id)` → node sizes.
   - For each decade d where d+1 exists, for each community c present in d and in d+1, emit link `(d,c) → (d+1,c)` with value = size in d+1.
   - Sort links by value descending, cap to `top_flows`, tiebreak by (source, target).
   - Build nodes: only include nodes that appear as source or target of kept links, OR that have a paper_count > 0 in decade with no successor (keep all nodes touched by a kept link + keep nodes with paper counts even if not linked — simplest: keep all `(decade, community)` that have papers).
   - Return `SankeyData`.
4. `load_synthetic(n: int = 10000, seed: int = 42) -> Iterator[PaperRow]` — deterministic RNG (`random.Random(seed)`), 20 communities, years uniform in `[1990, 2025]`, bibcode `f"synthetic-{i:06d}"`.
5. `load_from_db(dsn: str, resolution: str) -> Iterator[PaperRow]` — named cursor, `fetchmany(10_000)`, joins `papers p` on `paper_metrics pm`, filters on non-null year + non-null community column. Column name looked up in `COMMUNITY_COLUMNS` (allowlist).
6. `serialize_to_json(sd: SankeyData, path: Path) -> None` — writes JSON with sorted node list then link list, UTF-8, trailing newline.
7. `validate_sankey_data(obj: dict) -> SankeyData` — type-checks top-level keys "nodes"/"links"; each node has keys `id/decade/community_id/paper_count` with correct types; each link has `source/target/value`. Raises `ValueError` on mismatch. Returns frozen `SankeyData`.
8. `run(config: Config) -> SankeyData` — picks loader (synthetic vs DB), calls `aggregate`, returns the result. Does NOT write.
9. `main(argv) -> int` — parses args → Config → calls run() → logs summary → if not dry-run, calls `serialize_to_json`.

### CLI flags
- `--resolution {coarse,medium,fine}` (default medium).
- `--top-flows N` (int, default 500).
- `--output PATH` (default `data/viz/sankey.json`, resolved relative to repo root if relative).
- `--dsn` (default from `scix.db.DEFAULT_DSN`).
- `--dry-run` (flag, don't write).
- `--synthetic` (flag, skip DB).

### Flow semantics (chosen for determinism)
`flow[(d, c) → (d+1, c)] = count of papers in (d+1, c)` for every community
`c` that exists in both decade `d` and decade `d+1`. Produces at most
`|transitions|` links; ordered by value desc; capped at `--top-flows`.

### Top-flows cap
After sorting by value (desc), slice to `top_flows`. This exactly yields
`len(links) == min(all_possible_flows, top_flows)`. For the test to assert
`len(links) == N`, the synthetic dataset must produce > N flows — we will use
a generator in the test that produces many communities × many decades.

## Tests (`tests/test_build_temporal_sankey_data.py`)

### Test 1 — schema validation
Hand-build a minimal dict:
```python
raw = {
    "nodes": [{"id": "1990-0", "decade": 1990, "community_id": 0, "paper_count": 5}],
    "links": [{"source": "1990-0", "target": "2000-0", "value": 3}],
}
validate_sankey_data(raw)  # no exception
```
Also test failure: missing key raises `ValueError`.

### Test 2 — decade bucketing
`assert decade_of(1999) == 1990`, `assert decade_of(2000) == 2000`,
`assert decade_of(2025) == 2020`.

### Test 3 — top-flows cap
Build synthetic rows so that after aggregation we have > N possible links.
Use `aggregate` directly: construct rows with 50 communities × 5 decades
(so potential links ≈ 50 * 4 = 200 when each community persists across all
decades). Call `aggregate(rows, top_flows=10)` → assert `len(result.links) == 10`.

### Test 4 — dry-run writes nothing
Use `tmp_path` fixture. Call `main(["--dry-run", "--synthetic", "--output", str(tmp_path / "s.json")])`.
Assert returncode == 0 and path does not exist.

## Chmod +x
After writing the script, the commit stage uses
`git update-index --chmod=+x` to record the executable bit in the tree.

## Commit
`git add scripts/viz/ tests/test_build_temporal_sankey_data.py .claude/prd-build-artifacts/research-unit-v2-sankey-data.md .claude/prd-build-artifacts/plan-unit-v2-sankey-data.md .claude/prd-build-artifacts/test-unit-v2-sankey-data.md`
`git commit -m "prd-build: unit-v2-sankey-data — Temporal-community Sankey data builder"`.
