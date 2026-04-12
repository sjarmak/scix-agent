# Plan — u03-resolve-entities-service

## Step 1 — Sentinel module (`src/scix/_resolver_token.py`)

```python
class _ResolverToken:
    __slots__ = ()

_RESOLVER_INTERNAL = _ResolverToken()
```

Not re-exported from `src/scix/__init__.py`. Module starts with underscore to mark
it private. Anything importing the sentinel must do
`from scix._resolver_token import _RESOLVER_INTERNAL`, which is a structural
marker — we don't prevent Python-level access, but the convention + private name
is the signal, and the TypeError runtime guard is the teeth.

## Step 2 — EntityLink / EntityLinkSet (`src/scix/entity_link_set.py`)

```python
@dataclass(frozen=True)
class EntityLink:
    entity_id: int
    confidence: float          # 0..1
    link_type: str             # "mention", "citation", ...
    tier: int                  # 0..3 per migration 028
    lane: str                  # "static" | "jit_cache_hit" | "live_jit" | "local_ner"

class EntityLinkSet:
    def __init__(
        self,
        _token: _ResolverToken,
        *,
        bibcode: str,
        entities: frozenset[EntityLink],
        lane: str,
        model_version: str,
    ) -> None:
        if not isinstance(_token, _ResolverToken):
            raise TypeError(
                "EntityLinkSet can only be constructed by scix.resolve_entities; "
                "do not build instances directly."
            )
        object.__setattr__(self, "bibcode", bibcode)
        ...
```

`EntityLinkSet` is **not** `@dataclass` — a frozen dataclass auto-generates
`__init__(self, field1, field2, ...)` which bypasses our guard. Instead we make
it a regular class with a manual `__init__` that demands a positional sentinel,
then sets fields via `object.__setattr__` and marks `__slots__ = (...)` to
approximate immutability. Spec says `@dataclass(frozen=True)`, but the spec also
says the **init** must require the sentinel — those two are in tension for a
dataclass. We resolve by using a `@dataclass(frozen=True)` for the inner
`EntityLink` (which is the real data payload), and implement `EntityLinkSet` as
a hand-written frozen class to keep the sentinel guard. We expose a helper
`entity_ids()` that returns `frozenset[int]` for set-equality tests.

Actually — simpler: we _can_ use `@dataclass(frozen=True)` with the sentinel as
the first field, which means external callers must still provide the token
positionally, but we still want TypeError (not "missing argument"). We add a
`__post_init__` that `raise TypeError` if `_token` isn't the sentinel.
Tests can assert `TypeError` when the caller omits the token — pytest's
`raises(TypeError)` matches `TypeError: __init__() missing 1 required
positional argument: '_token'`, which IS a TypeError. So the frozen dataclass
form works and is spec-faithful. We'll go with that.

## Step 3 — `src/scix/resolve_entities.py`

Module layout:

```
_STATIC_MOCK: dict[str, frozenset[int]]
_JIT_CACHE_MOCK: dict[tuple[str, int, str], frozenset[int]]
_CONFIDENCE_NOISE: dict[str, float]   # per-lane offset ≤0.01

@dataclass(frozen=True)
class EntityResolveContext:
    candidate_set: frozenset[int]
    mode: Literal["static","jit","auto","live_jit","local_ner"]
    ttl_max: int = 3600
    budget_remaining: float = 1.0
    model_version: str = "v1"

def resolve_entities(bibcode: str, context: EntityResolveContext) -> EntityLinkSet:
    # route by mode; auto prefers static -> jit_cache -> live_jit -> local_ner
```

Four lane functions `_lane_static`, `_lane_jit_cache`, `_lane_live_jit`,
`_lane_local_ner`. Each returns an `EntityLinkSet` via the sentinel. Each takes
optional sleep hook injected via module-level `_LANE_LATENCIES` dict so the
benchmark can dial latency without touching real backends.

Helper `candidate_set_hash(context) -> int` = stable `hash` of a sorted tuple so
it's deterministic within a process (sufficient for test determinism).

Module-level test helpers (prefixed with `_seed_` and only called from tests):
`_seed_static(bibcode, ids)`, `_seed_jit_cache(bibcode, cset_hash, model_version, ids)`,
`_seed_live_jit(bibcode, cset_hash, ids)`, `_seed_local_ner(bibcode, ids)`,
`_reset_mocks()`. These are intentionally NOT underscore-private-module-only;
they live in the resolve_entities module so tests can drive all four lanes
with identical deterministic entity sets.

## Step 4 — AST lint (`scripts/ast_lint_resolver.py`)

Strategy:

1. `walk_py_files(Path("src"))` — skip `__pycache__`.
2. For each file parse with `libcst.parse_module(source)`.
3. A visitor finds `Call` nodes; when function is Attribute named `execute`
   `executemany`, `copy`, `copy_expert`, extract the first positional arg if it
   is `SimpleString`, `ConcatenatedString`, or `FormattedString`. Also walk
   top-level `SimpleString` statements that look like SQL (for `cur.execute(SQL)`
   patterns that assign the SQL to a variable — we handle the string
   assignment case separately by scanning all string literals in the file).
4. A **fallback** approach: scan ALL string literals (any SimpleString /
   ConcatenatedString / FormattedString in the module) against the forbidden
   regex set. This catches both inline and assigned SQL.
5. Forbidden regexes:
   - `r"\bINSERT\s+INTO\s+document_entities\b(?!_)"` — ban inserts to
     document_entities but NOT `document_entities_jit_cache` (handled
     separately) or `document_entities_canonical`.
   - `r"\bINSERT\s+INTO\s+document_entities_jit_cache\b"` — write ban.
   - `r"\bUPDATE\s+document_entities\b(?!_)"`
   - `r"\bDELETE\s+FROM\s+document_entities\b(?!_)"`
   - `r"\bFROM\s+document_entities_canonical\b"` — read ban.
6. Exempt file: `src/scix/resolve_entities.py`. Also exempt `scripts/`
   (migrations tools) and `migrations/` which are outside `src/` anyway.
7. Exempt lines with trailing `# noqa: resolver-lint` comment.

Return non-zero exit if any violation is found; print violations to stderr
with file:line.

## Step 5 — Unit tests (`tests/test_resolve_entities.py`)

Four tests: one per lane. Each seeds the corresponding mock, builds an
`EntityResolveContext` with the matching `mode`, calls `resolve_entities`,
and asserts:

- Returned `EntityLinkSet.entity_ids() == expected`.
- `.lane == expected_lane`.

Also tests `mode="auto"` path prefers static when seeded, falls back when not.

## Step 6 — Type guard test (`tests/test_resolve_entities_type_guard.py`)

Imports `EntityLinkSet` from `scix.entity_link_set` (NOT through
resolve_entities), tries `EntityLinkSet(entities=frozenset(), bibcode="x",
lane="static", model_version="v1")` without the token, asserts `TypeError`.

Also tries constructing with a fake sentinel value (e.g. `object()`), asserts
`TypeError` from `__post_init__`.

## Step 7 — Hypothesis invariant test (`tests/test_resolve_entities_invariant.py`)

```python
@given(
    bibcode=st.text(alphabet=string.ascii_letters + string.digits, min_size=5, max_size=19),
    candidate_ids=st.frozensets(st.integers(min_value=1, max_value=10000), min_size=0, max_size=8),
    model_version=st.sampled_from(["v1", "v2"]),
)
@settings(max_examples=120, deadline=None)
def test_lane_set_equality(bibcode, candidate_ids, model_version):
    resolve_entities_module._reset_mocks()
    # seed deterministic id-set across all four lanes
    seed_ids = frozenset(hash((bibcode, i)) & 0xFFFF for i in candidate_ids) or frozenset({1})
    _seed_static(bibcode, seed_ids)
    _seed_jit_cache(bibcode, stable_hash(candidate_ids), model_version, seed_ids)
    _seed_live_jit(bibcode, stable_hash(candidate_ids), seed_ids)
    _seed_local_ner(bibcode, seed_ids)

    ctx_base = EntityResolveContext(candidate_set=candidate_ids, mode="static", model_version=model_version)
    result_static = resolve_entities(bibcode, replace(ctx_base, mode="static"))
    result_jit    = resolve_entities(bibcode, replace(ctx_base, mode="jit"))
    result_live   = resolve_entities(bibcode, replace(ctx_base, mode="live_jit"))
    result_ner    = resolve_entities(bibcode, replace(ctx_base, mode="local_ner"))

    assert result_static.entity_ids() == result_jit.entity_ids() == result_live.entity_ids() == result_ner.entity_ids() == seed_ids
    for eid in seed_ids:
        confidences = [r.confidence_for(eid) for r in (result_static, result_jit, result_live, result_ner)]
        assert max(confidences) - min(confidences) <= 0.01 + 1e-9
```

`resolve_entities._lane_live_jit` must skip its real sleep in test mode so
Hypothesis runs fast — we expose `_LANE_LATENCIES["live_jit"] = 0.0` and let
the test set it to 0 at module scope. Same for local_ner.

## Step 8 — AST lint test (`tests/test_ast_lint_resolver.py`)

1. Run lint against current `src/` — assert exit 0.
2. Write a temp file with a planted `cursor.execute("INSERT INTO document_entities ...")`
   into a tmp dir that mimics `src/scix/bad.py`, invoke the lint module's
   public `run_lint(root: Path)` function (which returns a list of
   violations) and assert len > 0.
3. Write a temp file with the same SQL but trailing `# noqa: resolver-lint`
   and assert len == 0.

## Step 9 — Benchmark (`tests/bench_resolve_entities.py`)

Runs 100 calls per lane with mocked latencies. Computes p95 via
`statistics.quantiles(data, n=20)[-1]`. Writes markdown table to
`build-artifacts/m13_latency.md`. Asserts p95 budgets:

- static ≤ 5ms
- jit_cache ≤ 25ms
- live_jit ≤ 400ms (mock uses 300ms sleep, so p95 is ~300ms; budget asserts < 400)
- local_ner ≤ 300ms (mock uses 200ms sleep)

Actually 300ms \* 100 = 30s per lane. Too slow. Reduce to N=20 samples, and set
mock latencies via `_LANE_LATENCIES` dict to smaller deterministic values:

- static: 0.001 (1ms)
- jit_cache: 0.010 (10ms)
- live_jit: 0.050 (50ms)
- local_ner: 0.030 (30ms)

Budgets then:

- static p95 ≤ 5ms
- jit_cache p95 ≤ 25ms
- live_jit p95 ≤ 80ms
- local_ner p95 ≤ 60ms

This runs in < 5s total. Values are against mocks, not real backends — that
matches the spec.

## Step 10 — Deps & commit

Add `libcst` and `hypothesis` to `pyproject.toml` `dev` extras.
Stage everything, single commit per spec Phase 5.
