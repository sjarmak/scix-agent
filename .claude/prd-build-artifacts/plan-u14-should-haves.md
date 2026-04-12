# Plan — u14-should-haves

## Steps

1. **Migration 037** — `migrations/037_citation_consistency_and_disputes.sql`
   - `ALTER TABLE document_entities ADD COLUMN IF NOT EXISTS citation_consistency REAL;`
   - `CREATE TABLE IF NOT EXISTS entity_link_disputes (id BIGSERIAL PK, bibcode TEXT, entity_id BIGINT, reason TEXT, reported_at TIMESTAMPTZ DEFAULT now(), tier SMALLINT);`
   - Index on bibcode for lookup.

2. **src/scix/citation_consistency.py**
   - `compute_consistency(bibcode, entity_id, *, conn, link_type='mention', tier=None) -> float | None`
   - Query: outbound cites from `citation_edges` where `source_bibcode = %s`, left-join to `document_entities` filtered by entity_id/link_type (and tier if specified).
   - Return fraction = matched / total; None if total == 0.
   - Also provide `compute_batch` for lists (optional convenience, simple loop).
   - Only READS from `document_entities` — no lint concern.

3. **src/scix/query_expansion.py**
   - In-memory `_EntityIndex` dataclass: ndarray of shape (N, D) + list of entity_ids.
   - `expand(query, k=5, index=None) -> list[int]`: hash query to deterministic vector (seed np.random with stable hash), cosine-sim argpartition top-k, return entity_ids.
   - Default index initialized lazily from a module-global; tests can pass their own index.
   - Production TODO comment pointing at pgvector HNSW path on `entities` table.

4. **src/scix/feedback.py**
   - `_resolve_dsn(dsn)` identical to query_log pattern.
   - `report_incorrect_link(bibcode, entity_id, reason, tier, *, conn=None, dsn=None) -> None`
   - INSERT into entity_link_disputes.

5. **docs/mcp_dual_lane_contract.md**
   - Static-lane consumers list + latency budget.
   - JIT-lane consumers list + latency budget.
   - M13 resolver contract reference.

6. **Tests**
   - `tests/test_citation_consistency.py` — integration test (SCIX_TEST_DSN required). Seeds 5 papers + citation edges + entity links, asserts known closed-form.
   - `tests/test_query_expansion.py` — pure unit: builds 10-entity numpy index, asserts deterministic output, asserts latency < 20ms over 100-entity index.
   - `tests/test_feedback.py` — integration, asserts row written.

7. **Apply migration to scix_test** (for integration tests).

8. **Run tests + AST lint**.

9. **Commit**.

## Risks

- AST lint on SQL strings that mention `document_entities` in SELECT: verified lint only matches INSERT/UPDATE/DELETE, so SELECT is fine.
- Applying migration to scix_test: run psql before pytest.
