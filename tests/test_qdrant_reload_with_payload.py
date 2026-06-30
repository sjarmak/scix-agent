"""Unit tests for the pure transform in scripts/qdrant_reload_with_payload.py.

Covers _points_from_page — building upsert points from a Qdrant scroll page plus
PG enrichment, preserving id+vector and merging payload. The vector source is
Qdrant scroll (paper_embeddings was dropped 2026-06-14), so id+vector MUST be
carried through verbatim.
"""

from __future__ import annotations

from types import SimpleNamespace

from qdrant_reload_with_payload import _points_from_page


def _rec(point_id, vector, payload):
    return SimpleNamespace(id=point_id, vector=vector, payload=payload)


def _pg_row(bibcode, **over):
    base = {
        "bibcode": bibcode,
        "year": 2020,
        "doctype": "article",
        "arxiv_class": ["astro-ph.GA"],
        "bibstem": ["ApJ"],
        "title": "T",
        "first_author": "A",
        "citation_count": 5,
        "is_retracted": False,
        "community_semantic_coarse": 1,
        "community_semantic_medium": 2,
        "pagerank": 0.001,
    }
    base.update(over)
    return base


def test_enriched_point_carries_id_vector_and_full_payload():
    page = [_rec("id-1", [0.1, 0.2, 0.3], {"bibcode": "2020ApJ...1A"})]
    by_bibcode = {"2020ApJ...1A": _pg_row("2020ApJ...1A")}

    pts = _points_from_page(page, by_bibcode)

    assert len(pts) == 1
    p = pts[0]
    assert p.id == "id-1"
    assert p.vector == [0.1, 0.2, 0.3]  # vector preserved verbatim from scroll
    assert p.payload["bibcode"] == "2020ApJ...1A"
    assert p.payload["year"] == 2020
    assert p.payload["doctype"] == "article"
    # is_retracted False → omitted by _build_payload absence-semantics (ADR-008)
    assert "is_retracted" not in p.payload


def test_retracted_flag_written_when_true():
    page = [_rec("id-r", [1.0], {"bibcode": "bib-r"})]
    by_bibcode = {"bib-r": _pg_row("bib-r", is_retracted=True)}

    p = _points_from_page(page, by_bibcode)[0]

    assert p.payload["is_retracted"] is True


def test_point_without_pg_match_keeps_only_bibcode():
    # Point exists in Qdrant but the bibcode has no PG row — keep bibcode, no enrichment.
    page = [_rec("id-2", [0.4], {"bibcode": "missing"})]
    p = _points_from_page(page, {})[0]

    assert p.id == "id-2"
    assert p.vector == [0.4]
    assert p.payload == {"bibcode": "missing"}


def test_point_without_bibcode_payload_gets_empty_payload():
    # Defensive: a source point lacking a bibcode payload still copies id+vector.
    page = [_rec("id-3", [0.5], None)]
    p = _points_from_page(page, {})[0]

    assert p.id == "id-3"
    assert p.vector == [0.5]
    assert p.payload == {}


def test_preserves_page_order_and_count():
    page = [_rec(f"id-{i}", [float(i)], {"bibcode": f"b{i}"}) for i in range(5)]
    by_bibcode = {f"b{i}": _pg_row(f"b{i}") for i in range(5)}

    pts = _points_from_page(page, by_bibcode)

    assert [p.id for p in pts] == [f"id-{i}" for i in range(5)]
