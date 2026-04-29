"""Tests for tier-3 discipline-gated SsODNet target linker (xz4.p2v).

Two layers, mirroring tests/test_tier2.py:

* Pure-library tests of cohort SQL composition, the short-name
  disambiguator gate, confidence scoring, and config loading. No DB
  required.
* DB integration tests that seed a small planetary-cohort fixture and
  assert end-to-end linking behavior. Require ``SCIX_TEST_DSN`` set to
  a non-production DSN.
"""

from __future__ import annotations

import pathlib
import sys
import textwrap
from typing import Iterator

import psycopg
import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"
SRC_DIR = REPO_ROOT / "src"
for p in (SCRIPTS_DIR, SRC_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import link_targets_discipline_gated as gated  # noqa: E402
from scix.aho_corasick import EntityRow, build_automaton  # noqa: E402

from tests.helpers import get_test_dsn  # noqa: E402


# ---------------------------------------------------------------------------
# Pure library tests — config + cohort SQL
# ---------------------------------------------------------------------------


def test_load_cohort_config_default_file_parses() -> None:
    cfg = gated.load_cohort_config(gated.DEFAULT_CONFIG_PATH)
    assert "astro-ph.EP" in cfg.arxiv_classes
    assert "Icar" in cfg.bibstems
    assert cfg.min_surface_length_unconditional >= 4
    # All sentinel/disambiguator tokens must be lower-case post-load.
    assert all(tok == tok.lower() for tok in cfg.keyword_sentinels)
    assert all(tok == tok.lower() for tok in cfg.short_name_disambiguators)
    # extra_stop_words from YAML are merged into stop_words.
    assert "nasa" in cfg.stop_words
    assert "ccd" in cfg.stop_words


def test_load_cohort_config_custom_file(tmp_path: pathlib.Path) -> None:
    cfg_file = tmp_path / "cohort.yaml"
    cfg_file.write_text(
        textwrap.dedent(
            """
            arxiv_classes:
              - astro-ph.EP
            bibstems:
              - Icar
            keyword_sentinels:
              - asteroid
            min_surface_length_unconditional: 7
            short_name_disambiguators:
              - asteroid
              - comet
            stop_words_extra:
              - nasa
              - ccd
            """
        ).strip()
    )
    cfg = gated.load_cohort_config(cfg_file)
    assert cfg.arxiv_classes == ("astro-ph.EP",)
    assert cfg.bibstems == ("Icar",)
    assert cfg.keyword_sentinels == ("asteroid",)
    assert cfg.min_surface_length_unconditional == 7
    assert cfg.short_name_disambiguators == ("asteroid", "comet")
    assert "nasa" in cfg.stop_words
    assert "ccd" in cfg.stop_words


def test_is_excluded_surface_drops_stop_words() -> None:
    stop = frozenset({"the", "not", "may", "field"})
    assert gated._is_excluded_surface("The", stop_words=stop, min_canonical_len=5)
    assert gated._is_excluded_surface("NOT", stop_words=stop, min_canonical_len=5)
    assert gated._is_excluded_surface("Field", stop_words=stop, min_canonical_len=5)
    # Stop-word filter wins even if the surface would otherwise pass length.
    assert gated._is_excluded_surface("field", stop_words=stop, min_canonical_len=4)


def test_is_excluded_surface_min_length() -> None:
    # 4-char single-token name: dropped at min_len=7
    assert gated._is_excluded_surface("Eros", stop_words=frozenset(), min_canonical_len=7)
    # 7-char single-token name: kept at min_len=7
    assert not gated._is_excluded_surface(
        "Itokawa", stop_words=frozenset(), min_canonical_len=7
    )
    # Multi-token short name: kept regardless of length
    assert not gated._is_excluded_surface(
        "Comet 67P", stop_words=frozenset(), min_canonical_len=7
    )
    # Designation (digits): kept regardless of length
    assert not gated._is_excluded_surface(
        "2005 VA", stop_words=frozenset(), min_canonical_len=7
    )


def test_fetch_target_entity_rows_excludes_stop_words(
    tmp_path: pathlib.Path,
) -> None:
    # Pure-library check on the helper logic. Build a synthetic entity
    # set and verify the filter wires through.
    stop = frozenset({"the", "field", "nasa"})

    cases = [
        ("The", 7, frozenset({"the"}), True),
        ("Itokawa", 7, frozenset(), False),
        ("Bennu", 7, frozenset(), True),  # 5 chars → dropped
        ("Bennu", 5, frozenset(), False),  # 5 chars at min=5 → kept
        ("NASA", 7, stop, True),
    ]
    for surface, min_len, sw, expected in cases:
        got = gated._is_excluded_surface(
            surface, stop_words=sw, min_canonical_len=min_len
        )
        assert got is expected, f"surface={surface!r} min={min_len} sw={sw} expected={expected}"


def test_build_cohort_sql_includes_arxiv_bibstem_keywords() -> None:
    cfg = gated.CohortConfig(
        arxiv_classes=("astro-ph.EP",),
        bibstems=("Icar",),
        keyword_sentinels=("asteroid",),
        min_surface_length_unconditional=6,
        short_name_disambiguators=("asteroid",),
        stop_words=frozenset(),
        extra_stop_words=(),
    )
    sql, params = gated._build_cohort_sql(cfg, bibcode_prefix=None)
    assert "arxiv_class && %s" in sql
    assert "bibstem && %s" in sql
    assert "unnest" in sql  # keyword sentinels expanded via unnest
    assert ["astro-ph.EP"] in params
    assert ["Icar"] in params
    assert ["asteroid"] in params


def test_build_cohort_sql_with_bibcode_prefix() -> None:
    cfg = gated.CohortConfig(
        arxiv_classes=("astro-ph.EP",),
        bibstems=(),
        keyword_sentinels=(),
        min_surface_length_unconditional=6,
        short_name_disambiguators=(),
        stop_words=frozenset(),
        extra_stop_words=(),
    )
    sql, params = gated._build_cohort_sql(cfg, bibcode_prefix="test_p2v_")
    assert "bibcode LIKE %s" in sql
    assert "test_p2v_%" in params


def test_build_cohort_sql_rejects_empty_filter() -> None:
    cfg = gated.CohortConfig(
        arxiv_classes=(),
        bibstems=(),
        keyword_sentinels=(),
        min_surface_length_unconditional=6,
        short_name_disambiguators=(),
        stop_words=frozenset(),
        extra_stop_words=(),
    )
    with pytest.raises(ValueError):
        gated._build_cohort_sql(cfg, bibcode_prefix=None)


# ---------------------------------------------------------------------------
# Pure library tests — disambiguator gate + linking
# ---------------------------------------------------------------------------


def _planetary_cfg(
    *,
    min_len: int = 6,
    stop_words: frozenset[str] = frozenset(),
) -> gated.CohortConfig:
    return gated.CohortConfig(
        arxiv_classes=("astro-ph.EP",),
        bibstems=("Icar",),
        keyword_sentinels=("asteroid",),
        min_surface_length_unconditional=min_len,
        short_name_disambiguators=("asteroid", "comet", "moon", "spacecraft"),
        stop_words=stop_words,
        extra_stop_words=(),
    )


def _target_rows() -> list[EntityRow]:
    """Three target entities of varying name length."""
    return [
        # 5-char canonical: Bennu — needs a disambiguator.
        EntityRow(
            entity_id=10001,
            surface="Bennu",
            canonical_name="Bennu",
            ambiguity_class="unique",
            is_alias=False,
        ),
        # 7-char: Itokawa — over min_surface_length_unconditional=6.
        EntityRow(
            entity_id=10002,
            surface="Itokawa",
            canonical_name="Itokawa",
            ambiguity_class="unique",
            is_alias=False,
        ),
        # Long canonical: 16 chars
        EntityRow(
            entity_id=10003,
            surface="Comet Hale-Bopp",
            canonical_name="Comet Hale-Bopp",
            ambiguity_class="unique",
            is_alias=False,
        ),
    ]


def test_short_name_dropped_without_disambiguator() -> None:
    automaton = build_automaton(_target_rows())
    cfg = _planetary_cfg()
    out = gated.link_paper(
        bibcode="x_0001",
        title="Bennu in poetry",
        abstract="A literary discussion mentioning Bennu in passing.",
        automaton=automaton,
        cfg=cfg,
    )
    assert out == [], f"short name without disambiguator should drop, got {out}"


def test_short_name_kept_with_disambiguator() -> None:
    automaton = build_automaton(_target_rows())
    cfg = _planetary_cfg()
    out = gated.link_paper(
        bibcode="x_0002",
        title="Sample return from asteroid Bennu",
        abstract="OSIRIS-REx delivered samples from the asteroid Bennu in 2023.",
        automaton=automaton,
        cfg=cfg,
    )
    entity_ids = {link.entity_id for link in out}
    assert 10001 in entity_ids


def test_long_name_always_fires() -> None:
    automaton = build_automaton(_target_rows())
    cfg = _planetary_cfg()
    out = gated.link_paper(
        bibcode="x_0003",
        title="Itokawa observations",
        abstract="The surface of Itokawa shows boulders and gravel.",
        automaton=automaton,
        cfg=cfg,
    )
    entity_ids = {link.entity_id for link in out}
    assert 10002 in entity_ids


def test_confidence_scales_with_evidence() -> None:
    automaton = build_automaton(_target_rows())
    cfg = _planetary_cfg()
    # Long name with full canonical match, repeat hits, and disambiguator.
    out = gated.link_paper(
        bibcode="x_0004",
        title="Comet Hale-Bopp dust morphology",
        abstract=(
            "Comet Hale-Bopp showed strong jet activity. "
            "We tracked Comet Hale-Bopp during its 1997 perihelion. "
            "The comet's dust spectrum was unprecedented."
        ),
        automaton=automaton,
        cfg=cfg,
    )
    assert len(out) == 1
    link = out[0]
    assert link.entity_id == 10003
    assert link.confidence >= 0.90
    assert link.repeat_count >= 2
    assert link.context_hit is True


def test_word_boundary_substring_rejected() -> None:
    automaton = build_automaton(
        [
            EntityRow(
                entity_id=20001,
                surface="Mathilde",
                canonical_name="Mathilde",
                ambiguity_class="unique",
                is_alias=False,
            )
        ]
    )
    cfg = _planetary_cfg()
    # "Mathildes" should NOT match "Mathilde" because of word boundary.
    out = gated.link_paper(
        bibcode="x_0005",
        title="On asteroid surfaces",
        abstract="Mathildes are not a real word but should not match.",
        automaton=automaton,
        cfg=cfg,
    )
    assert out == []


def test_evidence_serialization_roundtrip() -> None:
    link = gated.GatedLink(
        bibcode="b1",
        entity_id=42,
        confidence=0.85,
        matched_surface="Itokawa",
        is_alias=False,
        surface_len=7,
        repeat_count=2,
        context_hit=True,
        field_seen=("abstract", "title"),
    )
    blob = gated._evidence_json(link)
    assert "Itokawa" in blob
    assert '"is_alias":false' in blob
    assert '"context_hit":true' in blob


def test_has_disambiguator_word_boundary() -> None:
    cfg = _planetary_cfg()
    # "asteroidal" should NOT match "asteroid" word-boundary token.
    assert not gated._has_disambiguator(
        "asteroidal cosmochemistry", cfg.short_name_disambiguators
    )
    # Plural "asteroids" is a separate sentinel and IS in our config? It
    # is not in the test cfg ("asteroid", "comet", "moon", "spacecraft")
    # — so "asteroids" should not match "asteroid".
    assert not gated._has_disambiguator(
        "asteroids in the main belt", ("asteroid",)
    )
    # Whole word match.
    assert gated._has_disambiguator(
        "the asteroid Bennu", ("asteroid",)
    )


def test_summary_writer_emits_yield_tables(tmp_path: pathlib.Path) -> None:
    stats = gated.GatedStats(
        papers_scanned=100,
        papers_with_links=42,
        candidates_generated=55,
        rows_inserted=55,
        short_name_dropped=3,
        per_arxiv_class_yield={"astro-ph.EP": 30, "physics.space-ph": 12},
        per_bibstem_yield={"Icar": 20, "M&PS": 10},
    )
    out = tmp_path / "summary.md"
    gated.write_summary(stats, out, wall_seconds=12.5, cohort_size=200)
    content = out.read_text()
    assert "astro-ph.EP" in content
    assert "Icar" in content
    assert "Cohort size (defined) | 200" in content
    assert "Wall time | 12s" in content


# ---------------------------------------------------------------------------
# Integration fixture against scix_test
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def dsn() -> str:
    test_dsn = get_test_dsn()
    if test_dsn is None:
        pytest.skip(
            "SCIX_TEST_DSN must be set to a non-production DSN for tier-3 tests"
        )
    return test_dsn


_SEED_PAPERS: list[tuple[str, list[str], list[str], str, str]] = [
    # (bibcode, arxiv_class, bibstem, title, abstract)
    (
        "test_p2v_0001",
        ["astro-ph.EP"],
        [],
        "OSIRIS-REx sample analysis",
        "Samples returned from the asteroid Bennu show carbonates and clays.",
    ),
    (
        "test_p2v_0002",
        [],
        ["Icar"],
        "Itokawa surface morphology",
        "We mapped the Itokawa surface using Hayabusa imagery.",
    ),
    (
        "test_p2v_0003",
        ["astro-ph.GA"],  # NOT in cohort
        [],
        "Galaxy-scale survey",
        "We discuss high-redshift galaxies and Bennu the bird god of Egypt.",
    ),
    (
        "test_p2v_0004",
        [],
        ["Icar"],
        "A trip back to Bennu",
        "Future missions to Bennu will collect more material from this asteroid.",
    ),
]

_SEED_TARGETS: list[tuple[str, str | None, str | None]] = [
    # (canonical_name, ambiguity_class, link_policy) — link_policy NULL exercises
    # the relaxed filter; ambiguity_class NULL exercises the entity_type/source
    # branch (this is the dominant case in production).
    ("Bennu", None, None),
    ("Itokawa", "domain_safe", None),
    ("Eros", "banned", None),  # banned — must NOT be added to automaton
    ("ZetaPolicy", None, "llm_only"),  # demoted — must NOT be added
]


def _cleanup(conn: psycopg.Connection) -> None:
    bibcodes = [p[0] for p in _SEED_PAPERS]
    canonicals = [t[0] for t in _SEED_TARGETS]
    with conn.cursor() as cur:
        cur.execute(
            "DELETE FROM document_entities WHERE bibcode = ANY(%s)",
            (bibcodes,),
        )
        cur.execute(
            "DELETE FROM entity_aliases "
            " WHERE entity_id IN ("
            "     SELECT id FROM entities "
            "      WHERE canonical_name = ANY(%s) AND source = 'unit_test_p2v'"
            " )",
            (canonicals,),
        )
        cur.execute(
            "DELETE FROM entities WHERE canonical_name = ANY(%s) AND source = 'unit_test_p2v'",
            (canonicals,),
        )
        cur.execute(
            "DELETE FROM papers WHERE bibcode = ANY(%s)",
            (bibcodes,),
        )
    conn.commit()


def _seed(conn: psycopg.Connection) -> dict[str, int]:
    _cleanup(conn)
    name_to_id: dict[str, int] = {}
    with conn.cursor() as cur:
        for bibcode, arxiv_class, bibstem, title, abstract in _SEED_PAPERS:
            cur.execute(
                "INSERT INTO papers (bibcode, arxiv_class, bibstem, title, abstract) "
                " VALUES (%s, %s, %s, %s, %s)",
                (bibcode, arxiv_class, bibstem, title, abstract),
            )
        for canonical, ambig, link_policy in _SEED_TARGETS:
            cur.execute(
                "INSERT INTO entities "
                "       (canonical_name, entity_type, source, "
                "        ambiguity_class, link_policy) "
                "VALUES (%s, 'target', 'unit_test_p2v', "
                "        %s::entity_ambiguity_class, %s::entity_link_policy) "
                "RETURNING id",
                (canonical, ambig, link_policy),
            )
            row = cur.fetchone()
            assert row is not None
            name_to_id[canonical] = int(row[0])
    conn.commit()
    return name_to_id


@pytest.fixture()
def seeded_conn(dsn: str) -> Iterator[tuple[psycopg.Connection, dict[str, int]]]:
    conn = psycopg.connect(dsn)
    try:
        ids = _seed(conn)
        yield conn, ids
    finally:
        try:
            _cleanup(conn)
        finally:
            conn.close()


# ---------------------------------------------------------------------------
# Integration tests — wired against the seeded fixture, but use a private
# fetch query so we only see the unit_test_p2v entities (avoids picking up
# the 1.49M production SsODNet rows when running against scix_test).
# ---------------------------------------------------------------------------


def _fetch_test_target_rows(
    conn: psycopg.Connection,
    *,
    stop_words: frozenset[str] = frozenset(),
    min_canonical_len: int = 0,
) -> list[EntityRow]:
    """Same shape as gated.fetch_target_entity_rows but scoped to the
    test fixture source so the integration test does not depend on a
    pristine entities table."""
    rows: list[EntityRow] = []
    with conn.cursor() as cur:
        cur.execute(
            "SELECT id, canonical_name FROM entities "
            " WHERE source = 'unit_test_p2v' "
            "   AND entity_type = 'target' "
            "   AND (link_policy IS NULL OR link_policy <> 'llm_only') "
            "   AND (ambiguity_class IS NULL OR ambiguity_class <> 'banned')"
        )
        for entity_id, canonical in cur.fetchall():
            rows.append(
                EntityRow(
                    entity_id=int(entity_id),
                    surface=canonical,
                    canonical_name=canonical,
                    ambiguity_class="unique",
                    is_alias=False,
                )
            )
    return rows


class TestDisciplineGatedIntegration:
    def test_seeded_targets_roundtrip(
        self, seeded_conn: tuple[psycopg.Connection, dict[str, int]]
    ) -> None:
        conn, ids = seeded_conn
        rows = _fetch_test_target_rows(conn)
        names = {r.canonical_name for r in rows}
        # Bennu + Itokawa are eligible. Eros (banned) and ZetaPolicy
        # (llm_only) must NOT show up.
        assert "Bennu" in names
        assert "Itokawa" in names
        assert "Eros" not in names
        assert "ZetaPolicy" not in names

    def test_cohort_filter_excludes_non_planetary_paper(
        self, seeded_conn: tuple[psycopg.Connection, dict[str, int]]
    ) -> None:
        conn, _ = seeded_conn
        cfg = _planetary_cfg()
        sql, params = gated._build_cohort_sql(cfg, bibcode_prefix="test_p2v_")
        with conn.cursor() as cur:
            cur.execute(sql, params)
            cohort_bibcodes = {r[0] for r in cur.fetchall()}
        # 0001 (astro-ph.EP), 0002 + 0004 (Icar) are in cohort.
        # 0003 (astro-ph.GA only) is NOT.
        assert "test_p2v_0001" in cohort_bibcodes
        assert "test_p2v_0002" in cohort_bibcodes
        assert "test_p2v_0004" in cohort_bibcodes
        assert "test_p2v_0003" not in cohort_bibcodes

    def test_run_inserts_tier3_rows_for_cohort_only(
        self, seeded_conn: tuple[psycopg.Connection, dict[str, int]]
    ) -> None:
        conn, ids = seeded_conn

        # Monkey-patch the entity fetcher to use only test entities so
        # the run does not depend on production SsODNet data being
        # present in scix_test.
        original_fetch = gated.fetch_target_entity_rows
        gated.fetch_target_entity_rows = _fetch_test_target_rows  # type: ignore[assignment]
        try:
            stats = gated.run(
                conn,
                cfg=_planetary_cfg(),
                bibcode_prefix="test_p2v_",
                dry_run=False,
            )
        finally:
            gated.fetch_target_entity_rows = original_fetch  # type: ignore[assignment]

        assert stats.papers_scanned == 3  # 0001, 0002, 0004 (cohort)
        assert stats.papers_with_links >= 2

        with conn.cursor() as cur:
            cur.execute(
                "SELECT bibcode, entity_id, link_type, tier, match_method "
                "  FROM document_entities "
                " WHERE bibcode LIKE %s "
                "   AND link_type = %s",
                ("test_p2v_%", gated.LINK_TYPE),
            )
            rows = cur.fetchall()

        assert rows, "expected at least one tier-3 row"
        seen_bibcodes = {r[0] for r in rows}
        assert "test_p2v_0003" not in seen_bibcodes  # excluded by cohort
        for _, _entity_id, link_type, tier, method in rows:
            assert link_type == gated.LINK_TYPE
            assert tier == gated.TIER
            assert method == gated.MATCH_METHOD

    def test_dry_run_rolls_back(
        self, seeded_conn: tuple[psycopg.Connection, dict[str, int]]
    ) -> None:
        conn, _ = seeded_conn
        original_fetch = gated.fetch_target_entity_rows
        gated.fetch_target_entity_rows = _fetch_test_target_rows  # type: ignore[assignment]
        try:
            stats = gated.run(
                conn,
                cfg=_planetary_cfg(),
                bibcode_prefix="test_p2v_",
                dry_run=True,
            )
        finally:
            gated.fetch_target_entity_rows = original_fetch  # type: ignore[assignment]

        assert stats.rows_inserted >= 1
        with conn.cursor() as cur:
            cur.execute(
                "SELECT count(*) FROM document_entities "
                " WHERE bibcode LIKE %s AND link_type = %s",
                ("test_p2v_%", gated.LINK_TYPE),
            )
            row = cur.fetchone()
            assert row is not None
            assert row[0] == 0  # dry-run rolled back
