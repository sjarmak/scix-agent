"""Integration tests for migration 052 + scripts/generate_community_labels.py.

These tests write to and delete from ``papers`` / ``paper_metrics`` /
``communities`` and therefore require ``SCIX_TEST_DSN`` to be set to a
non-production DB. They SKIP cleanly otherwise so ``pytest`` in a plain
checkout never touches the production ``scix`` database.

Covers:
    (a) TF-IDF ordering on a known fixture — terms unique to one
        community outrank terms present in all communities.
    (b) Label strings are byte-stable given deterministic input.
    (c) Unknown signal value is rejected by argparse (SystemExit).
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys

import psycopg
import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
SCRIPTS_DIR = REPO_ROOT / "scripts"
MIGRATION_051_PATH = REPO_ROOT / "migrations" / "051_community_semantic_columns.sql"
MIGRATION_052_PATH = REPO_ROOT / "migrations" / "052_communities_signal.sql"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tests.helpers import get_test_dsn  # noqa: E402

TEST_DSN = get_test_dsn()

pytestmark = pytest.mark.skipif(
    TEST_DSN is None,
    reason=(
        "SCIX_TEST_DSN is not set or points at production — "
        "generate_community_labels tests require a dedicated test DB"
    ),
)


TEST_BIBCODE_PREFIX = "COMMLABELTEST."
SEED = 42


def _load_script_module():
    mod_name = "generate_community_labels_script"
    if mod_name in sys.modules:
        return sys.modules[mod_name]
    spec = importlib.util.spec_from_file_location(
        mod_name,
        SCRIPTS_DIR / "generate_community_labels.py",
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def dsn() -> str:
    assert TEST_DSN is not None
    return TEST_DSN


@pytest.fixture(scope="module")
def applied_migrations(dsn: str) -> None:
    """Apply migrations 051 + 052 to the test DB (idempotent)."""
    for path in (MIGRATION_051_PATH, MIGRATION_052_PATH):
        sql = path.read_text()
        with psycopg.connect(dsn) as c:
            c.autocommit = True
            with c.cursor() as cur:
                cur.execute(sql)


# Community 1: 7 papers, arxiv=cs.LG, keywords={neural, transformer, common_kw}
# Community 2: 7 papers, arxiv=astro-ph.GA, keywords={galaxy, quasar, common_kw}
# Community 3: 7 papers, arxiv=physics.flu-dyn, keywords={vortex, common_kw}
# common_kw appears in all 3 communities -> IDF = log(3/(1+3)) < 0 vs
# terms unique to one community with IDF = log(3/(1+1)) = log(1.5) > 0.
_FIXTURE_SPEC: list[tuple[int, int, tuple[str, ...], tuple[str, ...]]] = [
    (1, 7, ("cs.LG",),          ("neural", "transformer", "common_kw")),
    (2, 7, ("astro-ph.GA",),    ("galaxy", "quasar", "common_kw")),
    (3, 7, ("physics.flu-dyn",),("vortex", "common_kw")),
]


def _insert_fixture(dsn: str) -> list[str]:
    """Insert a tiny deterministic corpus for 3 communities. Returns bibcodes."""
    all_bibcodes: list[str] = []
    with psycopg.connect(dsn) as c:
        c.autocommit = False
        with c.cursor() as cur:
            for cid, n_papers, arxiv, kws in _FIXTURE_SPEC:
                for i in range(n_papers):
                    bib = f"{TEST_BIBCODE_PREFIX}{cid:02d}.{i:03d}"
                    all_bibcodes.append(bib)
                    cur.execute(
                        "INSERT INTO papers (bibcode, title, arxiv_class, "
                        "keyword_norm, citation_count) "
                        "VALUES (%s, %s, %s, %s, %s) "
                        "ON CONFLICT (bibcode) DO UPDATE SET "
                        "  title = EXCLUDED.title, "
                        "  arxiv_class = EXCLUDED.arxiv_class, "
                        "  keyword_norm = EXCLUDED.keyword_norm, "
                        "  citation_count = EXCLUDED.citation_count",
                        (
                            bib,
                            f"fixture paper {bib}",
                            list(arxiv),
                            list(kws),
                            100 - i,  # deterministic citation order
                        ),
                    )
                    cur.execute(
                        "INSERT INTO paper_metrics "
                        "  (bibcode, community_id_coarse, "
                        "   community_id_medium, community_id_fine) "
                        "VALUES (%s, %s, %s, %s) "
                        "ON CONFLICT (bibcode) DO UPDATE SET "
                        "  community_id_coarse = EXCLUDED.community_id_coarse, "
                        "  community_id_medium = EXCLUDED.community_id_medium, "
                        "  community_id_fine   = EXCLUDED.community_id_fine",
                        (bib, cid, cid, cid),
                    )
        c.commit()
    return all_bibcodes


def _delete_fixture(dsn: str) -> None:
    with psycopg.connect(dsn) as c:
        c.autocommit = True
        with c.cursor() as cur:
            cur.execute(
                "DELETE FROM paper_metrics WHERE bibcode LIKE %s",
                (TEST_BIBCODE_PREFIX + "%",),
            )
            cur.execute(
                "DELETE FROM papers WHERE bibcode LIKE %s",
                (TEST_BIBCODE_PREFIX + "%",),
            )
            # Clean up any fixture-derived community rows.
            cur.execute(
                "DELETE FROM communities "
                "WHERE signal = 'citation' AND community_id IN (1,2,3) "
                "  AND resolution IN ('coarse','medium','fine')"
            )


@pytest.fixture
def fixture_data(dsn: str, applied_migrations: None):
    _delete_fixture(dsn)
    _insert_fixture(dsn)
    try:
        yield
    finally:
        _delete_fixture(dsn)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run_script(dsn: str, tmp_path: pathlib.Path, extra_args: list[str] | None = None) -> int:
    mod = _load_script_module()
    args = [
        "--dsn", dsn,
        "--signal", "citation",
        "--seed", str(SEED),
        "--spotcheck-n", "3",
        "--spotcheck-path", str(tmp_path / "spotcheck.md"),
        "--spotcheck-sample-path", str(tmp_path / "spotcheck_sample.md"),
    ]
    if extra_args:
        args.extend(extra_args)
    return mod.main(args)


def _fetch_community_labels(
    dsn: str,
) -> dict[tuple[str, str, int], tuple[str, list[str]]]:
    """Map (signal, resolution, community_id) -> (label, top_keywords)."""
    with psycopg.connect(dsn) as c, c.cursor() as cur:
        cur.execute(
            "SELECT signal, resolution, community_id, label, top_keywords "
            "FROM communities "
            "WHERE signal = 'citation' AND community_id IN (1,2,3) "
            "ORDER BY signal, resolution, community_id"
        )
        return {
            (row[0], row[1], row[2]): (row[3], list(row[4] or []))
            for row in cur.fetchall()
        }


# ---------------------------------------------------------------------------
# Test (a) — TF-IDF ordering on a known fixture
# ---------------------------------------------------------------------------


def test_tfidf_ordering_on_fixture(
    dsn: str,
    fixture_data: None,
    tmp_path: pathlib.Path,
) -> None:
    rc = _run_script(dsn, tmp_path)
    assert rc == 0, f"script returned non-zero exit code {rc}"

    labels = _fetch_community_labels(dsn)

    # Each (signal='citation', resolution, community) must have a label.
    for cid in (1, 2, 3):
        for resolution in ("coarse", "medium", "fine"):
            assert ("citation", resolution, cid) in labels, (
                f"missing label for citation/{resolution}/{cid}"
            )

    # TF-IDF check: terms unique to one community rank higher than the
    # shared 'common_kw' term.
    #
    # For community 1 the unique terms are 'neural' and 'transformer',
    # each with TF=7 and IDF=log(3/(1+1))=log(1.5) > 0.
    # 'common_kw' has TF=7 but IDF=log(3/(1+3))=log(0.75) < 0, so it must
    # rank below the unique terms (and may even be dropped from the top-10
    # if we had more terms — here we only have 3 so it appears last).
    _, c1_top = labels[("citation", "coarse", 1)]
    assert c1_top[0] in ("neural", "transformer"), (
        f"community 1 top term should be neural/transformer, got {c1_top!r}"
    )
    # common_kw must NOT be at position 0 (since it's present in every
    # community its IDF is smaller). It may appear later in the list.
    if "common_kw" in c1_top:
        assert c1_top.index("common_kw") > 0, (
            f"common_kw should not be the top-1 term, got {c1_top!r}"
        )

    # Community 2: top term is galaxy or quasar, not common_kw.
    _, c2_top = labels[("citation", "coarse", 2)]
    assert c2_top[0] in ("galaxy", "quasar"), (
        f"community 2 top term should be galaxy/quasar, got {c2_top!r}"
    )

    # Community 3: top term is vortex, not common_kw.
    _, c3_top = labels[("citation", "coarse", 3)]
    assert c3_top[0] == "vortex", (
        f"community 3 top term should be vortex, got {c3_top!r}"
    )


# ---------------------------------------------------------------------------
# Test (b) — label format is stable given deterministic input
# ---------------------------------------------------------------------------


def test_label_format_is_stable(
    dsn: str,
    fixture_data: None,
    tmp_path: pathlib.Path,
) -> None:
    # First run.
    rc1 = _run_script(dsn, tmp_path / "run1")
    assert rc1 == 0
    labels1 = _fetch_community_labels(dsn)

    # Second run with the same fixture — labels must be byte-equal.
    rc2 = _run_script(dsn, tmp_path / "run2")
    assert rc2 == 0
    labels2 = _fetch_community_labels(dsn)

    assert labels1 == labels2, (
        "labels changed between identical runs — not deterministic"
    )

    # Also check the pure-function label formatter directly.
    mod = _load_script_module()
    assert mod.make_label(["cs.LG"], ["neural", "transformer", "common_kw"]) == (
        "cs.LG · neural / transformer / common_kw"
    )
    assert mod.make_label([], ["vortex", "common_kw"]) == (
        "vortex / common_kw"
    )
    assert mod.make_label(["astro-ph.GA"], []) == "astro-ph.GA"
    assert mod.make_label([], []) == "unlabeled"
    # Truncation: only 2 arxiv + 3 keywords are used.
    assert mod.make_label(
        ["a", "b", "c", "d"],
        ["k1", "k2", "k3", "k4", "k5"],
    ) == "a + b · k1 / k2 / k3"


# ---------------------------------------------------------------------------
# Test (c) — unknown signal value is rejected
# ---------------------------------------------------------------------------


def test_unknown_signal_raises(tmp_path: pathlib.Path) -> None:
    mod = _load_script_module()
    with pytest.raises(SystemExit) as excinfo:
        mod.main([
            "--dsn", "dbname=scix_test",
            "--signal", "bogus",
            "--seed", "1",
            "--spotcheck-n", "1",
            "--spotcheck-path", str(tmp_path / "s.md"),
            "--spotcheck-sample-path", str(tmp_path / "ss.md"),
        ])
    # argparse uses exit code 2 for usage errors.
    assert excinfo.value.code == 2


# ---------------------------------------------------------------------------
# Test (d) — production DSN guard fires without --allow-prod
# ---------------------------------------------------------------------------


def test_refuses_production_dsn_without_allow_prod(tmp_path: pathlib.Path) -> None:
    mod = _load_script_module()
    rc = mod.main([
        "--dsn", "dbname=scix",  # synthetic prod DSN; guard fires first
        "--signal", "all",
        "--seed", "1",
        "--spotcheck-n", "1",
        "--spotcheck-path", str(tmp_path / "s.md"),
        "--spotcheck-sample-path", str(tmp_path / "ss.md"),
    ])
    assert rc == 2, "production DSN without --allow-prod must exit with code 2"
