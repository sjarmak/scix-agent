"""UAT (Unified Astronomy Thesaurus) concept hierarchy loader and mapper.

Downloads/parses the UAT SKOS vocabulary, loads it into PostgreSQL,
and maps paper keywords to UAT concepts.
"""

from __future__ import annotations

import logging
import time
import urllib.error
import urllib.request
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

import psycopg
from psycopg.rows import dict_row

from scix.db import get_connection

logger = logging.getLogger(__name__)

# UAT SKOS namespace constants
SKOS = "http://www.w3.org/2004/02/skos/core#"
RDF = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
RDFS = "http://www.w3.org/2000/01/rdf-schema#"
UAT_NS = "http://astrothesaurus.org/uat/"

UAT_SKOS_URL = "https://raw.githubusercontent.com/astrothesaurus/UAT/v6.0.0/UAT.rdf"

# Default download destination
_DEFAULT_DEST = Path("data/UAT.rdf")


@dataclass(frozen=True)
class UATConcept:
    """A single UAT concept with labels, definition, and hierarchy level."""

    concept_id: str  # e.g. "http://astrothesaurus.org/uat/1"
    preferred_label: str  # e.g. "Galaxies"
    alternate_labels: tuple[str, ...]
    definition: str | None
    level: int | None  # computed after relationship loading


@dataclass(frozen=True)
class UATRelationship:
    """A parent-child relationship between two UAT concepts."""

    parent_id: str
    child_id: str


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------


def download_uat(dest: Path | None = None) -> Path:
    """Download the UAT SKOS RDF/XML from the canonical location.

    Uses urllib.request (no external dependencies). Skips download if the
    destination file already exists and is non-empty.

    Args:
        dest: Path to save the downloaded file. Defaults to data/UAT-owl-skos.rdf.

    Returns:
        Path to the downloaded (or existing) file.

    Raises:
        urllib.error.URLError: If the download fails after retries.
    """
    if dest is None:
        dest = _DEFAULT_DEST

    if dest.exists() and dest.stat().st_size > 0:
        logger.info("UAT SKOS file already exists at %s, skipping download", dest)
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading UAT SKOS vocabulary from %s", UAT_SKOS_URL)

    max_retries = 3
    for attempt in range(1, max_retries + 1):
        try:
            req = urllib.request.Request(
                UAT_SKOS_URL,
                headers={"User-Agent": "scix-experiments/1.0"},
            )
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = resp.read()

            dest.write_bytes(data)
            logger.info(
                "Downloaded UAT SKOS vocabulary: %d bytes -> %s",
                len(data),
                dest,
            )
            return dest

        except (urllib.error.URLError, OSError) as exc:
            if attempt < max_retries:
                wait = 2**attempt
                logger.warning(
                    "Download attempt %d/%d failed: %s — retrying in %ds",
                    attempt,
                    max_retries,
                    exc,
                    wait,
                )
                time.sleep(wait)
            else:
                logger.error(
                    "Failed to download UAT SKOS after %d attempts: %s",
                    max_retries,
                    exc,
                )
                raise

    # Unreachable, but satisfies type checker
    raise RuntimeError("download_uat: unexpected exit from retry loop")


# ---------------------------------------------------------------------------
# SKOS Parsing
# ---------------------------------------------------------------------------


def parse_skos(path: Path) -> tuple[list[UATConcept], list[UATRelationship]]:
    """Parse a UAT SKOS RDF/XML file into concepts and relationships.

    Extracts concepts from ``<skos:Concept>`` and ``<rdf:Description>``
    elements whose ``rdf:about`` attribute starts with the UAT namespace.
    Uses only ``skos:broader`` to derive parent-child relationships (avoids
    duplicates from redundant ``skos:narrower``). Computes hierarchy levels
    via BFS from root concepts (those with no broader concept).

    Args:
        path: Path to the SKOS RDF/XML file.

    Returns:
        Tuple of (concepts, relationships).
    """
    tree = ET.parse(path)
    root = tree.getroot()

    # Collect raw concept data keyed by URI
    raw_concepts: dict[str, dict[str, Any]] = {}
    broader_map: dict[str, list[str]] = {}  # child_id -> [parent_ids]
    children_map: dict[str, list[str]] = {}  # parent_id -> [child_ids]

    # Find concept elements: <skos:Concept> or <rdf:Description>
    concept_tags = [f"{{{SKOS}}}Concept", f"{{{RDF}}}Description"]

    for tag in concept_tags:
        for elem in root.iter(tag):
            about = elem.get(f"{{{RDF}}}about", "")
            if not about.startswith(UAT_NS):
                continue

            pref_label_el = elem.find(f"{{{SKOS}}}prefLabel")
            pref_label = (
                pref_label_el.text.strip()
                if pref_label_el is not None and pref_label_el.text
                else ""
            )
            if not pref_label:
                continue

            alt_labels: list[str] = []
            for alt_el in elem.findall(f"{{{SKOS}}}altLabel"):
                if alt_el.text and alt_el.text.strip():
                    alt_labels.append(alt_el.text.strip())

            definition_el = elem.find(f"{{{SKOS}}}definition")
            definition = (
                definition_el.text.strip()
                if definition_el is not None and definition_el.text
                else None
            )

            raw_concepts[about] = {
                "preferred_label": pref_label,
                "alternate_labels": tuple(alt_labels),
                "definition": definition,
            }

            # broader relationships: this concept has a broader parent
            for broader_el in elem.findall(f"{{{SKOS}}}broader"):
                parent_uri = broader_el.get(f"{{{RDF}}}resource", "")
                if parent_uri.startswith(UAT_NS):
                    broader_map.setdefault(about, []).append(parent_uri)
                    children_map.setdefault(parent_uri, []).append(about)

    # Build relationships from broader (child -> parent means parent -> child edge)
    # Filter to only include relationships where both endpoints were parsed as
    # concepts.  The SKOS file can reference broader parents that lack a
    # prefLabel (deprecated/stub entries) and therefore aren't in raw_concepts.
    # Including those would cause FK violations when loading into the database.
    relationships: list[UATRelationship] = []
    seen_rels: set[tuple[str, str]] = set()
    skipped_rels = 0
    for child_id, parent_ids in broader_map.items():
        for parent_id in parent_ids:
            if parent_id not in raw_concepts or child_id not in raw_concepts:
                skipped_rels += 1
                continue
            pair = (parent_id, child_id)
            if pair not in seen_rels:
                seen_rels.add(pair)
                relationships.append(UATRelationship(parent_id=parent_id, child_id=child_id))

    if skipped_rels:
        logger.warning("Skipped %d relationships referencing unknown concepts", skipped_rels)

    # Compute levels via BFS from root concepts (no broader parent)
    all_concept_ids = set(raw_concepts.keys())
    root_ids = all_concept_ids - set(broader_map.keys())
    levels: dict[str, int] = {}

    queue: deque[tuple[str, int]] = deque()
    for rid in root_ids:
        queue.append((rid, 0))
        levels[rid] = 0

    while queue:
        current_id, current_level = queue.popleft()
        for child_id in children_map.get(current_id, []):
            if child_id not in levels:
                levels[child_id] = current_level + 1
                queue.append((child_id, current_level + 1))

    # Build final concept list
    concepts: list[UATConcept] = []
    for concept_id, data in raw_concepts.items():
        concepts.append(
            UATConcept(
                concept_id=concept_id,
                preferred_label=data["preferred_label"],
                alternate_labels=data["alternate_labels"],
                definition=data["definition"],
                level=levels.get(concept_id),
            )
        )

    logger.info(
        "Parsed %d concepts, %d relationships, %d root concepts",
        len(concepts),
        len(relationships),
        len(root_ids),
    )
    return concepts, relationships


# ---------------------------------------------------------------------------
# Database Loading
# ---------------------------------------------------------------------------

_CONCEPT_STAGING_DROP = "DROP TABLE IF EXISTS _uat_concept_staging"
_CONCEPT_STAGING_CREATE = (
    "CREATE TEMP TABLE _uat_concept_staging " "(LIKE uat_concepts INCLUDING DEFAULTS)"
)

_CONCEPT_MERGE_SQL = """
    INSERT INTO uat_concepts (concept_id, preferred_label, alternate_labels, definition, level)
    SELECT concept_id, preferred_label, alternate_labels, definition, level
    FROM _uat_concept_staging
    ON CONFLICT (concept_id) DO UPDATE SET
        preferred_label = EXCLUDED.preferred_label,
        alternate_labels = EXCLUDED.alternate_labels,
        definition = EXCLUDED.definition,
        level = EXCLUDED.level
"""

_REL_STAGING_DROP = "DROP TABLE IF EXISTS _uat_rel_staging"
_REL_STAGING_CREATE = "CREATE TEMP TABLE _uat_rel_staging (parent_id TEXT, child_id TEXT)"

# Count staging rows that would be dropped by the uat_concepts FK check.  Used
# for diagnostic logging so silent FK drops don't look like a successful load.
_REL_FK_DROP_COUNT_SQL = """
    SELECT COUNT(*) FROM (
        SELECT DISTINCT s.parent_id, s.child_id
        FROM _uat_rel_staging s
        WHERE NOT EXISTS (SELECT 1 FROM uat_concepts c WHERE c.concept_id = s.parent_id)
           OR NOT EXISTS (SELECT 1 FROM uat_concepts c WHERE c.concept_id = s.child_id)
    ) AS dropped
"""

# Insert only rows whose parent_id and child_id both exist in uat_concepts so
# the FK check never raises.  RETURNING 1 lets us count *newly-inserted* rows
# unambiguously — cursor.rowcount on INSERT ... ON CONFLICT DO NOTHING reports
# 0 on re-runs where the rows already exist, which masks successful loads.
_REL_MERGE_SQL = """
    WITH inserted AS (
        INSERT INTO uat_relationships (parent_id, child_id)
        SELECT DISTINCT s.parent_id, s.child_id
        FROM _uat_rel_staging s
        WHERE EXISTS (SELECT 1 FROM uat_concepts c WHERE c.concept_id = s.parent_id)
          AND EXISTS (SELECT 1 FROM uat_concepts c WHERE c.concept_id = s.child_id)
        ON CONFLICT (parent_id, child_id) DO NOTHING
        RETURNING 1
    )
    SELECT COUNT(*) FROM inserted
"""

# Authoritative "present after load" count: how many (parent, child) pairs
# from the staging table are now in uat_relationships.  This is what the
# caller actually wants — it stays correct across re-runs, unlike rowcount.
_REL_PRESENT_COUNT_SQL = """
    SELECT COUNT(*) FROM uat_relationships r
    WHERE EXISTS (
        SELECT 1 FROM _uat_rel_staging s
        WHERE s.parent_id = r.parent_id AND s.child_id = r.child_id
    )
"""


def load_concepts(conn: psycopg.Connection, concepts: list[UATConcept]) -> int:
    """Load UAT concepts into uat_concepts via staging table pattern.

    Args:
        conn: Database connection.
        concepts: List of UATConcept instances.

    Returns:
        Number of concepts upserted.
    """
    t0 = time.monotonic()

    with conn.cursor() as cur:
        cur.execute(_CONCEPT_STAGING_DROP)
        cur.execute(_CONCEPT_STAGING_CREATE)
        with cur.copy(
            "COPY _uat_concept_staging "
            "(concept_id, preferred_label, alternate_labels, definition, level) "
            "FROM STDIN"
        ) as copy:
            for c in concepts:
                # Format alternate_labels as PostgreSQL array literal
                alt_labels_pg = _pg_text_array(c.alternate_labels)
                copy.write_row(
                    (
                        c.concept_id,
                        c.preferred_label,
                        alt_labels_pg,
                        c.definition,
                        c.level,
                    )
                )
        cur.execute(_CONCEPT_MERGE_SQL)
        count = cur.rowcount

    conn.commit()
    elapsed = time.monotonic() - t0
    logger.info("Loaded %d concepts in %.1fs", count, elapsed)
    return count


def load_relationships(conn: psycopg.Connection, rels: list[UATRelationship]) -> int:
    """Load UAT relationships into uat_relationships via staging table pattern.

    Stages the input via a TEMP table + COPY, runs an FK-safe INSERT that skips
    pairs whose endpoints are missing from ``uat_concepts`` (instead of raising),
    then returns the count of staged pairs actually present in the target table.

    The returned value is the count of **matching rows now present**, not just
    newly-inserted rows.  ``ON CONFLICT DO NOTHING`` on an INSERT ... SELECT
    reports ``rowcount == 0`` when every row already exists, which would mask a
    successful load on re-runs and give the false impression that no rows landed.
    Using a present-count instead keeps the return value meaningful across
    idempotent loads and across FK drops (see WARN log below).

    Args:
        conn: Database connection.
        rels: List of UATRelationship instances.

    Returns:
        Number of input relationships present in ``uat_relationships`` after
        load.  Equals ``len(rels)`` when every input row references valid
        concepts; lower if any rows were skipped by the FK check.
    """
    t0 = time.monotonic()

    with conn.cursor() as cur:
        cur.execute(_REL_STAGING_DROP)
        cur.execute(_REL_STAGING_CREATE)
        with cur.copy("COPY _uat_rel_staging (parent_id, child_id) FROM STDIN") as copy:
            for r in rels:
                copy.write_row((r.parent_id, r.child_id))

        cur.execute(_REL_FK_DROP_COUNT_SQL)
        fk_dropped = cur.fetchone()[0]
        if fk_dropped:
            logger.warning(
                "Skipping %d relationships with parent_id or child_id "
                "missing from uat_concepts (FK preflight)",
                fk_dropped,
            )

        cur.execute(_REL_MERGE_SQL)
        newly_inserted = cur.fetchone()[0]

        cur.execute(_REL_PRESENT_COUNT_SQL)
        present = cur.fetchone()[0]

    conn.commit()
    elapsed = time.monotonic() - t0
    logger.info(
        "Loaded relationships: %d newly inserted, %d present in table, " "%d FK-dropped in %.1fs",
        newly_inserted,
        present,
        fk_dropped,
        elapsed,
    )
    return present


def _pg_text_array(values: tuple[str, ...] | list[str]) -> str:
    """Format a sequence of strings as a PostgreSQL text array literal.

    Escapes backslashes and double quotes within values.

    Examples:
        >>> _pg_text_array(())
        '{}'
        >>> _pg_text_array(("Galaxy",))
        '{"Galaxy"}'
        >>> _pg_text_array(("one", 'two "quoted"'))
        '{"one","two \\\\"quoted\\\\"}'
    """
    if not values:
        return "{}"
    escaped = []
    for v in values:
        v = v.replace("\\", "\\\\").replace('"', '\\"')
        escaped.append(f'"{v}"')
    return "{" + ",".join(escaped) + "}"


# ---------------------------------------------------------------------------
# Keyword Mapping
# ---------------------------------------------------------------------------


def map_keywords_exact(conn: psycopg.Connection, batch_size: int = 10_000) -> int:
    """Map papers.keywords to UAT concepts via case-insensitive exact match.

    Matches against both preferred labels and alternate labels.

    Args:
        conn: Database connection.
        batch_size: Not used for set-based operation but kept for API consistency.

    Returns:
        Total number of mappings created.
    """
    t0 = time.monotonic()
    total = 0

    with conn.cursor() as cur:
        # Build a materialized lookup table of all UAT labels (preferred + alternate)
        # so we can join efficiently against papers.keywords.
        cur.execute("""
            CREATE TEMP TABLE IF NOT EXISTS _uat_labels (
                concept_id TEXT NOT NULL,
                label_lower TEXT NOT NULL
            ) ON COMMIT PRESERVE ROWS
        """)
        cur.execute("TRUNCATE _uat_labels")
        cur.execute("""
            INSERT INTO _uat_labels (concept_id, label_lower)
            SELECT concept_id, lower(preferred_label) FROM uat_concepts
            UNION ALL
            SELECT concept_id, lower(al)
            FROM uat_concepts, LATERAL unnest(alternate_labels) AS al
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS _idx_uat_labels ON _uat_labels (label_lower)")
        conn.commit()
        logger.info("Built UAT label lookup table (%d entries)", cur.rowcount)

    # Process in year batches to avoid OOM on 32M-paper joins
    with conn.cursor() as cur:
        cur.execute("SELECT DISTINCT year FROM papers WHERE keywords IS NOT NULL ORDER BY year")
        years = [row[0] for row in cur.fetchall()]

    for year in years:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO paper_uat_mappings (bibcode, concept_id, match_type)
                SELECT DISTINCT p.bibcode, ul.concept_id, 'exact'
                FROM papers p,
                     LATERAL unnest(p.keywords) AS kw
                JOIN _uat_labels ul ON lower(kw) = ul.label_lower
                WHERE p.keywords IS NOT NULL AND p.year = %s
                ON CONFLICT DO NOTHING
            """,
                (year,),
            )
            batch_count = cur.rowcount
            total += batch_count
        conn.commit()
        if batch_count > 0:
            logger.info("Year %s: %d mappings (running total: %d)", year, batch_count, total)

    elapsed = time.monotonic() - t0
    logger.info("Total keyword mappings created: %d in %.1fs", total, elapsed)
    return total


# ---------------------------------------------------------------------------
# Hierarchical Search
# ---------------------------------------------------------------------------


def hierarchical_search(
    conn: psycopg.Connection, concept_id: str, limit: int = 50
) -> list[dict[str, Any]]:
    """Find papers matching a UAT concept or any of its descendants.

    Uses a recursive CTE to walk the concept hierarchy downward, then
    joins through paper_uat_mappings to return matching papers ordered
    by citation count.

    Args:
        conn: Database connection.
        concept_id: URI of the root concept to search from.
        limit: Maximum number of papers to return.

    Returns:
        List of dicts with keys: bibcode, title, first_author, year,
        citation_count.
    """
    query = """
        WITH RECURSIVE descendants AS (
            SELECT child_id AS concept_id
            FROM uat_relationships
            WHERE parent_id = %(concept_id)s
            UNION
            SELECT r.child_id
            FROM uat_relationships r
            JOIN descendants d ON r.parent_id = d.concept_id
        ),
        all_concepts AS (
            SELECT %(concept_id)s AS concept_id
            UNION ALL
            SELECT concept_id FROM descendants
        )
        SELECT DISTINCT p.bibcode, p.title, p.first_author, p.year, p.citation_count
        FROM all_concepts ac
        JOIN paper_uat_mappings m ON m.concept_id = ac.concept_id
        JOIN papers p ON p.bibcode = m.bibcode
        ORDER BY p.citation_count DESC NULLS LAST
        LIMIT %(limit)s
    """

    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(query, {"concept_id": concept_id, "limit": limit})
        return cur.fetchall()


# ---------------------------------------------------------------------------
# Full Pipeline
# ---------------------------------------------------------------------------


def run_pipeline(skos_path: Path | None = None, dsn: str | None = None) -> None:
    """Run the full UAT loading pipeline.

    Steps:
        1. Download UAT SKOS vocabulary (if not provided)
        2. Parse SKOS RDF/XML
        3. Load concepts into uat_concepts
        4. Load relationships into uat_relationships
        5. Map paper keywords to UAT concepts (exact match)
        6. Log summary statistics

    Args:
        skos_path: Path to a local SKOS file. Downloads if None.
        dsn: Database connection string. Uses SCIX_DSN or default if None.
    """
    t_pipeline = time.monotonic()

    # Step 1: Download
    if skos_path is None:
        skos_path = download_uat()
    elif not skos_path.exists():
        raise FileNotFoundError(f"SKOS file not found: {skos_path}")

    # Step 2: Parse
    logger.info("Parsing SKOS vocabulary from %s", skos_path)
    concepts, relationships = parse_skos(skos_path)
    logger.info("Parsed %d concepts, %d relationships", len(concepts), len(relationships))

    # Step 3-5: Load into database
    conn = get_connection(dsn)
    try:
        concept_count = load_concepts(conn, concepts)
        rel_count = load_relationships(conn, relationships)
        mapping_count = map_keywords_exact(conn)

        elapsed = time.monotonic() - t_pipeline
        logger.info(
            "UAT pipeline complete: %d concepts, %d relationships, " "%d keyword mappings in %.1fs",
            concept_count,
            rel_count,
            mapping_count,
            elapsed,
        )
    finally:
        conn.close()
