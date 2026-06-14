--
-- PostgreSQL database dump
--

\restrict lQ9BahwANnSvPg7JQojLPg2fvMXbWqLEu23BoesODgzFGtm8chXet7jrmG1SziY

-- Dumped from database version 16.14 (Ubuntu 16.14-0ubuntu0.24.04.1)
-- Dumped by pg_dump version 16.14 (Ubuntu 16.14-0ubuntu0.24.04.1)

SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET xmloption = content;
SET client_min_messages = warning;
SET row_security = off;

--
-- Name: staging; Type: SCHEMA; Schema: -; Owner: -
--

CREATE SCHEMA staging;


--
-- Name: vector; Type: EXTENSION; Schema: -; Owner: -
--

CREATE EXTENSION IF NOT EXISTS vector WITH SCHEMA public;


--
-- Name: EXTENSION vector; Type: COMMENT; Schema: -; Owner: -
--

COMMENT ON EXTENSION vector IS 'vector data type and ivfflat and hnsw access methods';


--
-- Name: entity_ambiguity_class; Type: TYPE; Schema: public; Owner: -
--

CREATE TYPE public.entity_ambiguity_class AS ENUM (
    'unique',
    'domain_safe',
    'homograph',
    'banned'
);


--
-- Name: entity_link_policy; Type: TYPE; Schema: public; Owner: -
--

CREATE TYPE public.entity_link_policy AS ENUM (
    'open',
    'context_required',
    'llm_only',
    'banned'
);


--
-- Name: embedding_outbox_enqueue(); Type: FUNCTION; Schema: public; Owner: -
--

CREATE FUNCTION public.embedding_outbox_enqueue() RETURNS trigger
    LANGUAGE plpgsql
    AS $$
BEGIN
    IF TG_OP = 'DELETE' THEN
        INSERT INTO embedding_outbox (bibcode, model_name, op)
            VALUES (OLD.bibcode, OLD.model_name, 'DELETE');
        RETURN OLD;
    ELSE
        INSERT INTO embedding_outbox (bibcode, model_name, op)
            VALUES (NEW.bibcode, NEW.model_name, TG_OP);
        RETURN NEW;
    END IF;
END
$$;


--
-- Name: FUNCTION embedding_outbox_enqueue(); Type: COMMENT; Schema: public; Owner: -
--

COMMENT ON FUNCTION public.embedding_outbox_enqueue() IS 'Row-level enqueue for embedding_outbox. Captures every paper_embeddings INSERT/UPDATE/DELETE so the Qdrant sync worker never misses a write (migration 070).';


--
-- Name: papers_external_ids_touch(); Type: FUNCTION; Schema: public; Owner: -
--

CREATE FUNCTION public.papers_external_ids_touch() RETURNS trigger
    LANGUAGE plpgsql
    AS $$
BEGIN
    NEW.updated_at := now();
    RETURN NEW;
END
$$;


SET default_tablespace = '';

SET default_table_access_method = heap;

--
-- Name: papers; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.papers (
    bibcode text NOT NULL,
    title text,
    abstract text,
    year smallint,
    doctype text,
    pub text,
    pub_raw text,
    volume text,
    issue text,
    page text[],
    authors text[],
    first_author text,
    affiliations text[],
    keywords text[],
    arxiv_class text[],
    database text[],
    doi text[],
    identifier text[],
    alternate_bibcode text[],
    bibstem text[],
    bibgroup text[],
    orcid_pub text[],
    orcid_user text[],
    property text[],
    copyright text,
    lang text,
    pubdate text,
    entry_date text,
    indexstamp text,
    citation_count integer,
    read_count integer,
    reference_count integer,
    tsv tsvector,
    body text,
    ack text,
    date text,
    eid text,
    entdate text,
    first_author_norm text,
    page_range text,
    pubnote text,
    series text,
    aff_id text[],
    alternate_title text[],
    author_norm text[],
    caption text[],
    comment text[],
    data text[],
    esources text[],
    facility text[],
    grant_facet text[],
    grant_agencies text[],
    grant_id text[],
    isbn text[],
    issn text[],
    keyword_norm text[],
    keyword_schema text[],
    links_data text[],
    nedid text[],
    nedtype text[],
    orcid_other text[],
    simbid text[],
    vizier text[],
    author_count integer,
    page_count integer,
    citation_count_norm real,
    cite_read_boost real,
    classic_factor real,
    openalex_id text,
    openalex_topics jsonb,
    correction_events jsonb DEFAULT '[]'::jsonb NOT NULL,
    retracted_at timestamp with time zone
);


--
-- Name: COLUMN papers.correction_events; Type: COMMENT; Schema: public; Owner: -
--

COMMENT ON COLUMN public.papers.correction_events IS 'JSONB array of correction events for this paper. Each element: {type, source, doi, date}. type in (retraction, erratum, correction, expression_of_concern, recalibration_supersession). source in (retraction_watch, openalex, crossref, journal_rss). Populated by scripts/ingest_corrections.py (PRD A3 / MH-3 broadened).';


--
-- Name: COLUMN papers.retracted_at; Type: COMMENT; Schema: public; Owner: -
--

COMMENT ON COLUMN public.papers.retracted_at IS 'Denormalized convenience: earliest event date among correction_events where type=retraction. NULL if no retraction event present. Kept in sync by scripts/ingest_corrections.py.';


--
-- Name: papers_is_oa_or_preprint(public.papers); Type: FUNCTION; Schema: public; Owner: -
--

CREATE FUNCTION public.papers_is_oa_or_preprint(p public.papers) RETURNS boolean
    LANGUAGE sql IMMUTABLE PARALLEL SAFE
    AS $$
    SELECT COALESCE('OPENACCESS' = ANY(p.property), FALSE)
        OR COALESCE(array_length(p.arxiv_class, 1) > 0, FALSE);
$$;


--
-- Name: FUNCTION papers_is_oa_or_preprint(p public.papers); Type: COMMENT; Schema: public; Owner: -
--

COMMENT ON FUNCTION public.papers_is_oa_or_preprint(p public.papers) IS 'OA/preprint gate for body-AI pipelines. TRUE iff property contains OPENACCESS or arxiv_class is non-empty. Single source of truth — body-AI scripts call this in WHERE clauses; matching index is idx_papers_is_oa.';


--
-- Name: papers_tsv_trigger(); Type: FUNCTION; Schema: public; Owner: -
--

CREATE FUNCTION public.papers_tsv_trigger() RETURNS trigger
    LANGUAGE plpgsql
    AS $$
BEGIN
    NEW.tsv :=
        setweight(to_tsvector('scix_english', coalesce(NEW.title, '')), 'A') ||
        setweight(to_tsvector('scix_english', coalesce(NEW.abstract, '')), 'B') ||
        setweight(to_tsvector('scix_english', coalesce(array_to_string(NEW.keywords, ' '), '')), 'C');
    RETURN NEW;
END
$$;


--
-- Name: promote_harvest(bigint, jsonb, numeric, numeric, integer); Type: FUNCTION; Schema: public; Owner: -
--

CREATE FUNCTION public.promote_harvest(run_id bigint, floors jsonb DEFAULT '{}'::jsonb, canonical_max numeric DEFAULT 0.02, alias_max numeric DEFAULT 0.05, orphan_threshold integer DEFAULT 1000) RETURNS jsonb
    LANGUAGE plpgsql
    AS $$
DECLARE
    result             JSONB := '{}'::jsonb;
    staging_total      BIGINT := 0;
    alias_staging_tot  BIGINT := 0;
    prod_entity_total  BIGINT := 0;
    prod_alias_total   BIGINT := 0;
    canonical_shrink   NUMERIC := 0;
    alias_shrink       NUMERIC := 0;
    floor_violations   JSONB := '[]'::jsonb;
    orphan_violations  JSONB := '[]'::jsonb;
    schema_errors      JSONB := '[]'::jsonb;
    per_source_json    JSONB := '{}'::jsonb;
    lock_acquired      BOOLEAN := FALSE;
    src_rec            RECORD;
    orphan_rec         RECORD;
    schema_rec         RECORD;
    n_promoted_ent     INTEGER := 0;
    n_promoted_ali     INTEGER := 0;
    n_promoted_ids     INTEGER := 0;
BEGIN
    -- 1. Advisory lock ------------------------------------------------------
    lock_acquired := pg_try_advisory_lock(hashtext('entities_promotion'));
    IF NOT lock_acquired THEN
        RETURN jsonb_build_object(
            'accepted', false,
            'reason', 'lock_unavailable',
            'diff', '{}'::jsonb
        );
    END IF;

    -- 2. Schema compatibility check ----------------------------------------
    -- Every non-metadata column present on entities_staging must also exist on
    -- public.entities (or be one of the staging-only bookkeeping columns).
    FOR schema_rec IN
        SELECT column_name
          FROM information_schema.columns
         WHERE table_schema = 'public'
           AND table_name = 'entities_staging'
           AND column_name NOT IN ('id', 'staging_run_id', 'created_at')
    LOOP
        IF NOT EXISTS (
            SELECT 1 FROM information_schema.columns
             WHERE table_schema = 'public'
               AND table_name = 'entities'
               AND column_name = schema_rec.column_name
        ) THEN
            schema_errors := schema_errors || to_jsonb(schema_rec.column_name);
        END IF;
    END LOOP;

    IF jsonb_array_length(schema_errors) > 0 THEN
        PERFORM pg_advisory_unlock(hashtext('entities_promotion'));
        UPDATE harvest_runs SET status = 'rejected_by_diff'
         WHERE id = run_id;
        RETURN jsonb_build_object(
            'accepted', false,
            'reason', 'schema_mismatch',
            'diff', jsonb_build_object('schema_errors', schema_errors)
        );
    END IF;

    -- 3. Counts -------------------------------------------------------------
    SELECT COUNT(*) INTO staging_total
      FROM entities_staging WHERE staging_run_id = run_id;
    SELECT COUNT(*) INTO alias_staging_tot
      FROM entity_aliases_staging WHERE staging_run_id = run_id;

    -- Only compare against production rows for the sources present in this
    -- staging run. Gives per-source granularity and avoids penalizing a
    -- single-source run against the total corpus.
    WITH run_sources AS (
        SELECT DISTINCT source FROM entities_staging WHERE staging_run_id = run_id
    )
    SELECT COUNT(*) INTO prod_entity_total
      FROM entities e
      JOIN run_sources rs ON rs.source = e.source;

    WITH run_sources AS (
        SELECT DISTINCT source FROM entities_staging WHERE staging_run_id = run_id
    )
    SELECT COUNT(*) INTO prod_alias_total
      FROM entity_aliases ea
      JOIN entities e ON ea.entity_id = e.id
      JOIN run_sources rs ON rs.source = e.source;

    -- Shrinkage (negative value means growth; positive means shrinkage).
    IF prod_entity_total > 0 THEN
        canonical_shrink := (prod_entity_total - staging_total)::NUMERIC
                            / prod_entity_total;
    END IF;
    IF prod_alias_total > 0 THEN
        alias_shrink := (prod_alias_total - alias_staging_tot)::NUMERIC
                        / prod_alias_total;
    END IF;

    -- Per-source breakdown + floor enforcement
    FOR src_rec IN
        SELECT source, COUNT(*) AS n
          FROM entities_staging
         WHERE staging_run_id = run_id
         GROUP BY source
    LOOP
        per_source_json := per_source_json
            || jsonb_build_object(src_rec.source, src_rec.n);

        IF floors ? src_rec.source THEN
            IF src_rec.n < (floors ->> src_rec.source)::BIGINT THEN
                floor_violations := floor_violations
                    || jsonb_build_object(
                        'source', src_rec.source,
                        'observed', src_rec.n,
                        'floor', (floors ->> src_rec.source)::BIGINT
                    );
            END IF;
        END IF;
    END LOOP;

    -- 4. Orphan check -------------------------------------------------------
    -- Entities currently in production (for sources present in this run)
    -- with >= orphan_threshold document_entities rows must have a matching
    -- natural key in the staging run. Otherwise promoting the run would
    -- implicitly "retire" them (they would no longer be reinforced by the
    -- harvest and their provenance would drift).
    FOR orphan_rec IN
        WITH run_sources AS (
            SELECT DISTINCT source FROM entities_staging WHERE staging_run_id = run_id
        ),
        heavy AS (
            SELECT e.id, e.canonical_name, e.entity_type, e.source,
                   COUNT(de.*) AS doc_count
              FROM entities e
              JOIN run_sources rs ON rs.source = e.source
              JOIN document_entities de ON de.entity_id = e.id
             GROUP BY e.id, e.canonical_name, e.entity_type, e.source
            HAVING COUNT(de.*) >= orphan_threshold
        )
        SELECT h.*
          FROM heavy h
         WHERE NOT EXISTS (
            SELECT 1 FROM entities_staging s
             WHERE s.staging_run_id = run_id
               AND s.canonical_name = h.canonical_name
               AND s.entity_type    = h.entity_type
               AND s.source         = h.source
         )
    LOOP
        orphan_violations := orphan_violations
            || jsonb_build_object(
                'id', orphan_rec.id,
                'canonical_name', orphan_rec.canonical_name,
                'entity_type', orphan_rec.entity_type,
                'source', orphan_rec.source,
                'doc_count', orphan_rec.doc_count
            );
    END LOOP;

    -- 5. Build the diff object for later inspection ------------------------
    result := jsonb_build_object(
        'staging_entity_count', staging_total,
        'staging_alias_count', alias_staging_tot,
        'prod_entity_count_for_sources', prod_entity_total,
        'prod_alias_count_for_sources', prod_alias_total,
        'canonical_shrinkage', canonical_shrink,
        'alias_shrinkage', alias_shrink,
        'per_source_counts', per_source_json,
        'floor_violations', floor_violations,
        'orphan_violations', orphan_violations
    );

    -- 6. Gate decisions -----------------------------------------------------
    IF canonical_shrink > canonical_max THEN
        PERFORM pg_advisory_unlock(hashtext('entities_promotion'));
        UPDATE harvest_runs SET status = 'rejected_by_diff'
         WHERE id = run_id;
        RETURN jsonb_build_object(
            'accepted', false,
            'reason', 'canonical_shrinkage',
            'diff', result
        );
    END IF;

    IF alias_shrink > alias_max THEN
        PERFORM pg_advisory_unlock(hashtext('entities_promotion'));
        UPDATE harvest_runs SET status = 'rejected_by_diff'
         WHERE id = run_id;
        RETURN jsonb_build_object(
            'accepted', false,
            'reason', 'alias_shrinkage',
            'diff', result
        );
    END IF;

    IF jsonb_array_length(floor_violations) > 0 THEN
        PERFORM pg_advisory_unlock(hashtext('entities_promotion'));
        UPDATE harvest_runs SET status = 'rejected_by_diff'
         WHERE id = run_id;
        RETURN jsonb_build_object(
            'accepted', false,
            'reason', 'floor_violation',
            'diff', result
        );
    END IF;

    IF jsonb_array_length(orphan_violations) > 0 THEN
        PERFORM pg_advisory_unlock(hashtext('entities_promotion'));
        UPDATE harvest_runs SET status = 'rejected_by_diff'
         WHERE id = run_id;
        RETURN jsonb_build_object(
            'accepted', false,
            'reason', 'orphan_violation',
            'diff', result
        );
    END IF;

    -- 7. Atomic upserts ----------------------------------------------------
    WITH ins AS (
        INSERT INTO entities (
            canonical_name, entity_type, discipline, source, source_version,
            ambiguity_class, link_policy, properties, harvest_run_id
        )
        SELECT
            s.canonical_name,
            s.entity_type,
            s.discipline,
            s.source,
            s.source_version,
            s.ambiguity_class::entity_ambiguity_class,
            s.link_policy::entity_link_policy,
            COALESCE(s.properties, '{}'::jsonb),
            run_id
          FROM entities_staging s
         WHERE s.staging_run_id = run_id
        ON CONFLICT (canonical_name, entity_type, source) DO UPDATE
            SET discipline     = COALESCE(EXCLUDED.discipline, entities.discipline),
                source_version = COALESCE(EXCLUDED.source_version, entities.source_version),
                ambiguity_class = COALESCE(EXCLUDED.ambiguity_class, entities.ambiguity_class),
                link_policy    = COALESCE(EXCLUDED.link_policy, entities.link_policy),
                properties     = entities.properties || EXCLUDED.properties,
                harvest_run_id = EXCLUDED.harvest_run_id,
                updated_at     = now()
        RETURNING 1
    )
    SELECT COUNT(*) INTO n_promoted_ent FROM ins;

    -- Aliases: resolve target entity via the natural key on the staging row.
    WITH resolved AS (
        SELECT DISTINCT e.id AS entity_id, sa.alias, sa.alias_source
          FROM entity_aliases_staging sa
          JOIN entities e
            ON e.canonical_name = sa.canonical_name
           AND e.entity_type    = sa.entity_type
           AND e.source         = sa.source
         WHERE sa.staging_run_id = run_id
           AND sa.alias IS NOT NULL
    ),
    ins AS (
        INSERT INTO entity_aliases (entity_id, alias, alias_source)
        SELECT entity_id, alias, alias_source FROM resolved
        ON CONFLICT (entity_id, alias) DO UPDATE
            SET alias_source = COALESCE(EXCLUDED.alias_source, entity_aliases.alias_source)
        RETURNING 1
    )
    SELECT COUNT(*) INTO n_promoted_ali FROM ins;

    WITH resolved AS (
        SELECT DISTINCT e.id AS entity_id, si.id_scheme, si.external_id,
               COALESCE(si.is_primary, false) AS is_primary
          FROM entity_identifiers_staging si
          JOIN entities e
            ON e.canonical_name = si.canonical_name
           AND e.entity_type    = si.entity_type
           AND e.source         = si.source
         WHERE si.staging_run_id = run_id
           AND si.id_scheme IS NOT NULL
           AND si.external_id IS NOT NULL
    ),
    ins AS (
        INSERT INTO entity_identifiers (entity_id, id_scheme, external_id, is_primary)
        SELECT entity_id, id_scheme, external_id, is_primary FROM resolved
        ON CONFLICT (id_scheme, external_id) DO UPDATE
            SET entity_id  = EXCLUDED.entity_id,
                is_primary = EXCLUDED.is_primary
        RETURNING 1
    )
    SELECT COUNT(*) INTO n_promoted_ids FROM ins;

    result := result || jsonb_build_object(
        'promoted_entities', n_promoted_ent,
        'promoted_aliases', n_promoted_ali,
        'promoted_identifiers', n_promoted_ids
    );

    UPDATE harvest_runs SET status = 'promoted', finished_at = now()
     WHERE id = run_id;

    PERFORM pg_advisory_unlock(hashtext('entities_promotion'));
    RETURN jsonb_build_object(
        'accepted', true,
        'reason', NULL,
        'diff', result
    );
EXCEPTION WHEN OTHERS THEN
    IF lock_acquired THEN
        PERFORM pg_advisory_unlock(hashtext('entities_promotion'));
    END IF;
    RAISE;
END
$$;


--
-- Name: FUNCTION promote_harvest(run_id bigint, floors jsonb, canonical_max numeric, alias_max numeric, orphan_threshold integer); Type: COMMENT; Schema: public; Owner: -
--

COMMENT ON FUNCTION public.promote_harvest(run_id bigint, floors jsonb, canonical_max numeric, alias_max numeric, orphan_threshold integer) IS 'Atomic promote of *_staging rows into public entity tables with shadow-diff gating (shrinkage, floor, orphan checks). Consolidates migration 030 stub + promote_harvest_v2 into single function.';


--
-- Name: tier_weight(smallint); Type: FUNCTION; Schema: public; Owner: -
--

CREATE FUNCTION public.tier_weight(tier smallint) RETURNS double precision
    LANGUAGE sql IMMUTABLE LEAKPROOF PARALLEL SAFE
    AS $$
    SELECT CASE tier
        WHEN 1::SMALLINT THEN 0.98::float8
        WHEN 2::SMALLINT THEN 0.85::float8
        WHEN 3::SMALLINT THEN 0.92::float8
        WHEN 4::SMALLINT THEN 0.50::float8
        WHEN 5::SMALLINT THEN 0.88::float8
        ELSE 0.50::float8
    END
$$;


--
-- Name: promote_entities(); Type: FUNCTION; Schema: staging; Owner: -
--

CREATE FUNCTION staging.promote_entities() RETURNS integer
    LANGUAGE plpgsql
    AS $$
DECLARE
    promoted_count INTEGER;
BEGIN
    -- 1. Upsert entities
    WITH upserted AS (
        INSERT INTO public.entities
            (canonical_name, entity_type, discipline, source, properties, created_at, updated_at)
        SELECT canonical_name, entity_type, discipline, source, properties, created_at, updated_at
        FROM staging.entities
        ON CONFLICT (canonical_name, entity_type, source)
        DO UPDATE SET
            properties = EXCLUDED.properties,
            updated_at = NOW()
        RETURNING 1
    )
    SELECT count(*) INTO promoted_count FROM upserted;

    -- 2. Upsert identifiers (remap entity_id through natural key)
    INSERT INTO public.entity_identifiers (entity_id, id_scheme, external_id, is_primary)
    SELECT pe.id, si.id_scheme, si.external_id, si.is_primary
    FROM staging.entity_identifiers si
    JOIN staging.entities se ON se.id = si.entity_id
    JOIN public.entities pe ON pe.canonical_name = se.canonical_name
                            AND pe.entity_type = se.entity_type
                            AND pe.source = se.source
    ON CONFLICT (id_scheme, external_id)
    DO UPDATE SET
        entity_id = EXCLUDED.entity_id,
        is_primary = EXCLUDED.is_primary;

    -- 3. Upsert aliases (remap entity_id through natural key)
    INSERT INTO public.entity_aliases (entity_id, alias, alias_source)
    SELECT pe.id, sa.alias, sa.alias_source
    FROM staging.entity_aliases sa
    JOIN staging.entities se ON se.id = sa.entity_id
    JOIN public.entities pe ON pe.canonical_name = se.canonical_name
                            AND pe.entity_type = se.entity_type
                            AND pe.source = se.source
    ON CONFLICT (entity_id, alias)
    DO NOTHING;

    -- 4. Clear staging tables
    TRUNCATE staging.entity_aliases;
    TRUNCATE staging.entity_identifiers;
    TRUNCATE staging.entities;

    RETURN promoted_count;
END;
$$;


--
-- Name: promote_extractions(); Type: FUNCTION; Schema: staging; Owner: -
--

CREATE FUNCTION staging.promote_extractions() RETURNS integer
    LANGUAGE plpgsql
    AS $$
DECLARE
    promoted_count INTEGER;
BEGIN
    -- Upsert from staging into public
    WITH upserted AS (
        INSERT INTO public.extractions (bibcode, extraction_type, extraction_version, payload, created_at)
        SELECT bibcode, extraction_type, extraction_version, payload, created_at
        FROM staging.extractions
        ON CONFLICT (bibcode, extraction_type, extraction_version)
        DO UPDATE SET
            payload = EXCLUDED.payload,
            created_at = EXCLUDED.created_at
        RETURNING 1
    )
    SELECT count(*) INTO promoted_count FROM upserted;

    -- Clear staging after successful promotion
    TRUNCATE staging.extractions;

    RETURN promoted_count;
END;
$$;


--
-- Name: simple_nostem; Type: TEXT SEARCH DICTIONARY; Schema: public; Owner: -
--

CREATE TEXT SEARCH DICTIONARY public.simple_nostem (
    TEMPLATE = pg_catalog.simple,
    stopwords = 'english' );


--
-- Name: scix_english; Type: TEXT SEARCH CONFIGURATION; Schema: public; Owner: -
--

CREATE TEXT SEARCH CONFIGURATION public.scix_english (
    PARSER = pg_catalog."default" );

ALTER TEXT SEARCH CONFIGURATION public.scix_english
    ADD MAPPING FOR asciiword WITH english_stem;

ALTER TEXT SEARCH CONFIGURATION public.scix_english
    ADD MAPPING FOR word WITH english_stem;

ALTER TEXT SEARCH CONFIGURATION public.scix_english
    ADD MAPPING FOR numword WITH simple;

ALTER TEXT SEARCH CONFIGURATION public.scix_english
    ADD MAPPING FOR email WITH simple;

ALTER TEXT SEARCH CONFIGURATION public.scix_english
    ADD MAPPING FOR url WITH simple;

ALTER TEXT SEARCH CONFIGURATION public.scix_english
    ADD MAPPING FOR host WITH simple;

ALTER TEXT SEARCH CONFIGURATION public.scix_english
    ADD MAPPING FOR sfloat WITH simple;

ALTER TEXT SEARCH CONFIGURATION public.scix_english
    ADD MAPPING FOR version WITH simple;

ALTER TEXT SEARCH CONFIGURATION public.scix_english
    ADD MAPPING FOR hword_numpart WITH simple;

ALTER TEXT SEARCH CONFIGURATION public.scix_english
    ADD MAPPING FOR hword_part WITH public.simple_nostem;

ALTER TEXT SEARCH CONFIGURATION public.scix_english
    ADD MAPPING FOR hword_asciipart WITH public.simple_nostem;

ALTER TEXT SEARCH CONFIGURATION public.scix_english
    ADD MAPPING FOR numhword WITH simple;

ALTER TEXT SEARCH CONFIGURATION public.scix_english
    ADD MAPPING FOR asciihword WITH english_stem;

ALTER TEXT SEARCH CONFIGURATION public.scix_english
    ADD MAPPING FOR hword WITH public.simple_nostem;

ALTER TEXT SEARCH CONFIGURATION public.scix_english
    ADD MAPPING FOR url_path WITH simple;

ALTER TEXT SEARCH CONFIGURATION public.scix_english
    ADD MAPPING FOR file WITH simple;

ALTER TEXT SEARCH CONFIGURATION public.scix_english
    ADD MAPPING FOR "float" WITH public.simple_nostem;

ALTER TEXT SEARCH CONFIGURATION public.scix_english
    ADD MAPPING FOR "int" WITH public.simple_nostem;

ALTER TEXT SEARCH CONFIGURATION public.scix_english
    ADD MAPPING FOR uint WITH public.simple_nostem;


--
-- Name: dataset_entities; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.dataset_entities (
    dataset_id integer NOT NULL,
    entity_id integer NOT NULL,
    relationship text NOT NULL
);


--
-- Name: datasets; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.datasets (
    id integer NOT NULL,
    name text NOT NULL,
    discipline text,
    source text NOT NULL,
    canonical_id text NOT NULL,
    description text,
    temporal_start date,
    temporal_end date,
    properties jsonb DEFAULT '{}'::jsonb,
    harvest_run_id integer,
    created_at timestamp with time zone DEFAULT now()
);


--
-- Name: document_datasets; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.document_datasets (
    bibcode text NOT NULL,
    dataset_id integer NOT NULL,
    link_type text NOT NULL,
    confidence real,
    match_method text,
    harvest_run_id integer
);


--
-- Name: entities; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.entities (
    id integer NOT NULL,
    canonical_name text NOT NULL,
    entity_type text NOT NULL,
    discipline text,
    source text NOT NULL,
    harvest_run_id integer,
    properties jsonb DEFAULT '{}'::jsonb,
    created_at timestamp with time zone DEFAULT now(),
    updated_at timestamp with time zone DEFAULT now(),
    ambiguity_class public.entity_ambiguity_class,
    link_policy public.entity_link_policy,
    source_version text,
    supersedes_id integer
);


--
-- Name: agent_dataset_context; Type: MATERIALIZED VIEW; Schema: public; Owner: -
--

CREATE MATERIALIZED VIEW public.agent_dataset_context AS
 SELECT d.id AS dataset_id,
    d.name AS dataset_name,
    d.source,
    d.discipline,
    d.description,
    COALESCE(jsonb_agg(DISTINCT jsonb_build_object('entity_id', e.id, 'name', e.canonical_name, 'type', e.entity_type, 'relationship', dse.relationship)) FILTER (WHERE (e.id IS NOT NULL)), '[]'::jsonb) AS linked_entities,
    COALESCE(jsonb_agg(DISTINCT jsonb_build_object('bibcode', p.bibcode, 'title', p.title, 'link_type', dd.link_type)) FILTER (WHERE (p.bibcode IS NOT NULL)), '[]'::jsonb) AS citing_papers
   FROM ((((public.datasets d
     LEFT JOIN public.dataset_entities dse ON ((dse.dataset_id = d.id)))
     LEFT JOIN public.entities e ON ((e.id = dse.entity_id)))
     LEFT JOIN public.document_datasets dd ON ((dd.dataset_id = d.id)))
     LEFT JOIN public.papers p ON ((p.bibcode = dd.bibcode)))
  GROUP BY d.id, d.name, d.source, d.discipline, d.description
  WITH NO DATA;


--
-- Name: document_entities; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.document_entities (
    bibcode text NOT NULL,
    entity_id integer NOT NULL,
    link_type text NOT NULL,
    confidence real,
    match_method text,
    evidence jsonb,
    harvest_run_id integer,
    tier smallint DEFAULT 0 NOT NULL,
    tier_version integer DEFAULT 1 NOT NULL,
    citation_consistency real
);


--
-- Name: COLUMN document_entities.citation_consistency; Type: COMMENT; Schema: public; Owner: -
--

COMMENT ON COLUMN public.document_entities.citation_consistency IS 'Fraction of outbound citations from bibcode that also link to entity_id (precision proxy, 0..1). NULL = not yet computed. See PRD §S1.';


--
-- Name: agent_document_context; Type: MATERIALIZED VIEW; Schema: public; Owner: -
--

CREATE MATERIALIZED VIEW public.agent_document_context AS
 SELECT p.bibcode,
    p.title,
    p.abstract,
    p.year,
    p.doctype,
    p.citation_count,
    p.reference_count,
    COALESCE(jsonb_agg(DISTINCT jsonb_build_object('entity_id', e.id, 'name', e.canonical_name, 'type', e.entity_type, 'discipline', e.discipline, 'link_type', de.link_type, 'confidence', de.confidence)) FILTER (WHERE (e.id IS NOT NULL)), '[]'::jsonb) AS linked_entities
   FROM ((public.papers p
     LEFT JOIN public.document_entities de ON ((de.bibcode = p.bibcode)))
     LEFT JOIN public.entities e ON ((e.id = de.entity_id)))
  GROUP BY p.bibcode, p.title, p.abstract, p.year, p.doctype, p.citation_count, p.reference_count
  WITH NO DATA;


--
-- Name: MATERIALIZED VIEW agent_document_context; Type: COMMENT; Schema: public; Owner: -
--

COMMENT ON MATERIALIZED VIEW public.agent_document_context IS 'Pre-joined paper + linked entities for single-call agent document context. Refresh after entity linking pipeline runs. See docs/prd/prd_agent_views_mcp.md.';


--
-- Name: entity_aliases; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.entity_aliases (
    entity_id integer NOT NULL,
    alias text NOT NULL,
    alias_source text
);


--
-- Name: entity_identifiers; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.entity_identifiers (
    entity_id integer,
    id_scheme text NOT NULL,
    external_id text NOT NULL,
    is_primary boolean DEFAULT false
);


--
-- Name: entity_relationships; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.entity_relationships (
    id integer NOT NULL,
    subject_entity_id integer,
    predicate text NOT NULL,
    object_entity_id integer,
    source text,
    harvest_run_id integer,
    confidence real DEFAULT 1.0,
    evidence jsonb
);


--
-- Name: COLUMN entity_relationships.evidence; Type: COMMENT; Schema: public; Owner: -
--

COMMENT ON COLUMN public.entity_relationships.evidence IS 'Optional derivation metadata: {"method": "...", "path": "...", "source_field": "..."}';


--
-- Name: agent_entity_context; Type: MATERIALIZED VIEW; Schema: public; Owner: -
--

CREATE MATERIALIZED VIEW public.agent_entity_context AS
 WITH de_counts AS (
         SELECT document_entities.entity_id,
            count(*) AS doc_count
           FROM public.document_entities
          GROUP BY document_entities.entity_id
        )
 SELECT e.id AS entity_id,
    e.canonical_name,
    e.entity_type,
    e.discipline,
    e.source,
    COALESCE(jsonb_agg(DISTINCT jsonb_build_object('scheme', ei.id_scheme, 'id', ei.external_id)) FILTER (WHERE (ei.external_id IS NOT NULL)), '[]'::jsonb) AS identifiers,
    COALESCE(array_agg(DISTINCT ea.alias) FILTER (WHERE (ea.alias IS NOT NULL)), ARRAY[]::text[]) AS aliases,
    COALESCE(jsonb_agg(DISTINCT jsonb_build_object('predicate', er.predicate, 'object_id', er.object_entity_id, 'confidence', er.confidence)) FILTER (WHERE (er.id IS NOT NULL)), '[]'::jsonb) AS relationships,
    COALESCE(dc.doc_count, (0)::bigint) AS citing_paper_count
   FROM ((((public.entities e
     LEFT JOIN public.entity_identifiers ei ON ((ei.entity_id = e.id)))
     LEFT JOIN public.entity_aliases ea ON ((ea.entity_id = e.id)))
     LEFT JOIN public.entity_relationships er ON ((er.subject_entity_id = e.id)))
     LEFT JOIN de_counts dc ON ((dc.entity_id = e.id)))
  GROUP BY e.id, e.canonical_name, e.entity_type, e.discipline, e.source, dc.doc_count
  WITH NO DATA;


--
-- Name: alerts; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.alerts (
    id bigint NOT NULL,
    severity text NOT NULL,
    source text NOT NULL,
    message text NOT NULL,
    created_at timestamp with time zone DEFAULT now() NOT NULL,
    acked_at timestamp with time zone,
    CONSTRAINT alerts_severity_check CHECK ((severity = ANY (ARRAY['info'::text, 'warn'::text, 'page'::text])))
);


--
-- Name: alerts_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.alerts_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: alerts_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.alerts_id_seq OWNED BY public.alerts.id;


--
-- Name: citation_contexts; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.citation_contexts (
    id integer NOT NULL,
    source_bibcode text NOT NULL,
    target_bibcode text NOT NULL,
    context_text text NOT NULL,
    char_offset integer,
    section_name text,
    intent text
);


--
-- Name: citation_contexts_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.citation_contexts_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: citation_contexts_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.citation_contexts_id_seq OWNED BY public.citation_contexts.id;


--
-- Name: citation_diff; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.citation_diff (
    source_bibcode text NOT NULL,
    target_bibcode text NOT NULL,
    in_ads boolean DEFAULT false NOT NULL,
    in_openalex boolean DEFAULT false NOT NULL,
    source_attrs jsonb
);


--
-- Name: TABLE citation_diff; Type: COMMENT; Schema: public; Owner: -
--

COMMENT ON TABLE public.citation_diff IS 'Full outer join of ADS citation_edges and OpenAlex works_references, joined via papers_external_ids crosswalk. Each row records whether a directed citation edge exists in ADS, OpenAlex, or both. Populated by scripts/analyze_citation_diff.py --populate. See paper Section 3.3.';


--
-- Name: citation_diff_by_journal; Type: MATERIALIZED VIEW; Schema: public; Owner: -
--

CREATE MATERIALIZED VIEW public.citation_diff_by_journal AS
 SELECT p.pub AS journal,
    count(*) AS total_edges,
    count(*) FILTER (WHERE (cd.in_ads AND cd.in_openalex)) AS both_count,
    count(*) FILTER (WHERE (cd.in_ads AND (NOT cd.in_openalex))) AS ads_only_count,
    count(*) FILTER (WHERE ((NOT cd.in_ads) AND cd.in_openalex)) AS openalex_only_count,
    round((((count(*) FILTER (WHERE (cd.in_ads AND cd.in_openalex)))::numeric / (NULLIF(count(*), 0))::numeric) * (100)::numeric), 2) AS overlap_pct
   FROM (public.citation_diff cd
     JOIN public.papers p ON ((p.bibcode = cd.source_bibcode)))
  GROUP BY p.pub
  ORDER BY (count(*)) DESC
  WITH NO DATA;


--
-- Name: citation_diff_by_year; Type: MATERIALIZED VIEW; Schema: public; Owner: -
--

CREATE MATERIALIZED VIEW public.citation_diff_by_year AS
 SELECT p.year AS pub_year,
    count(*) AS total_edges,
    count(*) FILTER (WHERE (cd.in_ads AND cd.in_openalex)) AS both_count,
    count(*) FILTER (WHERE (cd.in_ads AND (NOT cd.in_openalex))) AS ads_only_count,
    count(*) FILTER (WHERE ((NOT cd.in_ads) AND cd.in_openalex)) AS openalex_only_count,
    round((((count(*) FILTER (WHERE (cd.in_ads AND cd.in_openalex)))::numeric / (NULLIF(count(*), 0))::numeric) * (100)::numeric), 2) AS overlap_pct
   FROM (public.citation_diff cd
     JOIN public.papers p ON ((p.bibcode = cd.source_bibcode)))
  GROUP BY p.year
  ORDER BY p.year
  WITH NO DATA;


--
-- Name: citation_edges; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.citation_edges (
    source_bibcode text NOT NULL,
    target_bibcode text NOT NULL
);


--
-- Name: co_mention_runs; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.co_mention_runs (
    id integer NOT NULL,
    started_at timestamp with time zone DEFAULT now() NOT NULL,
    finished_at timestamp with time zone,
    refresh_kind text NOT NULL,
    n_papers_input bigint,
    n_pairs_output bigint,
    min_n_papers integer DEFAULT 2 NOT NULL,
    git_sha text,
    notes text,
    CONSTRAINT co_mention_runs_kind CHECK ((refresh_kind = ANY (ARRAY['full'::text, 'incremental'::text, 'pilot'::text])))
);


--
-- Name: TABLE co_mention_runs; Type: COMMENT; Schema: public; Owner: -
--

COMMENT ON TABLE public.co_mention_runs IS 'Audit log of co_mentions table rebuilds. Read this to assess staleness vs document_entities.';


--
-- Name: co_mention_runs_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.co_mention_runs_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: co_mention_runs_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.co_mention_runs_id_seq OWNED BY public.co_mention_runs.id;


--
-- Name: co_mentions; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.co_mentions (
    entity_a_id integer NOT NULL,
    entity_b_id integer NOT NULL,
    n_papers integer NOT NULL,
    first_year smallint,
    last_year smallint,
    CONSTRAINT co_mentions_a_lt_b CHECK ((entity_a_id < entity_b_id)),
    CONSTRAINT co_mentions_n_papers_ge CHECK ((n_papers >= 2)),
    CONSTRAINT co_mentions_year_order CHECK (((first_year IS NULL) OR (last_year IS NULL) OR (first_year <= last_year)))
);


--
-- Name: TABLE co_mentions; Type: COMMENT; Schema: public; Owner: -
--

COMMENT ON TABLE public.co_mentions IS 'Entity-entity co-mention edges. One row per unordered pair (a<b) with n_papers >= 2. Rebuilt by scripts/populate_co_mentions.py — see docs/prd/co_mentions.md.';


--
-- Name: COLUMN co_mentions.n_papers; Type: COMMENT; Schema: public; Owner: -
--

COMMENT ON COLUMN public.co_mentions.n_papers IS 'Distinct bibcodes where both entities are linked via document_entities (any match_method).';


--
-- Name: COLUMN co_mentions.first_year; Type: COMMENT; Schema: public; Owner: -
--

COMMENT ON COLUMN public.co_mentions.first_year IS 'Earliest papers.year across the supporting bibcodes (NULL when no supporting paper has a year).';


--
-- Name: COLUMN co_mentions.last_year; Type: COMMENT; Schema: public; Owner: -
--

COMMENT ON COLUMN public.co_mentions.last_year IS 'Latest papers.year across the supporting bibcodes (NULL when no supporting paper has a year).';


--
-- Name: communities; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.communities (
    community_id integer NOT NULL,
    resolution text NOT NULL,
    label text,
    paper_count integer DEFAULT 0 NOT NULL,
    top_keywords text[] DEFAULT '{}'::text[] NOT NULL,
    updated_at timestamp with time zone DEFAULT now() NOT NULL,
    signal text NOT NULL,
    CONSTRAINT communities_resolution_check CHECK ((resolution = ANY (ARRAY['coarse'::text, 'medium'::text, 'fine'::text]))),
    CONSTRAINT communities_signal_check CHECK (((signal IS NULL) OR (signal = ANY (ARRAY['citation'::text, 'semantic'::text, 'taxonomic'::text]))))
);


--
-- Name: concept_relationships; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.concept_relationships (
    vocabulary text NOT NULL,
    parent_id text NOT NULL,
    child_id text NOT NULL,
    relationship text DEFAULT 'broader'::text NOT NULL,
    CONSTRAINT concept_relationships_relationship_check CHECK ((relationship = ANY (ARRAY['broader'::text, 'narrower'::text, 'related'::text])))
);


--
-- Name: concepts; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.concepts (
    vocabulary text NOT NULL,
    concept_id text NOT NULL,
    preferred_label text NOT NULL,
    alternate_labels text[] DEFAULT '{}'::text[] NOT NULL,
    definition text,
    external_uri text,
    level integer,
    properties jsonb DEFAULT '{}'::jsonb NOT NULL
);


--
-- Name: core_promotion_log; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.core_promotion_log (
    id integer NOT NULL,
    entity_id integer NOT NULL,
    action text NOT NULL,
    query_hits_14d integer,
    reason text,
    ts timestamp with time zone DEFAULT now() NOT NULL,
    CONSTRAINT core_promotion_log_action_check CHECK ((action = ANY (ARRAY['promote'::text, 'demote'::text])))
);


--
-- Name: core_promotion_log_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.core_promotion_log_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: core_promotion_log_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.core_promotion_log_id_seq OWNED BY public.core_promotion_log.id;


--
-- Name: curated_entity_core; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.curated_entity_core (
    entity_id integer NOT NULL,
    query_hits_14d integer DEFAULT 0 NOT NULL,
    promoted_at timestamp with time zone DEFAULT now() NOT NULL
);


--
-- Name: datasets_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.datasets_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: datasets_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.datasets_id_seq OWNED BY public.datasets.id;


--
-- Name: document_entities_canonical; Type: MATERIALIZED VIEW; Schema: public; Owner: -
--

CREATE MATERIALIZED VIEW public.document_entities_canonical AS
 SELECT bibcode,
    entity_id,
    ((1)::double precision - exp(sum(ln(((1)::double precision - LEAST((0.9999)::double precision, GREATEST((0.0)::double precision, ((confidence)::double precision * public.tier_weight(tier))))))))) AS fused_confidence,
    count(*) AS link_count,
    array_agg(DISTINCT tier ORDER BY tier) AS contributing_tiers,
    max(tier_version) AS max_tier_version,
    max(harvest_run_id) AS latest_harvest_run_id
   FROM public.document_entities de
  WHERE (confidence IS NOT NULL)
  GROUP BY bibcode, entity_id
  WITH NO DATA;


--
-- Name: document_entities_jit_cache; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.document_entities_jit_cache (
    bibcode text NOT NULL,
    entity_id integer NOT NULL,
    link_type text NOT NULL,
    confidence real,
    match_method text,
    evidence jsonb,
    harvest_run_id integer,
    tier smallint DEFAULT 5 NOT NULL,
    tier_version integer DEFAULT 1 NOT NULL,
    candidate_set_hash text NOT NULL,
    model_version text NOT NULL,
    expires_at timestamp with time zone NOT NULL,
    created_at timestamp with time zone DEFAULT now() NOT NULL,
    CONSTRAINT document_entities_jit_cache_tier_check CHECK ((tier = 5))
)
PARTITION BY RANGE (expires_at);


--
-- Name: document_entities_jit_cache_default; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.document_entities_jit_cache_default (
    bibcode text NOT NULL,
    entity_id integer NOT NULL,
    link_type text NOT NULL,
    confidence real,
    match_method text,
    evidence jsonb,
    harvest_run_id integer,
    tier smallint DEFAULT 5 NOT NULL,
    tier_version integer DEFAULT 1 NOT NULL,
    candidate_set_hash text NOT NULL,
    model_version text NOT NULL,
    expires_at timestamp with time zone NOT NULL,
    created_at timestamp with time zone DEFAULT now() NOT NULL,
    CONSTRAINT document_entities_jit_cache_tier_check CHECK ((tier = 5))
);


--
-- Name: embedding_outbox; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.embedding_outbox (
    id bigint NOT NULL,
    bibcode text NOT NULL,
    model_name text NOT NULL,
    op text NOT NULL,
    enqueued_at timestamp with time zone DEFAULT now() NOT NULL,
    CONSTRAINT embedding_outbox_op_check CHECK ((op = ANY (ARRAY['INSERT'::text, 'UPDATE'::text, 'DELETE'::text, 'BACKFILL'::text])))
);


--
-- Name: TABLE embedding_outbox; Type: COMMENT; Schema: public; Owner: -
--

COMMENT ON TABLE public.embedding_outbox IS 'PG→Qdrant forward-write queue (migration 070, PRD MH-9). One row per paper_embeddings write, inserted by trigger trg_embedding_outbox. Drained and deleted by scripts/qdrant_outbox_sync.py. No ack column — drained rows are removed, so row count + oldest enqueued_at IS the sync-lag metric.';


--
-- Name: COLUMN embedding_outbox.op; Type: COMMENT; Schema: public; Owner: -
--

COMMENT ON COLUMN public.embedding_outbox.op IS 'INSERT/UPDATE → upsert current vector to Qdrant; DELETE → remove the point; BACKFILL → enqueued by the worker''s --backfill-since reconcile, treated as an upsert.';


--
-- Name: embedding_outbox_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

ALTER TABLE public.embedding_outbox ALTER COLUMN id ADD GENERATED ALWAYS AS IDENTITY (
    SEQUENCE NAME public.embedding_outbox_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1
);


--
-- Name: entities_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.entities_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: entities_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.entities_id_seq OWNED BY public.entities.id;


--
-- Name: entities_staging; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.entities_staging (
    id bigint NOT NULL,
    staging_run_id bigint NOT NULL,
    canonical_name text NOT NULL,
    entity_type text NOT NULL,
    discipline text,
    source text NOT NULL,
    source_version text,
    ambiguity_class text,
    link_policy text,
    properties jsonb DEFAULT '{}'::jsonb,
    created_at timestamp with time zone DEFAULT now()
);


--
-- Name: entities_staging_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.entities_staging_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: entities_staging_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.entities_staging_id_seq OWNED BY public.entities_staging.id;


--
-- Name: entity_aliases_staging; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.entity_aliases_staging (
    id bigint NOT NULL,
    staging_run_id bigint NOT NULL,
    staging_entity_id bigint,
    canonical_name text,
    entity_type text,
    source text,
    alias text NOT NULL,
    alias_source text
);


--
-- Name: entity_aliases_staging_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.entity_aliases_staging_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: entity_aliases_staging_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.entity_aliases_staging_id_seq OWNED BY public.entity_aliases_staging.id;


--
-- Name: entity_dictionary; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.entity_dictionary (
    id integer NOT NULL,
    canonical_name text NOT NULL,
    entity_type text NOT NULL,
    source text NOT NULL,
    external_id text,
    aliases text[] DEFAULT '{}'::text[] NOT NULL,
    metadata jsonb DEFAULT '{}'::jsonb NOT NULL,
    discipline text
);


--
-- Name: entity_dictionary_compat; Type: VIEW; Schema: public; Owner: -
--

CREATE VIEW public.entity_dictionary_compat AS
 SELECT e.id,
    e.canonical_name,
    e.entity_type,
    e.source,
    ei.external_id,
    COALESCE(( SELECT array_agg(ea.alias) AS array_agg
           FROM public.entity_aliases ea
          WHERE (ea.entity_id = e.id)), '{}'::text[]) AS aliases,
    e.properties AS metadata
   FROM (public.entities e
     LEFT JOIN public.entity_identifiers ei ON (((ei.entity_id = e.id) AND (ei.is_primary = true))));


--
-- Name: entity_dictionary_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.entity_dictionary_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: entity_dictionary_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.entity_dictionary_id_seq OWNED BY public.entity_dictionary.id;


--
-- Name: entity_identifiers_staging; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.entity_identifiers_staging (
    id bigint NOT NULL,
    staging_run_id bigint NOT NULL,
    staging_entity_id bigint,
    canonical_name text,
    entity_type text,
    source text,
    id_scheme text NOT NULL,
    external_id text NOT NULL,
    is_primary boolean DEFAULT false
);


--
-- Name: entity_identifiers_staging_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.entity_identifiers_staging_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: entity_identifiers_staging_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.entity_identifiers_staging_id_seq OWNED BY public.entity_identifiers_staging.id;


--
-- Name: entity_link_audits; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.entity_link_audits (
    tier smallint NOT NULL,
    bibcode text NOT NULL,
    entity_id bigint NOT NULL,
    annotator text NOT NULL,
    label text NOT NULL,
    note text,
    created_at timestamp with time zone DEFAULT now() NOT NULL,
    CONSTRAINT entity_link_audits_label_check CHECK ((label = ANY (ARRAY['correct'::text, 'incorrect'::text, 'ambiguous'::text])))
);


--
-- Name: entity_link_disputes; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.entity_link_disputes (
    id bigint NOT NULL,
    bibcode text,
    entity_id bigint,
    reason text,
    reported_at timestamp with time zone DEFAULT now() NOT NULL,
    tier smallint
);


--
-- Name: TABLE entity_link_disputes; Type: COMMENT; Schema: public; Owner: -
--

COMMENT ON TABLE public.entity_link_disputes IS 'Append-only researcher feedback on suspected incorrect document->entity links. Consumed by offline audit jobs, not the hot path. See PRD §S5.';


--
-- Name: entity_link_disputes_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.entity_link_disputes_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: entity_link_disputes_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.entity_link_disputes_id_seq OWNED BY public.entity_link_disputes.id;


--
-- Name: entity_merge_log; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.entity_merge_log (
    id integer NOT NULL,
    old_entity_id integer NOT NULL,
    new_entity_id integer NOT NULL,
    reason text,
    merged_by text,
    merged_at timestamp with time zone DEFAULT now()
);


--
-- Name: entity_merge_log_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.entity_merge_log_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: entity_merge_log_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.entity_merge_log_id_seq OWNED BY public.entity_merge_log.id;


--
-- Name: entity_relationships_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.entity_relationships_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: entity_relationships_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.entity_relationships_id_seq OWNED BY public.entity_relationships.id;


--
-- Name: entity_split_log; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.entity_split_log (
    id integer NOT NULL,
    parent_entity_id integer NOT NULL,
    child_entity_ids integer[] NOT NULL,
    reason text,
    split_by text,
    split_at timestamp with time zone DEFAULT now()
);


--
-- Name: entity_split_log_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.entity_split_log_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: entity_split_log_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.entity_split_log_id_seq OWNED BY public.entity_split_log.id;


--
-- Name: extraction_entity_links; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.extraction_entity_links (
    id bigint NOT NULL,
    extraction_id bigint,
    bibcode text NOT NULL,
    entity_type text NOT NULL,
    entity_id integer,
    entity_surface text NOT NULL,
    entity_canonical text,
    span_start integer,
    span_end integer,
    source text NOT NULL,
    confidence_tier smallint NOT NULL,
    confidence real,
    extraction_version text NOT NULL,
    payload jsonb,
    created_at timestamp with time zone DEFAULT now() NOT NULL
);


--
-- Name: extraction_entity_links_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.extraction_entity_links_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: extraction_entity_links_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.extraction_entity_links_id_seq OWNED BY public.extraction_entity_links.id;


--
-- Name: extractions; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.extractions (
    id integer NOT NULL,
    bibcode text NOT NULL,
    extraction_type text NOT NULL,
    extraction_version text NOT NULL,
    payload jsonb NOT NULL,
    created_at timestamp with time zone DEFAULT now(),
    source text DEFAULT 'llm'::text NOT NULL,
    confidence_tier text DEFAULT 'medium'::text NOT NULL,
    extraction_model text,
    CONSTRAINT chk_extractions_confidence_tier CHECK ((confidence_tier = ANY (ARRAY['high'::text, 'medium'::text, 'low'::text]))),
    CONSTRAINT chk_extractions_source CHECK ((source = ANY (ARRAY['metadata'::text, 'ner'::text, 'llm'::text, 'openalex'::text, 'citation_propagation'::text])))
);


--
-- Name: extractions_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.extractions_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: extractions_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.extractions_id_seq OWNED BY public.extractions.id;


--
-- Name: fusion_mv_state; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.fusion_mv_state (
    id integer DEFAULT 1 NOT NULL,
    dirty boolean DEFAULT true NOT NULL,
    last_refresh_at timestamp with time zone,
    CONSTRAINT fusion_mv_state_id_check CHECK ((id = 1))
);


--
-- Name: halfvec_backfill_progress; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.halfvec_backfill_progress (
    id integer NOT NULL,
    model_name text NOT NULL,
    last_bibcode text,
    rows_updated bigint DEFAULT 0 NOT NULL,
    started_at timestamp with time zone DEFAULT now() NOT NULL,
    updated_at timestamp with time zone DEFAULT now() NOT NULL,
    finished_at timestamp with time zone,
    note text
);


--
-- Name: halfvec_backfill_progress_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.halfvec_backfill_progress_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: halfvec_backfill_progress_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.halfvec_backfill_progress_id_seq OWNED BY public.halfvec_backfill_progress.id;


--
-- Name: harvest_runs; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.harvest_runs (
    id integer NOT NULL,
    source text NOT NULL,
    started_at timestamp with time zone DEFAULT now() NOT NULL,
    finished_at timestamp with time zone,
    status text DEFAULT 'running'::text NOT NULL,
    records_fetched integer DEFAULT 0 NOT NULL,
    records_upserted integer DEFAULT 0 NOT NULL,
    cursor_state jsonb,
    error_message text,
    config jsonb,
    counts jsonb DEFAULT '{}'::jsonb NOT NULL
);


--
-- Name: harvest_runs_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.harvest_runs_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: harvest_runs_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.harvest_runs_id_seq OWNED BY public.harvest_runs.id;


--
-- Name: ingest_log; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.ingest_log (
    filename text NOT NULL,
    records_loaded integer DEFAULT 0 NOT NULL,
    errors_skipped integer DEFAULT 0 NOT NULL,
    edges_loaded integer DEFAULT 0 NOT NULL,
    status text DEFAULT 'in_progress'::text NOT NULL,
    started_at timestamp with time zone DEFAULT now(),
    finished_at timestamp with time zone
);


--
-- Name: link_runs; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.link_runs (
    run_id bigint NOT NULL,
    max_entry_date timestamp with time zone,
    "timestamp" timestamp with time zone DEFAULT now() NOT NULL,
    rows_linked integer DEFAULT 0 NOT NULL,
    status text DEFAULT 'ok'::text NOT NULL,
    trip_count integer DEFAULT 0 NOT NULL,
    note text,
    CONSTRAINT link_runs_status_check CHECK ((status = ANY (ARRAY['ok'::text, 'tripped'::text, 'failed'::text])))
);


--
-- Name: link_runs_run_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.link_runs_run_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: link_runs_run_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.link_runs_run_id_seq OWNED BY public.link_runs.run_id;


--
-- Name: llm_cost_ledger; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.llm_cost_ledger (
    day date NOT NULL,
    total_usd numeric(12,6) DEFAULT 0 NOT NULL,
    call_count integer DEFAULT 0 NOT NULL
);


--
-- Name: paper_claims; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.paper_claims (
    claim_id uuid DEFAULT gen_random_uuid() NOT NULL,
    bibcode text NOT NULL,
    section_index integer NOT NULL,
    paragraph_index integer NOT NULL,
    char_span_start integer NOT NULL,
    char_span_end integer NOT NULL,
    claim_text text NOT NULL,
    claim_type text NOT NULL,
    subject text,
    predicate text,
    object text,
    confidence real,
    extraction_model text NOT NULL,
    extraction_prompt_version text NOT NULL,
    extracted_at timestamp with time zone DEFAULT now() NOT NULL,
    linked_entity_subject_id bigint,
    linked_entity_object_id bigint,
    CONSTRAINT paper_claims_claim_type_check CHECK ((claim_type = ANY (ARRAY['factual'::text, 'methodological'::text, 'comparative'::text, 'speculative'::text, 'cited_from_other'::text])))
);


--
-- Name: TABLE paper_claims; Type: COMMENT; Schema: public; Owner: -
--

COMMENT ON TABLE public.paper_claims IS 'Nanopub-inspired claim provenance. One row per scientific claim extracted from a paper. Provenance contract: (bibcode, section_index, paragraph_index, char_span_start, char_span_end) uniquely identifies the source span into papers_fulltext.sections[i].text.';


--
-- Name: COLUMN paper_claims.claim_id; Type: COMMENT; Schema: public; Owner: -
--

COMMENT ON COLUMN public.paper_claims.claim_id IS 'UUID primary key. Stable across re-extraction because new extractions should INSERT new rows (with a new claim_id and new extraction_model / extraction_prompt_version), not UPDATE existing ones.';


--
-- Name: COLUMN paper_claims.bibcode; Type: COMMENT; Schema: public; Owner: -
--

COMMENT ON COLUMN public.paper_claims.bibcode IS 'Source paper bibcode. FK to papers(bibcode).';


--
-- Name: COLUMN paper_claims.section_index; Type: COMMENT; Schema: public; Owner: -
--

COMMENT ON COLUMN public.paper_claims.section_index IS 'Index into papers_fulltext.sections[] (zero-based). Identifies which section this claim was extracted from.';


--
-- Name: COLUMN paper_claims.paragraph_index; Type: COMMENT; Schema: public; Owner: -
--

COMMENT ON COLUMN public.paper_claims.paragraph_index IS 'Paragraph offset within the section text (zero-based, paragraph-split is whatever the extractor uses; extractor must be deterministic so the contract holds).';


--
-- Name: COLUMN paper_claims.char_span_start; Type: COMMENT; Schema: public; Owner: -
--

COMMENT ON COLUMN public.paper_claims.char_span_start IS 'Inclusive character offset of the claim within papers_fulltext.sections[section_index].text.';


--
-- Name: COLUMN paper_claims.char_span_end; Type: COMMENT; Schema: public; Owner: -
--

COMMENT ON COLUMN public.paper_claims.char_span_end IS 'Exclusive character offset of the claim within papers_fulltext.sections[section_index].text.';


--
-- Name: COLUMN paper_claims.claim_text; Type: COMMENT; Schema: public; Owner: -
--

COMMENT ON COLUMN public.paper_claims.claim_text IS 'Verbatim text of the claim as extracted from the paper.';


--
-- Name: COLUMN paper_claims.claim_type; Type: COMMENT; Schema: public; Owner: -
--

COMMENT ON COLUMN public.paper_claims.claim_type IS 'One of: factual | methodological | comparative | speculative | cited_from_other. CHECK-constrained so unknown labels cannot slip in.';


--
-- Name: COLUMN paper_claims.subject; Type: COMMENT; Schema: public; Owner: -
--

COMMENT ON COLUMN public.paper_claims.subject IS 'Optional structured-claim subject (free text). Set when the extractor produced a subject-predicate-object decomposition.';


--
-- Name: COLUMN paper_claims.predicate; Type: COMMENT; Schema: public; Owner: -
--

COMMENT ON COLUMN public.paper_claims.predicate IS 'Optional structured-claim predicate (free text).';


--
-- Name: COLUMN paper_claims.object; Type: COMMENT; Schema: public; Owner: -
--

COMMENT ON COLUMN public.paper_claims.object IS 'Optional structured-claim object (free text).';


--
-- Name: COLUMN paper_claims.confidence; Type: COMMENT; Schema: public; Owner: -
--

COMMENT ON COLUMN public.paper_claims.confidence IS 'Optional extractor confidence in [0, 1]. Semantics defined by extraction_model + extraction_prompt_version.';


--
-- Name: COLUMN paper_claims.extraction_model; Type: COMMENT; Schema: public; Owner: -
--

COMMENT ON COLUMN public.paper_claims.extraction_model IS 'Model name + version that produced this claim (e.g. "claude-opus-4-7", "gpt-5.4-mini-2026-03-15").';


--
-- Name: COLUMN paper_claims.extraction_prompt_version; Type: COMMENT; Schema: public; Owner: -
--

COMMENT ON COLUMN public.paper_claims.extraction_prompt_version IS 'Prompt template version (e.g. "v1", "v2.1"). Combined with extraction_model uniquely identifies the extraction recipe.';


--
-- Name: COLUMN paper_claims.extracted_at; Type: COMMENT; Schema: public; Owner: -
--

COMMENT ON COLUMN public.paper_claims.extracted_at IS 'Timestamp of extraction.';


--
-- Name: COLUMN paper_claims.linked_entity_subject_id; Type: COMMENT; Schema: public; Owner: -
--

COMMENT ON COLUMN public.paper_claims.linked_entity_subject_id IS 'Optional link from claim subject into the entity graph. bigint, not REFERENCES-constrained: entities.id is currently SERIAL (int4) but we leave headroom, and we do not want claim INSERTs to fail when the entity-linking pass has not yet run.';


--
-- Name: COLUMN paper_claims.linked_entity_object_id; Type: COMMENT; Schema: public; Owner: -
--

COMMENT ON COLUMN public.paper_claims.linked_entity_object_id IS 'Optional link from claim object into the entity graph. See linked_entity_subject_id for type/FK rationale.';


--
-- Name: paper_embeddings; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.paper_embeddings (
    bibcode text NOT NULL,
    model_name text NOT NULL,
    embedding public.vector(768),
    input_type text DEFAULT 'title_abstract'::text NOT NULL,
    source_hash text,
    embedding_hv public.halfvec(768)
);


--
-- Name: COLUMN paper_embeddings.embedding_hv; Type: COMMENT; Schema: public; Owner: -
--

COMMENT ON COLUMN public.paper_embeddings.embedding_hv IS 'halfvec(768) shadow column for INDUS. Populated by scripts/backfill_halfvec.py and by scripts/embed.py on new writes. Replaces (embedding::vector(768)) as the canonical retrieval column for model_name=''indus''. Pilot models keep using embedding.';


--
-- Name: paper_metrics; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.paper_metrics (
    bibcode text NOT NULL,
    pagerank double precision,
    hub_score double precision,
    authority_score double precision,
    community_id_coarse integer,
    community_id_medium integer,
    community_id_fine integer,
    updated_at timestamp with time zone DEFAULT now() NOT NULL,
    community_taxonomic text,
    community_semantic_coarse integer,
    community_semantic_medium integer,
    community_semantic_fine integer
);


--
-- Name: paper_uat_mappings; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.paper_uat_mappings (
    bibcode text NOT NULL,
    concept_id text NOT NULL,
    match_type text NOT NULL,
    CONSTRAINT paper_uat_mappings_match_type_check CHECK ((match_type = ANY (ARRAY['exact'::text, 'fuzzy'::text, 'parent'::text])))
);


--
-- Name: papers_ads_body; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.papers_ads_body (
    bibcode text NOT NULL,
    body_text text NOT NULL,
    body_length integer NOT NULL,
    harvested_at timestamp with time zone NOT NULL,
    tsv tsvector GENERATED ALWAYS AS (to_tsvector('english'::regconfig, body_text)) STORED
);


--
-- Name: TABLE papers_ads_body; Type: COMMENT; Schema: public; Owner: -
--

COMMENT ON TABLE public.papers_ads_body IS 'Full text from the ADS `body` field, harvested ~55% of ADS papers. Populated by scripts/ingest_ads_body.py. body-only tsvector enables full-text search independent of the title/abstract tsv on papers.';


--
-- Name: COLUMN papers_ads_body.tsv; Type: COMMENT; Schema: public; Owner: -
--

COMMENT ON COLUMN public.papers_ads_body.tsv IS 'Generated stored tsvector using the built-in `english` config. See migration 039 preamble for the tradeoff vs scix_english.';


--
-- Name: papers_external_ids; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.papers_external_ids (
    bibcode text NOT NULL,
    doi text,
    arxiv_id text,
    openalex_id text,
    s2_corpus_id bigint,
    s2_paper_id text,
    pmcid text,
    pmid bigint,
    has_ads_body boolean DEFAULT false NOT NULL,
    has_arxiv_source boolean DEFAULT false NOT NULL,
    has_ar5iv_html boolean DEFAULT false NOT NULL,
    has_s2orc_body boolean DEFAULT false NOT NULL,
    openalex_has_pdf_url boolean DEFAULT false NOT NULL,
    updated_at timestamp with time zone DEFAULT now() NOT NULL
);


--
-- Name: TABLE papers_external_ids; Type: COMMENT; Schema: public; Owner: -
--

COMMENT ON TABLE public.papers_external_ids IS 'Crosswalk from ADS bibcode to external identifiers (OpenAlex / arXiv / Semantic Scholar / PubMed). Populated incrementally by PRD Build 5 work units W1-W6. has_* flags track which structured full-text sources have been ingested for each paper.';


--
-- Name: papers_fulltext; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.papers_fulltext (
    bibcode text NOT NULL,
    source text NOT NULL,
    sections jsonb NOT NULL,
    inline_cites jsonb NOT NULL,
    figures jsonb DEFAULT '[]'::jsonb NOT NULL,
    tables jsonb DEFAULT '[]'::jsonb NOT NULL,
    equations jsonb DEFAULT '[]'::jsonb NOT NULL,
    parser_version text NOT NULL,
    parsed_at timestamp with time zone DEFAULT now() NOT NULL,
    canonical_bibcode text,
    suppressed_by_publisher boolean DEFAULT false NOT NULL,
    source_version text,
    arxiv_version smallint,
    sections_tsv tsvector GENERATED ALWAYS AS (to_tsvector('english'::regconfig, ((COALESCE((jsonb_path_query_array(sections, '$[*]."text"'::jsonpath))::text, ''::text) || ' '::text) || COALESCE((jsonb_path_query_array(sections, '$[*]."heading"'::jsonpath))::text, ''::text)))) STORED
);


--
-- Name: TABLE papers_fulltext; Type: COMMENT; Schema: public; Owner: -
--

COMMENT ON TABLE public.papers_fulltext IS 'Structured full-text from multiple sources (ar5iv, arxiv_local, s2orc, ads_body, docling, abstract). Each row holds parsed sections, inline citations, figures, tables, and equations as JSONB arrays. Source column determines licensing treatment per ADR-006.';


--
-- Name: COLUMN papers_fulltext.source; Type: COMMENT; Schema: public; Owner: -
--

COMMENT ON COLUMN public.papers_fulltext.source IS 'Provenance tag: ar5iv | arxiv_local | s2orc | ads_body | docling | abstract. ar5iv/arxiv_local are LaTeX-derived (ADR-006 internal-use-only).';


--
-- Name: COLUMN papers_fulltext.sections; Type: COMMENT; Schema: public; Owner: -
--

COMMENT ON COLUMN public.papers_fulltext.sections IS 'Array of {heading, level, text, offset} objects representing the document structure. Level 1 = top-level section, 2 = subsection, etc.';


--
-- Name: COLUMN papers_fulltext.inline_cites; Type: COMMENT; Schema: public; Owner: -
--

COMMENT ON COLUMN public.papers_fulltext.inline_cites IS 'Array of {offset, bib_ref, target_bibcode_or_null} objects representing inline citation references found during parsing.';


--
-- Name: COLUMN papers_fulltext.canonical_bibcode; Type: COMMENT; Schema: public; Owner: -
--

COMMENT ON COLUMN public.papers_fulltext.canonical_bibcode IS 'The canonical papers.bibcode this full-text row logically belongs to. Nullable while the cross-bibcode resolver backfill is in progress. Populated by matching papers_fulltext.bibcode against papers.alternate_bibcode.';


--
-- Name: COLUMN papers_fulltext.suppressed_by_publisher; Type: COMMENT; Schema: public; Owner: -
--

COMMENT ON COLUMN public.papers_fulltext.suppressed_by_publisher IS 'True when the publisher has requested suppression of full-text serving for this paper. Default false. Partial index on the true subset.';


--
-- Name: COLUMN papers_fulltext.source_version; Type: COMMENT; Schema: public; Owner: -
--

COMMENT ON COLUMN public.papers_fulltext.source_version IS 'Upstream source version identifier (e.g. ar5iv build tag, s2orc snapshot id, docling version). NULL when not applicable.';


--
-- Name: COLUMN papers_fulltext.arxiv_version; Type: COMMENT; Schema: public; Owner: -
--

COMMENT ON COLUMN public.papers_fulltext.arxiv_version IS 'arXiv version number (v1, v2, ...) when parsed from an arXiv LaTeX source. NULL for non-arXiv sources.';


--
-- Name: COLUMN papers_fulltext.sections_tsv; Type: COMMENT; Schema: public; Owner: -
--

COMMENT ON COLUMN public.papers_fulltext.sections_tsv IS 'GENERATED tsvector over heading + text from sections JSONB. Backs the BM25 leg of section-grain retrieval. Bead scix_experiments-wqr.9.';


--
-- Name: papers_fulltext_failures; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.papers_fulltext_failures (
    bibcode text NOT NULL,
    parser_version text NOT NULL,
    failure_reason text,
    attempts integer DEFAULT 1 NOT NULL,
    first_attempt timestamp with time zone DEFAULT now() NOT NULL,
    last_attempt timestamp with time zone DEFAULT now() NOT NULL,
    retry_after timestamp with time zone NOT NULL
);


--
-- Name: TABLE papers_fulltext_failures; Type: COMMENT; Schema: public; Owner: -
--

COMMENT ON TABLE public.papers_fulltext_failures IS 'Negative cache for full-text parse failures. Each row records a bibcode whose parse failed, the parser version that failed, the failure reason, attempt count, and retry_after timestamp encoding R15 exponential backoff (24h -> 3d -> 7d -> 30d). Harvesters skip rows where now() < retry_after.';


--
-- Name: COLUMN papers_fulltext_failures.parser_version; Type: COMMENT; Schema: public; Owner: -
--

COMMENT ON COLUMN public.papers_fulltext_failures.parser_version IS 'Parser version string at the time of failure. Rows produced by an older parser_version may be ignored/reattempted when a newer parser is deployed.';


--
-- Name: COLUMN papers_fulltext_failures.failure_reason; Type: COMMENT; Schema: public; Owner: -
--

COMMENT ON COLUMN public.papers_fulltext_failures.failure_reason IS 'Human-readable classification of the failure (e.g. "no_source_found", "latex_parse_error", "pdf_ocr_failed"). Nullable to allow opaque failures.';


--
-- Name: COLUMN papers_fulltext_failures.attempts; Type: COMMENT; Schema: public; Owner: -
--

COMMENT ON COLUMN public.papers_fulltext_failures.attempts IS 'Number of parse attempts so far. Drives R15 backoff: 1 -> retry_after=+24h, 2 -> +3d, 3 -> +7d, >=4 -> +30d.';


--
-- Name: COLUMN papers_fulltext_failures.retry_after; Type: COMMENT; Schema: public; Owner: -
--

COMMENT ON COLUMN public.papers_fulltext_failures.retry_after IS 'Timestamp before which this bibcode should not be reattempted. Encodes R15 exponential backoff (24h -> 3d -> 7d -> 30d) based on attempts.';


--
-- Name: papers_openalex; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.papers_openalex (
    openalex_id text NOT NULL,
    doi text,
    title text,
    publication_year smallint,
    abstract text,
    topics jsonb,
    open_access jsonb,
    best_oa_location jsonb,
    cited_by_count integer,
    referenced_works_count integer,
    type text,
    updated_date date,
    created_date date
);


--
-- Name: TABLE papers_openalex; Type: COMMENT; Schema: public; Owner: -
--

COMMENT ON TABLE public.papers_openalex IS 'Pruned OpenAlex Work records from the S3 snapshot. Populated by src/scix/sources/openalex.py. CC0 licensed, ~260M works. The abstract field is reconstructed from abstract_inverted_index.';


--
-- Name: query_log; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.query_log (
    id integer NOT NULL,
    tool_name text NOT NULL,
    params_json jsonb,
    latency_ms real,
    success boolean NOT NULL,
    error_msg text,
    created_at timestamp with time zone DEFAULT now() NOT NULL,
    ts timestamp with time zone DEFAULT now() NOT NULL,
    tool text,
    query text,
    result_count integer,
    session_id text,
    is_test boolean DEFAULT false NOT NULL
);


--
-- Name: query_log_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.query_log_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: query_log_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.query_log_id_seq OWNED BY public.query_log.id;


--
-- Name: schema_migrations; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.schema_migrations (
    version integer NOT NULL,
    applied_at timestamp with time zone DEFAULT now() NOT NULL,
    filename text NOT NULL
);


--
-- Name: section_embeddings; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.section_embeddings (
    bibcode text NOT NULL,
    section_index integer NOT NULL,
    section_heading text,
    section_text_sha256 text NOT NULL,
    embedding public.halfvec(1024) NOT NULL
);


--
-- Name: TABLE section_embeddings; Type: COMMENT; Schema: public; Owner: -
--

COMMENT ON TABLE public.section_embeddings IS 'Per-section halfvec(1024) embeddings. One row per entry of papers_fulltext.sections, keyed by (bibcode, section_index). Bead scix_experiments-wqr.9.';


--
-- Name: COLUMN section_embeddings.section_index; Type: COMMENT; Schema: public; Owner: -
--

COMMENT ON COLUMN public.section_embeddings.section_index IS '0-based position of this section inside papers_fulltext.sections (the source JSONB array). Stable for a given parser_version.';


--
-- Name: COLUMN section_embeddings.section_text_sha256; Type: COMMENT; Schema: public; Owner: -
--

COMMENT ON COLUMN public.section_embeddings.section_text_sha256 IS 'SHA-256 of the exact text that was embedded. Used by the embedder script as a resumability key: skip rows whose hash matches the current section text; re-embed on mismatch after a parser bump.';


--
-- Name: COLUMN section_embeddings.embedding; Type: COMMENT; Schema: public; Owner: -
--

COMMENT ON COLUMN public.section_embeddings.embedding IS 'halfvec(1024). See header comment for the halfvec rationale and the paper_embeddings precedent (migrations 053/054, bead 0vy).';


--
-- Name: section_entities; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.section_entities (
    bibcode text NOT NULL,
    section_index integer NOT NULL,
    entity_id integer NOT NULL,
    link_type text NOT NULL,
    tier smallint DEFAULT 2 NOT NULL,
    tier_version integer DEFAULT 1 NOT NULL,
    confidence real,
    match_method text,
    section_heading text,
    section_role text,
    evidence jsonb,
    harvest_run_id integer,
    created_at timestamp with time zone DEFAULT now() NOT NULL
);


--
-- Name: TABLE section_entities; Type: COMMENT; Schema: public; Owner: -
--

COMMENT ON TABLE public.section_entities IS 'Section-grain entity links. One row per (bibcode, section_index, entity_id, link_type, tier). Pairs with section_embeddings (migration 061) and parallels paper-grain document_entities. Bead scix_experiments-67e.';


--
-- Name: COLUMN section_entities.section_index; Type: COMMENT; Schema: public; Owner: -
--

COMMENT ON COLUMN public.section_entities.section_index IS '0-based position inside papers_fulltext.sections. Same convention as section_embeddings.section_index.';


--
-- Name: COLUMN section_entities.section_role; Type: COMMENT; Schema: public; Owner: -
--

COMMENT ON COLUMN public.section_entities.section_role IS 'Canonical 5-role classification from scix.section_role: background, method, result, conclusion, other. NULL if the classifier declined.';


--
-- Name: COLUMN section_entities.evidence; Type: COMMENT; Schema: public; Owner: -
--

COMMENT ON COLUMN public.section_entities.evidence IS 'JSONB. Mirrors document_entities.evidence shape: matched_surface, start, end, ambiguity_class, is_alias, plus optional additional_spans when the same entity matched multiple surface forms in this section.';


--
-- Name: spdf_spase_crosswalk; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.spdf_spase_crosswalk (
    id integer NOT NULL,
    spdf_id text NOT NULL,
    spase_id text NOT NULL,
    source text DEFAULT 'spdf_harvest'::text NOT NULL,
    created_at timestamp with time zone DEFAULT now()
);


--
-- Name: spdf_spase_crosswalk_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.spdf_spase_crosswalk_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: spdf_spase_crosswalk_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.spdf_spase_crosswalk_id_seq OWNED BY public.spdf_spase_crosswalk.id;


--
-- Name: tier_weight_calibration_log; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.tier_weight_calibration_log (
    id integer NOT NULL,
    version text NOT NULL,
    weights jsonb NOT NULL,
    notes text,
    created_at timestamp with time zone DEFAULT now() NOT NULL
);


--
-- Name: tier_weight_calibration_log_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.tier_weight_calibration_log_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: tier_weight_calibration_log_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.tier_weight_calibration_log_id_seq OWNED BY public.tier_weight_calibration_log.id;


--
-- Name: uat_concepts; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.uat_concepts (
    concept_id text NOT NULL,
    preferred_label text NOT NULL,
    alternate_labels text[] DEFAULT '{}'::text[] NOT NULL,
    definition text,
    level integer
);


--
-- Name: uat_relationships; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.uat_relationships (
    parent_id text NOT NULL,
    child_id text NOT NULL
);


--
-- Name: v_claim_edges; Type: MATERIALIZED VIEW; Schema: public; Owner: -
--

CREATE MATERIALIZED VIEW public.v_claim_edges AS
 SELECT DISTINCT ON (cc.source_bibcode, cc.target_bibcode, cc.char_offset) cc.source_bibcode,
    cc.target_bibcode,
    SUBSTRING(cc.context_text FROM 1 FOR 1000) AS context_snippet,
    cc.intent,
    cc.section_name,
    sp.year AS source_year,
    tp.year AS target_year,
    cc.char_offset
   FROM (((public.citation_contexts cc
     JOIN public.citation_edges ce ON (((ce.source_bibcode = cc.source_bibcode) AND (ce.target_bibcode = cc.target_bibcode))))
     JOIN public.papers sp ON ((sp.bibcode = cc.source_bibcode)))
     JOIN public.papers tp ON ((tp.bibcode = cc.target_bibcode)))
  ORDER BY cc.source_bibcode, cc.target_bibcode, cc.char_offset, cc.intent, cc.ctid
  WITH NO DATA;


--
-- Name: vocabularies; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.vocabularies (
    vocabulary text NOT NULL,
    name text NOT NULL,
    description text,
    license text NOT NULL,
    license_url text,
    homepage_url text,
    source_url text NOT NULL,
    version text,
    record_count integer DEFAULT 0 NOT NULL,
    ingested_at timestamp with time zone DEFAULT now() NOT NULL,
    properties jsonb DEFAULT '{}'::jsonb NOT NULL
);


--
-- Name: works_references; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.works_references (
    source_openalex_id text NOT NULL,
    referenced_openalex_id text NOT NULL
);


--
-- Name: TABLE works_references; Type: COMMENT; Schema: public; Owner: -
--

COMMENT ON TABLE public.works_references IS 'Citation edges between OpenAlex works. source_openalex_id cites referenced_openalex_id. No FK to papers_openalex because referenced works may be outside the ingested corpus (xpac expansion).';


--
-- Name: entities; Type: TABLE; Schema: staging; Owner: -
--

CREATE TABLE staging.entities (
    id integer NOT NULL,
    canonical_name text NOT NULL,
    entity_type text NOT NULL,
    discipline text,
    source text NOT NULL,
    properties jsonb DEFAULT '{}'::jsonb,
    created_at timestamp with time zone DEFAULT now(),
    updated_at timestamp with time zone DEFAULT now()
);


--
-- Name: entities_id_seq; Type: SEQUENCE; Schema: staging; Owner: -
--

CREATE SEQUENCE staging.entities_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: entities_id_seq; Type: SEQUENCE OWNED BY; Schema: staging; Owner: -
--

ALTER SEQUENCE staging.entities_id_seq OWNED BY staging.entities.id;


--
-- Name: entity_aliases; Type: TABLE; Schema: staging; Owner: -
--

CREATE TABLE staging.entity_aliases (
    entity_id integer NOT NULL,
    alias text NOT NULL,
    alias_source text
);


--
-- Name: entity_identifiers; Type: TABLE; Schema: staging; Owner: -
--

CREATE TABLE staging.entity_identifiers (
    entity_id integer NOT NULL,
    id_scheme text NOT NULL,
    external_id text NOT NULL,
    is_primary boolean DEFAULT false
);


--
-- Name: extraction_entity_links; Type: TABLE; Schema: staging; Owner: -
--

CREATE TABLE staging.extraction_entity_links (
    id bigint NOT NULL,
    extraction_id bigint,
    bibcode text NOT NULL,
    entity_type text NOT NULL,
    entity_id integer,
    entity_surface text NOT NULL,
    entity_canonical text,
    span_start integer,
    span_end integer,
    source text NOT NULL,
    confidence_tier smallint NOT NULL,
    confidence real,
    extraction_version text NOT NULL,
    payload jsonb,
    created_at timestamp with time zone DEFAULT now() NOT NULL
)
PARTITION BY LIST (entity_type);


--
-- Name: extraction_entity_links_id_seq; Type: SEQUENCE; Schema: staging; Owner: -
--

CREATE SEQUENCE staging.extraction_entity_links_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: extraction_entity_links_id_seq; Type: SEQUENCE OWNED BY; Schema: staging; Owner: -
--

ALTER SEQUENCE staging.extraction_entity_links_id_seq OWNED BY staging.extraction_entity_links.id;


--
-- Name: extraction_entity_links_dataset; Type: TABLE; Schema: staging; Owner: -
--

CREATE TABLE staging.extraction_entity_links_dataset (
    id bigint DEFAULT nextval('staging.extraction_entity_links_id_seq'::regclass) NOT NULL,
    extraction_id bigint,
    bibcode text NOT NULL,
    entity_type text NOT NULL,
    entity_id integer,
    entity_surface text NOT NULL,
    entity_canonical text,
    span_start integer,
    span_end integer,
    source text NOT NULL,
    confidence_tier smallint NOT NULL,
    confidence real,
    extraction_version text NOT NULL,
    payload jsonb,
    created_at timestamp with time zone DEFAULT now() NOT NULL
);


--
-- Name: extraction_entity_links_default; Type: TABLE; Schema: staging; Owner: -
--

CREATE TABLE staging.extraction_entity_links_default (
    id bigint DEFAULT nextval('staging.extraction_entity_links_id_seq'::regclass) NOT NULL,
    extraction_id bigint,
    bibcode text NOT NULL,
    entity_type text NOT NULL,
    entity_id integer,
    entity_surface text NOT NULL,
    entity_canonical text,
    span_start integer,
    span_end integer,
    source text NOT NULL,
    confidence_tier smallint NOT NULL,
    confidence real,
    extraction_version text NOT NULL,
    payload jsonb,
    created_at timestamp with time zone DEFAULT now() NOT NULL
);


--
-- Name: extraction_entity_links_instrument; Type: TABLE; Schema: staging; Owner: -
--

CREATE TABLE staging.extraction_entity_links_instrument (
    id bigint DEFAULT nextval('staging.extraction_entity_links_id_seq'::regclass) NOT NULL,
    extraction_id bigint,
    bibcode text NOT NULL,
    entity_type text NOT NULL,
    entity_id integer,
    entity_surface text NOT NULL,
    entity_canonical text,
    span_start integer,
    span_end integer,
    source text NOT NULL,
    confidence_tier smallint NOT NULL,
    confidence real,
    extraction_version text NOT NULL,
    payload jsonb,
    created_at timestamp with time zone DEFAULT now() NOT NULL
);


--
-- Name: extraction_entity_links_method; Type: TABLE; Schema: staging; Owner: -
--

CREATE TABLE staging.extraction_entity_links_method (
    id bigint DEFAULT nextval('staging.extraction_entity_links_id_seq'::regclass) NOT NULL,
    extraction_id bigint,
    bibcode text NOT NULL,
    entity_type text NOT NULL,
    entity_id integer,
    entity_surface text NOT NULL,
    entity_canonical text,
    span_start integer,
    span_end integer,
    source text NOT NULL,
    confidence_tier smallint NOT NULL,
    confidence real,
    extraction_version text NOT NULL,
    payload jsonb,
    created_at timestamp with time zone DEFAULT now() NOT NULL
);


--
-- Name: extraction_entity_links_software; Type: TABLE; Schema: staging; Owner: -
--

CREATE TABLE staging.extraction_entity_links_software (
    id bigint DEFAULT nextval('staging.extraction_entity_links_id_seq'::regclass) NOT NULL,
    extraction_id bigint,
    bibcode text NOT NULL,
    entity_type text NOT NULL,
    entity_id integer,
    entity_surface text NOT NULL,
    entity_canonical text,
    span_start integer,
    span_end integer,
    source text NOT NULL,
    confidence_tier smallint NOT NULL,
    confidence real,
    extraction_version text NOT NULL,
    payload jsonb,
    created_at timestamp with time zone DEFAULT now() NOT NULL
);


--
-- Name: extractions; Type: TABLE; Schema: staging; Owner: -
--

CREATE TABLE staging.extractions (
    id integer NOT NULL,
    bibcode text NOT NULL,
    extraction_type text NOT NULL,
    extraction_version text NOT NULL,
    payload jsonb NOT NULL,
    created_at timestamp with time zone DEFAULT now(),
    source text,
    confidence_tier smallint,
    section_name text,
    char_offset integer
);


--
-- Name: extractions_id_seq; Type: SEQUENCE; Schema: staging; Owner: -
--

CREATE SEQUENCE staging.extractions_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: extractions_id_seq; Type: SEQUENCE OWNED BY; Schema: staging; Owner: -
--

ALTER SEQUENCE staging.extractions_id_seq OWNED BY staging.extractions.id;


--
-- Name: document_entities_jit_cache_default; Type: TABLE ATTACH; Schema: public; Owner: -
--

ALTER TABLE ONLY public.document_entities_jit_cache ATTACH PARTITION public.document_entities_jit_cache_default DEFAULT;


--
-- Name: extraction_entity_links_dataset; Type: TABLE ATTACH; Schema: staging; Owner: -
--

ALTER TABLE ONLY staging.extraction_entity_links ATTACH PARTITION staging.extraction_entity_links_dataset FOR VALUES IN ('dataset');


--
-- Name: extraction_entity_links_default; Type: TABLE ATTACH; Schema: staging; Owner: -
--

ALTER TABLE ONLY staging.extraction_entity_links ATTACH PARTITION staging.extraction_entity_links_default DEFAULT;


--
-- Name: extraction_entity_links_instrument; Type: TABLE ATTACH; Schema: staging; Owner: -
--

ALTER TABLE ONLY staging.extraction_entity_links ATTACH PARTITION staging.extraction_entity_links_instrument FOR VALUES IN ('instrument');


--
-- Name: extraction_entity_links_method; Type: TABLE ATTACH; Schema: staging; Owner: -
--

ALTER TABLE ONLY staging.extraction_entity_links ATTACH PARTITION staging.extraction_entity_links_method FOR VALUES IN ('method');


--
-- Name: extraction_entity_links_software; Type: TABLE ATTACH; Schema: staging; Owner: -
--

ALTER TABLE ONLY staging.extraction_entity_links ATTACH PARTITION staging.extraction_entity_links_software FOR VALUES IN ('software');


--
-- Name: alerts id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.alerts ALTER COLUMN id SET DEFAULT nextval('public.alerts_id_seq'::regclass);


--
-- Name: citation_contexts id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.citation_contexts ALTER COLUMN id SET DEFAULT nextval('public.citation_contexts_id_seq'::regclass);


--
-- Name: co_mention_runs id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.co_mention_runs ALTER COLUMN id SET DEFAULT nextval('public.co_mention_runs_id_seq'::regclass);


--
-- Name: core_promotion_log id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.core_promotion_log ALTER COLUMN id SET DEFAULT nextval('public.core_promotion_log_id_seq'::regclass);


--
-- Name: datasets id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.datasets ALTER COLUMN id SET DEFAULT nextval('public.datasets_id_seq'::regclass);


--
-- Name: entities id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.entities ALTER COLUMN id SET DEFAULT nextval('public.entities_id_seq'::regclass);


--
-- Name: entities_staging id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.entities_staging ALTER COLUMN id SET DEFAULT nextval('public.entities_staging_id_seq'::regclass);


--
-- Name: entity_aliases_staging id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.entity_aliases_staging ALTER COLUMN id SET DEFAULT nextval('public.entity_aliases_staging_id_seq'::regclass);


--
-- Name: entity_dictionary id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.entity_dictionary ALTER COLUMN id SET DEFAULT nextval('public.entity_dictionary_id_seq'::regclass);


--
-- Name: entity_identifiers_staging id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.entity_identifiers_staging ALTER COLUMN id SET DEFAULT nextval('public.entity_identifiers_staging_id_seq'::regclass);


--
-- Name: entity_link_disputes id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.entity_link_disputes ALTER COLUMN id SET DEFAULT nextval('public.entity_link_disputes_id_seq'::regclass);


--
-- Name: entity_merge_log id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.entity_merge_log ALTER COLUMN id SET DEFAULT nextval('public.entity_merge_log_id_seq'::regclass);


--
-- Name: entity_relationships id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.entity_relationships ALTER COLUMN id SET DEFAULT nextval('public.entity_relationships_id_seq'::regclass);


--
-- Name: entity_split_log id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.entity_split_log ALTER COLUMN id SET DEFAULT nextval('public.entity_split_log_id_seq'::regclass);


--
-- Name: extraction_entity_links id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.extraction_entity_links ALTER COLUMN id SET DEFAULT nextval('public.extraction_entity_links_id_seq'::regclass);


--
-- Name: extractions id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.extractions ALTER COLUMN id SET DEFAULT nextval('public.extractions_id_seq'::regclass);


--
-- Name: halfvec_backfill_progress id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.halfvec_backfill_progress ALTER COLUMN id SET DEFAULT nextval('public.halfvec_backfill_progress_id_seq'::regclass);


--
-- Name: harvest_runs id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.harvest_runs ALTER COLUMN id SET DEFAULT nextval('public.harvest_runs_id_seq'::regclass);


--
-- Name: link_runs run_id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.link_runs ALTER COLUMN run_id SET DEFAULT nextval('public.link_runs_run_id_seq'::regclass);


--
-- Name: query_log id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.query_log ALTER COLUMN id SET DEFAULT nextval('public.query_log_id_seq'::regclass);


--
-- Name: spdf_spase_crosswalk id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.spdf_spase_crosswalk ALTER COLUMN id SET DEFAULT nextval('public.spdf_spase_crosswalk_id_seq'::regclass);


--
-- Name: tier_weight_calibration_log id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.tier_weight_calibration_log ALTER COLUMN id SET DEFAULT nextval('public.tier_weight_calibration_log_id_seq'::regclass);


--
-- Name: entities id; Type: DEFAULT; Schema: staging; Owner: -
--

ALTER TABLE ONLY staging.entities ALTER COLUMN id SET DEFAULT nextval('staging.entities_id_seq'::regclass);


--
-- Name: extraction_entity_links id; Type: DEFAULT; Schema: staging; Owner: -
--

ALTER TABLE ONLY staging.extraction_entity_links ALTER COLUMN id SET DEFAULT nextval('staging.extraction_entity_links_id_seq'::regclass);


--
-- Name: extractions id; Type: DEFAULT; Schema: staging; Owner: -
--

ALTER TABLE ONLY staging.extractions ALTER COLUMN id SET DEFAULT nextval('staging.extractions_id_seq'::regclass);


--
-- Name: alerts alerts_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.alerts
    ADD CONSTRAINT alerts_pkey PRIMARY KEY (id);


--
-- Name: citation_contexts citation_contexts_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.citation_contexts
    ADD CONSTRAINT citation_contexts_pkey PRIMARY KEY (id);


--
-- Name: citation_diff citation_diff_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.citation_diff
    ADD CONSTRAINT citation_diff_pkey PRIMARY KEY (source_bibcode, target_bibcode);


--
-- Name: citation_edges citation_edges_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.citation_edges
    ADD CONSTRAINT citation_edges_pkey PRIMARY KEY (source_bibcode, target_bibcode);


--
-- Name: co_mention_runs co_mention_runs_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.co_mention_runs
    ADD CONSTRAINT co_mention_runs_pkey PRIMARY KEY (id);


--
-- Name: co_mentions co_mentions_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.co_mentions
    ADD CONSTRAINT co_mentions_pkey PRIMARY KEY (entity_a_id, entity_b_id);


--
-- Name: communities communities_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.communities
    ADD CONSTRAINT communities_pkey PRIMARY KEY (signal, resolution, community_id);


--
-- Name: concept_relationships concept_relationships_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.concept_relationships
    ADD CONSTRAINT concept_relationships_pkey PRIMARY KEY (vocabulary, parent_id, child_id, relationship);


--
-- Name: concepts concepts_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.concepts
    ADD CONSTRAINT concepts_pkey PRIMARY KEY (vocabulary, concept_id);


--
-- Name: core_promotion_log core_promotion_log_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.core_promotion_log
    ADD CONSTRAINT core_promotion_log_pkey PRIMARY KEY (id);


--
-- Name: curated_entity_core curated_entity_core_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.curated_entity_core
    ADD CONSTRAINT curated_entity_core_pkey PRIMARY KEY (entity_id);


--
-- Name: dataset_entities dataset_entities_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.dataset_entities
    ADD CONSTRAINT dataset_entities_pkey PRIMARY KEY (dataset_id, entity_id, relationship);


--
-- Name: datasets datasets_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.datasets
    ADD CONSTRAINT datasets_pkey PRIMARY KEY (id);


--
-- Name: datasets datasets_source_canonical_id_key; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.datasets
    ADD CONSTRAINT datasets_source_canonical_id_key UNIQUE (source, canonical_id);


--
-- Name: document_datasets document_datasets_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.document_datasets
    ADD CONSTRAINT document_datasets_pkey PRIMARY KEY (bibcode, dataset_id, link_type);


--
-- Name: document_entities_jit_cache document_entities_jit_cache_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.document_entities_jit_cache
    ADD CONSTRAINT document_entities_jit_cache_pkey PRIMARY KEY (bibcode, entity_id, link_type, candidate_set_hash, model_version, expires_at);


--
-- Name: document_entities_jit_cache_default document_entities_jit_cache_default_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.document_entities_jit_cache_default
    ADD CONSTRAINT document_entities_jit_cache_default_pkey PRIMARY KEY (bibcode, entity_id, link_type, candidate_set_hash, model_version, expires_at);


--
-- Name: document_entities document_entities_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.document_entities
    ADD CONSTRAINT document_entities_pkey PRIMARY KEY (bibcode, entity_id, link_type, tier);


--
-- Name: embedding_outbox embedding_outbox_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.embedding_outbox
    ADD CONSTRAINT embedding_outbox_pkey PRIMARY KEY (id);


--
-- Name: entities entities_canonical_name_entity_type_source_key; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.entities
    ADD CONSTRAINT entities_canonical_name_entity_type_source_key UNIQUE (canonical_name, entity_type, source);


--
-- Name: entities entities_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.entities
    ADD CONSTRAINT entities_pkey PRIMARY KEY (id);


--
-- Name: entities_staging entities_staging_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.entities_staging
    ADD CONSTRAINT entities_staging_pkey PRIMARY KEY (id);


--
-- Name: entity_aliases entity_aliases_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.entity_aliases
    ADD CONSTRAINT entity_aliases_pkey PRIMARY KEY (entity_id, alias);


--
-- Name: entity_aliases_staging entity_aliases_staging_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.entity_aliases_staging
    ADD CONSTRAINT entity_aliases_staging_pkey PRIMARY KEY (id);


--
-- Name: entity_dictionary entity_dictionary_canonical_name_entity_type_source_key; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.entity_dictionary
    ADD CONSTRAINT entity_dictionary_canonical_name_entity_type_source_key UNIQUE (canonical_name, entity_type, source);


--
-- Name: entity_dictionary entity_dictionary_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.entity_dictionary
    ADD CONSTRAINT entity_dictionary_pkey PRIMARY KEY (id);


--
-- Name: entity_identifiers entity_identifiers_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.entity_identifiers
    ADD CONSTRAINT entity_identifiers_pkey PRIMARY KEY (id_scheme, external_id);


--
-- Name: entity_identifiers_staging entity_identifiers_staging_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.entity_identifiers_staging
    ADD CONSTRAINT entity_identifiers_staging_pkey PRIMARY KEY (id);


--
-- Name: entity_link_audits entity_link_audits_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.entity_link_audits
    ADD CONSTRAINT entity_link_audits_pkey PRIMARY KEY (tier, bibcode, entity_id, annotator);


--
-- Name: entity_link_disputes entity_link_disputes_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.entity_link_disputes
    ADD CONSTRAINT entity_link_disputes_pkey PRIMARY KEY (id);


--
-- Name: entity_merge_log entity_merge_log_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.entity_merge_log
    ADD CONSTRAINT entity_merge_log_pkey PRIMARY KEY (id);


--
-- Name: entity_relationships entity_relationships_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.entity_relationships
    ADD CONSTRAINT entity_relationships_pkey PRIMARY KEY (id);


--
-- Name: entity_relationships entity_relationships_subject_entity_id_predicate_object_ent_key; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.entity_relationships
    ADD CONSTRAINT entity_relationships_subject_entity_id_predicate_object_ent_key UNIQUE (subject_entity_id, predicate, object_entity_id);


--
-- Name: entity_split_log entity_split_log_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.entity_split_log
    ADD CONSTRAINT entity_split_log_pkey PRIMARY KEY (id);


--
-- Name: extraction_entity_links extraction_entity_links_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.extraction_entity_links
    ADD CONSTRAINT extraction_entity_links_pkey PRIMARY KEY (id);


--
-- Name: extractions extractions_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.extractions
    ADD CONSTRAINT extractions_pkey PRIMARY KEY (id);


--
-- Name: fusion_mv_state fusion_mv_state_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.fusion_mv_state
    ADD CONSTRAINT fusion_mv_state_pkey PRIMARY KEY (id);


--
-- Name: halfvec_backfill_progress halfvec_backfill_progress_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.halfvec_backfill_progress
    ADD CONSTRAINT halfvec_backfill_progress_pkey PRIMARY KEY (id);


--
-- Name: harvest_runs harvest_runs_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.harvest_runs
    ADD CONSTRAINT harvest_runs_pkey PRIMARY KEY (id);


--
-- Name: ingest_log ingest_log_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.ingest_log
    ADD CONSTRAINT ingest_log_pkey PRIMARY KEY (filename);


--
-- Name: link_runs link_runs_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.link_runs
    ADD CONSTRAINT link_runs_pkey PRIMARY KEY (run_id);


--
-- Name: llm_cost_ledger llm_cost_ledger_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.llm_cost_ledger
    ADD CONSTRAINT llm_cost_ledger_pkey PRIMARY KEY (day);


--
-- Name: paper_claims paper_claims_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.paper_claims
    ADD CONSTRAINT paper_claims_pkey PRIMARY KEY (claim_id);


--
-- Name: paper_embeddings paper_embeddings_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.paper_embeddings
    ADD CONSTRAINT paper_embeddings_pkey PRIMARY KEY (bibcode, model_name);


--
-- Name: paper_metrics paper_metrics_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.paper_metrics
    ADD CONSTRAINT paper_metrics_pkey PRIMARY KEY (bibcode);


--
-- Name: paper_uat_mappings paper_uat_mappings_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.paper_uat_mappings
    ADD CONSTRAINT paper_uat_mappings_pkey PRIMARY KEY (bibcode, concept_id);


--
-- Name: papers_ads_body papers_ads_body_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.papers_ads_body
    ADD CONSTRAINT papers_ads_body_pkey PRIMARY KEY (bibcode);


--
-- Name: papers_external_ids papers_external_ids_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.papers_external_ids
    ADD CONSTRAINT papers_external_ids_pkey PRIMARY KEY (bibcode);


--
-- Name: papers_fulltext_failures papers_fulltext_failures_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.papers_fulltext_failures
    ADD CONSTRAINT papers_fulltext_failures_pkey PRIMARY KEY (bibcode);


--
-- Name: papers_fulltext papers_fulltext_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.papers_fulltext
    ADD CONSTRAINT papers_fulltext_pkey PRIMARY KEY (bibcode);


--
-- Name: papers_openalex papers_openalex_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.papers_openalex
    ADD CONSTRAINT papers_openalex_pkey PRIMARY KEY (openalex_id);


--
-- Name: papers papers_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.papers
    ADD CONSTRAINT papers_pkey PRIMARY KEY (bibcode);


--
-- Name: query_log query_log_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.query_log
    ADD CONSTRAINT query_log_pkey PRIMARY KEY (id);


--
-- Name: schema_migrations schema_migrations_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.schema_migrations
    ADD CONSTRAINT schema_migrations_pkey PRIMARY KEY (version);


--
-- Name: section_embeddings section_embeddings_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.section_embeddings
    ADD CONSTRAINT section_embeddings_pkey PRIMARY KEY (bibcode, section_index);


--
-- Name: section_entities section_entities_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.section_entities
    ADD CONSTRAINT section_entities_pkey PRIMARY KEY (bibcode, section_index, entity_id, link_type, tier);


--
-- Name: spdf_spase_crosswalk spdf_spase_crosswalk_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.spdf_spase_crosswalk
    ADD CONSTRAINT spdf_spase_crosswalk_pkey PRIMARY KEY (id);


--
-- Name: spdf_spase_crosswalk spdf_spase_crosswalk_spdf_id_spase_id_key; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.spdf_spase_crosswalk
    ADD CONSTRAINT spdf_spase_crosswalk_spdf_id_spase_id_key UNIQUE (spdf_id, spase_id);


--
-- Name: tier_weight_calibration_log tier_weight_calibration_log_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.tier_weight_calibration_log
    ADD CONSTRAINT tier_weight_calibration_log_pkey PRIMARY KEY (id);


--
-- Name: tier_weight_calibration_log tier_weight_calibration_log_version_key; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.tier_weight_calibration_log
    ADD CONSTRAINT tier_weight_calibration_log_version_key UNIQUE (version);


--
-- Name: uat_concepts uat_concepts_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.uat_concepts
    ADD CONSTRAINT uat_concepts_pkey PRIMARY KEY (concept_id);


--
-- Name: uat_relationships uat_relationships_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.uat_relationships
    ADD CONSTRAINT uat_relationships_pkey PRIMARY KEY (parent_id, child_id);


--
-- Name: extractions uq_extractions_bibcode_type_version; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.extractions
    ADD CONSTRAINT uq_extractions_bibcode_type_version UNIQUE (bibcode, extraction_type, extraction_version);


--
-- Name: extraction_entity_links uq_public_eel_bibcode_type_surface_version_source; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.extraction_entity_links
    ADD CONSTRAINT uq_public_eel_bibcode_type_surface_version_source UNIQUE (bibcode, entity_type, entity_surface, extraction_version, source);


--
-- Name: vocabularies vocabularies_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.vocabularies
    ADD CONSTRAINT vocabularies_pkey PRIMARY KEY (vocabulary);


--
-- Name: works_references works_references_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.works_references
    ADD CONSTRAINT works_references_pkey PRIMARY KEY (source_openalex_id, referenced_openalex_id);


--
-- Name: entities entities_canonical_name_entity_type_source_key; Type: CONSTRAINT; Schema: staging; Owner: -
--

ALTER TABLE ONLY staging.entities
    ADD CONSTRAINT entities_canonical_name_entity_type_source_key UNIQUE (canonical_name, entity_type, source);


--
-- Name: entities entities_pkey; Type: CONSTRAINT; Schema: staging; Owner: -
--

ALTER TABLE ONLY staging.entities
    ADD CONSTRAINT entities_pkey PRIMARY KEY (id);


--
-- Name: entity_aliases entity_aliases_pkey; Type: CONSTRAINT; Schema: staging; Owner: -
--

ALTER TABLE ONLY staging.entity_aliases
    ADD CONSTRAINT entity_aliases_pkey PRIMARY KEY (entity_id, alias);


--
-- Name: entity_identifiers entity_identifiers_pkey; Type: CONSTRAINT; Schema: staging; Owner: -
--

ALTER TABLE ONLY staging.entity_identifiers
    ADD CONSTRAINT entity_identifiers_pkey PRIMARY KEY (id_scheme, external_id);


--
-- Name: extraction_entity_links extraction_entity_links_pkey; Type: CONSTRAINT; Schema: staging; Owner: -
--

ALTER TABLE ONLY staging.extraction_entity_links
    ADD CONSTRAINT extraction_entity_links_pkey PRIMARY KEY (id, entity_type);


--
-- Name: extraction_entity_links_dataset extraction_entity_links_dataset_pkey; Type: CONSTRAINT; Schema: staging; Owner: -
--

ALTER TABLE ONLY staging.extraction_entity_links_dataset
    ADD CONSTRAINT extraction_entity_links_dataset_pkey PRIMARY KEY (id, entity_type);


--
-- Name: extraction_entity_links_default extraction_entity_links_default_pkey; Type: CONSTRAINT; Schema: staging; Owner: -
--

ALTER TABLE ONLY staging.extraction_entity_links_default
    ADD CONSTRAINT extraction_entity_links_default_pkey PRIMARY KEY (id, entity_type);


--
-- Name: extraction_entity_links_instrument extraction_entity_links_instrument_pkey; Type: CONSTRAINT; Schema: staging; Owner: -
--

ALTER TABLE ONLY staging.extraction_entity_links_instrument
    ADD CONSTRAINT extraction_entity_links_instrument_pkey PRIMARY KEY (id, entity_type);


--
-- Name: extraction_entity_links_method extraction_entity_links_method_pkey; Type: CONSTRAINT; Schema: staging; Owner: -
--

ALTER TABLE ONLY staging.extraction_entity_links_method
    ADD CONSTRAINT extraction_entity_links_method_pkey PRIMARY KEY (id, entity_type);


--
-- Name: extraction_entity_links_software extraction_entity_links_software_pkey; Type: CONSTRAINT; Schema: staging; Owner: -
--

ALTER TABLE ONLY staging.extraction_entity_links_software
    ADD CONSTRAINT extraction_entity_links_software_pkey PRIMARY KEY (id, entity_type);


--
-- Name: extractions extractions_pkey; Type: CONSTRAINT; Schema: staging; Owner: -
--

ALTER TABLE ONLY staging.extractions
    ADD CONSTRAINT extractions_pkey PRIMARY KEY (id);


--
-- Name: extractions uq_staging_extractions_bibcode_type_version; Type: CONSTRAINT; Schema: staging; Owner: -
--

ALTER TABLE ONLY staging.extractions
    ADD CONSTRAINT uq_staging_extractions_bibcode_type_version UNIQUE (bibcode, extraction_type, extraction_version);


--
-- Name: agent_document_context_bibcode_idx; Type: INDEX; Schema: public; Owner: -
--

CREATE UNIQUE INDEX agent_document_context_bibcode_idx ON public.agent_document_context USING btree (bibcode);


--
-- Name: agent_document_context_year_idx; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX agent_document_context_year_idx ON public.agent_document_context USING btree (year);


--
-- Name: idx_document_entities_jit_cache_lookup; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_document_entities_jit_cache_lookup ON ONLY public.document_entities_jit_cache USING btree (bibcode, candidate_set_hash, model_version);


--
-- Name: document_entities_jit_cache_d_bibcode_candidate_set_hash_mo_idx; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX document_entities_jit_cache_d_bibcode_candidate_set_hash_mo_idx ON public.document_entities_jit_cache_default USING btree (bibcode, candidate_set_hash, model_version);


--
-- Name: idx_document_entities_jit_cache_expires; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_document_entities_jit_cache_expires ON ONLY public.document_entities_jit_cache USING btree (expires_at);


--
-- Name: document_entities_jit_cache_default_expires_at_idx; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX document_entities_jit_cache_default_expires_at_idx ON public.document_entities_jit_cache_default USING btree (expires_at);


--
-- Name: idx_agent_dataset_ctx_id; Type: INDEX; Schema: public; Owner: -
--

CREATE UNIQUE INDEX idx_agent_dataset_ctx_id ON public.agent_dataset_context USING btree (dataset_id);


--
-- Name: idx_agent_doc_ctx_bibcode; Type: INDEX; Schema: public; Owner: -
--

CREATE UNIQUE INDEX idx_agent_doc_ctx_bibcode ON public.agent_document_context USING btree (bibcode);


--
-- Name: idx_agent_entity_ctx_id; Type: INDEX; Schema: public; Owner: -
--

CREATE UNIQUE INDEX idx_agent_entity_ctx_id ON public.agent_entity_context USING btree (entity_id);


--
-- Name: idx_alerts_source; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_alerts_source ON public.alerts USING btree (source);


--
-- Name: idx_alerts_unacked_severity; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_alerts_unacked_severity ON public.alerts USING btree (severity, created_at DESC) WHERE (acked_at IS NULL);


--
-- Name: idx_citation_diff_by_journal_pk; Type: INDEX; Schema: public; Owner: -
--

CREATE UNIQUE INDEX idx_citation_diff_by_journal_pk ON public.citation_diff_by_journal USING btree (journal);


--
-- Name: idx_citation_diff_by_year_pk; Type: INDEX; Schema: public; Owner: -
--

CREATE UNIQUE INDEX idx_citation_diff_by_year_pk ON public.citation_diff_by_year USING btree (pub_year);


--
-- Name: idx_citation_diff_provenance; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_citation_diff_provenance ON public.citation_diff USING btree (in_ads, in_openalex);


--
-- Name: idx_citation_diff_source; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_citation_diff_source ON public.citation_diff USING btree (source_bibcode);


--
-- Name: idx_citation_diff_target; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_citation_diff_target ON public.citation_diff USING btree (target_bibcode);


--
-- Name: idx_citctx_source_target; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_citctx_source_target ON public.citation_contexts USING btree (source_bibcode, target_bibcode);


--
-- Name: idx_citctx_target; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_citctx_target ON public.citation_contexts USING btree (target_bibcode);


--
-- Name: idx_cite_target; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_cite_target ON public.citation_edges USING btree (target_bibcode);


--
-- Name: idx_concept_rel_child; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_concept_rel_child ON public.concept_relationships USING btree (vocabulary, child_id);


--
-- Name: idx_concept_rel_parent; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_concept_rel_parent ON public.concept_relationships USING btree (vocabulary, parent_id);


--
-- Name: idx_concepts_alt_labels; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_concepts_alt_labels ON public.concepts USING gin (alternate_labels);


--
-- Name: idx_concepts_external_uri; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_concepts_external_uri ON public.concepts USING btree (external_uri) WHERE (external_uri IS NOT NULL);


--
-- Name: idx_concepts_label_lower; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_concepts_label_lower ON public.concepts USING btree (vocabulary, lower(preferred_label));


--
-- Name: idx_concepts_vocab; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_concepts_vocab ON public.concepts USING btree (vocabulary);


--
-- Name: idx_core_promotion_log_entity; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_core_promotion_log_entity ON public.core_promotion_log USING btree (entity_id);


--
-- Name: idx_core_promotion_log_ts; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_core_promotion_log_ts ON public.core_promotion_log USING btree (ts);


--
-- Name: idx_crosswalk_spase; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_crosswalk_spase ON public.spdf_spase_crosswalk USING btree (spase_id);


--
-- Name: idx_crosswalk_spdf; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_crosswalk_spdf ON public.spdf_spase_crosswalk USING btree (spdf_id);


--
-- Name: idx_curated_entity_core_hits; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_curated_entity_core_hits ON public.curated_entity_core USING btree (query_hits_14d);


--
-- Name: idx_dec_bibcode; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_dec_bibcode ON public.document_entities_canonical USING btree (bibcode);


--
-- Name: idx_dec_bibcode_entity; Type: INDEX; Schema: public; Owner: -
--

CREATE UNIQUE INDEX idx_dec_bibcode_entity ON public.document_entities_canonical USING btree (bibcode, entity_id);


--
-- Name: idx_dec_entity_fused; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_dec_entity_fused ON public.document_entities_canonical USING btree (entity_id, fused_confidence DESC);


--
-- Name: idx_document_entities_gliner; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_document_entities_gliner ON public.document_entities USING btree (bibcode) WHERE (match_method = 'gliner'::text);


--
-- Name: idx_document_entities_tier; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_document_entities_tier ON public.document_entities USING btree (tier);


--
-- Name: idx_embed_hnsw; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_embed_hnsw ON public.paper_embeddings USING hnsw (embedding public.vector_cosine_ops) WITH (m='16', ef_construction='200');


--
-- Name: idx_embed_hnsw_indus; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_embed_hnsw_indus ON public.paper_embeddings USING hnsw (embedding public.vector_cosine_ops) WITH (m='16', ef_construction='64') WHERE (model_name = 'indus'::text);


--
-- Name: idx_embed_hnsw_indus_hv; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_embed_hnsw_indus_hv ON public.paper_embeddings USING hnsw (embedding_hv public.halfvec_cosine_ops) WITH (m='16', ef_construction='64') WHERE (model_name = 'indus'::text);


--
-- Name: idx_embed_hnsw_nomic; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_embed_hnsw_nomic ON public.paper_embeddings USING hnsw (embedding public.vector_cosine_ops) WITH (m='16', ef_construction='64') WHERE (model_name = 'nomic'::text);


--
-- Name: idx_embed_hnsw_specter2; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_embed_hnsw_specter2 ON public.paper_embeddings USING hnsw (embedding public.vector_cosine_ops) WITH (m='16', ef_construction='64') WHERE (model_name = 'specter2'::text);


--
-- Name: idx_embedding_outbox_drain; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_embedding_outbox_drain ON public.embedding_outbox USING btree (model_name, enqueued_at, id);


--
-- Name: idx_entities_canonical_lower; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_entities_canonical_lower ON public.entities USING btree (lower(canonical_name));


--
-- Name: idx_entities_discipline; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_entities_discipline ON public.entities USING btree (discipline);


--
-- Name: idx_entities_entity_type; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_entities_entity_type ON public.entities USING btree (entity_type);


--
-- Name: idx_entities_gliner_lookup; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_entities_gliner_lookup ON public.entities USING btree (lower(canonical_name), entity_type) WHERE (source = 'gliner'::text);


--
-- Name: idx_entities_properties; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_entities_properties ON public.entities USING gin (properties jsonb_path_ops);


--
-- Name: idx_entities_source_version; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_entities_source_version ON public.entities USING btree (source, source_version) WHERE (source_version IS NOT NULL);


--
-- Name: idx_entities_staging_natural_key; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_entities_staging_natural_key ON public.entities_staging USING btree (canonical_name, entity_type, source);


--
-- Name: idx_entities_staging_run; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_entities_staging_run ON public.entities_staging USING btree (staging_run_id);


--
-- Name: idx_entities_supersedes_id; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_entities_supersedes_id ON public.entities USING btree (supersedes_id) WHERE (supersedes_id IS NOT NULL);


--
-- Name: idx_entity_aliases_lower; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_entity_aliases_lower ON public.entity_aliases USING btree (lower(alias));


--
-- Name: idx_entity_aliases_staging_run; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_entity_aliases_staging_run ON public.entity_aliases_staging USING btree (staging_run_id);


--
-- Name: idx_entity_dict_aliases; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_entity_dict_aliases ON public.entity_dictionary USING gin (aliases);


--
-- Name: idx_entity_dict_canonical_lower; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_entity_dict_canonical_lower ON public.entity_dictionary USING btree (lower(canonical_name));


--
-- Name: idx_entity_dict_discipline; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_entity_dict_discipline ON public.entity_dictionary USING btree (discipline);


--
-- Name: idx_entity_dict_type; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_entity_dict_type ON public.entity_dictionary USING btree (entity_type);


--
-- Name: idx_entity_identifiers_entity_id; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_entity_identifiers_entity_id ON public.entity_identifiers USING btree (entity_id);


--
-- Name: idx_entity_identifiers_staging_run; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_entity_identifiers_staging_run ON public.entity_identifiers_staging USING btree (staging_run_id);


--
-- Name: idx_entity_link_audits_annotator; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_entity_link_audits_annotator ON public.entity_link_audits USING btree (annotator);


--
-- Name: idx_entity_link_audits_tier_label; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_entity_link_audits_tier_label ON public.entity_link_audits USING btree (tier, label);


--
-- Name: idx_entity_link_disputes_bibcode; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_entity_link_disputes_bibcode ON public.entity_link_disputes USING btree (bibcode);


--
-- Name: idx_entity_link_disputes_entity; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_entity_link_disputes_entity ON public.entity_link_disputes USING btree (entity_id);


--
-- Name: idx_entity_link_disputes_reported_at; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_entity_link_disputes_reported_at ON public.entity_link_disputes USING btree (reported_at);


--
-- Name: idx_entity_merge_log_at; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_entity_merge_log_at ON public.entity_merge_log USING btree (merged_at);


--
-- Name: idx_entity_merge_log_new; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_entity_merge_log_new ON public.entity_merge_log USING btree (new_entity_id);


--
-- Name: idx_entity_merge_log_old; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_entity_merge_log_old ON public.entity_merge_log USING btree (old_entity_id);


--
-- Name: idx_entity_relationships_evidence; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_entity_relationships_evidence ON public.entity_relationships USING gin (evidence jsonb_path_ops);


--
-- Name: idx_entity_relationships_object; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_entity_relationships_object ON public.entity_relationships USING btree (object_entity_id);


--
-- Name: idx_entity_relationships_predicate; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_entity_relationships_predicate ON public.entity_relationships USING btree (predicate);


--
-- Name: idx_entity_relationships_source; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_entity_relationships_source ON public.entity_relationships USING btree (source);


--
-- Name: idx_entity_split_log_at; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_entity_split_log_at ON public.entity_split_log USING btree (split_at);


--
-- Name: idx_entity_split_log_parent; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_entity_split_log_parent ON public.entity_split_log USING btree (parent_entity_id);


--
-- Name: idx_extractions_bibcode; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_extractions_bibcode ON public.extractions USING btree (bibcode);


--
-- Name: idx_extractions_confidence_tier; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_extractions_confidence_tier ON public.extractions USING btree (confidence_tier);


--
-- Name: idx_extractions_source; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_extractions_source ON public.extractions USING btree (source);


--
-- Name: idx_extractions_type; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_extractions_type ON public.extractions USING btree (extraction_type);


--
-- Name: idx_halfvec_backfill_progress_model; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_halfvec_backfill_progress_model ON public.halfvec_backfill_progress USING btree (model_name, started_at DESC);


--
-- Name: idx_harvest_runs_source; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_harvest_runs_source ON public.harvest_runs USING btree (source);


--
-- Name: idx_link_runs_max_entry_date_desc; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_link_runs_max_entry_date_desc ON public.link_runs USING btree (max_entry_date DESC NULLS LAST);


--
-- Name: idx_link_runs_timestamp_desc; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_link_runs_timestamp_desc ON public.link_runs USING btree ("timestamp" DESC);


--
-- Name: idx_papers_ads_body_tsv; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_papers_ads_body_tsv ON public.papers_ads_body USING gin (tsv);


--
-- Name: idx_papers_arxiv; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_papers_arxiv ON public.papers USING gin (arxiv_class);


--
-- Name: idx_papers_author_count; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_papers_author_count ON public.papers USING btree (author_count);


--
-- Name: idx_papers_author_norm; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_papers_author_norm ON public.papers USING gin (author_norm);


--
-- Name: idx_papers_authors; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_papers_authors ON public.papers USING gin (authors);


--
-- Name: idx_papers_correction_events; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_papers_correction_events ON public.papers USING gin (correction_events);


--
-- Name: idx_papers_data; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_papers_data ON public.papers USING gin (data);


--
-- Name: idx_papers_doctype; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_papers_doctype ON public.papers USING btree (doctype);


--
-- Name: idx_papers_doi; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_papers_doi ON public.papers USING gin (doi);


--
-- Name: idx_papers_esources; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_papers_esources ON public.papers USING gin (esources);


--
-- Name: idx_papers_external_ids_arxiv_id; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_papers_external_ids_arxiv_id ON public.papers_external_ids USING btree (arxiv_id);


--
-- Name: idx_papers_external_ids_doi; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_papers_external_ids_doi ON public.papers_external_ids USING btree (doi);


--
-- Name: idx_papers_external_ids_doi_lower; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_papers_external_ids_doi_lower ON public.papers_external_ids USING btree (lower(doi)) WHERE (doi IS NOT NULL);


--
-- Name: idx_papers_external_ids_openalex_id; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_papers_external_ids_openalex_id ON public.papers_external_ids USING btree (openalex_id);


--
-- Name: idx_papers_external_ids_s2_corpus_id; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_papers_external_ids_s2_corpus_id ON public.papers_external_ids USING btree (s2_corpus_id);


--
-- Name: idx_papers_facility; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_papers_facility ON public.papers USING gin (facility);


--
-- Name: idx_papers_first_author; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_papers_first_author ON public.papers USING btree (first_author);


--
-- Name: idx_papers_fulltext_canonical_bibcode; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_papers_fulltext_canonical_bibcode ON public.papers_fulltext USING btree (canonical_bibcode);


--
-- Name: idx_papers_fulltext_failures_retry_after; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_papers_fulltext_failures_retry_after ON public.papers_fulltext_failures USING btree (retry_after);


--
-- Name: idx_papers_fulltext_sections_tsv; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_papers_fulltext_sections_tsv ON public.papers_fulltext USING gin (sections_tsv);


--
-- Name: idx_papers_fulltext_source; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_papers_fulltext_source ON public.papers_fulltext USING btree (source);


--
-- Name: idx_papers_fulltext_suppressed_by_publisher; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_papers_fulltext_suppressed_by_publisher ON public.papers_fulltext USING btree (suppressed_by_publisher) WHERE (suppressed_by_publisher = true);


--
-- Name: idx_papers_is_oa; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_papers_is_oa ON public.papers USING btree (public.papers_is_oa_or_preprint(papers.*)) WHERE (body IS NOT NULL);


--
-- Name: idx_papers_keyword_norm; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_papers_keyword_norm ON public.papers USING gin (keyword_norm);


--
-- Name: idx_papers_keywords; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_papers_keywords ON public.papers USING gin (keywords);


--
-- Name: idx_papers_nedid; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_papers_nedid ON public.papers USING gin (nedid);


--
-- Name: idx_papers_openalex_doi; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_papers_openalex_doi ON public.papers_openalex USING btree (doi) WHERE (doi IS NOT NULL);


--
-- Name: idx_papers_openalex_id; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_papers_openalex_id ON public.papers USING btree (openalex_id);


--
-- Name: idx_papers_openalex_updated; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_papers_openalex_updated ON public.papers_openalex USING btree (updated_date);


--
-- Name: idx_papers_openalex_year; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_papers_openalex_year ON public.papers_openalex USING btree (publication_year);


--
-- Name: idx_papers_retracted_at; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_papers_retracted_at ON public.papers USING btree (retracted_at) WHERE (retracted_at IS NOT NULL);


--
-- Name: idx_papers_simbid; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_papers_simbid ON public.papers USING gin (simbid);


--
-- Name: idx_papers_tsv; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_papers_tsv ON public.papers USING gin (tsv);


--
-- Name: idx_papers_year; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_papers_year ON public.papers USING btree (year);


--
-- Name: idx_pm_community_coarse; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_pm_community_coarse ON public.paper_metrics USING btree (community_id_coarse);


--
-- Name: idx_pm_community_fine; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_pm_community_fine ON public.paper_metrics USING btree (community_id_fine);


--
-- Name: idx_pm_community_medium; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_pm_community_medium ON public.paper_metrics USING btree (community_id_medium);


--
-- Name: idx_pm_community_semantic_coarse; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_pm_community_semantic_coarse ON public.paper_metrics USING btree (community_semantic_coarse);


--
-- Name: idx_pm_community_semantic_fine; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_pm_community_semantic_fine ON public.paper_metrics USING btree (community_semantic_fine);


--
-- Name: idx_pm_community_semantic_medium; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_pm_community_semantic_medium ON public.paper_metrics USING btree (community_semantic_medium);


--
-- Name: idx_pm_community_taxonomic; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_pm_community_taxonomic ON public.paper_metrics USING btree (community_taxonomic);


--
-- Name: idx_pm_pagerank; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_pm_pagerank ON public.paper_metrics USING btree (pagerank DESC);


--
-- Name: idx_public_eel_bibcode; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_public_eel_bibcode ON public.extraction_entity_links USING btree (bibcode);


--
-- Name: idx_public_eel_entity_id; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_public_eel_entity_id ON public.extraction_entity_links USING btree (entity_id) WHERE (entity_id IS NOT NULL);


--
-- Name: idx_public_eel_entity_type; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_public_eel_entity_type ON public.extraction_entity_links USING btree (entity_type);


--
-- Name: idx_pum_concept; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_pum_concept ON public.paper_uat_mappings USING btree (concept_id);


--
-- Name: idx_pum_match_type; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_pum_match_type ON public.paper_uat_mappings USING btree (match_type);


--
-- Name: idx_query_log_created_at; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_query_log_created_at ON public.query_log USING btree (created_at);


--
-- Name: idx_query_log_session_id; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_query_log_session_id ON public.query_log USING btree (session_id);


--
-- Name: idx_query_log_tool; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_query_log_tool ON public.query_log USING btree (tool);


--
-- Name: idx_query_log_tool_name; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_query_log_tool_name ON public.query_log USING btree (tool_name);


--
-- Name: idx_query_log_ts; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_query_log_ts ON public.query_log USING btree (ts);


--
-- Name: idx_section_embeddings_hnsw; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_section_embeddings_hnsw ON public.section_embeddings USING hnsw (embedding public.halfvec_cosine_ops) WITH (m='16', ef_construction='64');


--
-- Name: idx_section_entities_bibcode_entity; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_section_entities_bibcode_entity ON public.section_entities USING btree (bibcode, entity_id);


--
-- Name: idx_section_entities_entity_id; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_section_entities_entity_id ON public.section_entities USING btree (entity_id);


--
-- Name: idx_section_entities_role; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_section_entities_role ON public.section_entities USING btree (section_role) WHERE (section_role IS NOT NULL);


--
-- Name: idx_uat_alternate_labels; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_uat_alternate_labels ON public.uat_concepts USING gin (alternate_labels);


--
-- Name: idx_uat_preferred_label; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_uat_preferred_label ON public.uat_concepts USING btree (preferred_label);


--
-- Name: idx_uat_rel_child; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_uat_rel_child ON public.uat_relationships USING btree (child_id);


--
-- Name: idx_v_claim_edges_pk; Type: INDEX; Schema: public; Owner: -
--

CREATE UNIQUE INDEX idx_v_claim_edges_pk ON public.v_claim_edges USING btree (source_bibcode, target_bibcode, char_offset) NULLS NOT DISTINCT;


--
-- Name: idx_v_claim_edges_source_intent; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_v_claim_edges_source_intent ON public.v_claim_edges USING btree (source_bibcode, intent);


--
-- Name: idx_v_claim_edges_target_intent; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_v_claim_edges_target_intent ON public.v_claim_edges USING btree (target_bibcode, intent);


--
-- Name: idx_works_references_target; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_works_references_target ON public.works_references USING btree (referenced_openalex_id);


--
-- Name: ix_co_mention_runs_started_at; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_co_mention_runs_started_at ON public.co_mention_runs USING btree (started_at DESC);


--
-- Name: ix_co_mentions_a_npapers; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_co_mentions_a_npapers ON public.co_mentions USING btree (entity_a_id, n_papers DESC);


--
-- Name: ix_co_mentions_b_npapers; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_co_mentions_b_npapers ON public.co_mentions USING btree (entity_b_id, n_papers DESC);


--
-- Name: ix_paper_claims_bibcode_section; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_paper_claims_bibcode_section ON public.paper_claims USING btree (bibcode, section_index);


--
-- Name: ix_paper_claims_claim_text_tsv; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_paper_claims_claim_text_tsv ON public.paper_claims USING gin (to_tsvector('english'::regconfig, claim_text));


--
-- Name: ix_paper_claims_claim_type; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_paper_claims_claim_type ON public.paper_claims USING btree (claim_type);


--
-- Name: ix_paper_claims_linked_entity_object_id; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_paper_claims_linked_entity_object_id ON public.paper_claims USING btree (linked_entity_object_id);


--
-- Name: ix_paper_claims_linked_entity_subject_id; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_paper_claims_linked_entity_subject_id ON public.paper_claims USING btree (linked_entity_subject_id);


--
-- Name: ix_papers_alternate_bibcode_gin; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_papers_alternate_bibcode_gin ON public.papers USING gin (alternate_bibcode);


--
-- Name: ux_paper_claims_provenance_text; Type: INDEX; Schema: public; Owner: -
--

CREATE UNIQUE INDEX ux_paper_claims_provenance_text ON public.paper_claims USING btree (bibcode, section_index, paragraph_index, char_span_start, char_span_end, md5(claim_text));


--
-- Name: idx_staging_eel_source_tier_version; Type: INDEX; Schema: staging; Owner: -
--

CREATE INDEX idx_staging_eel_source_tier_version ON ONLY staging.extraction_entity_links USING btree (source, confidence_tier, extraction_version);


--
-- Name: extraction_entity_links_datas_source_confidence_tier_extrac_idx; Type: INDEX; Schema: staging; Owner: -
--

CREATE INDEX extraction_entity_links_datas_source_confidence_tier_extrac_idx ON staging.extraction_entity_links_dataset USING btree (source, confidence_tier, extraction_version);


--
-- Name: idx_staging_eel_bibcode; Type: INDEX; Schema: staging; Owner: -
--

CREATE INDEX idx_staging_eel_bibcode ON ONLY staging.extraction_entity_links USING btree (bibcode);


--
-- Name: extraction_entity_links_dataset_bibcode_idx; Type: INDEX; Schema: staging; Owner: -
--

CREATE INDEX extraction_entity_links_dataset_bibcode_idx ON staging.extraction_entity_links_dataset USING btree (bibcode);


--
-- Name: idx_staging_eel_created_at; Type: INDEX; Schema: staging; Owner: -
--

CREATE INDEX idx_staging_eel_created_at ON ONLY staging.extraction_entity_links USING btree (created_at);


--
-- Name: extraction_entity_links_dataset_created_at_idx; Type: INDEX; Schema: staging; Owner: -
--

CREATE INDEX extraction_entity_links_dataset_created_at_idx ON staging.extraction_entity_links_dataset USING btree (created_at);


--
-- Name: extraction_entity_links_defau_source_confidence_tier_extrac_idx; Type: INDEX; Schema: staging; Owner: -
--

CREATE INDEX extraction_entity_links_defau_source_confidence_tier_extrac_idx ON staging.extraction_entity_links_default USING btree (source, confidence_tier, extraction_version);


--
-- Name: extraction_entity_links_default_bibcode_idx; Type: INDEX; Schema: staging; Owner: -
--

CREATE INDEX extraction_entity_links_default_bibcode_idx ON staging.extraction_entity_links_default USING btree (bibcode);


--
-- Name: extraction_entity_links_default_created_at_idx; Type: INDEX; Schema: staging; Owner: -
--

CREATE INDEX extraction_entity_links_default_created_at_idx ON staging.extraction_entity_links_default USING btree (created_at);


--
-- Name: extraction_entity_links_instr_source_confidence_tier_extrac_idx; Type: INDEX; Schema: staging; Owner: -
--

CREATE INDEX extraction_entity_links_instr_source_confidence_tier_extrac_idx ON staging.extraction_entity_links_instrument USING btree (source, confidence_tier, extraction_version);


--
-- Name: extraction_entity_links_instrument_bibcode_idx; Type: INDEX; Schema: staging; Owner: -
--

CREATE INDEX extraction_entity_links_instrument_bibcode_idx ON staging.extraction_entity_links_instrument USING btree (bibcode);


--
-- Name: extraction_entity_links_instrument_created_at_idx; Type: INDEX; Schema: staging; Owner: -
--

CREATE INDEX extraction_entity_links_instrument_created_at_idx ON staging.extraction_entity_links_instrument USING btree (created_at);


--
-- Name: extraction_entity_links_metho_source_confidence_tier_extrac_idx; Type: INDEX; Schema: staging; Owner: -
--

CREATE INDEX extraction_entity_links_metho_source_confidence_tier_extrac_idx ON staging.extraction_entity_links_method USING btree (source, confidence_tier, extraction_version);


--
-- Name: extraction_entity_links_method_bibcode_idx; Type: INDEX; Schema: staging; Owner: -
--

CREATE INDEX extraction_entity_links_method_bibcode_idx ON staging.extraction_entity_links_method USING btree (bibcode);


--
-- Name: extraction_entity_links_method_created_at_idx; Type: INDEX; Schema: staging; Owner: -
--

CREATE INDEX extraction_entity_links_method_created_at_idx ON staging.extraction_entity_links_method USING btree (created_at);


--
-- Name: extraction_entity_links_softw_source_confidence_tier_extrac_idx; Type: INDEX; Schema: staging; Owner: -
--

CREATE INDEX extraction_entity_links_softw_source_confidence_tier_extrac_idx ON staging.extraction_entity_links_software USING btree (source, confidence_tier, extraction_version);


--
-- Name: extraction_entity_links_software_bibcode_idx; Type: INDEX; Schema: staging; Owner: -
--

CREATE INDEX extraction_entity_links_software_bibcode_idx ON staging.extraction_entity_links_software USING btree (bibcode);


--
-- Name: extraction_entity_links_software_created_at_idx; Type: INDEX; Schema: staging; Owner: -
--

CREATE INDEX extraction_entity_links_software_created_at_idx ON staging.extraction_entity_links_software USING btree (created_at);


--
-- Name: idx_staging_extractions_bibcode; Type: INDEX; Schema: staging; Owner: -
--

CREATE INDEX idx_staging_extractions_bibcode ON staging.extractions USING btree (bibcode);


--
-- Name: idx_staging_extractions_section_name; Type: INDEX; Schema: staging; Owner: -
--

CREATE INDEX idx_staging_extractions_section_name ON staging.extractions USING btree (section_name) WHERE (section_name IS NOT NULL);


--
-- Name: idx_staging_extractions_source; Type: INDEX; Schema: staging; Owner: -
--

CREATE INDEX idx_staging_extractions_source ON staging.extractions USING btree (source);


--
-- Name: idx_staging_extractions_type; Type: INDEX; Schema: staging; Owner: -
--

CREATE INDEX idx_staging_extractions_type ON staging.extractions USING btree (extraction_type);


--
-- Name: document_entities_jit_cache_d_bibcode_candidate_set_hash_mo_idx; Type: INDEX ATTACH; Schema: public; Owner: -
--

ALTER INDEX public.idx_document_entities_jit_cache_lookup ATTACH PARTITION public.document_entities_jit_cache_d_bibcode_candidate_set_hash_mo_idx;


--
-- Name: document_entities_jit_cache_default_expires_at_idx; Type: INDEX ATTACH; Schema: public; Owner: -
--

ALTER INDEX public.idx_document_entities_jit_cache_expires ATTACH PARTITION public.document_entities_jit_cache_default_expires_at_idx;


--
-- Name: document_entities_jit_cache_default_pkey; Type: INDEX ATTACH; Schema: public; Owner: -
--

ALTER INDEX public.document_entities_jit_cache_pkey ATTACH PARTITION public.document_entities_jit_cache_default_pkey;


--
-- Name: extraction_entity_links_datas_source_confidence_tier_extrac_idx; Type: INDEX ATTACH; Schema: staging; Owner: -
--

ALTER INDEX staging.idx_staging_eel_source_tier_version ATTACH PARTITION staging.extraction_entity_links_datas_source_confidence_tier_extrac_idx;


--
-- Name: extraction_entity_links_dataset_bibcode_idx; Type: INDEX ATTACH; Schema: staging; Owner: -
--

ALTER INDEX staging.idx_staging_eel_bibcode ATTACH PARTITION staging.extraction_entity_links_dataset_bibcode_idx;


--
-- Name: extraction_entity_links_dataset_created_at_idx; Type: INDEX ATTACH; Schema: staging; Owner: -
--

ALTER INDEX staging.idx_staging_eel_created_at ATTACH PARTITION staging.extraction_entity_links_dataset_created_at_idx;


--
-- Name: extraction_entity_links_dataset_pkey; Type: INDEX ATTACH; Schema: staging; Owner: -
--

ALTER INDEX staging.extraction_entity_links_pkey ATTACH PARTITION staging.extraction_entity_links_dataset_pkey;


--
-- Name: extraction_entity_links_defau_source_confidence_tier_extrac_idx; Type: INDEX ATTACH; Schema: staging; Owner: -
--

ALTER INDEX staging.idx_staging_eel_source_tier_version ATTACH PARTITION staging.extraction_entity_links_defau_source_confidence_tier_extrac_idx;


--
-- Name: extraction_entity_links_default_bibcode_idx; Type: INDEX ATTACH; Schema: staging; Owner: -
--

ALTER INDEX staging.idx_staging_eel_bibcode ATTACH PARTITION staging.extraction_entity_links_default_bibcode_idx;


--
-- Name: extraction_entity_links_default_created_at_idx; Type: INDEX ATTACH; Schema: staging; Owner: -
--

ALTER INDEX staging.idx_staging_eel_created_at ATTACH PARTITION staging.extraction_entity_links_default_created_at_idx;


--
-- Name: extraction_entity_links_default_pkey; Type: INDEX ATTACH; Schema: staging; Owner: -
--

ALTER INDEX staging.extraction_entity_links_pkey ATTACH PARTITION staging.extraction_entity_links_default_pkey;


--
-- Name: extraction_entity_links_instr_source_confidence_tier_extrac_idx; Type: INDEX ATTACH; Schema: staging; Owner: -
--

ALTER INDEX staging.idx_staging_eel_source_tier_version ATTACH PARTITION staging.extraction_entity_links_instr_source_confidence_tier_extrac_idx;


--
-- Name: extraction_entity_links_instrument_bibcode_idx; Type: INDEX ATTACH; Schema: staging; Owner: -
--

ALTER INDEX staging.idx_staging_eel_bibcode ATTACH PARTITION staging.extraction_entity_links_instrument_bibcode_idx;


--
-- Name: extraction_entity_links_instrument_created_at_idx; Type: INDEX ATTACH; Schema: staging; Owner: -
--

ALTER INDEX staging.idx_staging_eel_created_at ATTACH PARTITION staging.extraction_entity_links_instrument_created_at_idx;


--
-- Name: extraction_entity_links_instrument_pkey; Type: INDEX ATTACH; Schema: staging; Owner: -
--

ALTER INDEX staging.extraction_entity_links_pkey ATTACH PARTITION staging.extraction_entity_links_instrument_pkey;


--
-- Name: extraction_entity_links_metho_source_confidence_tier_extrac_idx; Type: INDEX ATTACH; Schema: staging; Owner: -
--

ALTER INDEX staging.idx_staging_eel_source_tier_version ATTACH PARTITION staging.extraction_entity_links_metho_source_confidence_tier_extrac_idx;


--
-- Name: extraction_entity_links_method_bibcode_idx; Type: INDEX ATTACH; Schema: staging; Owner: -
--

ALTER INDEX staging.idx_staging_eel_bibcode ATTACH PARTITION staging.extraction_entity_links_method_bibcode_idx;


--
-- Name: extraction_entity_links_method_created_at_idx; Type: INDEX ATTACH; Schema: staging; Owner: -
--

ALTER INDEX staging.idx_staging_eel_created_at ATTACH PARTITION staging.extraction_entity_links_method_created_at_idx;


--
-- Name: extraction_entity_links_method_pkey; Type: INDEX ATTACH; Schema: staging; Owner: -
--

ALTER INDEX staging.extraction_entity_links_pkey ATTACH PARTITION staging.extraction_entity_links_method_pkey;


--
-- Name: extraction_entity_links_softw_source_confidence_tier_extrac_idx; Type: INDEX ATTACH; Schema: staging; Owner: -
--

ALTER INDEX staging.idx_staging_eel_source_tier_version ATTACH PARTITION staging.extraction_entity_links_softw_source_confidence_tier_extrac_idx;


--
-- Name: extraction_entity_links_software_bibcode_idx; Type: INDEX ATTACH; Schema: staging; Owner: -
--

ALTER INDEX staging.idx_staging_eel_bibcode ATTACH PARTITION staging.extraction_entity_links_software_bibcode_idx;


--
-- Name: extraction_entity_links_software_created_at_idx; Type: INDEX ATTACH; Schema: staging; Owner: -
--

ALTER INDEX staging.idx_staging_eel_created_at ATTACH PARTITION staging.extraction_entity_links_software_created_at_idx;


--
-- Name: extraction_entity_links_software_pkey; Type: INDEX ATTACH; Schema: staging; Owner: -
--

ALTER INDEX staging.extraction_entity_links_pkey ATTACH PARTITION staging.extraction_entity_links_software_pkey;


--
-- Name: paper_embeddings trg_embedding_outbox; Type: TRIGGER; Schema: public; Owner: -
--

CREATE TRIGGER trg_embedding_outbox AFTER INSERT OR DELETE OR UPDATE ON public.paper_embeddings FOR EACH ROW EXECUTE FUNCTION public.embedding_outbox_enqueue();


--
-- Name: papers_external_ids trig_papers_external_ids_updated_at; Type: TRIGGER; Schema: public; Owner: -
--

CREATE TRIGGER trig_papers_external_ids_updated_at BEFORE UPDATE ON public.papers_external_ids FOR EACH ROW EXECUTE FUNCTION public.papers_external_ids_touch();


--
-- Name: papers trig_papers_tsv; Type: TRIGGER; Schema: public; Owner: -
--

CREATE TRIGGER trig_papers_tsv BEFORE INSERT OR UPDATE OF title, abstract, keywords ON public.papers FOR EACH ROW EXECUTE FUNCTION public.papers_tsv_trigger();


--
-- Name: concept_relationships concept_relationships_vocabulary_child_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.concept_relationships
    ADD CONSTRAINT concept_relationships_vocabulary_child_id_fkey FOREIGN KEY (vocabulary, child_id) REFERENCES public.concepts(vocabulary, concept_id) ON DELETE CASCADE;


--
-- Name: concept_relationships concept_relationships_vocabulary_parent_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.concept_relationships
    ADD CONSTRAINT concept_relationships_vocabulary_parent_id_fkey FOREIGN KEY (vocabulary, parent_id) REFERENCES public.concepts(vocabulary, concept_id) ON DELETE CASCADE;


--
-- Name: concepts concepts_vocabulary_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.concepts
    ADD CONSTRAINT concepts_vocabulary_fkey FOREIGN KEY (vocabulary) REFERENCES public.vocabularies(vocabulary) ON DELETE CASCADE;


--
-- Name: curated_entity_core curated_entity_core_entity_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.curated_entity_core
    ADD CONSTRAINT curated_entity_core_entity_id_fkey FOREIGN KEY (entity_id) REFERENCES public.entities(id) ON DELETE CASCADE;


--
-- Name: dataset_entities dataset_entities_dataset_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.dataset_entities
    ADD CONSTRAINT dataset_entities_dataset_id_fkey FOREIGN KEY (dataset_id) REFERENCES public.datasets(id) ON DELETE CASCADE;


--
-- Name: dataset_entities dataset_entities_entity_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.dataset_entities
    ADD CONSTRAINT dataset_entities_entity_id_fkey FOREIGN KEY (entity_id) REFERENCES public.entities(id) ON DELETE CASCADE;


--
-- Name: datasets datasets_harvest_run_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.datasets
    ADD CONSTRAINT datasets_harvest_run_id_fkey FOREIGN KEY (harvest_run_id) REFERENCES public.harvest_runs(id);


--
-- Name: document_datasets document_datasets_dataset_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.document_datasets
    ADD CONSTRAINT document_datasets_dataset_id_fkey FOREIGN KEY (dataset_id) REFERENCES public.datasets(id) ON DELETE CASCADE;


--
-- Name: document_datasets document_datasets_harvest_run_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.document_datasets
    ADD CONSTRAINT document_datasets_harvest_run_id_fkey FOREIGN KEY (harvest_run_id) REFERENCES public.harvest_runs(id);


--
-- Name: document_entities document_entities_entity_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.document_entities
    ADD CONSTRAINT document_entities_entity_id_fkey FOREIGN KEY (entity_id) REFERENCES public.entities(id) ON DELETE CASCADE;


--
-- Name: document_entities document_entities_harvest_run_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.document_entities
    ADD CONSTRAINT document_entities_harvest_run_id_fkey FOREIGN KEY (harvest_run_id) REFERENCES public.harvest_runs(id);


--
-- Name: entities entities_harvest_run_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.entities
    ADD CONSTRAINT entities_harvest_run_id_fkey FOREIGN KEY (harvest_run_id) REFERENCES public.harvest_runs(id);


--
-- Name: entities entities_supersedes_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.entities
    ADD CONSTRAINT entities_supersedes_id_fkey FOREIGN KEY (supersedes_id) REFERENCES public.entities(id) ON DELETE SET NULL;


--
-- Name: entity_aliases entity_aliases_entity_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.entity_aliases
    ADD CONSTRAINT entity_aliases_entity_id_fkey FOREIGN KEY (entity_id) REFERENCES public.entities(id) ON DELETE CASCADE;


--
-- Name: entity_identifiers entity_identifiers_entity_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.entity_identifiers
    ADD CONSTRAINT entity_identifiers_entity_id_fkey FOREIGN KEY (entity_id) REFERENCES public.entities(id) ON DELETE CASCADE;


--
-- Name: entity_merge_log entity_merge_log_new_entity_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.entity_merge_log
    ADD CONSTRAINT entity_merge_log_new_entity_id_fkey FOREIGN KEY (new_entity_id) REFERENCES public.entities(id);


--
-- Name: entity_relationships entity_relationships_harvest_run_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.entity_relationships
    ADD CONSTRAINT entity_relationships_harvest_run_id_fkey FOREIGN KEY (harvest_run_id) REFERENCES public.harvest_runs(id);


--
-- Name: entity_relationships entity_relationships_object_entity_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.entity_relationships
    ADD CONSTRAINT entity_relationships_object_entity_id_fkey FOREIGN KEY (object_entity_id) REFERENCES public.entities(id) ON DELETE CASCADE;


--
-- Name: entity_relationships entity_relationships_subject_entity_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.entity_relationships
    ADD CONSTRAINT entity_relationships_subject_entity_id_fkey FOREIGN KEY (subject_entity_id) REFERENCES public.entities(id) ON DELETE CASCADE;


--
-- Name: extraction_entity_links extraction_entity_links_bibcode_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.extraction_entity_links
    ADD CONSTRAINT extraction_entity_links_bibcode_fkey FOREIGN KEY (bibcode) REFERENCES public.papers(bibcode);


--
-- Name: extractions extractions_bibcode_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.extractions
    ADD CONSTRAINT extractions_bibcode_fkey FOREIGN KEY (bibcode) REFERENCES public.papers(bibcode);


--
-- Name: paper_claims paper_claims_bibcode_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.paper_claims
    ADD CONSTRAINT paper_claims_bibcode_fkey FOREIGN KEY (bibcode) REFERENCES public.papers(bibcode);


--
-- Name: paper_embeddings paper_embeddings_bibcode_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.paper_embeddings
    ADD CONSTRAINT paper_embeddings_bibcode_fkey FOREIGN KEY (bibcode) REFERENCES public.papers(bibcode);


--
-- Name: paper_metrics paper_metrics_bibcode_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.paper_metrics
    ADD CONSTRAINT paper_metrics_bibcode_fkey FOREIGN KEY (bibcode) REFERENCES public.papers(bibcode);


--
-- Name: paper_uat_mappings paper_uat_mappings_bibcode_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.paper_uat_mappings
    ADD CONSTRAINT paper_uat_mappings_bibcode_fkey FOREIGN KEY (bibcode) REFERENCES public.papers(bibcode);


--
-- Name: paper_uat_mappings paper_uat_mappings_concept_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.paper_uat_mappings
    ADD CONSTRAINT paper_uat_mappings_concept_id_fkey FOREIGN KEY (concept_id) REFERENCES public.uat_concepts(concept_id);


--
-- Name: papers_ads_body papers_ads_body_bibcode_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.papers_ads_body
    ADD CONSTRAINT papers_ads_body_bibcode_fkey FOREIGN KEY (bibcode) REFERENCES public.papers(bibcode);


--
-- Name: papers_external_ids papers_external_ids_bibcode_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.papers_external_ids
    ADD CONSTRAINT papers_external_ids_bibcode_fkey FOREIGN KEY (bibcode) REFERENCES public.papers(bibcode);


--
-- Name: papers_fulltext papers_fulltext_bibcode_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.papers_fulltext
    ADD CONSTRAINT papers_fulltext_bibcode_fkey FOREIGN KEY (bibcode) REFERENCES public.papers(bibcode);


--
-- Name: section_embeddings section_embeddings_bibcode_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.section_embeddings
    ADD CONSTRAINT section_embeddings_bibcode_fkey FOREIGN KEY (bibcode) REFERENCES public.papers(bibcode);


--
-- Name: section_entities section_entities_bibcode_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.section_entities
    ADD CONSTRAINT section_entities_bibcode_fkey FOREIGN KEY (bibcode) REFERENCES public.papers(bibcode);


--
-- Name: section_entities section_entities_entity_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.section_entities
    ADD CONSTRAINT section_entities_entity_id_fkey FOREIGN KEY (entity_id) REFERENCES public.entities(id) ON DELETE CASCADE;


--
-- Name: section_entities section_entities_harvest_run_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.section_entities
    ADD CONSTRAINT section_entities_harvest_run_id_fkey FOREIGN KEY (harvest_run_id) REFERENCES public.harvest_runs(id);


--
-- Name: uat_relationships uat_relationships_child_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.uat_relationships
    ADD CONSTRAINT uat_relationships_child_id_fkey FOREIGN KEY (child_id) REFERENCES public.uat_concepts(concept_id);


--
-- Name: uat_relationships uat_relationships_parent_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.uat_relationships
    ADD CONSTRAINT uat_relationships_parent_id_fkey FOREIGN KEY (parent_id) REFERENCES public.uat_concepts(concept_id);


--
-- PostgreSQL database dump complete
--

\unrestrict lQ9BahwANnSvPg7JQojLPg2fvMXbWqLEu23BoesODgzFGtm8chXet7jrmG1SziY

