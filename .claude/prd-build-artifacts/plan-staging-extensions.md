# Plan: Staging Extensions for Entity Graph

## Migration 022_staging_entities.sql

### 1. Public Tables (created first, needed for FK references and promote targets)

**public.entities**

- id SERIAL PK, canonical_name TEXT NOT NULL, entity_type TEXT NOT NULL, discipline TEXT, source TEXT NOT NULL
- properties JSONB DEFAULT '{}'
- created_at TIMESTAMPTZ DEFAULT NOW(), updated_at TIMESTAMPTZ DEFAULT NOW()
- UNIQUE(canonical_name, entity_type, source)
- Indexes on entity_type, source

**public.entity_identifiers**

- entity_id INT NOT NULL REFERENCES public.entities(id)
- id_scheme TEXT NOT NULL, external_id TEXT NOT NULL, is_primary BOOLEAN DEFAULT false
- PK(id_scheme, external_id)

**public.entity_aliases**

- entity_id INT NOT NULL REFERENCES public.entities(id)
- alias TEXT NOT NULL, alias_source TEXT
- PK(entity_id, alias)

### 2. Staging Tables (no FK enforcement)

Mirror public structure but in staging schema, no FK constraints.

### 3. staging.promote_entities() Function

1. Upsert entities: INSERT INTO public.entities SELECT FROM staging.entities ON CONFLICT DO UPDATE
2. Upsert identifiers: JOIN staging.entity_identifiers with staging.entities on entity_id, then JOIN public.entities on natural key to get correct public entity_id
3. Upsert aliases: Same join pattern as identifiers, ON CONFLICT DO NOTHING
4. TRUNCATE all 3 staging tables
5. Return count of promoted entities

### 4. Tests

Static tests verifying SQL structure: table definitions, function definition, ON CONFLICT, TRUNCATE.
