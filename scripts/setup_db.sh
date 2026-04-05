#!/usr/bin/env bash
# Setup the scix PostgreSQL database and apply all migrations.
# Idempotent: safe to re-run — tracks applied migrations in schema_migrations.
#
# For peer auth (default on Ubuntu):
#   bash scripts/setup_db.sh
#
# For password auth or custom user:
#   SCIX_DB_USER=scix SCIX_DB_PASS=scix bash scripts/setup_db.sh

set -euo pipefail

DB_NAME="${SCIX_DB_NAME:-scix}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MIGRATIONS_DIR="${SCRIPT_DIR}/../migrations"

# Build psql args
PSQL_ARGS=()
if [ -n "${SCIX_DB_USER:-}" ]; then
    PSQL_ARGS+=(-U "$SCIX_DB_USER")
fi
if [ -n "${SCIX_DB_HOST:-}" ]; then
    PSQL_ARGS+=(-h "$SCIX_DB_HOST")
fi
if [ -n "${SCIX_DB_PORT:-}" ]; then
    PSQL_ARGS+=(-p "$SCIX_DB_PORT")
fi

# Export password for psql if provided
if [ -n "${SCIX_DB_PASS:-}" ]; then
    export PGPASSWORD="$SCIX_DB_PASS"
fi

run_psql() {
    psql "${PSQL_ARGS[@]}" "$@"
}

echo "==> Setting up database: ${DB_NAME}"

# 1. Create database if not exists
if ! run_psql -d postgres -tAc "SELECT 1 FROM pg_database WHERE datname='${DB_NAME}'" | grep -q 1; then
    run_psql -d postgres -c "CREATE DATABASE ${DB_NAME};"
    echo "    Created database: ${DB_NAME}"
else
    echo "    Database already exists: ${DB_NAME}"
fi

# 2. Ensure pgvector extension
run_psql -d "${DB_NAME}" -c "CREATE EXTENSION IF NOT EXISTS vector;" 2>/dev/null || true

# 3. Ensure schema_migrations table exists (bootstrap — before applying any migration)
run_psql -d "${DB_NAME}" -c "
CREATE TABLE IF NOT EXISTS schema_migrations (
    version INT PRIMARY KEY,
    applied_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    filename TEXT NOT NULL
);"

# 4. Apply migrations in numeric order
APPLIED=0
SKIPPED=0

for migration_file in $(ls "${MIGRATIONS_DIR}"/*.sql | sort -t/ -k999 -V); do
    filename="$(basename "$migration_file")"
    # Extract version number from prefix (e.g., 001_initial_schema.sql -> 1)
    version=$(echo "$filename" | grep -oP '^\d+' | sed 's/^0*//' )
    if [ -z "$version" ]; then
        echo "    WARN: Could not parse version from ${filename}, skipping"
        continue
    fi

    # Check if already recorded in schema_migrations
    already_applied=$(run_psql -d "${DB_NAME}" -tAc "SELECT 1 FROM schema_migrations WHERE version = ${version};" 2>/dev/null || echo "")
    if [ "$already_applied" = "1" ]; then
        SKIPPED=$((SKIPPED + 1))
        continue
    fi

    # Apply migration — all migrations use IF NOT EXISTS so re-application is safe
    echo "    Applying: ${filename} (version ${version})"
    if run_psql -d "${DB_NAME}" -f "$migration_file" > /dev/null 2>&1; then
        # Record in schema_migrations
        run_psql -d "${DB_NAME}" -c "INSERT INTO schema_migrations (version, filename) VALUES (${version}, '${filename}') ON CONFLICT (version) DO NOTHING;"
        APPLIED=$((APPLIED + 1))
    else
        # Try again with output visible for debugging
        echo "    WARN: Error applying ${filename}, retrying with output..."
        if run_psql -d "${DB_NAME}" -f "$migration_file"; then
            run_psql -d "${DB_NAME}" -c "INSERT INTO schema_migrations (version, filename) VALUES (${version}, '${filename}') ON CONFLICT (version) DO NOTHING;"
            APPLIED=$((APPLIED + 1))
        else
            echo "    ERROR: Failed to apply ${filename}"
            exit 1
        fi
    fi
done

echo "==> Done. Applied ${APPLIED} migrations, skipped ${SKIPPED} (already applied)."
echo "    Database '${DB_NAME}' is ready."
echo "    Connect with: psql -d ${DB_NAME}"
