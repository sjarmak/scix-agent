#!/usr/bin/env bash
# Setup the scix PostgreSQL database and apply the schema.
# Idempotent: all CREATE statements use IF NOT EXISTS.
#
# For peer auth (default on Ubuntu):
#   bash scripts/setup_db.sh
#
# For password auth or custom user:
#   SCIX_DB_USER=scix SCIX_DB_PASS=scix bash scripts/setup_db.sh

set -euo pipefail

DB_NAME="${SCIX_DB_NAME:-scix}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SCHEMA_FILE="${SCRIPT_DIR}/../schema.sql"

if [ ! -f "$SCHEMA_FILE" ]; then
    echo "ERROR: schema.sql not found at ${SCHEMA_FILE}"
    exit 1
fi

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

# 2. Apply schema (all statements are IF NOT EXISTS)
echo "    Applying schema.sql..."
run_psql -d "${DB_NAME}" -f "$SCHEMA_FILE" > /dev/null 2>&1

echo "==> Done. Database '${DB_NAME}' is ready."
echo "    Connect with: psql -d ${DB_NAME}"
