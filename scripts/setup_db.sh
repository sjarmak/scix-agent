#!/usr/bin/env bash
# Setup the scix PostgreSQL database with pgvector.
# Idempotent: safe to re-run.
#
# For peer auth (default on Ubuntu):
#   bash scripts/setup_db.sh
#
# For password auth or custom user:
#   SCIX_DB_USER=scix SCIX_DB_PASS=scix bash scripts/setup_db.sh

set -euo pipefail

DB_NAME="${SCIX_DB_NAME:-scix}"

echo "==> Setting up database: ${DB_NAME}"

# Create database if not exists
if ! psql -d postgres -tAc "SELECT 1 FROM pg_database WHERE datname='${DB_NAME}'" | grep -q 1; then
    psql -d postgres -c "CREATE DATABASE ${DB_NAME};"
    echo "    Created database: ${DB_NAME}"
else
    echo "    Database already exists: ${DB_NAME}"
fi

# Apply schema migration
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MIGRATION_FILE="${SCRIPT_DIR}/../migrations/001_initial_schema.sql"
if [ ! -f "${MIGRATION_FILE}" ]; then
    echo "ERROR: Migration file not found: ${MIGRATION_FILE}"
    exit 1
fi

echo "==> Applying migration: 001_initial_schema.sql"
psql -d "${DB_NAME}" -f "${MIGRATION_FILE}"

echo "==> Done. Database '${DB_NAME}' is ready."
echo "    Connect with: psql -d ${DB_NAME}"
