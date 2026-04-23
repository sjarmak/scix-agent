#!/usr/bin/env bash
# scripts/viz/run.sh — one-command runner for the SciX viz demo.
#
# Builds the Sankey and UMAP JSON payloads (with --synthetic when the DB
# is unreachable or the operator asks for it) and then launches the
# FastAPI viz server via uvicorn on localhost.
#
# Usage:
#   ./scripts/viz/run.sh                   # build any missing data + serve
#   ./scripts/viz/run.sh --synthetic       # force synthetic rebuild + serve
#   ./scripts/viz/run.sh --build-only --synthetic
#   ./scripts/viz/run.sh --no-build --port 9000
#
# See docs/viz/DEMO.md for the talk narrative and prepared scenarios.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

VENV_PY="${VENV_PY:-.venv/bin/python}"

HOST="127.0.0.1"
PORT="8765"
NO_BUILD=0
BUILD_ONLY=0
SYNTHETIC=0

print_usage() {
    cat <<'USAGE'
Usage: scripts/viz/run.sh [--host HOST] [--port PORT] [--no-build]
                          [--build-only] [--synthetic] [-h|--help]

Builds data/viz/sankey.json and data/viz/umap.json (if missing, or if
--synthetic is set) and launches the SciX viz FastAPI app under uvicorn.

Options:
  --host HOST     Bind address (default: 127.0.0.1).
  --port PORT     Listen port (default: 8765).
  --no-build      Skip regenerating data/viz/*.json even if files are missing.
  --build-only    Regenerate data files then exit without starting the server.
  --synthetic     Force synthetic (no-DB) rebuild of both data files.
                  Use this when Postgres is unreachable or for a quick demo.
  -h, --help      Print this help and exit.

Environment overrides:
  VENV_PY         Python interpreter to use (default: .venv/bin/python).

Examples:
  scripts/viz/run.sh --synthetic              # demo mode, no DB needed
  scripts/viz/run.sh --build-only --synthetic # refresh data files only
  scripts/viz/run.sh --no-build --port 9000   # reuse existing JSON, alt port
USAGE
}

log() {
    printf '[viz-demo] %s\n' "$*" >&2
}

die() {
    printf 'scripts/viz/run.sh: %s\n' "$*" >&2
    exit 2
}

while [ $# -gt 0 ]; do
    case "$1" in
        --help|-h)
            print_usage
            exit 0
            ;;
        --host)
            [ $# -ge 2 ] || die "--host requires a value"
            HOST="$2"
            shift 2
            ;;
        --host=*)
            HOST="${1#--host=}"
            shift
            ;;
        --port)
            [ $# -ge 2 ] || die "--port requires a value"
            PORT="$2"
            shift 2
            ;;
        --port=*)
            PORT="${1#--port=}"
            shift
            ;;
        --no-build)
            NO_BUILD=1
            shift
            ;;
        --build-only)
            BUILD_ONLY=1
            shift
            ;;
        --synthetic)
            SYNTHETIC=1
            shift
            ;;
        --)
            shift
            break
            ;;
        *)
            print_usage >&2
            die "unknown argument: $1"
            ;;
    esac
done

SANKEY_JSON="data/viz/sankey.json"
UMAP_JSON="data/viz/umap.json"

build_sankey() {
    local args=()
    if [ "$SYNTHETIC" -eq 1 ]; then
        args+=(--synthetic)
    fi
    log "building $SANKEY_JSON${args[*]:+ (${args[*]})}"
    "$VENV_PY" scripts/viz/build_temporal_sankey_data.py "${args[@]}"
}

build_umap() {
    local args=()
    if [ "$SYNTHETIC" -eq 1 ]; then
        # 2000 synthetic points is enough for a lively UMAP scatter while
        # keeping the build under ~30s even on modest CPUs.
        args+=(--synthetic 2000)
    fi
    log "building $UMAP_JSON${args[*]:+ (${args[*]})}"
    "$VENV_PY" scripts/viz/project_embeddings_umap.py "${args[@]}"
}

if [ "$NO_BUILD" -ne 1 ]; then
    if [ "$SYNTHETIC" -eq 1 ] || [ ! -f "$SANKEY_JSON" ]; then
        build_sankey
    else
        log "reusing existing $SANKEY_JSON"
    fi

    if [ "$SYNTHETIC" -eq 1 ] || [ ! -f "$UMAP_JSON" ]; then
        build_umap
    else
        log "reusing existing $UMAP_JSON"
    fi
else
    log "--no-build set: skipping data regeneration"
fi

if [ "$BUILD_ONLY" -eq 1 ]; then
    log "--build-only set: done (skipping server)"
    exit 0
fi

log "starting uvicorn scix.viz.server:app on http://$HOST:$PORT"
log "open http://$HOST:$PORT/viz/  (sankey.html, umap_browser.html, agent_trace.html)"
exec "$VENV_PY" -m uvicorn scix.viz.server:app --host "$HOST" --port "$PORT"
