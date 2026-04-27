#!/usr/bin/env bash
# preflight_qdrant_security.sh — verify the running scix-qdrant container
# is not exposing its API to LAN or the public internet.
#
# Per docs/prd/qdrant_nas_migration.md MH-1, Qdrant must bind ONLY to
# 127.0.0.1. The default docker behaviour ("6333:6333" with no IP prefix)
# maps to 0.0.0.0 and silently exposes the API to anyone who can reach
# the host on port 6333 — bead s1a documents the live drift this script
# was created to detect.
#
# Exits 0 if every port binding has a non-empty HostIp that is not
# 0.0.0.0 (treats 127.0.0.1 / ::1 as the only acceptable values).
# Exits 1 otherwise. Prints the offending bindings to stderr.
#
# Usage:
#   scripts/preflight_qdrant_security.sh                # checks scix-qdrant
#   scripts/preflight_qdrant_security.sh other-name     # checks <other-name>
#
# Designed for use in:
#   - deploy/run.sh (or whatever brings the stack up) as a post-start gate
#   - CI smoke job before any test that talks to Qdrant
#   - cron health check (loud-fail mode)

set -euo pipefail

CONTAINER="${1:-scix-qdrant}"

if ! command -v docker >/dev/null 2>&1; then
    echo "preflight: docker not on PATH" >&2
    exit 2
fi

if ! docker inspect "$CONTAINER" >/dev/null 2>&1; then
    echo "preflight: container '$CONTAINER' not found (is it running?)" >&2
    exit 2
fi

# Pull the port-binding map. Format: "<container_port> <HostIp> <HostPort>"
# per binding entry. Empty HostIp means 0.0.0.0 (LAN-reachable).
BINDINGS=$(docker inspect "$CONTAINER" --format '
{{- range $port, $configs := .HostConfig.PortBindings -}}
    {{- range $configs -}}
        {{ $port }} {{ .HostIp }} {{ .HostPort }}{{ printf "\n" -}}
    {{- end -}}
{{- end -}}
')

if [[ -z "$BINDINGS" ]]; then
    echo "preflight: '$CONTAINER' has no port bindings — that's fine for an internal-only deployment, exiting OK" >&2
    exit 0
fi

violations=0
while IFS=' ' read -r container_port host_ip host_port; do
    [[ -z "$container_port" ]] && continue
    case "$host_ip" in
        127.0.0.1|::1)
            ;;
        ""|0.0.0.0|"::")
            echo "VIOLATION: $CONTAINER $container_port -> ${host_ip:-<empty>}:${host_port} is LAN-reachable; bind to 127.0.0.1 per PRD MH-1" >&2
            violations=$((violations + 1))
            ;;
        *)
            echo "VIOLATION: $CONTAINER $container_port -> $host_ip:$host_port has unexpected HostIp; only 127.0.0.1 / ::1 are acceptable" >&2
            violations=$((violations + 1))
            ;;
    esac
done <<< "$BINDINGS"

if [[ $violations -gt 0 ]]; then
    echo "preflight: $violations binding violation(s) detected on '$CONTAINER'" >&2
    exit 1
fi

echo "preflight: '$CONTAINER' port bindings are loopback-only ✓"
exit 0
