#!/usr/bin/env bash
# check_batch_context.sh — READ-ONLY diagnostic for the scix-batch discipline.
# Prints which guard markers are visible in this environment, the current
# cgroup, the host's oomd kill settings, and any live mem-batch scopes.
# Changes nothing; safe to run anywhere, wrapped or unwrapped.
#
# Usage:
#   bash .claude/skills/scix-memory-and-batch-discipline/scripts/check_batch_context.sh
#   scix-batch bash .claude/skills/scix-memory-and-batch-discipline/scripts/check_batch_context.sh
set -u

echo "== Guard markers (what the scripts/ self-enforcement checks see) =="
if [[ -n "${INVOCATION_ID:-}" ]]; then
    echo "INVOCATION_ID : SET (${INVOCATION_ID}) -> --allow-prod scope checks PASS"
else
    echo "INVOCATION_ID : unset -> --allow-prod scope checks REFUSE (wrap in scix-batch)"
fi
if [[ -n "${SYSTEMD_SCOPE:-}" ]]; then
    echo "SYSTEMD_SCOPE : SET (${SYSTEMD_SCOPE}) -> --require-batch-scope PASSES"
else
    echo "SYSTEMD_SCOPE : unset -> --require-batch-scope REFUSES (prepend SYSTEMD_SCOPE=1; it is never auto-set)"
fi

echo
echo "== Current cgroup (am I inside a mem-batch scope?) =="
cat /proc/self/cgroup
case "$(cat /proc/self/cgroup)" in
    *mem-batch-*) echo "-> inside a mem-batch/scix-batch scope" ;;
    *) echo "-> NOT inside a mem-batch scope (heavy work here risks oomd collateral kills)" ;;
esac

echo
echo "== Host oomd policy on the shared user slice =="
systemctl show user@"$(id -u)".service \
    -p ManagedOOMMemoryPressure,ManagedOOMMemoryPressureLimit 2>/dev/null \
    || echo "(systemctl show unavailable)"
echo "(ManagedOOMMemoryPressureLimit=2147483648 raw == 50% of 2^32)"

echo
echo "== Wrapper on PATH =="
if command -v scix-batch >/dev/null 2>&1; then
    ls -la "$(command -v scix-batch)"
else
    echo "scix-batch NOT on PATH — expected at ~/.local/bin/scix-batch -> mem-batch"
fi

echo
echo "== Live mem-batch scopes =="
systemctl --user list-units 'mem-batch-*' --no-pager 2>/dev/null || echo "(no user manager reachable)"

echo
echo "== Memory headroom =="
free -g
