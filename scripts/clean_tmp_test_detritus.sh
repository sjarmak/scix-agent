#!/usr/bin/env bash
# clean_tmp_test_detritus.sh
#
# Periodic cleanup for /tmp scratch dirs left behind by gascity / beads
# integration tests. Without this, ~64k stub dirs accumulate from
# gc-sling-test, gc-test-binary, eb-*, gc-bd-* etc and bloat /tmp inode
# count + directory metadata.
#
# Why a custom script instead of /etc/tmpfiles.d:
#   systemd-tmpfiles 'r' / 'R' types do NOT honor the age field
#   (tmpfiles.d(5): "The age field only applies to lines starting with
#   d, D, e, v, q, Q, C, x and X.") so a tmpfiles.d policy cannot
#   express "delete /tmp/gc-sling-test-* once they reach 7 days old".
#
# Safety:
#   - Only acts on files owned by the invoking user (never touches other
#     users' /tmp/, never needs sudo).
#   - mtime > 7 days only — today's active runs are preserved.
#   - Pattern list is explicit and matches the gascity / beads test
#     suites' known scratch-dir naming.
#
# Bound to systemd timer scix-tmp-detritus.timer (weekly).

set -u

AGE_DAYS="${SCIX_TMP_DETRITUS_AGE_DAYS:-7}"
DRY_RUN="${SCIX_TMP_DETRITUS_DRY_RUN:-0}"

PATTERNS=(
  "gc-sling-test*"
  "gc-bd-*"
  "gc-supervisor-*"
  "gc-test-*"
  "gc-fake-worker-*"
  "gc-invalid-*"
  "gc-rename-*"
  "gc-testscript-*"
  "gc-integration-*"
  "gc-home-*"
  "eb-*"
)

# Build the find expression: -name P1 -o -name P2 -o ...
expr=()
for i in "${!PATTERNS[@]}"; do
  if [[ $i -gt 0 ]]; then
    expr+=("-o")
  fi
  expr+=("-name" "${PATTERNS[$i]}")
done

USER_NAME="$(id -un)"
COUNT_FILE="$(mktemp)"
trap 'rm -f "$COUNT_FILE"' EXIT

if [[ "$DRY_RUN" = "1" ]]; then
  count=0
  while IFS= read -r -d '' path; do
    count=$((count + 1))
    if [[ $count -le 5 ]]; then
      echo "DRY_RUN: would remove $path"
    fi
  done < <(find /tmp -maxdepth 1 -mindepth 1 \
    -mtime "+${AGE_DAYS}" \
    -user "$USER_NAME" \
    \( "${expr[@]}" \) \
    -print0 2>/dev/null)
  echo "scix-tmp-detritus: ${count} entries match (>${AGE_DAYS}d, user=${USER_NAME}); DRY_RUN=1"
  exit 0
fi

# Stream find -print0 directly into xargs rm -rf; no intermediate string.
find /tmp -maxdepth 1 -mindepth 1 \
  -mtime "+${AGE_DAYS}" \
  -user "$USER_NAME" \
  \( "${expr[@]}" \) \
  -print0 2>/dev/null \
  | tee >(tr -d -c '\0' | wc -c > "$COUNT_FILE") \
  | xargs -0 -r rm -rf

count=$(cat "$COUNT_FILE")
echo "scix-tmp-detritus: removed ${count} entries (>${AGE_DAYS}d, user=${USER_NAME})"
