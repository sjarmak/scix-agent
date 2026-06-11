#!/usr/bin/env bash
# bd_close_with_branch_archive.sh — close a bead and archive its branch(es).
#
# Background
# ----------
# `bd close <id>` closes the bead but leaves any `bd/<id>-*` branch alive on
# disk. Over time the local repo accumulates stale branches whose beads are
# already closed (the 2026-05-02 cleanup found 34 such branches, 5 of which
# carried conflicting migration 063 numbers — see bead scix_experiments-ku4r).
#
# bd does not expose an after-close hook (its hooks are git-side only:
# pre-commit, post-merge, pre-push, post-checkout, prepare-commit-msg), so
# branch hygiene has to be done by a wrapper around `bd close`. This is that
# wrapper. Companion to scripts/sweep_merged_branches.sh, which handles the
# merged-branch case.
#
# Behaviour
# ---------
# 1. Find local branches that match the bead's branch-naming conventions:
#    - dotted form: `bd/<id>-*` (e.g. `bd/dbl.6-vocab-boost-labels`)
#    - undotted form: `bd/<id-without-dots>-*` (e.g. `bd/dbl6-vocab-boost-labels`)
# 2. Skip any branch currently checked out in any worktree (would orphan the
#    worktree). Print a warning and continue with the others.
# 3. For each remaining match, archive to `refs/archive/<branch>` (idempotent
#    — refuses to overwrite an existing archive ref) and then `git branch -D`.
#    Recovery: `git branch <name> refs/archive/<name>`.
# 4. Run `bd close <bead-id> [extra args]` and forward its exit code.
# 5. If no matching branch exists, the script is effectively `bd close <id>`.
#
# Usage
# -----
#   bash scripts/bd_close_with_branch_archive.sh <bead-id> [bd close flags...]
#
# Examples:
#   bash scripts/bd_close_with_branch_archive.sh scix_experiments-dbl.6
#   bash scripts/bd_close_with_branch_archive.sh scix_experiments-7m98 --reason "done in commit e182094"
#
# Exit codes
# ----------
#   0  — bd close succeeded (regardless of how many branches were archived)
#   1  — usage error or bd close itself failed
#   2  — internal git error while archiving (bd close is NOT attempted)

set -eu

if [ $# -lt 1 ]; then
  echo "Usage: $0 <bead-id> [bd close args...]" >&2
  echo "  e.g.  $0 scix_experiments-dbl.6 --reason 'merged'" >&2
  exit 1
fi

bead_id="$1"
shift  # remaining args are forwarded to `bd close`

# --- Derive branch name patterns ----------------------------------------------
#
# Bead IDs in this repo come in two shapes that have shown up on real branches:
#   * `scix_experiments-dbl.6`   → `bd/dbl.6-*`  AND  `bd/dbl6-*`
#   * `scix_experiments-7m98`    → `bd/7m98-*`   (no dots, only one shape)
#
# We strip the project prefix (`scix_experiments-`) for matching since branches
# don't carry it.

# Strip the project prefix if present.
short_id="${bead_id#scix_experiments-}"

# Build the two short-id forms. If short_id has no dots, both forms are equal.
dotted_id="$short_id"
undotted_id="${short_id//./}"

# Collect candidate branches via `git for-each-ref` (faster + safer than
# parsing `git branch`). Use exact-prefix pattern matching.
candidates=()
while IFS= read -r ref; do
  [ -z "$ref" ] && continue
  candidates+=("$ref")
done < <(git for-each-ref \
  --format='%(refname:short)' \
  "refs/heads/bd/${dotted_id}-*" \
  "refs/heads/bd/${undotted_id}-*")

# Dedupe (the dotted/undotted patterns can both match the same branch when the
# short id has no dots).
if [ ${#candidates[@]} -gt 0 ]; then
  mapfile -t candidates < <(printf '%s\n' "${candidates[@]}" | sort -u)
fi

# --- Identify branches that are currently checked out in any worktree --------
#
# `git worktree list --porcelain` emits `branch refs/heads/<name>` lines for
# every worktree that has a branch checked out. Detached worktrees emit no
# such line, so they don't pin anything.

declare -A checked_out_set=()
while IFS= read -r line; do
  case "$line" in
    "branch refs/heads/"*)
      br="${line#branch refs/heads/}"
      checked_out_set["$br"]=1
      ;;
  esac
done < <(git worktree list --porcelain)

# --- Archive + delete ---------------------------------------------------------

archived=0
skipped_checked_out=0
errored=0

for br in "${candidates[@]:-}"; do
  [ -z "$br" ] && continue

  if [ -n "${checked_out_set[$br]:-}" ]; then
    echo "skip: '$br' is checked out in a worktree (would orphan it)" >&2
    skipped_checked_out=$((skipped_checked_out + 1))
    continue
  fi

  archive_ref="refs/archive/$br"

  # Idempotency: never overwrite an existing archive entry.
  if git show-ref --verify --quiet "refs/archive/$br"; then
    echo "skip: archive ref '$archive_ref' already exists; manual intervention needed" >&2
    errored=$((errored + 1))
    continue
  fi

  # Resolve the branch tip; bail if anything looks off.
  tip=$(git rev-parse --verify "refs/heads/$br" 2>/dev/null) || {
    echo "error: cannot resolve refs/heads/$br" >&2
    errored=$((errored + 1))
    continue
  }

  # Create the archive ref then delete the branch. If branch -D fails the
  # archive is still in place, so re-running is safe (will skip on the
  # 'already exists' check above and we won't double-archive).
  if ! git update-ref "$archive_ref" "$tip"; then
    echo "error: failed to create $archive_ref" >&2
    errored=$((errored + 1))
    continue
  fi

  if ! git branch -D "$br" >/dev/null; then
    echo "error: archived to $archive_ref but failed to delete branch '$br'" >&2
    errored=$((errored + 1))
    continue
  fi

  echo "archived: $br → $archive_ref"
  archived=$((archived + 1))
done

if [ "$errored" -gt 0 ]; then
  echo "error: $errored branch(es) had problems; not running 'bd close'" >&2
  exit 2
fi

# --- Forward to bd close ------------------------------------------------------

echo "running: bd close $bead_id $*"
bd close "$bead_id" "$@"
bd_rc=$?

# Summary line for operators / log scrapers.
echo "summary: bead=$bead_id archived=$archived skipped_checked_out=$skipped_checked_out bd_close_rc=$bd_rc"

exit "$bd_rc"
