#!/usr/bin/env bash
# Sweep local branches that are fully merged into main.
#
# Uses `git branch --merged main` so a branch is only deleted if every
# commit on it is reachable from main — the same safety check `git branch -d`
# applies. Archive-then-delete is unnecessary here: if it's merged, the
# commits stay reachable through main forever.
#
# Run as `bash scripts/sweep_merged_branches.sh` for a dry preview, then
# `bash scripts/sweep_merged_branches.sh --apply` to actually delete.
#
# Companion to refs/archive/ workflow used for unmerged-but-stale branches:
# this script does NOT touch unmerged branches. See bead for `bd_close_with_
# branch_archive.sh` (per-bead-close wrapper) and the worktree expiry
# script for the unmerged path.
set -eu

mode="dry-run"
if [ "${1:-}" = "--apply" ]; then mode="apply"; fi

# Excludes: main, anything currently checked out (in any worktree), the
# refs/archive/ namespace.
checked_out=$(git worktree list --porcelain | awk '/^branch / {sub(/^refs\/heads\//, "", $2); print $2}')
exclude_pattern="^(main$"
for br in $checked_out; do exclude_pattern="${exclude_pattern}|${br}$"; done
exclude_pattern="${exclude_pattern})"

merged=$(git branch --merged main --format='%(refname:short)' | grep -vE "$exclude_pattern" || true)

if [ -z "$merged" ]; then
  echo "No merged branches to sweep."
  exit 0
fi

echo "Merged branches eligible for deletion:"
echo "$merged" | sed 's/^/  /'
echo

if [ "$mode" = "dry-run" ]; then
  echo "(dry-run) re-run with --apply to delete."
  exit 0
fi

count=0
echo "$merged" | while read -r br; do
  [ -z "$br" ] && continue
  git branch -d "$br" >/dev/null
  count=$((count + 1))
done
echo "Deleted $(echo "$merged" | grep -c .) merged branches."
