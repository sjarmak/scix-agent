#!/usr/bin/env bash
# Flag (and optionally prune) stale git worktrees.
#
# Worktrees accumulate without an expiry signal. A stale, dirty, unmerged
# worktree hides duplicate or abandoned work until someone audits by hand —
# e.g. the scix_experiments-viz / viz-ux-fixes worktree found in the 2026-05-02
# audit, 53 commits ahead of main with a dirty draft of work already shipped
# elsewhere. This script makes that condition visible.
#
# For every linked worktree (the primary worktree is never touched) it reports:
#   worktree path | branch | last-commit-age | dirty-files | merged-status
# and a suggested action:
#   review-and-commit  — stale AND dirty AND unmerged (work may be unsaved)
#   discard-and-remove — clean AND merged AND stale (safe to prune)
#   keep               — anything else
#
# Run as `bash scripts/worktree_expiry_check.sh` for a dry-run report.
# `--prune-clean-stale` removes the discard-and-remove worktrees (the only
# destructive mode; clean+merged+stale only, so no commits are ever lost).
# `--days N` overrides the 30-day staleness threshold.
#
# Companion to `scripts/sweep_merged_branches.sh` (the merged-branch path).
# `git worktree prune` removes admin files for already-deleted directories;
# this script complements it for actively-tracked but stale worktrees.
set -eu

threshold_days=30
prune=false

usage() {
  echo "usage: $0 [--days N] [--prune-clean-stale]" >&2
  exit "${1:-0}"
}

while [ $# -gt 0 ]; do
  case "$1" in
    --days)
      [ $# -ge 2 ] || { echo "error: --days needs a value" >&2; usage 1; }
      threshold_days="$2"
      shift 2
      ;;
    --days=*) threshold_days="${1#*=}"; shift ;;
    --prune-clean-stale) prune=true; shift ;;
    -h|--help) usage 0 ;;
    *) echo "error: unknown argument '$1'" >&2; usage 1 ;;
  esac
done

case "$threshold_days" in
  ''|*[!0-9]*) echo "error: --days must be a non-negative integer" >&2; exit 1 ;;
esac

if ! main_ref=$(git rev-parse --verify --quiet refs/heads/main); then
  echo "error: no local 'main' branch — cannot compute merged-status" >&2
  exit 1
fi

now=$(date +%s)
threshold_secs=$((threshold_days * 86400))

# Parse `git worktree list --porcelain` into newline records of
# "path<TAB>branch<TAB>head-sha". The first record is the primary worktree
# and is dropped — expiry never applies to the main checkout. A detached
# worktree has no branch; awk emits the literal "-" so the field is never
# empty (TAB is IFS-whitespace, so an empty field would collapse adjacent
# tabs and misalign the columns when the loop below reads it).
records=$(git worktree list --porcelain | awk '
  /^worktree / { path = substr($0, 10); branch = "-"; head = "" }
  /^HEAD /     { head = $2 }
  /^branch /   { sub(/^refs\/heads\//, "", $2); branch = $2 }
  /^$/         { if (path != "") print path "\t" branch "\t" head; path = "" }
  END          { if (path != "") print path "\t" branch "\t" head }
' | tail -n +2)

if [ -z "$records" ]; then
  echo "No linked worktrees to check."
  exit 0
fi

printf '%-48s  %-32s  %-12s  %-11s  %-9s  %s\n' \
  "WORKTREE" "BRANCH" "LAST-COMMIT" "DIRTY-FILES" "MERGED" "SUGGESTED-ACTION"

prune_targets=""

while IFS=$(printf '\t') read -r path branch head; do
  [ -z "$path" ] && continue

  [ "$branch" = "-" ] && branch_label="(detached)" || branch_label="$branch"

  if commit_secs=$(git -C "$path" log -1 --format=%ct 2>/dev/null); then
    age_secs=$((now - commit_secs))
    age_days=$((age_secs / 86400))
    age_label="${age_days}d"
  else
    age_secs=0
    age_label="?"
  fi

  dirty_count=$(git -C "$path" status --short 2>/dev/null | grep -c . || true)

  if git merge-base --is-ancestor "$head" "$main_ref" 2>/dev/null; then
    merged_label="merged"
    merged=true
  else
    merged_label="unmerged"
    merged=false
  fi

  is_stale=false
  [ "$age_label" != "?" ] && [ "$age_secs" -gt "$threshold_secs" ] && is_stale=true
  is_dirty=false
  [ "$dirty_count" -gt 0 ] && is_dirty=true

  if $is_stale && $is_dirty && ! $merged; then
    action="review-and-commit"
  elif $is_stale && ! $is_dirty && $merged; then
    action="discard-and-remove"
    prune_targets="${prune_targets}${path}"$'\n'
  else
    action="keep"
  fi

  printf '%-48s  %-32s  %-12s  %-11s  %-9s  %s\n' \
    "$path" "$branch_label" "$age_label" "$dirty_count" "$merged_label" "$action"
done <<EOF
$records
EOF

echo
if ! $prune; then
  echo "(dry-run) re-run with --prune-clean-stale to remove discard-and-remove worktrees."
  exit 0
fi

if [ -z "$prune_targets" ]; then
  echo "No clean+merged+stale worktrees to prune."
  exit 0
fi

# The report loop reads from a here-doc (current shell, not a pipe), so the
# paths it classified discard-and-remove survive in $prune_targets — act on
# that list directly. `git worktree remove` (no --force) is the backstop: it
# refuses a worktree that became dirty or locked since the report.
removed=0
while IFS= read -r path; do
  [ -z "$path" ] && continue
  git worktree remove "$path"
  echo "removed $path"
  removed=$((removed + 1))
done <<EOF
$prune_targets
EOF

echo "Pruned $removed clean+merged+stale worktree(s)."
