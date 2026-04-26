#!/usr/bin/env bash
# prd_queue_lifecycle.sh — worktree lifecycle for /prd-queue
#
# Subcommands:
#   setup PRD_PATH      Create worktree + integration branch + symlinks.
#                       Prints: <slug>\t<worktree_path>\t<branch_name>
#   teardown SLUG       FF integration branch into main, remove worktree, delete branch.
#                       Refuses if branch diverged from main (keeps everything for manual merge).
#   keep SLUG           Print worktree path; leave everything intact for inspection.
#   status SLUG         Print one-line status: <slug> <wt_exists> <branch_exists> <branch_position>
#
# All paths are absolute. Idempotent where possible. Never destructive without confirmation.

set -euo pipefail

PRIMARY_REPO="${SCIX_PRIMARY_REPO:-/home/ds/projects/scix_experiments}"
WT_BASE_DIR="$(dirname "$PRIMARY_REPO")"
WT_PREFIX="scix-wt"

# Symlinks each worktree gets (relative paths inside primary repo)
SYMLINKS=(.venv models ads_metadata_by_year_picard)

slugify() {
    basename "$1" .md | sed -E 's/^prd_//' | tr '_' '-'
}

worktree_path() {
    printf '%s/%s-%s\n' "$WT_BASE_DIR" "$WT_PREFIX" "$1"
}

branch_name() {
    printf 'prd-build/%s\n' "$1"
}

cmd_setup() {
    local prd="${1:?usage: setup PRD_PATH}"
    local slug wt branch
    slug="$(slugify "$prd")"
    wt="$(worktree_path "$slug")"
    branch="$(branch_name "$slug")"

    # Validate PRD exists relative to primary repo
    if [[ ! -f "$PRIMARY_REPO/$prd" && ! -f "$prd" ]]; then
        echo "ERROR: PRD not found: $prd (looked in cwd and $PRIMARY_REPO)" >&2
        return 2
    fi

    # Idempotent worktree creation
    if [[ -d "$wt/.git" || -f "$wt/.git" ]]; then
        echo "INFO: worktree already at $wt (reusing)" >&2
    else
        # If branch exists, reuse it; else create from main
        if git -C "$PRIMARY_REPO" rev-parse --verify "$branch" >/dev/null 2>&1; then
            git -C "$PRIMARY_REPO" worktree add "$wt" "$branch" >&2
        else
            git -C "$PRIMARY_REPO" worktree add -b "$branch" "$wt" main >&2
        fi
    fi

    # Idempotent symlinks
    for link in "${SYMLINKS[@]}"; do
        if [[ ! -e "$wt/$link" && -e "$PRIMARY_REPO/$link" ]]; then
            ln -s "$PRIMARY_REPO/$link" "$wt/$link"
        fi
    done

    # Stage PRD into worktree. PRDs are gitignored in this repo (docs/prd/ in
    # .gitignore), so they don't propagate to fresh worktree checkouts. The
    # build worker runs `cd <worktree> && /prd-build <prd-path>` — without
    # this stage step, it sees a missing file. Copy (not symlink) so the
    # build sees a stable snapshot even if the primary copy is edited mid-run.
    # Skip when the user passed an absolute path (already accessible from anywhere).
    if [[ "$prd" != /* ]]; then
        local prd_src prd_dst
        if [[ -f "$PRIMARY_REPO/$prd" ]]; then
            prd_src="$PRIMARY_REPO/$prd"
        else
            prd_src="$(realpath "$prd")"
        fi
        prd_dst="$wt/$prd"

        mkdir -p "$(dirname "$prd_dst")"
        if [[ ! -f "$prd_dst" ]] || ! cmp -s "$prd_src" "$prd_dst"; then
            cp "$prd_src" "$prd_dst"
            echo "INFO: staged PRD into worktree: $prd_dst" >&2
        fi
    fi

    printf '%s\t%s\t%s\n' "$slug" "$wt" "$branch"
}

cmd_teardown() {
    local slug="${1:?usage: teardown SLUG}"
    local wt branch
    wt="$(worktree_path "$slug")"
    branch="$(branch_name "$slug")"

    # Sanity check: primary repo must be on main with a clean index. The /prd-build
    # worker is supposed to leave the primary alone, but in practice a sub-agent has
    # been observed to detach HEAD or stage conflicts there (e.g. from `git checkout
    # <sha>` run with bad cwd). Bail out here rather than failing mid-merge.
    local primary_ref
    primary_ref="$(git -C "$PRIMARY_REPO" symbolic-ref --short -q HEAD || echo "DETACHED")"
    if [[ "$primary_ref" != "main" ]]; then
        echo "ERROR: primary repo HEAD is '$primary_ref', expected 'main'." >&2
        echo "       A /prd-build sub-agent likely checked out something in the primary repo." >&2
        echo "       Recover with:  git -C $PRIMARY_REPO switch main  (resolve any conflicts first)" >&2
        return 4
    fi
    if [[ -n "$(git -C "$PRIMARY_REPO" diff --name-only --diff-filter=U 2>/dev/null)" ]]; then
        echo "ERROR: primary repo has unmerged paths in index — cannot teardown." >&2
        git -C "$PRIMARY_REPO" diff --name-only --diff-filter=U >&2
        return 4
    fi

    # FF integration branch into main IF it has commits not on main
    if git -C "$PRIMARY_REPO" rev-parse --verify "$branch" >/dev/null 2>&1; then
        local branch_sha main_sha
        branch_sha="$(git -C "$PRIMARY_REPO" rev-parse "$branch")"
        main_sha="$(git -C "$PRIMARY_REPO" rev-parse main)"

        if [[ "$branch_sha" == "$main_sha" ]]; then
            echo "INFO: $branch == main; nothing to merge" >&2
        elif git -C "$PRIMARY_REPO" merge-base --is-ancestor "$branch" main; then
            echo "INFO: $branch already in main (commits FF'd by harness during build)" >&2
        elif git -C "$PRIMARY_REPO" merge-base --is-ancestor main "$branch"; then
            git -C "$PRIMARY_REPO" merge --ff-only "$branch" >&2
            echo "INFO: FF-merged $branch into main" >&2
        else
            echo "ERROR: $branch diverged from main; refusing teardown. Inspect $wt manually." >&2
            return 3
        fi
    fi

    # Remove worktree (--force handles dirty state since real work is committed)
    if [[ -d "$wt" ]]; then
        git -C "$PRIMARY_REPO" worktree remove --force "$wt" 2>/dev/null || rm -rf "$wt"
        echo "INFO: removed worktree $wt" >&2
    fi

    # Delete branch (if still exists after FF — it might be the current branch in another worktree)
    if git -C "$PRIMARY_REPO" rev-parse --verify "$branch" >/dev/null 2>&1; then
        git -C "$PRIMARY_REPO" branch -D "$branch" 2>/dev/null && echo "INFO: deleted branch $branch" >&2 || \
            echo "WARN: could not delete $branch (may still be in use)" >&2
    fi
}

cmd_keep() {
    local slug="${1:?usage: keep SLUG}"
    local wt
    wt="$(worktree_path "$slug")"
    echo "INFO: keeping $wt for inspection (worktree + branch left intact)" >&2
    echo "$wt"
}

cmd_status() {
    local slug="${1:?usage: status SLUG}"
    local wt branch wt_exists branch_pos
    wt="$(worktree_path "$slug")"
    branch="$(branch_name "$slug")"

    [[ -d "$wt" ]] && wt_exists="yes" || wt_exists="no"

    if git -C "$PRIMARY_REPO" rev-parse --verify "$branch" >/dev/null 2>&1; then
        branch_pos="$(git -C "$PRIMARY_REPO" rev-parse --short "$branch")"
        if git -C "$PRIMARY_REPO" merge-base --is-ancestor "$branch" main 2>/dev/null; then
            branch_pos="${branch_pos} (in-main)"
        elif git -C "$PRIMARY_REPO" merge-base --is-ancestor main "$branch" 2>/dev/null; then
            local ahead
            ahead="$(git -C "$PRIMARY_REPO" rev-list --count "main..$branch")"
            branch_pos="${branch_pos} (${ahead}-ahead)"
        else
            branch_pos="${branch_pos} (diverged)"
        fi
    else
        branch_pos="(no branch)"
    fi

    printf '%s\twt=%s\tbranch=%s\n' "$slug" "$wt_exists" "$branch_pos"
}

case "${1:-}" in
    setup) shift; cmd_setup "$@" ;;
    teardown) shift; cmd_teardown "$@" ;;
    keep) shift; cmd_keep "$@" ;;
    status) shift; cmd_status "$@" ;;
    *)
        cat >&2 <<'USAGE'
Usage: prd_queue_lifecycle.sh <subcommand> [args]

  setup PRD_PATH       Create worktree + branch + symlinks.
                       Prints: <slug>\t<worktree_path>\t<branch_name> on stdout.
  teardown SLUG        FF integration branch into main, remove worktree, delete branch.
                       Refuses if diverged.
  keep SLUG            Print worktree path; leave everything intact for inspection.
  status SLUG          One-line status: existence + branch position vs main.

Env:
  SCIX_PRIMARY_REPO    Override primary repo path (default: /home/ds/projects/scix_experiments)
USAGE
        exit 1
        ;;
esac
