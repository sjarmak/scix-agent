# /tmp test-detritus cleanup

## Problem

Gascity sling tests, beads ergonomic-builder tests (`eb-*`), and various
gc/bd integration tests leave behind small scratch directories under `/tmp/`
that are never cleaned up. Over weeks, ~64,000 of these accumulate
(`gc-sling-test-formulas-*`, `gc-sling-test-city-*`, `gc-test-binary-*`,
`gc-bd-*`, `eb-*` …), bloating `/tmp` inode count and directory metadata.

`scix-worker-4` discovered this during epic `yjda` Phase 2; bead
`scix_experiments-zas8` covered the cleanup + prophylactic policy.

## One-time cleanup (2026-04-29)

Bead `scix_experiments-zas8` removed 65,083 stale entries (>7d) matching
the gc/bd/eb test patterns:

```bash
find /tmp -maxdepth 1 -mindepth 1 -mtime +7 \
  \( -name "gc-sling-test*" -o -name "gc-bd-*" -o -name "gc-supervisor-*" \
     -o -name "gc-invalid-*" -o -name "gc-rename-*" -o -name "gc-fake-worker-*" \
     -o -name "gc-testscript-*" -o -name "gc-integration-*" -o -name "gc-home-*" \
     -o -name "gc-test-*" -o -name "eb-*" \) \
  -depth -print0 \
  | xargs -0 -r rm -rf
```

Bytes freed: 877 MB. The bead author estimated 90 GB but that estimate
appears to have come from filesystem-allocated overhead (296k directory
entries in `/tmp` itself); the actual content of those stub dirs was tiny
(most were empty 6-byte dirs). The bulk of `/tmp`'s 96 GB sits in `<7d`
active dirs (`gc-test-binary-*` ~24 GB, `eval-worktrees` 5 GB,
`go-build*` ~10 GB, `livedocs/sourcegraph/oss_scan` ~5 GB) which are
correctly preserved.

## Recurring policy: weekly user systemd timer

The bead asked for a `tmpfiles.d` aging policy at
`/etc/tmpfiles.d/scix-test-detritus.conf`. That doesn't fit the use case:
**`tmpfiles.d`'s `r` and `R` types do not honor the age field**
(see `tmpfiles.d(5)`: *"The age field only applies to lines starting with
d, D, e, v, q, Q, C, x and X."*). The example `d /tmp/gc-sling-test - - - 7d`
would clean *contents* of `/tmp/gc-sling-test/` — but actual detritus lives
as siblings (`/tmp/gc-sling-test-formulas-*`), not children. systemd-tmpfiles
simply can't express "delete top-level entries matching glob X older than N days".

Instead, a user-level systemd timer + service runs the same `find … -mtime +7
| xargs rm -rf` weekly. No sudo required. Owned by user `ds`, restricted to
files owned by the invoking user (so it can never touch other users' files
in /tmp).

### Files

| Path | Purpose |
|---|---|
| `scripts/clean_tmp_test_detritus.sh` | Cleanup script (in repo) |
| `~/.local/share/systemd/user/scix-tmp-detritus.service` | Oneshot service |
| `~/.local/share/systemd/user/scix-tmp-detritus.timer` | Weekly timer (Sun 04:30) |

### Operating

```bash
# Status
systemctl --user status scix-tmp-detritus.timer
systemctl --user list-timers scix-tmp-detritus.timer

# Run now (e.g. after a stress test that left a lot of detritus)
systemctl --user start scix-tmp-detritus.service
journalctl --user -u scix-tmp-detritus.service -n 20

# Dry-run (no deletion)
SCIX_TMP_DETRITUS_DRY_RUN=1 ./scripts/clean_tmp_test_detritus.sh

# Custom age threshold
SCIX_TMP_DETRITUS_AGE_DAYS=14 ./scripts/clean_tmp_test_detritus.sh
```

### Verifying the policy works

End-to-end test (as run during install):

```bash
# Set up a stale canary + an active canary
mkdir -p /tmp/gc-sling-test-canary-stale /tmp/gc-sling-test-canary-active
touch -d "10 days ago" /tmp/gc-sling-test-canary-stale

# Trigger the service
systemctl --user start scix-tmp-detritus.service

# Stale canary should be GONE; active canary should remain
ls -ld /tmp/gc-sling-test-canary-stale 2>&1   # No such file
ls -ld /tmp/gc-sling-test-canary-active 2>&1  # Still there

# Cleanup
rm -rf /tmp/gc-sling-test-canary-active
```

### Adding new patterns

Edit `scripts/clean_tmp_test_detritus.sh`, add to the `PATTERNS=( … )` array,
then dry-run before deploying:

```bash
SCIX_TMP_DETRITUS_DRY_RUN=1 ./scripts/clean_tmp_test_detritus.sh
```

Patterns are passed to `find -name`, so shell globs (`*`, `?`) work.
Each entry must match a *top-level* `/tmp/<pattern>` (no nested subpaths).

## Long-term: have tests clean up after themselves

This script is a backstop. The right fix is for gascity / beads test
suites to register cleanup hooks that remove their scratch dirs when a
test finishes (success OR failure). Until then, this timer keeps `/tmp`
bounded.
