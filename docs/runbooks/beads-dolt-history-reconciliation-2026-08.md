# Beads Dolt history reconciliation, August 2026

## Recovery decision

The live city database was the authority for current issue values because it
contained August work, including `scix_experiments-d0j`. The Git-backed Dolt
remote stopped on May 29, 2026 and did not contain that issue. The histories
had no common ancestor, so a normal pull could not merge them.

The recovery keeps both lineages without force-pushing or importing the passive
JSONL export:

- `pre-reconcile-live-20260806` preserves the original live tip
  `n3vokehunu7rl5727isd6qdqlc24cpr6`.
- `remotes/origin/main` preserves the remote tip
  `qqgfqera44u0up85hoqq6a9ss3nktfru`.
- `reconcile-union-20260806` is rooted at the remote tip and adds a union commit
  `lmkhnpfr04a1ue434ackao10dr7qp75n`.
- Local `main` adds deterministic dependency re-keying at
  `qjlf20o20gptaq3kibf2n0c9nkrl696d`.

Because Dolt does not merge unrelated roots, the old live commit chain remains
on its named preservation branch instead of becoming a parent of the union
commit. Its current issue state was migrated into the union.

## Union rules

The two sources contained 170 live issues and 240 remote issues, with five IDs
in common. The union contains 405 issues:

- all 165 live-only issues were retained;
- all 235 remote-only issues were restored;
- the live row won for the five overlapping issue IDs;
- comments, events, labels, and dependencies were unioned by their logical
  identities;
- old dependency rows were converted to the current typed-target schema;
- live schema, indexes, checks, views, and foreign-key rules were retained; and
- clone-local tables remained excluded from Dolt history.

The preserved `scix_experiments-d0j` notes have SHA-256
`f1eae07c0ed3c8a3ce89e52cf7628011fd089d9b7ef9a3717c32f013dbbce5f7`,
matching the pre-reconciliation live branch.

## Verification evidence

Before advancing local `main`, the union showed:

- zero missing live issue IDs;
- zero missing remote issue IDs;
- zero missing events, labels, or dependencies from either source;
- zero Dolt constraint violations; and
- five computed blocked issues matching five persisted `is_blocked` flags.

After advancing local `main`, `bd status` reported 405 issues, `bd doctor`
reported 79 passing checks and no database warnings, the server-mode health
check passed all six checks, and repeated `bd dolt pull` operations completed.
`bd doctor --fix` re-keyed 253 legacy dependency rows to the clone-stable key
format required for cross-clone pulls.

## Publication and rollback boundary

The local reconciliation is recoverable until publication by resetting `main`
to `pre-reconcile-live-20260806`. Do not delete that branch or either preserved
Dolt directory until a fresh clone has verified the published union.

Publishing requires an explicitly approved `bd dolt push`. After that push,
verify a fresh clone contains 405 issues, the `scix_experiments-d0j` notes hash
above, and a clean pull/push cycle. Do not use `bd import`, JSONL replacement,
or a force push for this recovery.
