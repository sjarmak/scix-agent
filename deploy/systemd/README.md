# systemd units (user-level)

Drafts for `~/.config/systemd/user/`. Not auto-installed — operator
copies the units, reloads, and enables.

## scix-citation-contexts-backfill (bead scix_experiments-6hr7)

Daily timer that runs one shard of the citation_contexts backfill plus
the post-shard chain (intent classification + `v_claim_edges` refresh).
Shard rotation is day-of-year mod 4, so all four shards are touched
every four days.

```bash
mkdir -p ~/.config/systemd/user
cp deploy/systemd/scix-citation-contexts-backfill.service ~/.config/systemd/user/
cp deploy/systemd/scix-citation-contexts-backfill.timer   ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now scix-citation-contexts-backfill.timer

# Verify the timer is armed:
systemctl --user list-timers scix-citation-contexts-backfill.timer

# Manual one-off run (any shard 0..3):
bash scripts/run_citation_contexts_shard.sh 0

# Watch the next firing land in the journal:
journalctl --user -u scix-citation-contexts-backfill.service -f
```

### Pre-flight checklist before enabling

1. `~/.local/bin/scix-batch` exists and is on PATH (CLAUDE.md §Memory isolation).
2. `.venv` resolves to a Python with the project installed (`pip install -e .`).
3. `psql` connects to the prod DSN without credential prompt (`.pgpass`
   or pg_hba trust for the local user).
4. Free disk on `/` is ≥ 50 GB at install time. The shard run aborts via
   `enforce_free_disk_guard` if it later drops below.
5. PR #5 (sjarmak/scix-agent#5, squash-merged as `810b8eb`) is in main —
   the wrapper assumes `--shard` and `--allow-prod` flags exist on
   `scripts/extract_citation_contexts.py`.

### Expected journal output per run

```
... [shard 0/4] step BEGIN: extract_citation_contexts
... [shard 0/4] step OK:    extract_citation_contexts
... [shard 0/4] step BEGIN: backfill_citation_intent
... [shard 0/4] step OK:    backfill_citation_intent
... [shard 0/4] step BEGIN: refresh_v_claim_edges
... [shard 0/4] step OK:    refresh_v_claim_edges
... [shard 0/4] shard 0/4 complete
```

A FAIL line short-circuits the chain; the timer remains armed for the
next day. State in `citation_contexts` and `ingest_log` is preserved.
