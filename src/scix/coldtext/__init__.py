"""Cold-text tier (ADR-016).

Per-year read-only SQLite shards on NAS holding the bulky ``papers_fulltext``
JSONB sealed out of Postgres for closed years. ``sections_tsv`` is NOT sealed —
it stays in Postgres so section BM25 keeps working for sealed years.
"""

from scix.coldtext.route import (
    HOT_WINDOW_START_YEAR,
    clear_cache,
    coldtext_root,
    fetch_sections_cold,
)
from scix.coldtext.shard import (
    SHARD_SCHEMA_VERSION,
    ColdTextReader,
    ShardStats,
    sections_checksum,
    shard_path,
    write_shard,
)

__all__ = [
    "HOT_WINDOW_START_YEAR",
    "SHARD_SCHEMA_VERSION",
    "ColdTextReader",
    "ShardStats",
    "clear_cache",
    "coldtext_root",
    "fetch_sections_cold",
    "sections_checksum",
    "shard_path",
    "write_shard",
]
