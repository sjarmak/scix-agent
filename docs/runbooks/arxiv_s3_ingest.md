# arXiv S3 src/ Ingest Runbook

## Overview

Sync arXiv LaTeX source tarballs (~3 TB) from the `arxiv` S3 bucket
(requester-pays, `us-east-1`) to a local staging cache, then convert
papers missing ar5iv HTML using a local LaTeXML runner (ar5ivist Docker image).

## Prerequisites

- AWS credentials configured (`aws configure` or IAM role on EC2)
- Docker installed (for Part B — local LaTeXML conversion)
- Python environment with `boto3` installed

## Part A — S3 Bulk Sync

### Step 1: Launch EC2 in us-east-1

**Critical**: Always run from `us-east-1` to avoid cross-region transfer costs.
Intra-region S3 transfer is free for requester-pays buckets.

```bash
# Recommended instance: m5.xlarge (4 vCPU, 16 GB RAM)
# Attach a 4 TB gp3 EBS volume for staging
```

### Step 2: Sync manifest

```python
from pathlib import Path
from scix.sources.arxiv_s3 import ArxivS3Config, ArxivS3Client

cfg = ArxivS3Config(cache_dir=Path("/data/arxiv_src"))
client = ArxivS3Client(cfg)

# Download and parse manifest
entries = client.sync_manifest()
print(f"Total tars in manifest: {len(entries)}")
```

### Step 3: Delta sync (identify new tars)

```python
new_entries = client.delta_sync(entries)
print(f"New tars to download: {len(new_entries)}")
```

### Step 4: Download and extract

```python
for entry in new_entries:
    tar_path = client.download_tar(entry)
    extracted = client.extract_papers(tar_path)
    print(f"{entry.filename}: extracted {len(extracted)} papers")
```

### Step 5: Verify

```bash
# Count extracted papers
find /data/arxiv_src/raw_latex -name '*.tar.gz' | wc -l
```

## Part B — Local LaTeXML Conversion

### Step 1: Pull Docker image

```bash
docker pull ghcr.io/ar5iv/ar5ivist:latest
# Note: Update the digest in ar5ivist_local.py after verifying
```

### Step 2: Convert papers

```python
from pathlib import Path
from scix.sources.ar5ivist_local import ArxivLocalConfig, ArxivLocalConverter

cfg = ArxivLocalConfig(
    cache_dir=Path("/data/arxiv_src"),
    dsn="dbname=scix",
    yes_production=True,  # explicit opt-in for production
    workers=4,
    timeout_seconds=120,
)
converter = ArxivLocalConverter(cfg)

# Get list of papers needing conversion (query your DB for papers
# with has_arxiv_source=true AND no ar5iv fulltext row)
paper_ids = [...]  # populate from DB query

results = converter.batch_convert(paper_ids, workers=4)
succeeded = sum(1 for r in results if r.success)
print(f"Converted {succeeded}/{len(results)} papers")
```

### Step 3: Ingest to papers_fulltext

```python
for result in results:
    if result.success and result.html:
        converter.ingest_to_fulltext(
            html=result.html,
            arxiv_id=result.arxiv_id,
            bibcode=arxiv_id_to_bibcode[result.arxiv_id],  # your mapping
        )
```

## Monthly Delta Resync

Run Steps A2-A4 monthly. The manifest diff (`delta_sync`) ensures only
new tars are downloaded. Cost is effectively $0 for intra-region transfer.

```bash
# Cron example (first Sunday of each month)
0 2 1-7 * 0 /path/to/venv/bin/python scripts/arxiv_s3_monthly_sync.py
```

## Cost Estimate

| Component                          | Cost       |
| ---------------------------------- | ---------- |
| S3 GET requests (manifest)         | ~$0.01     |
| S3 data transfer (intra-region)    | $0.00      |
| S3 GET requests (tars, initial)    | ~$5-10     |
| EC2 m5.xlarge (initial sync, ~24h) | ~$5        |
| EBS gp3 4TB                        | ~$320/mo   |
| **Monthly delta**                  | **~$0.10** |

## Troubleshooting

### MD5 mismatch on download

Re-download the tar. If persistent, check for S3 bucket updates
(manifest may be stale).

### LaTeXML conversion timeout

Increase `timeout_seconds`. Some papers with complex macros take longer.
Default 120s handles >98% of papers.

### Docker permission denied

Ensure the user is in the `docker` group or use `sudo`.
