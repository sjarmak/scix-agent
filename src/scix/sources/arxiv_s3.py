"""arXiv S3 ``src/`` bulk ingest — manifest-driven download and extraction.

Downloads arXiv LaTeX source tarballs from the ``arxiv`` S3 bucket
(requester-pays, ``us-east-1``). The bucket contains tar-of-tars: each outer
tar ``arXiv_src_YYMM_NNN.tar`` holds per-paper ``.tar.gz`` archives.

This module:
    1. Parses ``arXiv_src_manifest.xml`` for content inventory + MD5 checksums
    2. Downloads outer tars with MD5 verification (resumable via manifest diff)
    3. Extracts per-paper ``.tar.gz`` into a content-addressed cache
       (``raw_latex/{arxiv_id}.tar.gz``)
    4. Provides delta sync via manifest diff (skip already-downloaded tars)

SAFETY:
    * All S3 operations use ``RequestPayer='requester'``.
    * Path traversal in tar extraction is prevented by rejecting ``..`` in
      member names.
    * No DB writes — this module only populates a local file cache.

See also:
    - ``src/scix/sources/ar5ivist_local.py`` (consumes the cache)
    - ``docs/runbooks/arxiv_s3_ingest.md`` (operational guide)
"""

from __future__ import annotations

import hashlib
import logging
import re
import tarfile
import defusedxml.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Regex for new-style arXiv IDs: 2301.00001, 2301.00001v2
_NEW_STYLE_RE = re.compile(r"^(\d{4}\.\d{4,5}(?:v\d+)?)")

# Regex for old-style arXiv IDs: astro-ph0001001 → astro-ph/0001001
_OLD_STYLE_RE = re.compile(r"^([a-z][\w.-]+?)(\d{7})(?:v\d+)?")


# ---------------------------------------------------------------------------
# Data classes (all frozen / immutable)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ManifestEntry:
    """A single file entry from ``arXiv_src_manifest.xml``."""

    filename: str
    yymm: str
    seq_num: int
    num_items: int
    md5sum: str
    content_md5sum: str
    size: int
    first_item: str
    last_item: str
    timestamp: str


@dataclass(frozen=True)
class ArxivS3Config:
    """Immutable configuration for the arXiv S3 client."""

    cache_dir: Path
    bucket: str = "arxiv"
    region: str = "us-east-1"


# ---------------------------------------------------------------------------
# Manifest XML parsing
# ---------------------------------------------------------------------------


def parse_manifest_xml(xml_text: str) -> list[ManifestEntry]:
    """Parse ``arXiv_src_manifest.xml`` into a sorted list of ManifestEntry.

    Entries are sorted by (yymm, seq_num) for deterministic processing order.
    """
    root = ET.fromstring(xml_text)
    entries: list[ManifestEntry] = []

    for file_elem in root.findall("file"):
        filename = _xml_text(file_elem, "filename")
        if not filename:
            continue

        entries.append(
            ManifestEntry(
                filename=filename,
                yymm=_xml_text(file_elem, "yymm"),
                seq_num=int(_xml_text(file_elem, "seq_num") or "0"),
                num_items=int(_xml_text(file_elem, "num_items") or "0"),
                md5sum=_xml_text(file_elem, "md5sum"),
                content_md5sum=_xml_text(file_elem, "content_md5sum"),
                size=int(_xml_text(file_elem, "size") or "0"),
                first_item=_xml_text(file_elem, "first_item"),
                last_item=_xml_text(file_elem, "last_item"),
                timestamp=_xml_text(file_elem, "timestamp"),
            )
        )

    entries.sort(key=lambda e: (e.yymm, e.seq_num))
    return entries


def _xml_text(parent: ET.Element, tag: str) -> str:
    """Extract text content from a child element, defaulting to empty string."""
    elem = parent.find(tag)
    if elem is not None and elem.text:
        return elem.text.strip()
    return ""


# ---------------------------------------------------------------------------
# arXiv ID extraction from inner tar member filenames
# ---------------------------------------------------------------------------


def extract_arxiv_id_from_filename(member_name: str) -> str | None:
    """Extract an arXiv ID from an inner tar member filename.

    Handles:
        - ``2301.00001.tar.gz`` → ``2301.00001``
        - ``2301.00001v2.tar.gz`` → ``2301.00001v2``
        - ``astro-ph0001001.gz`` → ``astro-ph/0001001``
        - ``2301.00001`` (bare .tex) → ``2301.00001``

    Returns None for non-paper files (README, etc.).
    """
    if not member_name:
        return None

    # Strip known extensions
    base = member_name
    for ext in (".tar.gz", ".gz", ".pdf"):
        if base.endswith(ext):
            base = base[: -len(ext)]
            break

    # Try new-style first
    m = _NEW_STYLE_RE.match(base)
    if m:
        return m.group(1)

    # Try old-style (category prefix + 7-digit ID)
    m = _OLD_STYLE_RE.match(base)
    if m:
        return f"{m.group(1)}/{m.group(2)}"

    return None


# ---------------------------------------------------------------------------
# S3 Client
# ---------------------------------------------------------------------------


class ArxivS3Client:
    """Client for the arXiv requester-pays S3 bucket.

    Provides manifest sync, tar download, extraction, and delta sync.
    """

    def __init__(self, config: ArxivS3Config) -> None:
        self._cfg = config
        self._s3_client: Any = None

    def _get_s3_client(self) -> Any:
        """Lazily create and cache the boto3 S3 client."""
        if self._s3_client is None:
            import boto3

            self._s3_client = boto3.client("s3", region_name=self._cfg.region)
        return self._s3_client

    # -- Manifest ----------------------------------------------------------

    def sync_manifest(self) -> list[ManifestEntry]:
        """Download and parse ``arXiv_src_manifest.xml`` from S3.

        The manifest is cached locally at ``{cache_dir}/arXiv_src_manifest.xml``
        for offline inspection and delta sync.
        """
        s3 = self._get_s3_client()
        response = s3.get_object(
            Bucket=self._cfg.bucket,
            Key="src/arXiv_src_manifest.xml",
            RequestPayer="requester",
        )
        manifest_bytes: bytes = response["Body"].read()
        manifest_text = manifest_bytes.decode("utf-8")

        # Cache locally
        self._cfg.cache_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = self._cfg.cache_dir / "arXiv_src_manifest.xml"
        manifest_path.write_text(manifest_text, encoding="utf-8")
        logger.info("Manifest saved to %s", manifest_path)

        entries = parse_manifest_xml(manifest_text)
        logger.info("Parsed %d manifest entries", len(entries))
        return entries

    # -- Delta sync --------------------------------------------------------

    def delta_sync(self, entries: list[ManifestEntry]) -> list[ManifestEntry]:
        """Return manifest entries whose tars have not been downloaded yet.

        Checks for the presence of the outer tar in ``{cache_dir}/tars/``.
        """
        tars_dir = self._cfg.cache_dir / "tars"
        new_entries: list[ManifestEntry] = []
        for entry in entries:
            tar_name = Path(entry.filename).name
            if not (tars_dir / tar_name).exists():
                new_entries.append(entry)
        logger.info(
            "Delta sync: %d new of %d total entries",
            len(new_entries),
            len(entries),
        )
        return new_entries

    # -- Download ----------------------------------------------------------

    def download_tar(self, entry: ManifestEntry) -> Path:
        """Download an outer tar from S3 with MD5 verification.

        Returns the local path to the downloaded tar.
        """
        tars_dir = self._cfg.cache_dir / "tars"
        tars_dir.mkdir(parents=True, exist_ok=True)
        tar_name = Path(entry.filename).name
        tar_path = tars_dir / tar_name

        s3 = self._get_s3_client()
        logger.info("Downloading %s (%d bytes)", entry.filename, entry.size)
        s3.download_file(
            Bucket=self._cfg.bucket,
            Key=entry.filename,
            Filename=str(tar_path),
            ExtraArgs={"RequestPayer": "requester"},
        )

        if not self.verify_md5(tar_path, entry.md5sum):
            logger.error(
                "MD5 mismatch for %s — deleting corrupt download",
                tar_name,
            )
            tar_path.unlink(missing_ok=True)
            raise ValueError(f"MD5 verification failed for {tar_name}")

        logger.info("Downloaded and verified %s", tar_name)
        return tar_path

    # -- MD5 verification --------------------------------------------------

    @staticmethod
    def verify_md5(filepath: Path, expected_md5: str) -> bool:
        """Verify that a file's MD5 matches the expected hash."""
        md5 = hashlib.md5()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                md5.update(chunk)
        actual = md5.hexdigest()
        if actual != expected_md5:
            logger.warning(
                "MD5 mismatch: expected %s, got %s for %s",
                expected_md5,
                actual,
                filepath,
            )
            return False
        return True

    # -- Extraction --------------------------------------------------------

    def extract_papers(self, tar_path: Path) -> list[str]:
        """Extract per-paper ``.tar.gz`` from an outer tar into the cache.

        Returns a list of newly extracted arXiv IDs (skips already-cached).
        Rejects tar members with path traversal (``..`` in name).
        """
        cache_dir = self._cfg.cache_dir / "raw_latex"
        cache_dir.mkdir(parents=True, exist_ok=True)

        extracted_ids: list[str] = []

        with tarfile.open(tar_path, "r") as outer:
            for member in outer.getmembers():
                # Security: reject path traversal, symlinks, and hardlinks
                if ".." in member.name or member.name.startswith("/"):
                    logger.warning(
                        "Rejected tar member with unsafe path: %s",
                        member.name,
                    )
                    continue

                if member.issym() or member.islnk():
                    logger.warning(
                        "Rejected tar member with symlink/hardlink: %s",
                        member.name,
                    )
                    continue

                if not member.isfile():
                    continue

                arxiv_id = extract_arxiv_id_from_filename(member.name)
                if arxiv_id is None:
                    continue

                # Content-addressed cache: use safe filename
                safe_name = arxiv_id.replace("/", "_") + ".tar.gz"
                cache_path = cache_dir / safe_name

                # Skip if already cached
                if cache_path.exists():
                    continue

                # Extract member content to cache
                fileobj = outer.extractfile(member)
                if fileobj is None:
                    continue

                data = fileobj.read()
                cache_path.write_bytes(data)
                extracted_ids.append(arxiv_id)

        logger.info("Extracted %d papers from %s", len(extracted_ids), tar_path.name)
        return extracted_ids
