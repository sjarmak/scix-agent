"""Unit tests for src/scix/sources/arxiv_s3.py — arXiv S3 src/ bulk ingest.

Covers:
- Manifest XML parsing
- ManifestEntry immutability
- Tar-of-tars extraction logic (per-paper .tar.gz from outer tar)
- Content-addressed cache (skip already-extracted papers)
- MD5 verification on download
- Delta sync (manifest diff: new files only)
- arXiv ID extraction from inner tar member names
- Requester-pays S3 configuration
- Path traversal prevention in tar extraction

No actual S3 or network access. All external deps are mocked.
"""

from __future__ import annotations

import hashlib
import io
import tarfile
import textwrap
from dataclasses import FrozenInstanceError
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from scix.sources.arxiv_s3 import (
    ArxivS3Config,
    ArxivS3Client,
    ManifestEntry,
    parse_manifest_xml,
    extract_arxiv_id_from_filename,
)

# ---------------------------------------------------------------------------
# Fixture: minimal manifest XML
# ---------------------------------------------------------------------------

MINIMAL_MANIFEST_XML = textwrap.dedent("""\
    <?xml version="1.0" encoding="UTF-8"?>
    <arXivSRC>
      <file>
        <content_md5sum>d41d8cd98f00b204e9800998ecf8427e</content_md5sum>
        <filename>src/arXiv_src_2301_001.tar</filename>
        <first_item>2301.00001</first_item>
        <last_item>2301.00500</last_item>
        <md5sum>abc123def456</md5sum>
        <num_items>500</num_items>
        <seq_num>1</seq_num>
        <size>1073741824</size>
        <timestamp>2023-02-15 00:00:00</timestamp>
        <yymm>2301</yymm>
      </file>
      <file>
        <content_md5sum>e41d8cd98f00b204e9800998ecf8427e</content_md5sum>
        <filename>src/arXiv_src_2301_002.tar</filename>
        <first_item>2301.00501</first_item>
        <last_item>2301.01000</last_item>
        <md5sum>def456ghi789</md5sum>
        <num_items>500</num_items>
        <seq_num>2</seq_num>
        <size>2147483648</size>
        <timestamp>2023-02-15 00:00:00</timestamp>
        <yymm>2301</yymm>
      </file>
    </arXivSRC>
""")


# ---------------------------------------------------------------------------
# ManifestEntry immutability
# ---------------------------------------------------------------------------


class TestManifestEntry:
    def test_is_frozen(self) -> None:
        entry = ManifestEntry(
            filename="src/arXiv_src_2301_001.tar",
            yymm="2301",
            seq_num=1,
            num_items=500,
            md5sum="abc123",
            content_md5sum="def456",
            size=1024,
            first_item="2301.00001",
            last_item="2301.00500",
            timestamp="2023-02-15 00:00:00",
        )
        with pytest.raises(FrozenInstanceError):
            entry.filename = "mutated"  # type: ignore[misc]

    def test_all_fields_present(self) -> None:
        entry = ManifestEntry(
            filename="src/arXiv_src_2301_001.tar",
            yymm="2301",
            seq_num=1,
            num_items=500,
            md5sum="abc123",
            content_md5sum="def456",
            size=1024,
            first_item="2301.00001",
            last_item="2301.00500",
            timestamp="2023-02-15 00:00:00",
        )
        assert entry.yymm == "2301"
        assert entry.seq_num == 1
        assert entry.num_items == 500
        assert entry.size == 1024


# ---------------------------------------------------------------------------
# Manifest XML parsing
# ---------------------------------------------------------------------------


class TestParseManifestXml:
    def test_parses_two_entries(self) -> None:
        entries = parse_manifest_xml(MINIMAL_MANIFEST_XML)
        assert len(entries) == 2

    def test_first_entry_fields(self) -> None:
        entries = parse_manifest_xml(MINIMAL_MANIFEST_XML)
        e = entries[0]
        assert e.filename == "src/arXiv_src_2301_001.tar"
        assert e.yymm == "2301"
        assert e.seq_num == 1
        assert e.num_items == 500
        assert e.md5sum == "abc123def456"
        assert e.size == 1073741824
        assert e.first_item == "2301.00001"
        assert e.last_item == "2301.00500"

    def test_sorted_by_yymm_then_seq(self) -> None:
        entries = parse_manifest_xml(MINIMAL_MANIFEST_XML)
        assert entries[0].seq_num < entries[1].seq_num

    def test_empty_manifest(self) -> None:
        xml = '<?xml version="1.0"?><arXivSRC></arXivSRC>'
        entries = parse_manifest_xml(xml)
        assert entries == []


# ---------------------------------------------------------------------------
# ArxivS3Config
# ---------------------------------------------------------------------------


class TestArxivS3Config:
    def test_is_frozen(self) -> None:
        cfg = ArxivS3Config(
            cache_dir=Path("/tmp/test"),
            bucket="arxiv",
            region="us-east-1",
        )
        with pytest.raises(FrozenInstanceError):
            cfg.bucket = "mutated"  # type: ignore[misc]

    def test_defaults(self) -> None:
        cfg = ArxivS3Config(cache_dir=Path("/tmp/test"))
        assert cfg.bucket == "arxiv"
        assert cfg.region == "us-east-1"


# ---------------------------------------------------------------------------
# arXiv ID extraction from inner tar member filenames
# ---------------------------------------------------------------------------


class TestExtractArxivId:
    def test_new_style_id(self) -> None:
        assert extract_arxiv_id_from_filename("2301.00001.tar.gz") == "2301.00001"

    def test_new_style_with_version(self) -> None:
        assert extract_arxiv_id_from_filename("2301.00001v2.tar.gz") == "2301.00001v2"

    def test_old_style_id(self) -> None:
        # In the tar, old-style IDs have slash replaced with underscore-or-path
        assert extract_arxiv_id_from_filename("astro-ph0001001.gz") == "astro-ph/0001001"

    def test_tex_file(self) -> None:
        # Some papers are single .tex files, not tarballs
        assert extract_arxiv_id_from_filename("2301.00001") == "2301.00001"

    def test_none_for_non_paper(self) -> None:
        assert extract_arxiv_id_from_filename("README") is None
        assert extract_arxiv_id_from_filename("") is None


# ---------------------------------------------------------------------------
# ArxivS3Client — sync_manifest (mocked S3)
# ---------------------------------------------------------------------------


class TestSyncManifest:
    def test_downloads_and_parses_manifest(self, tmp_path: Path) -> None:
        cfg = ArxivS3Config(cache_dir=tmp_path)
        client = ArxivS3Client(cfg)

        mock_s3 = MagicMock()
        manifest_bytes = MINIMAL_MANIFEST_XML.encode("utf-8")
        mock_s3.get_object.return_value = {
            "Body": io.BytesIO(manifest_bytes),
        }

        with patch.object(client, "_get_s3_client", return_value=mock_s3):
            entries = client.sync_manifest()

        assert len(entries) == 2
        mock_s3.get_object.assert_called_once()
        call_kwargs = mock_s3.get_object.call_args[1]
        assert call_kwargs["RequestPayer"] == "requester"

    def test_caches_manifest_locally(self, tmp_path: Path) -> None:
        cfg = ArxivS3Config(cache_dir=tmp_path)
        client = ArxivS3Client(cfg)

        mock_s3 = MagicMock()
        manifest_bytes = MINIMAL_MANIFEST_XML.encode("utf-8")
        mock_s3.get_object.return_value = {
            "Body": io.BytesIO(manifest_bytes),
        }

        with patch.object(client, "_get_s3_client", return_value=mock_s3):
            client.sync_manifest()

        manifest_path = tmp_path / "arXiv_src_manifest.xml"
        assert manifest_path.exists()
        assert manifest_path.read_text(encoding="utf-8") == MINIMAL_MANIFEST_XML


# ---------------------------------------------------------------------------
# ArxivS3Client — delta_sync (manifest diff)
# ---------------------------------------------------------------------------


class TestDeltaSync:
    def test_returns_only_new_entries(self, tmp_path: Path) -> None:
        """delta_sync should return entries not already downloaded."""
        cfg = ArxivS3Config(cache_dir=tmp_path)
        client = ArxivS3Client(cfg)

        entries = parse_manifest_xml(MINIMAL_MANIFEST_XML)
        # Simulate: first tar already downloaded
        tars_dir = tmp_path / "tars"
        tars_dir.mkdir()
        (tars_dir / "arXiv_src_2301_001.tar").touch()

        new_entries = client.delta_sync(entries)
        assert len(new_entries) == 1
        assert new_entries[0].seq_num == 2


# ---------------------------------------------------------------------------
# ArxivS3Client — extract_papers (tar-of-tars)
# ---------------------------------------------------------------------------


def _make_inner_tarball(arxiv_id: str, content: str = "\\documentclass{article}") -> bytes:
    """Create a minimal .tar.gz containing a single .tex file."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as inner:
        tex_bytes = content.encode("utf-8")
        info = tarfile.TarInfo(name=f"{arxiv_id}/main.tex")
        info.size = len(tex_bytes)
        inner.addfile(info, io.BytesIO(tex_bytes))
    return buf.getvalue()


def _make_outer_tar(members: dict[str, bytes]) -> bytes:
    """Create an outer tar with named inner members."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as outer:
        for name, data in members.items():
            info = tarfile.TarInfo(name=name)
            info.size = len(data)
            outer.addfile(info, io.BytesIO(data))
    return buf.getvalue()


class TestExtractPapers:
    def test_extracts_inner_tarballs(self, tmp_path: Path) -> None:
        cfg = ArxivS3Config(cache_dir=tmp_path)
        client = ArxivS3Client(cfg)

        inner1 = _make_inner_tarball("2301.00001")
        inner2 = _make_inner_tarball("2301.00002")
        outer = _make_outer_tar(
            {
                "2301.00001.tar.gz": inner1,
                "2301.00002.tar.gz": inner2,
            }
        )
        outer_path = tmp_path / "arXiv_src_2301_001.tar"
        outer_path.write_bytes(outer)

        extracted = client.extract_papers(outer_path)
        assert len(extracted) == 2

        cache_dir = tmp_path / "raw_latex"
        assert (cache_dir / "2301.00001.tar.gz").exists()
        assert (cache_dir / "2301.00002.tar.gz").exists()

    def test_skips_already_cached(self, tmp_path: Path) -> None:
        cfg = ArxivS3Config(cache_dir=tmp_path)
        client = ArxivS3Client(cfg)

        inner1 = _make_inner_tarball("2301.00001")
        outer = _make_outer_tar({"2301.00001.tar.gz": inner1})
        outer_path = tmp_path / "arXiv_src_2301_001.tar"
        outer_path.write_bytes(outer)

        # Pre-populate cache
        cache_dir = tmp_path / "raw_latex"
        cache_dir.mkdir(parents=True)
        (cache_dir / "2301.00001.tar.gz").write_bytes(inner1)

        extracted = client.extract_papers(outer_path)
        # Should report 0 newly extracted (skipped due to cache)
        assert len(extracted) == 0

    def test_rejects_path_traversal_in_tar(self, tmp_path: Path) -> None:
        """Tar members with '..' must be rejected to prevent path traversal."""
        cfg = ArxivS3Config(cache_dir=tmp_path)
        client = ArxivS3Client(cfg)

        evil_data = b"evil content"
        outer = _make_outer_tar({"../../etc/passwd.tar.gz": evil_data})
        outer_path = tmp_path / "evil.tar"
        outer_path.write_bytes(outer)

        # Should not extract anything (path traversal rejected)
        extracted = client.extract_papers(outer_path)
        assert len(extracted) == 0
        assert not (tmp_path / "etc").exists()


# ---------------------------------------------------------------------------
# ArxivS3Client — MD5 verification
# ---------------------------------------------------------------------------


class TestMd5Verification:
    def test_verify_md5_passes(self, tmp_path: Path) -> None:
        cfg = ArxivS3Config(cache_dir=tmp_path)
        client = ArxivS3Client(cfg)

        content = b"test content for md5"
        expected_md5 = hashlib.md5(content).hexdigest()

        test_file = tmp_path / "test_file.tar"
        test_file.write_bytes(content)

        # Should not raise
        assert client.verify_md5(test_file, expected_md5) is True

    def test_verify_md5_fails(self, tmp_path: Path) -> None:
        cfg = ArxivS3Config(cache_dir=tmp_path)
        client = ArxivS3Client(cfg)

        test_file = tmp_path / "test_file.tar"
        test_file.write_bytes(b"some content")

        assert client.verify_md5(test_file, "wrong_md5") is False


# ---------------------------------------------------------------------------
# ArxivS3Client — download_tar (mocked S3)
# ---------------------------------------------------------------------------


class TestDownloadTar:
    def test_downloads_with_requester_pays(self, tmp_path: Path) -> None:
        cfg = ArxivS3Config(cache_dir=tmp_path)
        client = ArxivS3Client(cfg)

        entry = ManifestEntry(
            filename="src/arXiv_src_2301_001.tar",
            yymm="2301",
            seq_num=1,
            num_items=500,
            md5sum=hashlib.md5(b"fake tar content").hexdigest(),
            content_md5sum="def456",
            size=1024,
            first_item="2301.00001",
            last_item="2301.00500",
            timestamp="2023-02-15 00:00:00",
        )

        mock_s3 = MagicMock()
        mock_s3.download_file = MagicMock()

        tars_dir = tmp_path / "tars"
        tars_dir.mkdir(parents=True)
        tar_path = tars_dir / "arXiv_src_2301_001.tar"

        # Pre-create the file so verify_md5 works
        tar_path.write_bytes(b"fake tar content")

        with patch.object(client, "_get_s3_client", return_value=mock_s3):
            with patch.object(client, "verify_md5", return_value=True):
                result = client.download_tar(entry)

        assert result == tar_path
        mock_s3.download_file.assert_called_once()
        call_args = mock_s3.download_file.call_args
        assert call_args[1].get("ExtraArgs", {}).get("RequestPayer") == "requester"
