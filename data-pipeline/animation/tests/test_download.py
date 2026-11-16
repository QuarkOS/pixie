"""Downloader transport fixtures never contact an external server."""
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest.mock import patch

from animation.download import download, main, record_manual


class Response(io.BytesIO):
    def __init__(self, data, length=None, final_url="https://example.org/archive.zip"):
        super().__init__(data)
        self.headers = {"Content-Length": str(length if length is not None else len(data))}
        self.final_url = final_url

    def geturl(self):
        return self.final_url


class DownloadTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.output = Path(self.temporary.name) / "download"

    def test_bytes_urls_hash_and_no_extraction(self):
        data = b"not automatically extracted"
        calls = []
        def opener(request, timeout):
            calls.append((request.full_url, timeout))
            return Response(data)
        record = download("https://example.org/old.zip", "https://example.org/pack", self.output, opener=opener)
        self.assertEqual(record["sha256"], hashlib.sha256(data).hexdigest())
        self.assertEqual((self.output / "asset.bin").read_bytes(), data)
        self.assertEqual(record["download_url"], "https://example.org/archive.zip")
        self.assertEqual(record["requested_url"], "https://example.org/old.zip")
        self.assertFalse(record["archive_extracted"])
        self.assertEqual(len(calls), 1)
        with self.assertRaises(FileExistsError):
            download("https://example.org/old.zip", "https://example.org/pack", self.output, opener=opener)
        self.assertEqual(len(calls), 1)

    def test_bound_size_and_expected_hash_fail_without_manifest(self):
        with self.assertRaisesRegex(ValueError, "size"):
            download("https://example.org/a", "https://example.org/pack", self.output, max_bytes=3, opener=lambda *a, **k: Response(b"four"))
        self.assertFalse((self.output / "download.json").exists())
        other = self.output.parent / "mismatch"
        with self.assertRaisesRegex(ValueError, "SHA256"):
            download("https://example.org/a", "https://example.org/pack", other, expected_sha256="0" * 64, opener=lambda *a, **k: Response(b"content"))
        self.assertFalse((other / "download.json").exists())

    def test_credentials_or_query_do_not_reach_transport(self):
        def forbidden(*args, **kwargs):
            self.fail("transport must not run")
        for url in ["https://user:secret@example.org/a", "https://example.org/a?token=secret", "file:///asset.zip"]:
            with self.assertRaisesRegex(ValueError, "public"):
                download(url, "https://example.org/pack", self.output, opener=forbidden)
        self.assertFalse(self.output.exists())

    def test_truncated_download_does_not_publish_manifest(self):
        with self.assertRaisesRegex(ValueError, "length"):
            download("https://example.org/a", "https://example.org/pack", self.output, opener=lambda *a, **k: Response(b"abc", length=4))
        self.assertFalse((self.output / "download.json").exists())

    def test_manual_file_records_public_page_and_exact_bytes_without_network(self):
        source = self.output.parent / "operator-local-archive.zip"
        data = b"PK fixture archive, not extracted"
        source.write_bytes(data)
        before = hashlib.sha256(data).hexdigest()
        with patch("animation.download.urlopen") as network, patch.object(Path, "read_bytes", side_effect=AssertionError("must stream")):
            record = record_manual(source, "https://pixelfrog-assets.itch.io/pixel-adventure-1", self.output)
        network.assert_not_called()
        self.assertEqual(record["acquisition"], "manual_download")
        self.assertEqual(record["sha256"], before)
        self.assertEqual(record["bytes"], len(data))
        self.assertEqual((self.output / "asset.bin").read_bytes(), data)
        self.assertEqual(source.read_bytes(), data)
        self.assertFalse(record["archive_extracted"])
        self.assertNotIn("requested_url", record)
        self.assertNotIn("download_url", record)
        serialized = (self.output / "download.json").read_text()
        self.assertNotIn(str(source), serialized)
        self.assertNotIn(source.name, serialized)
        self.assertEqual(json.loads(serialized), record)

    def test_manual_cli_and_exclusive_acquisition_options(self):
        source = self.output.parent / "archive.zip"
        source.write_bytes(b"fixture")
        stdout = io.StringIO()
        args = ["--local-file", str(source), "--source-url", "https://example.org/pack", "--output", str(self.output)]
        with redirect_stdout(stdout), patch("animation.download.urlopen") as network:
            main(args)
        network.assert_not_called()
        self.assertEqual(json.loads(stdout.getvalue())["sha256"], hashlib.sha256(b"fixture").hexdigest())
        for conflicting in (args + ["--url", "https://example.org/archive"], args[2:]):
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as exit:
                main(conflicting)
            self.assertEqual(exit.exception.code, 2)

    def test_manual_hash_size_empty_missing_and_signed_source_refuse_before_output(self):
        source = self.output.parent / "archive.zip"
        source.write_bytes(b"fixture")
        for kwargs, message in [({"expected_sha256": "0" * 64}, "SHA256"), ({"max_bytes": 2}, "size"),
                                ({"max_bytes": 0}, "size"), ({"expected_sha256": "invalid"}, "SHA256")]:
            with self.assertRaisesRegex(ValueError, message):
                record_manual(source, "https://example.org/pack", self.output, **kwargs)
            self.assertFalse(self.output.exists())
        for source_url in ("https://example.org/pack?token=secret", "https://user:secret@example.org/pack"):
            with self.assertRaisesRegex(ValueError, "public"):
                record_manual(source, source_url, self.output)
            self.assertFalse(self.output.exists())
        source.write_bytes(b"")
        with self.assertRaisesRegex(ValueError, "empty"):
            record_manual(source, "https://example.org/pack", self.output)
        for missing in (source.parent / "missing.zip", source.parent):
            with self.assertRaisesRegex(ValueError, "regular file"):
                record_manual(missing, "https://example.org/pack", self.output)
        self.assertFalse(self.output.exists())

    def test_manual_existing_output_is_never_overwritten(self):
        source = self.output.parent / "archive.zip"
        source.write_bytes(b"fixture")
        self.output.mkdir()
        marker = self.output / "keep.txt"
        marker.write_text("keep")
        with self.assertRaises(FileExistsError):
            record_manual(source, "https://example.org/pack", self.output)
        self.assertEqual(marker.read_text(), "keep")
        self.assertFalse((self.output / "asset.bin").exists())
        self.assertEqual(source.read_bytes(), b"fixture")

    def test_manual_source_change_does_not_publish_provenance(self):
        source = self.output.parent / "archive.zip"
        source.write_bytes(b"fixture")
        with patch("animation.download.sha256", return_value="0" * 64):
            with self.assertRaisesRegex(ValueError, "changed while copying"):
                record_manual(source, "https://example.org/pack", self.output)
        self.assertFalse((self.output / "download.json").exists())


if __name__ == "__main__":
    unittest.main()
