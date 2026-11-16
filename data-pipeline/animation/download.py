"""Record provenance for a public download or a manually downloaded asset."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import Request, urlopen

from .common import new_output, sha256
from .ingest import public_url


def _validate_hash(expected_sha256):
    if expected_sha256 is not None and (len(expected_sha256) != 64 or any(c not in "0123456789abcdef" for c in expected_sha256)):
        raise ValueError("expected SHA256 must be lowercase hexadecimal")


def _write_record(root, record):
    (root / "download.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    return record


def download(url, source_url, output, max_bytes=256 * 1024 * 1024, timeout=30, expected_sha256=None, opener=urlopen):
    public_url(url)
    public_url(source_url)
    if type(max_bytes) is not int or max_bytes <= 0 or timeout <= 0:
        raise ValueError("download size and timeout must be positive")
    _validate_hash(expected_sha256)
    root = new_output(output)
    payload = root / "asset.bin"
    with opener(Request(url, headers={"User-Agent": "Pixie-animation-asset-downloader/1"}), timeout=timeout) as response:
        final_url = public_url(response.geturl())
        declared = response.headers.get("Content-Length")
        if declared is not None and (not declared.isdigit() or int(declared) > max_bytes):
            raise ValueError("asset exceeds the download size limit")
        count = 0
        with payload.open("xb") as stream:
            while block := response.read(min(1 << 20, max_bytes - count + 1)):
                count += len(block)
                if count > max_bytes:
                    raise ValueError("asset exceeds the download size limit")
                stream.write(block)
        if count == 0:
            raise ValueError("asset download is empty")
        if declared is not None and count != int(declared):
            raise ValueError("asset download length differs from Content-Length")
    digest = sha256(payload)
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError("asset SHA256 differs from expected hash")
    record = {"schema": "pixie-animation-download-v1", "acquisition": "http_download", "source_url": source_url, "requested_url": url, "download_url": final_url, "path": "asset.bin", "sha256": digest, "bytes": count, "archive_extracted": False}
    return _write_record(root, record)


def record_manual(local_file, source_url, output, max_bytes=256 * 1024 * 1024, expected_sha256=None):
    """Copy an existing regular file without recording its private location."""
    public_url(source_url)
    _validate_hash(expected_sha256)
    if type(max_bytes) is not int or max_bytes <= 0:
        raise ValueError("asset size limit must be positive")
    source = Path(local_file)
    if source.is_symlink() or not source.is_file():
        raise ValueError("manual asset must be an existing regular file, not a symlink")
    size = source.stat().st_size
    if size == 0:
        raise ValueError("manual asset is empty")
    if size > max_bytes:
        raise ValueError("asset exceeds the download size limit")
    if Path(output).exists():
        raise FileExistsError("output already exists; choose a new folder")
    expected = sha256(source)
    if expected_sha256 is not None and expected != expected_sha256:
        raise ValueError("asset SHA256 differs from expected hash")
    root = new_output(output)
    digest = hashlib.sha256()
    count = 0
    with source.open("rb") as incoming, (root / "asset.bin").open("xb") as destination:
        while block := incoming.read(min(1 << 20, max_bytes - count + 1)):
            count += len(block)
            if count > max_bytes:
                raise ValueError("asset exceeds the download size limit")
            destination.write(block)
            digest.update(block)
    if count != size or digest.hexdigest() != expected:
        raise ValueError("manual asset changed while copying; no provenance record was published")
    record = {"schema": "pixie-animation-download-v1", "acquisition": "manual_download",
              "source_url": source_url, "path": "asset.bin", "sha256": digest.hexdigest(),
              "bytes": count, "archive_extracted": False}
    return _write_record(root, record)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    acquisition = parser.add_mutually_exclusive_group(required=True)
    acquisition.add_argument("--url", help="Direct public download URL")
    acquisition.add_argument("--local-file", type=Path, help="Existing manually downloaded archive or asset")
    parser.add_argument("--source-url", required=True, help="Public asset page URL")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-bytes", type=int, default=256 * 1024 * 1024)
    parser.add_argument("--timeout", type=float, default=30)
    parser.add_argument("--expected-sha256")
    args = parser.parse_args(argv)
    if args.local_file is not None:
        record = record_manual(args.local_file, args.source_url, args.output, args.max_bytes, args.expected_sha256)
    else:
        record = download(args.url, args.source_url, args.output, args.max_bytes, args.timeout, args.expected_sha256)
    print(json.dumps({"sha256": record["sha256"], "bytes": record["bytes"]}))


if __name__ == "__main__":
    main()
