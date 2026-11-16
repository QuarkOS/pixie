# Record a manually downloaded pack

Some itch.io asset downloads use signed URLs. Download the file through the
browser, then register the local archive against its public asset page.
The downloader keeps rejecting signed URLs, credentials and query strings.
It does not automate accounts or extract archives.

Run this command from `data-pipeline` with the project venv interpreter in
`$PY`. The output directory must be new:

```powershell
& $PY -m animation.download `
  --local-file "$DATA_ROOT/manual-downloads/pixel-adventure.zip" `
  --source-url https://pixelfrog-assets.itch.io/pixel-adventure-1 `
  --output "$DATA_ROOT/download-records/pixel-adventure-v1"
```

Use the public pack page as `--source-url`, not the browser's temporary download
URL. The exported record contains that page URL, a SHA256 hash, byte count,
`acquisition: manual_download` and the copied `asset.bin` path. It contains no
local source path or filename, signed download URL, or session token. Original
file bytes remain untouched. License metadata is not an admission gate.

`--expected-sha256` can verify a previously recorded hash before creating the
output. `--max-bytes` bounds input size. The tool streams the file, checks the
copy against the original hash, and publishes `download.json` only after a
complete copy. If the source changes during the copy, a partial output folder
may remain without that record. Choose another new folder after investigating
the failure. No existing output is overwritten.

For an ordinary public direct download, use `--url` instead of `--local-file`.
The options are mutually exclusive. Retain `asset.bin` and `download.json`
alongside the extracted pack. Ingest separately hashes every selected PNG,
GIF or JSON source file and preserves those bytes. The download archive hash
identifies the received pack revision; it does not replace per-asset hashes.

From the repository root, run the CPU fixtures:

```powershell
& $PY -m unittest discover -s data-pipeline/animation/tests -t data-pipeline -p test_download.py
```
