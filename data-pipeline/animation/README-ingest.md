# Import animated packs into a new collection

Use the project venv. These tools require Pillow and NumPy. They do not train,
change P32 data, or modify holdout files. About 500 curated four-frame sequences
is the collection goal, not a measured result.

Run commands from `data-pipeline`. Use a fresh output directory for every run.

The holdout policy checks the frozen original byte hashes `7524db9c...` and
`a1d6893e...`. JSON reformatting or Windows newline conversion fails before
import. Supply the original files with `--blind50` and `--eval50` if needed.
Pulling Git attributes into an existing checkout does not always replace older
CRLF working copies. The tools refuse mismatched bytes and never rewrite them
or adjust fingerprints to accept converted copies.

```sh
$PY -m animation.download --source-url https://example.org/asset-page --url https://example.org/pack.zip --output "$DATA_ROOT/downloads/pack-v1"
$PY -m animation.ingest --input "$DATA_ROOT/packs/pack-v1" --spec "$DATA_ROOT/packs/pack-v1/ingest.json" --output "$DATA_ROOT/animations/pack-v1" --canvas 32 32
```

The downloader writes `asset.bin` and `download.json` with requested URL, final
URL, source page, byte count and SHA256. Set `--expected-sha256` to verify a
known download. It rejects credential-bearing and query-bearing URLs. Download
archives through a stable public link or download them manually. Archives are
not extracted automatically. Unpack selected files into a separate input folder.
For manual downloads, provide their public asset page in the ingest spec.

Create a JSON spec with explicit sequence metadata. For a frame folder:

```json
{
  "sequences": [{
    "sequence_id": "copper-beetle-walk",
    "source_url": "https://example.org/asset-page",
    "lineage_group": "copper-beetle",
    "subject": "a copper beetle",
    "caption": "a copper beetle walking right",
    "action": "walk",
    "direction": "right",
    "loop": true,
    "anchor_px": [16, 30],
    "format": "folder",
    "frames": [
      {"path": "walk/pose-a.png", "duration_ms": 100},
      {"path": "walk/pose-b.png", "duration_ms": 100}
    ]
  }]
}
```

All input references are relative to `--input`. `anchor_px` is a shared source
canvas reference. Use null for an unknown anchor. The importer never estimates
an anchor from individual frames. Set `lineage_group` to the same value for all
actions, directions and recolors of an asset. Source URLs and per-file hashes
are required provenance. License metadata does not gate this importer.

Replace the format-specific fields for other inputs:

| Format | Required fields |
|---|---|
| `sheet` | `path`, `cell_px: [width,height]`, explicit `order: [[column,row],...]`, and `durations_ms`, `duration_ms` or the CLI timing default |
| `aseprite` | JSON `path`; optional `image` override, `tag`, and `frame_order` for JSON-hash exports. Durations come from JSON unless an explicit `duration_ms` fallback is supplied. |
| `gif` | GIF `path`; decoded source durations must be positive, or supply an explicit `duration_ms` fallback when missing. |
| `folder` | Ordered `frames` entries with relative paths and positive durations from each entry, the sequence or the CLI timing default. File names never determine order. |

Aseprite trim offsets restore each frame to `sourceSize` before normalization.
JSON arrays follow array order. JSON-hash exports follow recorded key order
unless `frame_order` names a different order. Named tags select actions and
respect forward, reverse and ping-pong directions. Rotated packed exports are
rejected. Export without rotation. Exports with multiple action tags require
an explicit `tag`. GIF frames use Pillow's composited playback
with disposal handling; they retain decoded durations.

By default, the importer fits the entire shared source canvas into the target
canvas with nearest-neighbor resize and common padding. All source frames must
share a canvas. It preserves raw input bytes under `originals/`, reconstructed
source-size RGBA frames under `masters/`, and normalized RGBA frames under
`frames/`. Masters retain the original alpha and RGB before working-frame
keying or cropping. No option recenters individual frames or refits a palette.

## Key solid backgrounds before normalization

Opaque magenta sheets need an explicit color key. Use `--key-rgb 255 0 255`
for exact magenta. Only RGB matches become transparent. The key has no tolerance
and does not remove near-magenta pink or purple. Existing partial alpha in
nonmatching pixels remains unchanged. A key match retains its RGB while its
alpha becomes zero. Raw files and reconstructed masters are not keyed.

The sequence spec can set `key_rgb: [255, 0, 255]` instead. A sequence value
overrides the CLI default. Explicit `key_rgb: null` disables the CLI key for
that sequence. No key is applied by default. Review the result because an
exactly matching foreground color would also be removed. This is an opaque
background key, not recovery of ground-truth partial alpha from a composite.
An entirely invisible sequence after keying fails before creating an output
folder. Individual blank frames remain in order with their durations.

## Preserve content size with one shared crop or a larger canvas

Large padded source cells can make subjects tiny when fitted into 32x32.
`--crop-to-content` takes the union of visible alpha across the whole sequence
after keying. It applies that same rectangle to every frame before fitting.
The crop includes the common source anchor and `--crop-padding` source pixels,
clamped to the original canvas. Translation and changes in pose remain visible.
This is not an independent crop or center operation on each frame.

```sh
$PY -m animation.ingest --input "$DATA_ROOT/packs/birds" --spec "$DATA_ROOT/packs/birds/ingest.json" --output "$DATA_ROOT/animations/birds-keyed-cropped" --key-rgb 255 0 255 --crop-to-content --crop-padding 1 --canvas 32 32
```

Use `--canvas 64 64` to preserve more source detail in a separate collection.
That option changes the normalized working frames, not the original masters.
The downstream sheet builder must explicitly support the selected frame size.
It does not turn a larger source into native 32x32 artwork. The sequence spec
can override `crop_to_content` and `crop_padding` for individual sequences.
Padding without a content crop is an error.

The normalization record retains original and working canvas sizes, shared
crop rectangle, padding, exact key, keyed pixel counts per frame, resize,
offset and transformed anchor. Cropping and keying are opt-in operations.

## Annotate untimed pose tiles explicitly

For pose tiles without source durations, supply `--default-duration-ms 100`
alongside an explicit frame order. The number is an illustrative playback
annotation, not a measured or recovered source duration. Kenney pose tiles
remain subject to whole-sequence review and are not automatically accepted as
motion. Repeating an unrelated pose does not establish a useful animation.

The timing priority is source GIF or Aseprite metadata, explicit frame duration,
sequence `duration_ms`, then the CLI default. Source durations are not overridden
by the default. Missing or zero source durations can use an explicit fallback.
Negative, boolean or malformed durations fail. Annotations have
`timing_origin: annotated`; a CLI default fallback has `timing_origin: default`. Untouched source timing retains `source_metadata`.
The supplied default is recorded in normalization metadata.

```sh
$PY -m animation.ingest --input "$DATA_ROOT/packs/kenney" --spec "$DATA_ROOT/packs/kenney/ingest.json" --output "$DATA_ROOT/animations/kenney-timed" --default-duration-ms 100 --crop-to-content --canvas 32 32
```

`sources.jsonl` records every input URL and hash. `sequences.jsonl` records
frame order, durations, canvas, anchor, lineage, action and frame count.
`manifest.json` binds the files to SHA256 hashes and records the original
Blind-50 and Eval-50 fingerprints. The existing punctuation-normalized holdout
policy rejects matching subjects, captions or actions before output is created.
Imported examples start as `unreviewed` and `unassigned`; ingest does not admit
them to training. Outputs must be outside the input pack and spec directories.

Set `kind` to `generated` when importing a P32 or Krea sprite; the default is
`artist`. Optional `provenance` records can retain model hashes, prompt and seed
context. They must not contain private paths or credentials. Import does not
infer a generated sprite's model or quality from its appearance.
Set `color_space` when the source convention is known. Otherwise it records
`source_unspecified`; ingest performs no implicit color-space conversion.

Keep human review in a separate JSON file instead of editing the collection:

```json
{
  "collection_sha256": "<SHA256 of the collection manifest.json>",
  "reviews": [
    {"sequence_id": "copper-beetle-walk", "status": "accepted"}
  ]
}
```

`common.reviewed_sequences(collection, review_file)` applies this overlay in
memory. Unlisted examples keep their original status. It rejects a changed
collection hash, unknown or duplicate IDs, and malformed statuses. A human
must judge sequence quality before setting `accepted`. Automated consistency
scores do not set this status.

Run the CPU fixtures from the repository root:

```sh
$PY -m unittest discover -s data-pipeline/animation/tests -t data-pipeline -p "test_ingest.py"
$PY -m unittest discover -s data-pipeline/animation/tests -t data-pipeline -p "test_download.py"
```

The fixtures create synthetic packs. No public asset download, real corpus
yield or GPU performance is measured by these tests.
