# Curate and ingest downloaded LPC sheets

The LPC source guide is a queue, not a claim that archives are present in this
repository. This command is for packs already downloaded by the operator. It
does not download, extract, recenter, recolor, or infer frame order. A curator
must select complete four-phase actions from the source sheet and record the
shared canvas and anchor.

The official LPC fish sheet rows are **up, left, right, down** (rows 0–3).
The fish curation validator requires that row for the named direction, so the
older up/left/down/right order fails before import. On the LPC horse sheet,
group 0 (rows 0–3) is gallop and group 1 (rows 4–7) is walk. Horse rows must
use integer source-canvas coordinates, set `anchor_policy: "source_ground_center"`,
and set `source_motion` equal to the captioned action. For those two groups the
action has to name the sheet group.
The integer source anchor is transformed once by ingest for a 128→64 import,
which keeps ground contact fixed instead of deriving a floating anchor from a
resized preview.

Create one curation file per pack. The archive hash is the hash of the original
downloaded archive; each imported source file is hashed again by the normal
ingest manifest.

```json
{
  "schema": "pixie-lpc-curation-v1",
  "pack_id": "lpc-fish-v1",
  "source_url": "https://opengameart.org/content/lpc-fish",
  "archive_sha256": "<lowercase SHA256 of the downloaded archive>",
  "sequences": [
    {
      "sequence_id": "fish-trout-swim-south",
      "lineage_group": "lpc-fish-trout",
      "subject": "trout",
      "action": "swim",
      "direction": "south",
      "loop": true,
      "anchor_px": [16, 28],
      "format": "sheet",
      "path": "sprites/trout.png",
      "cell_px": [32, 32],
      "order": [[0, 3], [1, 3], [2, 3], [3, 3]],
      "durations_ms": [150, 150, 150, 150]
    }
  ]
}
```

Run from `data-pipeline` with a new output folder:

```sh
$PY -m animation.lpc_curate \
  --input "$DATA_ROOT/downloads/lpc-fish-v1/extracted" \
  --curation "$DATA_ROOT/curation/lpc-fish-v1.json" \
  --output "$DATA_ROOT/animations/lpc-fish-v1" \
  --canvas 32 32
```

For larger LPC cells, use `--canvas 64 64` and keep that collection separate.
Use `--default-duration-ms 100` only when the source has no duration metadata;
the resulting sequence records `timing_origin: default`. Explicit durations
remain `annotated`, and source GIF/Aseprite timing remains `source_metadata`.

Soft-edged LPC animals can be admitted to the opaque sheet format only with an
explicit alpha cutout. For example, `--alpha-threshold 128` makes source alpha
values below 128 transparent and all remaining values opaque. It records the
threshold and changed-pixel counts, preserves the original masters, and can
make `alpha_mode` binary so the sheet builder can accept the sequence. This is
lossy and opt-in; compare the cutout against the original before review.

The command writes a generated ingest spec in a sibling `*.ingest-spec/`
folder next to the requested output and removes it if ingestion fails. The collection records the curation file hash,
pack ID, public source URL and archive hash. It refuses an existing output,
duplicate IDs, non-artist rows, malformed hashes/URLs, paths outside the input,
and Blind-50/Eval-50 matches through the shared ingest policy.

Recommended first packs are LPC Character Bases, Goblin Full Sheet, Wild Boar,
Bears/Deer/Lions, Fish and Birds. Keep shared bases, recolors and mirrored
directions in one lineage group; do not count them as independent artist
motion. Record the measured accepted count after review. The source guide's
116–206 LPC action range is an estimate, not a measurement.

CPU fixture:

```sh
$PY -m unittest data-pipeline/animation/tests/test_lpc_curate.py
```
