# Create procedural motion derivatives

These motions provide controlled derivatives of reviewed static P32 or Krea
sprites. They do not establish artist-authored motion quality. Each result is
tagged `procedural` and `unreviewed`, retains its parent lineage and split, and
records the parent PNG hash, source URL, algorithm version, effective parameters
and seed. The parent collection stays unchanged.

First ingest the sprite as a one-frame sequence with an explicit anchor and
`kind: generated` for P32 or Krea outputs. Use a manifest-bound human review
file to accept it, as described in `README-ingest.md`. Run commands from `data-pipeline` using
the project venv interpreter in `$PY`:

```sh
$PY -m animation.motions --input "$DATA_ROOT/accepted-static" --reviews "$DATA_ROOT/static-reviews.json" --output "$DATA_ROOT/bob-pilot" --motion bob --frames 4 --duration-ms 100 --seed 42
```

`--motion` accepts `bob`, `blink`, `pulse`, `recoil`, `spin` and `fade`.
Frame-count options are 4, 9 and 16. These are output formats, not measured
quality claims. All frames keep the shared canvas and anchor. Geometry uses
nearest-neighbor sampling and does not fit palettes. Fade preserves foreground
RGB and changes alpha. Recoil and fade are non-looping annotations.

Pass `--params` with a JSON file to set motion parameters. Bob accepts
`amplitude_px` and `phase`, pulse accepts `amplitude` and `phase`, recoil accepts
`amplitude_px` and `axis`, spin accepts `turns`, and fade accepts `start_opacity`
and `end_opacity`. Missing cyclic phases are deterministic from the recorded
seed. Some motions are deterministic without randomness but still record it.

Blink requires explicit eye rectangles and colors. It does not guess eye
locations. An illustrative annotation is:

```json
{"eye_boxes": [[14, 13, 16, 16]], "fill_rgba": [80, 140, 220, 255], "ink_rgba": [10, 10, 10, 255], "closed_frames": [1]}
```

Rectangles use exclusive right and bottom bounds. Closed-frame indices start
at zero. Without `closed_frames`, the seed chooses a closed frame. Review the
result because these annotations encode an intended edit, not an eye detector.

## Inspect per-sequence skips

Clipping skips that parent sequence and continues with the remaining parents.
The output manifest records `counts.requested`, `counts.created`,
`counts.skipped` and a `skipped` array. Each skip names the parent ID, source
URL, parent PNG hash, motion version, effective parameters, seed and first
clipping frame index. Frame indices start at zero. A skipped sequence does not
leave partial frames in the output.

The manifest status is `complete`, `complete_with_skips` or `all_skipped`.
The command exits successfully when it creates any sequence, and returns exit
code 3 when every requested sequence clips. An all-skipped run still writes
an empty collection and its report to the new output folder. Existing output
folders remain a hard error, including a retry of an all-skipped run.

Only the dedicated foreground-clipping failure is skipped. Invalid motion
parameters, unaccepted parents, missing anchors, holdout matches and corrupted
source hashes still fail the command before output creation. Inspect and fix
those inputs rather than treating them as skipped examples.

Spin always refuses a transform that would crop foreground. A foot anchor can
send the upper body outside the canvas during rotation. Supply a larger padded
parent canvas with the anchor adjusted to that same padding, or keep the
reported skip. The tool does not silently move the anchor, shrink the sprite
or refit the canvas to make a spin pass. `allow_clipping: true` is rejected for
spin. For other motions, it remains an explicit recorded alternative for an
intentional effect that leaves the canvas.

A seed makes an operation repeatable; it does not make a symmetric
sprite's spin or a tiny pulse visibly useful. The sequence review must catch
static or poor results. Procedural examples remain distinguishable from real
sequences when building LoRA sheets.

From the repository root, run the CPU fixtures:

```sh
$PY -m unittest discover -s data-pipeline/animation/tests -t data-pipeline -p test_motions.py
```
