# Collect animation LoRA data

The first-round goal is approximately 500 curated four-frame sequences.
This is a target, not a measured collection or evidence of LoRA quality.
Start with real animation packs. Procedural and reviewed generated sequences
are optional later additions with separate provenance.

Use the project venv interpreter in `$PY`. Core tools need NumPy and Pillow.
The tools write only new output folders. They do not modify P32 data, R1,
holdout bytes, running training or generation services.

## Run the collection stages

1. Choose a pack from [the public source shortlist](README-sources.md).
   [Download and ingest it](README-ingest.md), retaining source URLs, hashes,
   original files and RGBA masters. License metadata is optional.
2. [Export consistency review](README-score.md) and inspect playback. Scores
   are `UNCALIBRATED`. Identity needs hash-bound embeddings or an existing local
   vision model. Without either, identity is explicitly unavailable.
3. Save a human-review sidecar bound to the collection manifest. Accept whole
   sequences after checking identity, anchor motion, action and loop intent.
   Do not edit the immutable collection records to mark acceptance.
4. [Build LoRA sheets](README-sheets.md) from accepted sequences. The default
   uses four frames in a 2x2 magenta sheet, one common scale and ordered
   captions. Train/eval outputs are separated by lineage and subject groups.
   Use a frozen evaluation-subject file for comparisons across data revisions.
5. If useful, [create procedural derivatives](README-motions.md) from reviewed
   static P32 or Krea sprites ingested as generated sources. Review these new
   sequences independently. Including them in sheets requires explicit opt-in.
6. [Plan or run the Krea baseline pilot](README-pilot.md) on a separate frozen
   animation audit set. A dry run makes no requests. Execution uses the current
   LoRA on an already-running operator-specified server. Keep its evaluation
   sheets out of training, including optional generated-data rounds.

The sheet builder supports 3x3 and 4x4 formats, but the
[proposal's promotion targets](../../docs/animation-alpha-data-plan.md#increase-frame-count-only-after-the-shorter-layout-passes)
are untested. Curate real longer sequences and measure quality, memory and
generation cost before adopting a longer layout. This pipeline does not train
or configure a new LoRA.

## Run CPU fixtures before real data

From the repository root:

```sh
$PY -m unittest discover -s data-pipeline/animation/tests -t data-pipeline
```

The fixture suite creates its own PNG sheets, Aseprite JSON, GIFs and frame
folders in temporary directories. It also drives the CLIs through ingest,
review, sheet assembly, scoring, procedural opt-in and a network-free pilot.
Mock transports exercise the generation request protocol and errors.
No real asset, checkpoint, model download or GPU is required for these tests.

CLI commands in the stage guides run from `data-pipeline` so `animation` is
importable. Pass a new output folder on every invocation. A hash mismatch is a
failure to investigate, not a reason to reseal a manifest or rewrite holdouts.
The local vision-model path and actual sd.cpp execution still need validation
on the operator's hardware. CPU fixture results do not measure animation
quality, collection yield or generation throughput.
