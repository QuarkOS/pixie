# Krea animation baseline pilot

`pilot.py` creates a held-out baseline for the current Krea 2 LoRA. It asks an
already-running `sd.cpp` server for one 2x2, 512px sheet per prompt. The four
frames use row-major order, a fixed camera and scale, and a magenta background.
The output remains a review artifact. The script does not extract frames or
admit them to training.

The default command is a dry run. It validates the frozen audit prompts,
training-subject file, Blind-50 and Eval-50 policy, hashes, seeds and output
layout without making a network request:

```powershell
& $PY -m animation.pilot `
  --prompts $DATA_ROOT/animation/audit-prompts.json `
  --training-subjects $DATA_ROOT/animation/frozen-training-subjects.json `
  --output $RUNS/animation-baseline-plan-20261115-01 `
  --n 2 --seed 20261115 `
  --model-sha256 "$MODEL_SHA256" `
  --lora-sha256 "$LORA_SHA256" `
  --blind50 $DATA_ROOT/holdout/blind-50.json `
  --eval50 $DATA_ROOT/holdout/eval-50.json
```

Run the real pilot only after reviewing the dry-run manifest. Supply the
server-side identifiers and the loopback server explicitly. The server must
already be running; this script does not start or restart it:

```powershell
& $PY -m animation.pilot `
  --prompts $DATA_ROOT/animation/audit-prompts.json `
  --training-subjects $DATA_ROOT/animation/frozen-training-subjects.json `
  --output $RUNS/animation-baseline-run-20261115-01 `
  --n 2 --seed 20261115 `
  --model-sha256 "$MODEL_SHA256" `
  --lora-sha256 "$LORA_SHA256" `
  --model "$MODEL_ID" --lora "$LORA_ID" `
  --server http://127.0.0.1:17890 --execute `
  --blind50 $DATA_ROOT/holdout/blind-50.json `
  --eval50 $DATA_ROOT/holdout/eval-50.json
```

The prompt file can start with this small frozen audit fixture:

```json
{
  "schema": "pixie-animation-baseline-prompts-v1",
  "audit_set": {"id": "animation-audit-v1", "frozen": true},
  "heldout_subjects": ["silver beetle", "red fox"],
  "prompts": [
    {"subject": "silver beetle", "action": "walk", "direction": "right", "loop": true},
    {"subject": "red fox", "action": "blink", "direction": "left", "loop": false}
  ]
}
```

The training-subject file can use this frozen list format:

```json
{
  "frozen": true,
  "subjects": ["blue slime", "stone golem"]
}
```

The prompt file must use `pixie-animation-baseline-prompts-v1`, declare a
frozen `audit_set` id or `heldout_subjects`, and list `prompts` with `subject`,
`action`, `direction` and `loop`. The training-subject JSON must be a frozen
array or an object with a `subjects` array. Prompt subjects must be absent from
that training list. The normalized matching policy also rejects any prompt or
action that matches Blind-50 or Eval-50 before output creation or a network
call.

Keep the output directory outside the directories that contain the frozen
prompt and training-subject files. The script refuses an output nested inside
either input directory.

Set `--n` to a positive value no greater than the number of frozen prompt
entries. Each run requires a new output directory. It contains the exact prompt and
training-subject bytes, `requests.jsonl`, `events.jsonl`, `sheets/`, and a
manifest. `plan.json` is written before any network call and retains every
planned prompt, seed and effective sampler setting if execution is interrupted.
The manifest records model and LoRA hashes, optional public source
URLs, request settings, requested and returned seeds, file hashes and the
review-only status. It never records the server URL or server-side model and
LoRA identifiers. If the server does not return a seed, the record says
`requested_only`; that seed is not proof of replayability.

Model and LoRA hashes are operator-supplied identifiers. The script does not
attest which weights the server has loaded. Verify those on the generation
machine before interpreting the baseline comparison.

The first-round collection target is about 500 curated four-frame sequences.
That is a goal, not a measurement. Review every sheet and run the consistency
scorer before any later data admission.

## Interpret the current baseline

Operator-measured on the 3080 Ti: the initial pilot completed 64/64 sheets at
approximately 3.1 seconds per sheet. An informal review found visible motion
in approximately 2 of 15 inspected sheets. That small review is not a
calibrated acceptance estimate for the full pilot. These are operator-reported
results, not measurements rerun by this change. The current LoRA therefore
remains a comparison baseline; prioritize the artist-made collection in
[the source guide](README-sources.md) before another generated-data round.

Keep palette diagnostics at a declared resolution. A 512x512 2x2 pilot has
256x256 source cells. Ingest those cells with `--canvas 256 256` for native-cell
scoring; use an explicit `order` of `[[0,0],[1,0],[0,1],[1,1]]`, common anchor,
action and annotated playback timing. For a magenta composite, opt in to
`--key-rgb 255 0 255` and inspect fringe colors. Generated records must use
`kind: generated` and retain their pilot prompt/seed/model/LoRA provenance.

Score that diagnostic collection before any downscale. The scorer measures
stored frames before preview resizing and records `measurement_space`.
Alternatively compare baseline and candidate after identical 32x32 ingest
settings and label both as normalized-target metrics. Native-cell palette
counts and previously downscaled palette counts are not comparable. Resizing
can remove rare colors and thin details; a lower count is not evidence of
better palette consistency. See [scoring guidance](README-score.md).
