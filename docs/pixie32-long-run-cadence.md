# P32 long-run reporting cadence

The proposed next-run cadence is full eval every **250,000** images, a sample
grid every **500,000**, and recovery saves every **100,000** initially. These
are **untested recommendations**, not new defaults. Keep the running training
unchanged. Use a fresh-folder branch if the operator chooses to change cadence.
Keep b80, LR `1e-4`, 1,500-step warmup, 12M-image WSD budget and the final 2M
decay to `1e-5`; changing LR alongside kernels would confound the comparison.

Compared with periodic reporting every 100k, this removes **60%** of periodic
eval events and **80%** of periodic grids (**calculated**, ignoring finalization
and alignment). The trainer retains a final eval and grid. Reporting's measured
73% share of a 100-update smoke segment includes mandatory finalization; it
does not establish the long-run share or predict a 73% saving.

At the operator's **measured** 0.221-second median b80 update, 100k/250k/500k
views correspond to approximately **4.6 / 11.5 / 23.0 minutes of update time**
(**estimated**, using a median as a rate and excluding reporting/startup).
Successful periodic recovery at 100k therefore limits lost update work to roughly
five minutes under those assumptions (about half that on average for uniformly
timed interruptions). Do not switch saves to 250k without accepting
that roughly twelve-minute recovery window. Finalization and best-model writes
are additional checkpoint work.

The current save flag couples one permanent model snapshot and a full mutable
`last.pt`. At 100k, a 12M recipe produces 120 periodic model snapshots; model
tensor storage alone is about **31.9 GB decimal**, **calculated** from
120 × 265,697,280 bytes. Actual serialized files differ. This proposal does not
delete any files. A later isolated change can decouple model retention (for
example every 1M) from full recovery (100k), after measuring write pauses and
agreeing a recovery budget. Never remove an existing checkpoint to save space.

Read a completed `paused.json` or `train_summary.json` from the real run:

```bash
"$PY" -m train.pixie32.report_cadence \
  --summary "$RUNS/p32-observed/paused.json" \
  --remaining-images 11000000 --batch 80 --steady-update-seconds 0.221 \
  --eval-every-images 250000 --sample-every-images 500000 \
  --save-every-images 100000 --out "$RUNS/cadence-analysis-new"
```

`11000000` is an **example input**, not the current remaining budget. Supply
the actual target minus `images_seen`. This read-only analysis reports measured
segment phase shares and separately labelled projections from the segment's
per-event means. It accounts for global cadence boundaries and final eval/grid.
Set `--batch` to the projected continuation's batch size; final-batch boundary
crossings suppress duplicate final events just as the trainer does.
If a phase has no measured events, its projection is unknown. Use the supplied
steady rate of **0.221 seconds/update** for this b80 recipe. The report retains
the measured segment mean as evidence but uses the explicit steady rate for
planning, so compile time in a short segment does not inflate run length.
Do not reuse this rate for a different batch or an unmeasured kernel variant.
Without the flag, the tool still labels its fallback as the segment mean.

At b80, the full 12M-view budget is **150,000 updates**, or **9.21 hours of
update time**, calculated from the supplied steady rate. Each remaining 1M
views costs about **46.0 minutes** of updates. These estimates exclude startup,
evaluation, grids and writes. A partial final update is conservatively charged
as a full update. Use the actual remaining views; no current checkpoint count
is inferred here.

The checkpoint phase mixes model, full recovery, final and best-model writes.
It cannot yield a reliable per-cycle cost by dividing its aggregate event
count. Only pass `--checkpoint-cycle-seconds` after measuring an entire periodic
model-plus-full-save cycle. Its projection excludes mandatory final and
best-model writes; the tool deliberately offers no total wall-time forecast.

Measure that periodic write cycle on the target run storage, while the GPU is
idle, using the paused full checkpoint and a fresh output folder:

```bash
"$PY" -m train.pixie32.bench_checkpoint \
  --checkpoint "$RUNS/paused/last.pt" --device cuda:0 --trials 5 \
  --out "$RUNS/checkpoint-write-bench-new"
```

This performs no training. It uses the trainer's actual temporary-save/replace
function to write one model snapshot and one full recovery checkpoint per
trial. It checks round-trip model/optimizer values outside timing and leaves
all ten new files for inspection. The real P32 tensor sizes imply roughly
**5.3 GB** for five pairs (calculated; serialization overhead differs), so choose
storage with room. The command checks free space and never deletes checkpoints.
`--device cpu` measures CPU serialization only and cannot establish the real
GPU-to-host write pause. CUDA mode mirrors the current non-fused optimizer;
fused/capturable checkpoint states are refused until their transfer layout is
explicitly supported.

Use `cycle_write_seconds.median` from `checkpoint-benchmark.json` as
`--checkpoint-cycle-seconds`. This is the trainer-return cost, which can include
OS page-cache buffering. The separate `post_write_file_fsync_seconds` and
`cycle_write_plus_file_fsync_seconds` expose additional storage flush cost; they
do not imply the trainer currently fsyncs or guarantee directory durability.
Disk-cache pressure and background I/O can affect later trials; inspect ranges.
Source load, verification and checkpoint hashing are excluded from timings.
No real-checkpoint write cost was measured in this workspace.

Measure the real long-run share from a naturally completed or paused segment;
do not stop an active run to manufacture this report. For a controlled short
reporting experiment on an idle GPU, the existing `bench_reporting` command
provides five fresh-folder smoke segments. Such smoke measurements still include
mandatory final reporting and are not a long-run cadence benchmark.

```bash
"$PY" -m unittest train.pixie32.test_report_cadence train.pixie32.test_bench_checkpoint
```

No GPU performance was measured here. The tests exercise actual CLI output,
event alignment, unknown-phase handling, invalid timings and overwrite refusal
using explicitly synthetic timings. Holdout files and trainer behavior are
unchanged.
