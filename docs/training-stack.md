# Training and inference stack

How the Pixie training, sprite-generation (Krea 2) and chat-inference stack is set up, with the numbers measured on the real machines. Use it to find and propose optimizations.

Scope and rules:

- Gathered read-only on 2026-11-15. Every number below was measured on these machines unless it is marked **(estimate)** or **not measured**. Host inventory that would identify a specific computer (CPU model, kernel build, driver build, disk sizes) is not included.
- No credentials, hosts or remote access are included or needed. Machines are named by role. Paths use placeholders:

| Placeholder | Meaning |
|---|---|
| `$PIXIE_ROOT` | Code, venv, v3 data, CLIP cache and v3 runs on the training box's SSD |
| `$DATA_ROOT` | Long-term HDD on the training box (new datasets, Krea output, Pixie-32 runs) |
| `$RUNS` | A run parent folder (v3 runs live under `$PIXIE_ROOT/runs`, Pixie-32 runs under `$DATA_ROOT/pixelai-runs`) |
| `$KREA_ROOT` | sd.cpp binaries, Krea models and the generator app on a given box |
| `$CHAT_ROOT` | Chat worker install on the inference box |

## 1. Machines

| | Training box (12 GB GPU) | Inference box (8 GB GPU) | Helper box (16 GB GPU) |
|---|---|---|---|
| Role | Pixie training + bulk Krea 2 generation | Live chat inference; Krea 2 overnight when chat is paused | Part-time Krea 2 via the Windows installer |
| GPU / VRAM | 12 GB consumer GPU (11,961 MiB visible to CUDA) | 8 GB consumer GPU | 16 GB consumer GPU (Blackwell, `sm_120`) |
| CPU | not published | not published | not recorded |
| RAM | 16 GB (+8 GB swap) | 32 GB | not recorded |
| OS | Linux, headless | Windows desktop | Windows |
| NVIDIA driver | not published | not published | installer requires driver CUDA ≥ 12.8 |
| Python / torch | uv venv, Python 3.12.3, torch 2.9.0+cu128 (CUDA 12.8, cuDNN 9.10.2), triton 3.5.0; uv 0.13.7 | Chat venv: Python 3.12.8, torch 2.9.0+cu128, torchvision 0.24.0+cu128 | Installer venv via uv 0.13.8, Python 3.12 (stdlib + numpy/Pillow for the generator) |
| Disks | local SSD for the checkout and a separate data disk | local disks | not recorded |

Other box settings that matter:

- Training box: sleep/suspend masked, NVIDIA persistence mode on, user lingering on (user services run without a login). There is no desktop session, so the whole 12 GB is free for one job.
- Inference box: a desktop with normal apps open, which used about 1.6 GB of VRAM before the Krea fit test.
- Only one heavy GPU job per card. Launchers refuse to start when `nvidia-smi` shows any compute app on the selected GPU.

### Where data lives (training box)

```
$PIXIE_ROOT/                     # checkout, venv, v3 data, caches, runs
  .venv/                         # uv venv used by training and tools
  v3/                            # v3 code copy, data (npz), caches, launchers, results/
    clip/                        # CLIP-L text weights (11 layers, fp32)
    tok/tokenizer.json
    clipcache_s7_t18/            # v3 CLIP feature cache, 74.9 GB fp16
  data/subset150k/palette.npy    # 4097-colour palette [4097,3] uint8
  runs/<run>/                    # v3 runs (checkpoints, logs, samples)
  chat-worker-*/                 # old Linux chat worker trees (service disabled)
  tools/push_runs.py             # admin panel sync (see 2.6)
$DATA_ROOT/                      # HDD
  krea2-gen-3080ti/              # Linux generator: app/, data/{sheets,sprites32,events.jsonl,labels.jsonl}, logs/
  krea2-3x3/                     # 3x3 LoRA training output
  pixie-krea2-combined-20261115/ # combined 32x32 dataset (see docs/dataset-krea2-combined.md)
  pixie32-ds-krea2-r1-20261115/  # built Pixie-32 training arrays (299 MB)
  pixelai-runs/<run>/            # Pixie-32 runs
  pixie32-logs/                  # Pixie-32 launch scripts, smoke and run logs
$KREA_ROOT/                      # sd.cpp build (sd-server, sd-cli, lib/), models/ (9.0 GB), Python env for the generator
```

## 2. Training

### 2.1 Pixie-32 (current)

Code: `train/pixie32/` on the PR branches (`feat/pixie32-native`, then `fix/pixie32-review-followup`). Launcher `train/pixie32/run_p32.sh` takes all paths from env (`PY`, `CLIP_DIR`, `TOKENIZER`, `DATASET_OUT`, `RUNS`, `RUN_NAME`, `GPU`, `V3_CKPT`) and refuses a busy GPU or an existing run folder.

Config of the full run launched on the training box:

```
--batch 64 --no-gradient-checkpointing --ce-chunks 16
--images 12000000 --decay-images 2000000 --warmup 1500 (default)
--lr 1e-4 --lr-min 1e-5        # WSD: warmup, constant, then linear decay over the last 2M images
--v3-ckpt <v3 condboost 64M checkpoint>   # warm start, fresh optimizer
```

Eval and saves every 100k images (defaults). Text goes through **online CLIP** each step; Pixie-32 has no CLIP feature cache yet. Data is a set of small `.npy` arrays (32×32 `uint16` palette indices, CLIP id rows of 18 `uint16`, cond and split arrays) plus `records.json` and a sha256 `manifest.json`. The whole built dataset is 299 MB, so it fits in RAM; disk IO is not a factor for Pixie-32.

Smoke tests on the RTX 3080 Ti (100 measured steps unless noted; `peak` covers training, eval and sampling):

| Batch | Grad checkpointing | step/s | images/s | Peak allocated | Peak reserved |
|---:|---|---:|---:|---:|---:|
| 32 | on | 4.474 (200 steps) | 143.2 | 2.09 GB | 2.28 GB |
| 64 | on | 2.900 | 185.6 | 3.07 GB | 3.58 GB |
| 96 | on | 2.021 | 194.0 | 4.05 GB | 4.85 GB |
| 128 | on | 1.595 | 204.2 | 5.04 GB | 6.06 GB |
| 192 | on | 1.076 | 206.6 | 7.01 GB | 8.94 GB |
| 256 | on | 0.816 | 209.0 | 8.99 GB | 11.59 GB |
| **64** | **off** | **3.486** | **223.1** | **9.07 GB** | **9.55 GB** |
| 128 | off | OOM | – | – | – |

(1 GB = 10^9 bytes.) The first two "no checkpointing" attempts did not pass the flag, so their numbers matched the checkpointed rows; only the rows marked "off" really had it off.

Long run with that config (PR #4 code): 5,492 steps / 351k images at **3.551 step/s, 227.3 images/s**, peak 9.07 GB allocated / 9.55 GB reserved, iid NLL 0.527 at the point it was stopped for the PR #5 restart. `nvidia-smi` showed 97% GPU utilization during a Pixie-32 run (single sample). The PR #5 code, eager, batch 64, 300-step smoke: 3.521 step/s. `--compile` smokes (batch 64/96/128) were running when this was written; no results yet.

At 227 images/s, the 12M-image run takes about 14.7 h (estimate from the measured rate, before eval pauses).

### 2.2 v3 / s7 (16×16, live model line)

Recipe and flags are in `AGENTS.md` and `train/run_v3_full_linux.sh`. Speed history on 12 GB cards, batch 128:

| Setup | step/s | Note |
|---|---:|---|
| Same box under Windows, eager, pilot config | 6.31 | flash SDPA not available on Windows ("No available kernel") |
| Linux, eager, same config | 6.65 | +5% from the OS alone |
| Linux, `torch.compile` | 8.35 | +32% vs Windows; peak 7.3 GB; NLL matched |
| Linux, compile + CUDA graphs | 8.44 | +1%, not worth it |
| Linux, compile, batch 192 | 5.73 (1,100 img/s) | +3% images/s, not worth the LR change |
| Linux, compile, batch 256 | OOM | needs about 13 GB without chunked CE |
| s7 items, online CLIP (bf16, compiled, 16 tokens, blank rows skipped) | 8.90 | peak 6.5 GB |
| + 4-way chunked CE | 8.65 | slower; saves 0.3 GB |
| **s7 with cached CLIP features** | **9.60** | used for the full run; prefetch wait 0.01 ms/step |
| cached + chunked CE | 9.30 | not used |
| cached, static dummy batch | 9.61 | data path and RNG add about 0 overhead |

The full s7 run averaged 9.59 step/s (about 1.2k images/s) with peak 6.48 GB. Fine-tunes on it (condboost) ran at 9.54 step/s, 6,477 MiB.

CLIP feature cache: 2,867,034 distinct captions (main, synonym, typo, subject and blank) at 17 positions, fp16, 74.9 GB, built in 688 s. It sits on the USB SSD. Cold random 26 KB reads gave 2,835 rows/s with 1 thread and 11,944 rows/s with 6 threads, which is about 104 batches/s against roughly 10 needed, so the cache is not the bottleneck.

### 2.3 Where IO hurt before

- An older pixel-token trainer did an `fsync` every step while writing to the HDD. That, not the CPU, was the bottleneck, and the run used `--no-fsync`.
- The root SSD is USB-attached and 99% full. Anything large (datasets, runs, Krea output) has to go on the HDD. A new v3-style CLIP cache would not fit on the SSD now.
- The HDD is slow for random access. Pixie-32 avoids it by loading its arrays into RAM.

### 2.4 Process supervision

- Training runs in `tmux` sessions on the training box, one per run, each with its own new output folder. Launchers check that the GPU is free and refuse to reuse a run name.
- Live chat used to run on this box as a systemd user service with `Restart=always`. It is now stopped and disabled so the 12 GB card is free. It needed about 4.5 to 6 GB, so it cannot share the card with batch-128 training.
- The Linux Krea generator is a systemd user service (`krea2-gen-3080ti.service`, `Restart=on-failure`) wrapping `start.sh`. That script supervises `sd-server`, the dashboard and the runner, restarts `sd-server` if it dies, and frees the GPU when the queue is done.

### 2.5 Checkpoint backup

A daily copy-only pull at 04:00 copies listed run folders from the training box to the inference box's HDD (it skips same-size files and never deletes).

### 2.6 Admin panel sync (`push_runs`)

`tools/push_runs.py` runs from a systemd user timer every 5 minutes (oneshot, `Nice=10`, idle IO class). It reads run folders under `$PIXIE_ROOT/runs` read-only, compares against a baselines file and pushes run history to the web app's admin training panel with an ingest token read from a file. Pixie-32 runs write to `$DATA_ROOT/pixelai-runs`, which this job does not watch yet.

## 3. Sprite generation (Krea 2 Turbo via sd.cpp)

Engine: `stable-diffusion.cpp` `sd-server`, release `master-1047-984eb06`. On Windows this is the CUDA 12.8.1 build, which includes `sm_120`. Models: `krea2_turbo-iq4_xs.gguf` (6,500 MB), `Qwen3-VL-4B-Instruct-Q4_K_M.gguf` text encoder (2,376 MB), `wan_2.1_vae.safetensors` (242 MB), 9.1 GB of params in total. Server flags: `--lora-apply-mode at_runtime --disable-image-metadata --conditioning-cache-size 64`, loopback only. 5 sampling steps.

| Layout | Size | LoRA | Cells |
|---|---|---|---|
| 2×2 | 512 | `pix32-final` (464/528 tensors applied) | 4 sprites/sheet |
| 3×3 | 576 | `p3-1500` | 9 sprites/sheet |

Generator app (`data-pipeline/krea2-gen`, `data-pipeline/krea2-win/installer`): `runner.py` sends N requests in flight, recompresses each sheet losslessly, cuts the cells (`extract.py`), then appends one line per sheet to `events.jsonl` (`t`, `sheet`, `gen_s`, `extract_s`, `sprites`, `usable`) plus per-cell `labels.jsonl`. Resume skips sheets already logged without error. `dashboard.py` reads those files and polls `nvidia-smi` every 2 s, but it keeps GPU samples **in memory only**, so GPU util, VRAM, temperature and power are lost when it stops.

### 3.1 Measured throughput per GPU

| GPU | Layout | in flight | Wall s/sheet | Sprites/s | Usable (auto gate) | Notes |
|---|---|---:|---:|---:|---|---|
| RTX 3080 Ti (Linux) | 3×3 @ 576 | 2 | 3.648 | 2.467 | 33,058 / 90,000 with the old gate | 10,000 sheets in 10.13 h, 0 errors. Request median 6.14 s, extract median 1.15 s (CPU). 10,414 MiB VRAM. All weights on GPU. |
| RTX 3080 Ti (Linux) | 2×2 @ 512 | – | 3.07 | 1.30 usable/s | – | earlier baseline; 10.2 GB VRAM |
| RTX 5070 Ti (Windows) | 2×2 @ 512 | 3 | 4.805 | 0.832 | 33,600 / 35,280 (95.2%) | 8,820 sheets in 11.77 h, 0 errors. Request median 13.35 s, extract median 0.38 s. `--flash-attn on`. GPU stats not kept. |
| RTX 3060 Ti (Windows) | 3×3 @ 576 | 1 | 12.50 | 0.72 | 5,522 / 14,544 with the old gate | 1,616 sheets in 5.61 h. Text encoder on CPU. Per sheet (medians): text encode 2.37 s, sampling 6.81 s, VAE decode 0.85 s. |

The 5070 Ti was **slower** than the 3080 Ti on the same 2×2 setup (4.80 vs 3.07 s/sheet), even though it has about the same memory bandwidth and much more compute. Its settings (`in_flight 3`, `--flash-attn on`) were never benchmarked on that card, and its server logs were not kept, so the cause is unknown.

### 3.2 RTX 3060 Ti lessons

- **Fit in 8 GB:** `--backend te=cpu --vae-tiling --conditioning-cache-size 4` fits without `--offload-to-cpu`. Peak was 7,820 MiB used with the desktop holding about 1.6 GB. First-sheet warmup took 327 s (2×2) and 548 s (3×3).
- **Cached-prompt timing trap:** the fit test reused one prompt for every timed sheet, so the conditioning cache skipped text encoding. It measured 8.98 s/sheet for 3×3 (8.05 for 2×2). With unique prompts overnight, the steady request time was 10.7 s and the wall time 12.5 s/sheet, because the CPU text encoder alone costs about 2.4 s per new prompt. Always benchmark with unique prompts.
- **Model load from the HDD:** with models on the 4 TB HDD, sd.cpp's loader crawled (the progress bar showed 0.01 to 0.05 MB/s, and tensor-load steps took a median 282 s). Moving the models to the NVMe SSD fixed it (0.88 s median per load).
- **Task priority starvation:** Windows Scheduled Tasks default to priority 7 (below normal), which starved the run while the desktop was busy. The Krea task now runs at priority 4, and the script raises `sd-server` and the runner to AboveNormal (dashboard BelowNormal).
- The first 200 sheets averaged 16.0 s (median 11.8, max 291 s, including warmup). After that the median held at 10.7 s per request for the rest of the night.

### 3.3 Extraction (`extract_v2`)

`data-pipeline/extract_v2.py` (+ `reextract.py`), tuned for 3×3 @ 576 (6 screen px per art px). It keys magenta from a seed rule on the 32×32 (tolerance 90 plus a magenta-tint rule) and widens by flood fill from keyed magenta into near-magenta on the raw 192 px cell (`grow_tol=120`, 6 px). It then strips the outline fringe twice at tol ≤ 100. Gates: `fg` 0.04 to 1.00, `spill` ≤ 0.08, `main_comp` ≥ 0.55, `purity` ≥ 0.35. `edge_touch` is a flag only. Re-cutting all 20,436 overnight sheets with these gates gave 139,824 raw cells and 138,010 final sprites after the gates, holdout and duplicate removal (details in `docs/dataset-krea2-combined.md` on the handoff branch). Extraction runs on the CPU. On the 4-core training box it is about 1.15 s per 3×3 sheet, about a third of the per-sheet GPU wall time.

## 4. Chat inference

- **Where:** the inference box's RTX 3060 Ti. The worker runs from `$CHAT_ROOT` with its own venv and starts from a Scheduled Task at logon (interactive user session). A `.cmd` watchdog restarts it 5 s after any exit except code 3 (lock port taken, which means another worker is running), and Task Scheduler restarts the task on failure (10 times, 1 min apart). A `HOLD` file makes a loaded worker stop polling without exiting.
- **Model:** v3 s7 condboost 64M checkpoint, CLIP-L text, CFG 2.0, temperature 0.8, top-k 32. It samples 64 candidates per job and ranks them with the reward-v7 scorer (fallback `feat_linear`), then returns the top 4. `fast_sample.py` uses CUDA graphs on the live path.
- **Job flow:** the worker long-polls `POST /api/worker/next` (server holds up to 20 s, client timeout 40 s), runs the job and posts `/api/worker/result`. The web app runs one job at a time with up to 12 waiting (`CHAT_QUEUE_MAX`), a 60 s job timeout, and drops queued jobs whose page stopped polling for 45 s. Browsers long-poll `/api/chat/status?wait=1` (25 s).
- **Measured latency on the RTX 3060 Ti:** 1,108 to 1,783 ms per job end to end inside the worker, of which sampling is 517 to 790 ms (11 logged jobs). Round trip from submit to result in the backup test was 1.83 to 2.13 s.
- **VRAM:** about 4.5 to 6 GB measured on the 12 GB card (`AGENTS.md`). Not measured on the 3060 Ti. Chat and Krea can't share the 8 GB card, so chat is paused for overnight Krea runs.

## 5. Known bottlenecks and open ideas

Measured bottlenecks:

1. **Pixie-32 throughput is about 5× lower than v3 in images/s** (227 vs about 1,200), at 4× the tokens per image. Batch 64 without checkpointing uses 9.5 GB, and batch 128 without checkpointing runs out of memory. Turning checkpointing on frees memory but costs more than bigger batches gain (best checkpointed result: 209 images/s at batch 256).
2. **Pixie-32 runs CLIP online every step.** On v3, caching CLIP features gave +8% step/s (8.90 → 9.60). A cache for the Pixie-32 captions (about 89k distinct captions plus augment variants) would be small next to the 74.9 GB v3 cache. It still has to go on the HDD or the RAM budget (16 GB), because the SSD is full.
3. **`torch.compile` for Pixie-32** is untested at the time of writing. It gave +26% on v3 (6.65 → 8.35 step/s).
4. **Krea on the 5070 Ti underperformed** the 3080 Ti on the same layout (4.80 vs 3.07 s/sheet). Next time, benchmark `in_flight` 1/2/3 with and without `--flash-attn`, using unique prompts, and keep the sd-server logs and an `nvidia-smi` CSV.
5. **CPU-bound pieces on the training box (4 threads):** Krea cell extraction (1.15 s per 3×3 sheet) and the data side of training. The extraction work could move off the training box or overlap better with generation.
6. **3060 Ti Krea:** the CPU text encoder costs about 2.4 s of each 10.7 s request. The diffusion model is 6.5 GB, so moving the text encoder to the GPU needs a smaller TE quant or offloading something else (not measured).
7. **Dashboard GPU stats are not persisted.** Appending each 2 s `nvidia-smi` sample to a `gpu.jsonl` would make every run reconstructable.
8. **Disk:** the training box root SSD has 2.9 GB free. Any new cache or run folder must go on the HDD, and anything that writes to the SSD can fail mid-run.

Ideas not yet measured, so no expected numbers:

- bf16 autocast review and fused optimizers for Pixie-32 (check what is already on).
- A shared "unique-prompt" Krea benchmark script per GPU so per-card configs get chosen from data.
- Persist `push_runs` coverage for the HDD runs folder so Pixie-32 runs appear in the admin panel.
