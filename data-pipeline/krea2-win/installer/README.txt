Krea 2 32x32 sprite generator - Windows one-click package (target: RTX 5070 Ti 16 GB)
==================================================================================

1. Unzip anywhere with >= 20 GB free (a short path without spaces is safest).
2. Double-click install.bat. It
   - checks the NVIDIA driver supports CUDA 12.8+ (needed for RTX 50 / Blackwell),
   - downloads (resumable, SHA256-checked, ~9.9 GB):
       stable-diffusion.cpp master-1047-984eb06 Windows CUDA 12 build (built with CUDA 12.8.1, includes sm_120)
       + CUDA runtime DLLs (same release), uv 0.13.8 (portable Python 3.12),
       krea2_turbo-iq4_xs.gguf, Qwen3-VL-4B-Instruct-Q4_K_M.gguf, wan_2.1_vae.safetensors
   - installs Python + numpy/pillow/scipy inside this folder only (no global installs, no admin),
   - creates desktop shortcuts "Krea2 Sprites - Start" / "- Stop",
   - starts generation and opens the dashboard.
   Re-running install.bat is safe; it skips finished downloads.
3. Dashboard: http://127.0.0.1:8765  (sprites total, usable %, sprites/s now/avg, sheets/min,
   GPU util/VRAM/temp/power, queue + ETA, graphs, latest sprites).

Output: data\sprites32\ok\NNN\*.png  (true 32x32 RGBA, magenta keyed out; NNN = sheet_id // 1000, max 1000 sheets per folder)
        data\sprites32\rejected\NNN\*.png (failed the automatic usability check)
        data\sheets\NNN\*.png (raw 512x512 2x2 sheets), data\events.jsonl (per-sheet stats incl. per-cell scores)
Logs:   logs\ and install.log

QUEUE: queue.jsonl = 20,000 sheets = 80,000 sprites (~10 h at ~2.2 sprites/s - estimate, unmeasured on 5070 Ti),
from dataset2m/gen_prompts.py --n 80000 --grid 2 --seed 20261114 (deterministic; Blind-50/Eval-50/probe subjects excluded;
no repeated subject group within a sheet). Each line has the sheet prompt, seed and the 4 cell phrases.
Labels: data\labels.jsonl, one line per sprite: file (relative to data\sprites32), phrase (the label), subject, cat,
sheet_id, cell, seed, usable, sheet. Written together with the per-sheet event, so resume never duplicates labels.
To use a plain subject list instead, remove "queue" from config.json (then subjects.txt is used, 4 per sheet).
The runner resumes: finished sheets are skipped. Delete data\ to start a fresh run.
When the queue is done, sd-server is stopped (GPU freed) and the dashboard shows "finished".

HOW TO STOP (e.g. in the morning): double-click "Krea2 Sprites - Stop" on the desktop (or Stop-Krea2.bat).
Nothing is lost; "Krea2 Sprites - Start" continues where it stopped. While running, the PC is kept awake
(runner.py blocks sleep, no admin needed). Keep the black launcher window open/minimised: it restarts
sd-server if it crashes. Tip: also pause Windows Update for the night so it doesn't reboot.

DISK: needs ~10 GB for models/tools + ~5 GB output for the full queue (estimate):
  raw sheets ~220 KB each (lossless PNG re-compressed; measured on our 3080 Ti sheets) x 20,000 = ~4.4 GB
  sprites ~1.6 KB each but ~4 KB on disk (NTFS cluster) x 80,000 = ~0.3 GB
  labels.jsonl ~20 MB, events.jsonl ~25 MB.   Plan for >= 20 GB free in total.
  (Set "recompress_sheets": false in config.json to skip re-compression: ~345 KB/sheet = ~6.9 GB.)
Unattended: if sd-server crashes, the launcher restarts it; failed sheets are retried on next start.

Pipeline (same as the measured 3080 Ti setup): 2x2 sheet @ 512x512, 256 px cells, 5 Euler steps,
fixed sigmas 1.0,0.9266,0.8257,0.6780,0.4412,0, txt CFG 1.0, our LoRA pix32-final x1.0 applied at runtime,
3 requests in flight (reviewer safe default; 2 was measured on the 3080 Ti), cell -> 32x32 via mode pooling + palette cleanup + magenta key.

5070 Ti choices (ALL UNTESTED on that card - no measurements exist):
  - everything on GPU, no CPU offload, model stays loaded, conditioning cache 64, 3 in flight, --flash-attn on,
    CUDA_MODULE_LOADING=LAZY, CUDA_DEVICE_ORDER=PCI_BUS_ID, AboveNormal priority (sd-server + runner), no -v logging.
  - If sd-server goes down, the runner waits and retries the SAME sheet (never marks it failed for that).
  - optional higher-quality text encoder: run  install.bat -Q8TextEncoder  then set
    "text_encoder": "Qwen3-VL-4B-Instruct-Q8_0.gguf" in config.json (expected ~+1.8 GB VRAM; quality/speed effect unmeasured).
  - "extra_server_args" in config.json holds extra sd-server flags (default "--flash-attn on"; never use --diffusion-fa).
  No higher-precision Krea 2 turbo GGUF than IQ4_XS is published (nvfp4 exists but is untested with sd.cpp/LoRA).

Our LoRA (loras\pix32-final.safetensors) is private - do not share/upload it.
