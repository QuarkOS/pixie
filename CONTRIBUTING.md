# Contributing

Pixie is the source for a small pixel-art sprite model, the data tools around
it, and a local web app. The code in this repository is under the Apache
License 2.0. See [LICENSE](LICENSE). Checkpoints and the training corpus are
not included, and their licenses are tracked in [MODEL_CARD.md](MODEL_CARD.md).

## What to send

- A focused change with a short explanation of why.
- Tests for behaviour you change. Python checks live in `train/test_v3.py`
  and `train/pixie32/test_*.py`. Web checks live in `web/tests/`.
- Commands you actually ran, and numbers you actually measured. Label
  estimates as estimates.

## What not to send

- Checkpoints, datasets, sheet archives, CLIP or Krea weights, `.env` files,
  logs, or API keys. `.gitignore` is there so those stay local.
- Training runs that read the Blind-50 / Eval-50 holdout files. They are not in
  this tree (put your own in `holdout/` or point `PIXIE_HOLDOUT_DIR` at them;
  tests that need them skip otherwise). Those files are evaluation-only. Do not add their phrases to a training
  list, a typo augmentation pass, or a generation queue.
- A new training run written into someone else's output folder. Pick a new
  directory.
- Hostnames, home-directory paths, account ids, or credentials. Take paths
  from environment variables (`PIXIE_ROOT`, `DATA_DIR`, `DATA_ROOT`, and the
  flags on each script).

## Local setup

Use a project virtual environment. Do not install PyTorch into the system
Python. The web app reads `web/.env.example`; copy it to `web/.env` and keep
the copy untracked.

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m unittest discover -s train/pixie32 -p 'test_*.py'
```

GPU training and sheet generation need local weights and a dataset. The
README lists the variables. If you do not have those files, review and test
the code paths that run without them.

## What this tree does not redistribute

- Krea 2, Qwen3-VL, Wan 2.1, and CLIP weights. The code downloads or loads
  them from paths you set. Their licenses are upstream. Training and sampling
  need CLIP; sheet generation also needs the Krea stack. The unit tests do
  not.
- Artist packs named under `data-pipeline/animation/` (LPC, Kenney, itch.io,
  OpenGameArt). The code stores a source URL and a hash after you download
  a pack yourself. Kenney's public packs are CC0; other packs are not all
  CC0. Nothing in this repository is those files.
- The Pixie training sprites (generated with Krea 2 Turbo), the Blind-50 /
  Eval-50 holdout lists, and the preference-vote fit behind
  `worker/feat_linear.json`. Without that file the worker's fallback scorer is
  neutral; set `PIXEL_FEAT_LINEAR` to use your own fit.
