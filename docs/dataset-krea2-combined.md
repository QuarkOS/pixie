# Dataset: Krea 2 combined 32×32 sprites (2026-11-15)

Synthetic 32×32 RGBA sprites cut from Krea 2 Turbo sprite sheets generated on
three GPUs, re-extracted with one extractor (`extract_v2`) so every source gets
the same gates. The full set lives outside this repo at `$DATA_ROOT` and is not released.
Commands that take `$SAMPLE_DIR` expect a small folder with the same layout.

## Counts

| Source | Grid / sheet px | LoRA | Sheets | Raw sprites | Pass extract_v2 | Final | After R1 |
|---|---|---|---:|---:|---:|---:|---:|
| `3080ti` | 3×3 @ 576 | `p3-1500` | 10,000 | 90,000 | 88,997 | 88,997 | 85,745 |
| `3060ti` | 3×3 @ 576 | `p3-1500` | 1,616 | 14,544 | 14,403 | 14,403 | 13,937 |
| `5070ti` | 2×2 @ 512 | `pix32-final` | 8,820 | 35,280 | 34,613 | 34,610 | 34,006 |
| **Total** | | | **20,436** | **139,824** | **138,013** | **138,010** | **133,688** |

- Final = passed extract_v2, not a holdout match (0 matched), not an exact pixel
  duplicate (3 removed, all `5070ti`: blank or crossword-like images).
- After R1: 4,892 subjects, 86,746 unique captions. Re-checked against Blind-50
  and Eval-50 (100 phrases, caption **and** subject, exact or word-boundary
  contains): **0 hits**.
- Sheet generation: Krea 2 Turbo (sd.cpp, iq4_xs, Qwen3-VL-4B text encoder,
  wan 2.1 VAE, 5 steps) on a magenta key background.

## Layout of `$DATA_ROOT`

```
$DATA_ROOT/
  final/                      train-ready set (138,010 sprites); do not modify
    images/<source>/<NNN>/<id>.png   32×32 RGBA, alpha is binary {0,255}
    images/<source>/<NNN>/<id>.txt   caption (one line)
    captions.tsv              "<file_name>\t<caption>" per sprite
    metadata.jsonl            one row per sprite (schema below)
  manifest.jsonl              one row per raw cell (139,824), every verdict
  rejected.jsonl              1,814 cells that failed extract_v2 (with reasons)
  flags_edge_touch.jsonl      77,187 final sprites whose opaque pixels touch the 32 px border
  summary.json                per-source counts (copy in data-pipeline/krea2-combined/reports/)
  spotcheck/                  spot-check outputs and candidate filter lists
    cand_R1_mag_outline<=2_and_no_tiny_islands.txt   133,688 paths (the training list)
    cand_R2_R1_and_purity>=0.5.txt                   133,111
    cand_R3_R2_and_edge_px<32.txt                    119,835
    cand_flag_edge_px_ge32.txt                       13,642 (13,283 of them are in R1)
    cand_magenta_outline_cleanup.txt                 13,931
    cand_mag_outline<=2_only.txt                     135,846
    cand_H_purity0.5_mc0.85_edge8.txt                87,142
  extract/<source>/{ok,purity_fail,rejected}/...     every extracted cell
  raw/<source>/               copied generation runs (sheets/, labels.jsonl, state.json)
  holdout/                    Blind-50 / Eval-50 phrase lists (not in the public tree)
  code/                       the scripts in data-pipeline/krea2-combined/
```

Candidate lists hold paths relative to `$DATA_ROOT` (`final/images/...png`).

### `final/metadata.jsonl`

`file_name` (relative to `final/`), `text` (caption), `id`, `source`, `grid`
(`"3x3"` / `"2x2"`), `seed`, `edge_touch`.

### `manifest.jsonl` (one row per raw cell)

| Field | Meaning |
|---|---|
| `id` | `<source>-s<sheet:06d>-c<cell>` (unique) |
| `source`, `grid`, `sheet_px`, `sheet`, `sheet_id`, `cell`, `seed` | provenance |
| `caption`, `subject`, `cat`, `subject_id` | prompt; subject is the noun phrase it was built from |
| `fg`, `border`, `main_comp`, `n_comp`, `spill`, `purity` | extract_v2 metrics |
| `edge_touch`, `fringe_dropped` | flag; magenta fringe pixels removed |
| `bin` | `ok` / `purity_fail` / `rejected` |
| `pass_extract_v2`, `reject_reasons`, `extractor` (`extract_v2_3x3_576`) | verdict |
| `sprite_path` | extracted cell (`extract/...`) |
| `pixel_hash` | sha1 of the 32×32 RGBA pixels (exact dedup key) |
| `orig_file`, `orig_usable` | verdict of the overnight (legacy) extractor |
| `holdout` | null, or `{match: exact/contains, phrase, sets}` |
| `dup_of` | id of the first sprite with the same `pixel_hash` |
| `final`, `status`, `final_path` | in `final/` or not, and why |

`rejected.jsonl`: `id, source, status, bin, reject_reasons, holdout, dup_of,
caption, sprite_path`. `flags_edge_touch.jsonl`: `id, final_path, caption`.

## extract_v2 gates (3×3 @ 576, 6 screen px per art px)

| Gate | extract_v2 | Legacy overnight gate |
|---|---|---|
| `fg` (foreground fraction) | 0.04 – 1.00 | 0.04 – 0.90 |
| `spill` | ≤ 0.08 | ≤ 0.08 |
| `main_comp` | ≥ 0.55 | ≥ 0.85 |
| `purity` | ≥ 0.35 | ≥ 0.75 |
| `edge_touch` | flag only | flag only |

Also: magenta key tolerance 110 (was 90) and an outline-only fringe cleanup
(2 rounds) that removes magenta-tinted outline pixels without touching
interior pinks and purples.

## Spot-check findings

- The legacy purity ≥ 0.75 gate was **tuned for 8 screen px per art px** and
  is miscalibrated for 3×3 @ 576 (6 px per art px): purity there has a median of
  ~0.73 (p5 0.585, p95 0.857), so the old gate rejected about half of the
  3×3 cells. On the `3080ti` run, 51,698 cells failed **only** purity < 0.75;
  in total 65,839 cells were rejected by the legacy gates but kept by extract_v2.
- In a visual check of 200 sprites from that group, about 3% were clearly bad
  (about 7% counting borderline ones), against about 2% in 100 sprites that
  passed both gate sets. These are visual estimates from small samples, not
  measured rates.
- Remaining defects: faint magenta specks on outlines (13,931 sprites with at
  least one magenta outline pixel) and tiny stray pixel islands (2,518 sprites
  with a component of ≤ 3 px).
- 77,187 final sprites touch the 32 px border (median 13 border pixels among
  those). Dropping all of them would leave 60,823, so they are kept.

## R1 rule (the training list)

Keep a final sprite if

- it has **≤ 2 magenta outline pixels**: opaque pixels with `g < 110, r > 140,
  b > 140, |r − b| < 90` that are adjacent to a transparent pixel, and
- it has **no tiny islands**: no connected component of opaque pixels (scipy
  `ndimage.label` on alpha > 0) of ≤ 3 pixels.

Result: 133,688 sprites (`spotcheck/cand_R1_mag_outline<=2_and_no_tiny_islands.txt`).
The definitions are in `spotcheck.py` (`mag_outline_px`, `small_comp32`) and
`spotcheck2.py`. The list itself was written from those per-sprite stats;
recompute them to reproduce it.

## Edge-heavy sprites: low-weight bucket (optional)

`cand_flag_edge_px_ge32.txt`: sprites with ≥ 32 opaque pixels on the 32 px
border (13,642; 13,283 are in R1). Many are cropped or full-frame objects.
Suggested handling: keep them, but sample them at reduced weight (for
example 0.5× in the per-row sampling weights, combined multiplicatively with
the rare-subject balance weight and renormalised to mean 1). If the loader has
no cheap per-sample weights, train on R1 unweighted and note it.

## Scripts

`data-pipeline/krea2-combined/`:

- `build_combined.py`: re-extracts every raw sheet with extract_v2 (resumable,
  per-group, multiprocess), writes one row per cell.
- `extract_v2.py`: the extractor used here (3×3 @ 576 gates above).
- `finalize.py`: holdout filter, exact pixel dedup, `final/`, manifest, contact sheet.
- `run_all.sh`, `run_rest.sh`: resumable drivers.
- `spotcheck.py`, `spotcheck2.py`: read-only spot checks; write to `spotcheck/` only.
- `tests/`: extraction checks used while building (geometry, memo equivalence).
- `reports/`: `summary.json`, `spotcheck_summary.json`, `spotcheck2_summary.json`, `options_union.json`.

Paths use `$DATA_ROOT` (this folder), `$GEN_ROOT` (generation run root) and
`$PY` (Python with numpy, Pillow and scipy).
