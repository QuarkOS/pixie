# 3x3 @ 576 rejection/acceptance calibration

Time: 14 Nov 2026, 21:27 local time.

The running 3080 Ti job was **not stopped or written to**. Nothing was deleted. The new gate is a **separate** extractor (`extract_v2.py`) plus a re-extract script that was dry-run on the sample into a new folder.

## Labelled reference set (assistant visual judgement)

378 sprites: 72 eval (`sample_eval/p3_s1500_m1.0`, p3-1500 / 3x3 @ 576) + 306 from 34 finished raw sheets of the 3080 Ti job (copied read-only to `sample_gen/`). I inspected the contact sheets and individual 32×32 previews.

**These labels are my judgement, not ground truth.** Taxonomy: good, or bad with reason in {blob, wrong subject unclear, cropped, fringe, background kept, multi-piece}.

| source | good | bad | total |
|---|---:|---:|---:|
| eval | 72 | 0 | 72 |
| gen sample | 293 | 13 | 306 |
| **all** | **365** | **13** | **378** |

Judged **bad** (13):

| id | phrase | reason | note |
|---|---|---|---|
| g-004-1 | colored pencil set | cropped | box sides cut off |
| g-004-7 | calculator | cropped | fills the cell, edges cut off |
| g-004-8 | fairy | multi-piece | fairy plus a separate black silhouette |
| g-007-8 | blue electric shaver on | wrong subject unclear | reads as a closed jar |
| g-010-8 | walking green stegosaurus | wrong subject unclear | reads as foliage |
| g-012-8 | white person, front view | blob | featureless white silhouette |
| g-014-5 | sleeping blue ice dragon | wrong subject unclear | ice mass, not a dragon |
| g-016-2 | black common squirrel monkey | multi-piece | two animals instead of one |
| g-021-2 | brown catfish | wrong subject unclear | reads as a log/branch |
| g-024-5 | red boot down symbol | background kept | boot on an opaque sign plate, not keyed |
| g-028-3 | plastic explosive | wrong subject unclear | two bottles |
| g-032-2 | closed blue store | cropped | sign text cut off (STREE) |
| g-033-7 | paper dynamite stick | wrong subject unclear | plain beige stick |

Eval 72 were all judged good, including white objects, mild 32px edge-touch (frog, bike, tools), and the unpatterned soccer ball. Full-frame but complete subjects (maze, writing desk, checkerboard, emerald block, library, organ, nightstand, whiteboard, broken pill as two halves) were judged **good**. 32px edge-touch is common (256/378 under the old extract, 256/378 under v2: eval 53, gen 203) and is a **flag**, not a reject.

Labels: `labels.jsonl` (`label_is_judgement: true`).

## Fringe cleanup — what was tried, what was kept

Goal: strip magenta-tinted edge pixels and widen key tolerance, then measure the effect.

**Unsafe (measured, not used):** raising the global key to tol=110 and applying a loose magenta tint on the 32×32 outline. That cut leftover-magenta-like pixels but **ate subject colour**:

| sprite | v1 fg32 | unsafe fg32 | delta |
|---|---:|---:|---:|
| jelly donut | 547 | 355 | −192 (frosting punched out) |
| pink nail polish | 572 | 480 | −92 |
| purple crystal dragon | 539 | 421 | −118 |
| fresh grape | 492 | 412 | −80 |

124 / 378 sprites lost foreground vs v1. Contact: `contacts/contact_fringe_safe_variants.png` (last column) and the writeup pair sheet.

**Kept in extract_v2:**

1. Seed key on the 32×32 stays v1 (tol=90 + the original tint `(g<70)&(r>150)&(b>150)&(|r-b|<80)`).
2. **Widen** only by flood-filling from already-keyed magenta into adjacent near-magenta (`grow_tol=120`, 6 px) on the **raw 192 cell**, used for spill / fg / main_comp.
3. **Strip** only near-exact magenta (tol≤100) on the 32×32 outline, 2 rounds.

Effect of the safe strip vs v1 (tight leftover = pixels with max-norm to (255,0,255) ≤ 100 still in the FG):

| | v1 key only | v2 grown score + safe strip |
|---|---:|---:|
| sprites with any outline pixels stripped | 0 | 13 |
| mean pixels stripped | 0 | 0.138 |
| max stripped (jelly donut rim) | 0 | 34 |
| leftover near-exact magenta, sum | 73 | 21 |
| mean 32×32 FG pixel delta vs v1 | 0 | −0.143 |

Pink/purple subjects stay intact (donut −36 rim pixels, still a complete iced donut; dragon −2; polish/grapes/heel/garlic unchanged). Contact: `contacts/contact_fringe_before_after.png`.

The grown key is what makes the **spill** gate usable at 6 px:

| sprite | judgement | v1 spill | v2 grown spill |
|---|---|---:|---:|
| peach | good | 0.167 (old reject) | **0.068** |
| nightstand | good | 0.137 (old reject) | **0.054** |
| calculator | bad / cropped | 0.108 | **0.106** |
| fairy | bad / multi-piece | 0.161 | **0.159** |

So spill≤0.08 can be kept: it no longer false-rejects peach/nightstand, and still catches calculator + fairy.

## Threshold choice

Sweep: purity ∈ {0.35…0.75}, fg_max ∈ {0.90,0.95,0.98,0.995,1.00}, spill_max ∈ {0.08…1.0}, main_comp_min ∈ {0.50…0.85}, fg_min=0.04, on the v2 extract. Objective: first keep all judged-good sprites, then reject as many judged-bad as those features allow, then prefer a readable gate.

Locked thresholds (also the preferred rounded set that keeps all goods):

```
fg_min=0.04  fg_max=1.00  spill_max=0.08  main_comp_min=0.55  purity_min=0.35
```

Why each change vs v1:

- **purity 0.75 → 0.35:** 0.75 was tuned for 8 px/art px. At 6 px, 33/72 eval sprites and 167/306 gen sprites were `purity_fail` but almost all were judged good (eval min purity among goods is well above 0.35; lowest good is scattered hay at 0.364, then squash racket 0.416). A floor of 0.45 would drop hay. Purity is a weak reject signal at 6 px; 0.35 only guards empties / mush below the worst good.
- **fg_max 0.90 → 1.00:** 0.90 rejected complete full-frame subjects (maze, desk, checkerboard, emerald, library, organ, safe, whiteboard). Cropped full-bleed cases (store, pencil set, boot-on-plate) look the same to this feature, so they stay accepted.
- **spill stays 0.08**, but scored on the grown key (see above).
- **main_comp 0.85 → 0.55:** 0.85 rejected the correctly broken pill (main=0.614, two halves). Fairy is still caught by spill (main=0.929, not by this gate).
- **edge_touch:** flag only. Most goods touch the 32 px border.

Full old-vs-new count table: `gate_table.md`. Sweep dump: `sweep.json`. Per-sprite scores and labels: `per_sprite.jsonl`.

## Precision / recall-style counts (judgement labels)

| | old v1 | new v2 |
|---|---:|---:|
| good kept / all good (recall) | 160 / 365 = 0.438 | **365 / 365 = 1.000** |
| bad rejected / all bad | 11 / 13 = 0.846 | 2 / 13 = 0.154 |
| precision of `ok` | 160 / 162 = 0.988 | **365 / 376 = 0.971** |
| eval auto-ok | 39 / 72 | **72 / 72** |
| gen auto-ok | 123 / 306 | **304 / 306** |

Recovered 205 sprites the old gate did not mark `ok` (194 former `purity_fail` goods + 11 former `rejected` goods). The 11 remaining false accepts are listed in the labelled-bad table (all `new=ok` except calculator and fairy). Structural gates will not catch “shaver looks like a jar” without a semantic model.

## Extractor v2 and re-extract (dry-run already done)

- New extractor (does not replace the live job’s `app/extract.py`): `data-pipeline/extract_v2.py` (the copy that was run lived under `$KREA_CALIB_ROOT`)
- Re-extract script (later pass over **all** kept raw sheets, into a **new** folder): `data-pipeline/reextract.py`

Dry-run on the sample, 14 Nov 2026 21:26 local time, into `$KREA_CALIB_ROOT/dryrun_sample/`:

| group | cells | ok | rejected | yield |
|---|---:|---:|---:|---:|
| eval | 72 | 72 | 0 | 1.000 |
| gen | 306 | 304 | 2 (calculator, fairy) | 0.993 |
| total | 378 | 376 | 2 | 0.995 |

Later, after the 3080 Ti job finishes, re-extract **all** raw sheets without touching the live `data/` outputs:

```
python3 reextract.py \
  --jobs "$DATA_ROOT/krea2-gen/data/jobs.jsonl" \
  --src  "$DATA_ROOT/krea2-gen/data/sheets" \
  --out  "$DATA_ROOT/krea2-gen/reextract_v2" \
  --px 32
```

## Contact sheets

| file | what |
|---|---|
| `contacts/contact_eval_72.png` | eval 72 (inspection) |
| `contacts/contact_gen_000-007.png` … `032-033.png` | gen sample (inspection) |
| `contacts/contact_new_ok.png` | all 376 new-gate `ok` (red bar = judged bad) |
| `contacts/contact_new_rejected.png` | both new-gate rejects, with judgement + gate reason |
| `contacts/contact_labelled_bad.png` | all 13 judged bad, with new-gate decision |
| `contacts/contact_new_ok_but_labelled_bad.png` | 11 false accepts |
| `contacts/contact_recovered_old_reject_now_ok.png` | recovered vs old gate |
| `contacts/contact_fringe_before_after.png` | v1 key vs v2 safe strip |
| `contacts/contact_fringe_safe_variants.png` | safe vs unsafe fringe variants |

## Paths on the box

```
$KREA_CALIB_ROOT/
  labels.jsonl
  extract_v2.py
  reextract.py
  run_calib.py
  REPORT.md
  gate_table.md
  sweep.json
  fringe_effect.json
  features.jsonl
  per_sprite.jsonl
  contacts/
  dryrun_sample/          # dry-run output (eval/ + gen/, each with ok|purity_fail|rejected)
  sample_eval/
  sample_gen/             # read-only copy of 34 raw sheets + jobs/labels
  preview_sprites/
```
