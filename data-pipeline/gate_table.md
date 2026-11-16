# Old vs new extractor gate (3x3 @ 576, 6 px per art px)

Labels are **assistant visual judgements**, not ground truth. Set: 72 eval + 306 gen sample = 378 sprites. Judged **365 good / 13 bad**.

## Thresholds

| gate | old (v1, tuned for 8 px) | new (extract_v2) |
|---|---|---|
| key seed tolerance | 90 + global magenta tint | 90 + same tint (unchanged on the 32×32) |
| key widen | (none) | flood-fill from known magenta into near-magenta, grow_tol=120, 6 px, **on the raw 192 cell only** (spill / fg / main_comp) |
| fringe cleanup | tint applied globally in `is_bg` | plus 2 rounds of **outline-only** near-exact magenta (tol≤100) on the 32×32 |
| fg (size) | 0.04 ≤ fg ≤ **0.90** | 0.04 ≤ fg ≤ **1.00** |
| spill | ≤ **0.08** (on v1 key) | ≤ **0.08** (on grown key) |
| main_comp (single-piece) | ≥ **0.85** | ≥ **0.55** |
| purity | ≥ **0.75** | ≥ **0.35** |
| 32px edge_touch | flag only (not a reject) | flag only (not a reject) |

A naive global widen (tol=110 + loose pink/purple tint on the outline) was measured and **rejected**: it punched holes in pink/purple subjects (donut frosting, crystal dragon, nail polish, grapes).

## Counts on the labelled set (judgement)

Positive class = **keep** (`ok`).

| | old v1 | new v2 |
|---|---:|---:|
| ok | 162 | 376 |
| purity_fail | 200 | 0 |
| rejected | 16 | 2 |
| good kept (recall) | 160 / 365 = **0.438** | 365 / 365 = **1.000** |
| good rejected | 205 | 0 |
| bad rejected (recall of bads) | 11 / 13 = 0.846 | 2 / 13 = **0.154** |
| bad kept (false accepts) | 2 | 11 |
| precision (ok are good) | 160 / 162 = 0.988 | 365 / 376 = **0.971** |
| eval ok | 39 / 72 | **72 / 72** |
| gen ok | 123 / 306 | **304 / 306** |

The old gate looks “better” at rejecting bads only because it also dumped 205 goods (mostly `purity_fail` at 6 px, plus full-frame objects with fg>0.90). The new gate keeps every judged-good sprite and still rejects the two **structural** bads (calculator spill=0.106, fairy spill=0.159). The other 11 judged bads are semantically wrong or softly cropped but structurally clean — size/purity/spill/main_comp cannot catch them without also dropping goods.

## New gate rejects (both judged bad)

| id | phrase | judgement | gate reason |
|---|---|---|---|
| g-004-7 | calculator | cropped | spill 0.106 > 0.08 |
| g-004-8 | fairy | multi-piece (fairy + black silhouette) | spill 0.159 > 0.08 |
