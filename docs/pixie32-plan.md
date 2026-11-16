# Pixie-32 plan (native 32×32)

Status: approved by the operator on 2026-11-13. This file restates the approved
plan points. The original planning document was not available for this repo, so
treat the bullets below as the binding summary and the details as guidance.

## Goal

A native 32×32 text-to-sprite model in the v3 family, trained on the combined
Krea 2 sprite set (see `docs/dataset-krea2-combined.md`), that keeps the 16×16
v3 line's palette, text encoder and conditioning so the chat worker can serve
both.

## Approved points

1. **Data: filter synthetic, drop game-icons.** Use only sets that passed the
   extraction gates and the R1 spot-check rule. Do not add icon-style packs.
2. **Keep the 4097-entry palette.** Index 0 = transparent, 1..4096 = colours of
   `$PALETTE` (`[4097,3] uint8` `.npy`). Vocab stays
   padded to 4160, ids ≥ 4097 masked. `NO_PALETTE = 4097` for empty palette
   cond slots. 32×32 RGBA is quantized with alpha < 128 → 0 and nearest-RGB
   for opaque pixels (same method as `data-pipeline/pixie32-prep/build_cache.py`).
3. **Warm-start from v3.** Initialise from the best v3 s7 checkpoint (the
   condboost model, 64M views, P50 d512 × L16, model-only `img_064M.pt`, or the
   decayed 40M model). Load every shape-compatible tensor; re-map only the
   position tables (see below). Log which tensors were copied, remapped, or new.
4. **Causal intra-patch head.** Keep 2×2 patches and the slot-aware causal MLP
   head (`SlotCausalHead`, slots TL, TR, BL, BR; slot k sees slots < k). At 32×32
   that is 16×16 = 256 patches, sequence = `[BOS | text 18 | cond 4 | SEP | 255 patches]`
   = 279 positions (v3: 87).
5. **2D-RoPE.** Replace the 1D learned position over 64 patches with 2D axial
   rotary embeddings over (patch_row, patch_col) ∈ 0..15 on q/k. Prefix tokens
   (BOS, text, cond) get the identity rotation. Use the coordinates of the
   *target* patch for each output position, so relative offsets are consistent.
   A partial rotary (e.g. half the head dim, split between row and col) keeps the
   warm-started weights closer to v3; measure it.
6. **Tighter dedup and holdout.**
   - Exact pixel dedup is already done; also dedup on the quantized palette
     tokens (two RGB sprites can collapse to the same tokens).
   - Blind-50 and Eval-50: exclude any row whose normalized caption **or
     subject** equals or contains a holdout phrase (word boundaries), and check
     the CLIP-trimmed (16-token) caption too. Re-run this after every filter.
   - Eval split by sheet (all cells of a sheet together) plus a small
     unseen-subject probe; drop train rows whose caption exactly equals an eval
     caption.

## Position-table warm start (suggested)

| v3 tensor | shape | 32×32 |
|---|---|---|
| `pos.weight[:23]` (prefix) | `[23,512]` | copy |
| `pos.weight[23:87]` (64 patch outputs) | `[64,512]` | upsample 8×8 → 16×16 nearest, or drop in favour of RoPE |
| `row_in/col_in/row_out/col_out` | `[8,512]` | `repeat_interleave(2)` → `[16,512]` |
| everything else | same | copy |

## Out of scope for this run

Non-square aspect ratios, masked/outpaint training, animation frames, and the
16→32 refiner. 2D-RoPE is the shared base those features will build on later.

Future collection options are documented in the
[animation and transparency data proposal](animation-alpha-data-plan.md).
That proposal is separate from this run and does not change its dataset policy.
