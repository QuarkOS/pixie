# Pixie-32 real-sprite prep (earlier work, 2026-11-13)

Earlier 32×32 data prep from CC0 real-sprite sources (pixl data, Kenney,
OpenGameArt): fetch, filter, dedup, palette-quantize to the 4097 palette
(`build_cache.py` → `img_idx.npy [N,1024] uint16`), and CLIP text caches for
32×32 rows (`build_clip_text_cache*.py`). Recaptioning scripts are not included.

Not used by the Krea 2 combined run, but `build_cache.py` (quantization) and
`build_clip_text_cache_v2.py` (18-position CLIP features) are the reference
implementations for the 32×32 caches. Paths use `$PIXIE_ROOT`.
