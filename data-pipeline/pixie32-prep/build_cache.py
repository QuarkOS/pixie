#!/usr/bin/env python3
"""Build palette-index token cache for Pixie-32. CPU only.

Writes out/cache/img_idx.npy uint16 [N, 1024] using the existing 4097-colour
subset150k palette (0 = transparent). Reports CLIP-feature cache size, does
not build the 70G-style teacher cache (would need CLIP-L; GPU reserved for chat).
"""
from __future__ import annotations

import json, os, time
from pathlib import Path
import numpy as np

ROOT = Path(os.path.expandvars("$PIXIE_ROOT/data32"))
OUT = ROOT / "out"
PAL = Path(os.path.expandvars("$PIXIE_ROOT/data/subset150k/palette.npy"))


def log(*a):
    print(*a, flush=True)


def main():
    t0 = time.time()
    recs = [json.loads(l) for l in (OUT / "meta.jsonl").open() if l.strip()]
    n = len(recs)
    token_bytes = n * 1024 * 2
    # v3 clipcache_s7_t18 feats_t18_f16.bin is 70G for ~1,356,348 rows
    clip_est = int(n * (70 * (1024 ** 3) / 1_356_348))
    import shutil
    free = shutil.disk_usage("/").free
    report = {
        "n": n,
        "token_cache_bytes": token_bytes,
        "clip_l14_text_t18_est_bytes": clip_est,
        "disk_free_bytes": free,
        "will_build_token_cache": True,
        "will_build_clip_cache": False,
        "clip_skip_reason": (
            "Estimated from v3 clipcache_s7_t18 (70 GiB / 1.36M rows). "
            "Needs CLIP-L forwards; skipped so the GPU stays free for chat."
        ),
    }
    if token_bytes + 200_000_000 > free - 8 * (1024 ** 3):
        report["will_build_token_cache"] = False
        report["token_skip_reason"] = "not enough free disk with 8 GiB safety margin"
        (OUT / "cache_estimate.json").write_text(json.dumps(report, indent=2))
        log("skip token cache", report)
        return
    if not PAL.exists():
        report["will_build_token_cache"] = False
        report["token_skip_reason"] = "palette.npy not found"
        (OUT / "cache_estimate.json").write_text(json.dumps(report, indent=2))
        return

    pal = np.load(PAL, allow_pickle=False)  # (4097, 3) uint8
    assert pal.shape[1] == 3
    # packed RGB -> palette id. index 0 is the transparent sentinel.
    packed_pal = (pal[:, 0].astype(np.uint32) << 16) | (pal[:, 1].astype(np.uint32) << 8) | pal[:, 2].astype(np.uint32)
    # map pal[0] only as last resort; opaque pixels should not become 0 if another colour is closer
    pal_rgb = pal.astype(np.int16)
    imgs = np.load(OUT / "images.npy", mmap_mode="r")
    assert imgs.shape == (n, 32, 32, 4), imgs.shape

    # unique opaque colours (this set has millions — nearest must be vectorised)
    opaque = imgs[..., 3] >= 128
    rgb = imgs[..., :3]
    packed = (rgb[..., 0].astype(np.uint32) << 16) | (rgb[..., 1].astype(np.uint32) << 8) | rgb[..., 2].astype(np.uint32)
    uniq = np.unique(packed[opaque]) if opaque.any() else np.array([], dtype=np.uint32)
    log(f"unique opaque colours {len(uniq)}")

    pal_exact = {}
    for i, v in enumerate(packed_pal):
        if i == 0:
            continue
        pal_exact[int(v)] = i
    exact_mask = np.fromiter((int(u) in pal_exact for u in uniq), dtype=np.bool_, count=len(uniq))
    mapping = np.empty(len(uniq), dtype=np.uint16)
    exact = int(exact_mask.sum())
    nearest = int((~exact_mask).sum())
    if exact:
        mapping[exact_mask] = np.fromiter((pal_exact[int(u)] for u in uniq[exact_mask]), dtype=np.uint16, count=exact)
    miss = uniq[~exact_mask]
    # vectorised nearest, skip pal[0]
    CH = 4096
    miss_ids = []
    for s in range(0, len(miss), CH):
        e = min(len(miss), s + CH)
        cols = miss[s:e]
        cr = ((cols >> 16) & 255).astype(np.int16)
        cg = ((cols >> 8) & 255).astype(np.int16)
        cb = (cols & 255).astype(np.int16)
        # [M, 4097]
        ss = ((pal_rgb[None, :, 0].astype(np.int32) - cr[:, None].astype(np.int32)) ** 2 +
              (pal_rgb[None, :, 1].astype(np.int32) - cg[:, None].astype(np.int32)) ** 2 +
              (pal_rgb[None, :, 2].astype(np.int32) - cb[:, None].astype(np.int32)) ** 2)
        ss[:, 0] = 1 << 30
        miss_ids.append(np.argmin(ss, axis=1).astype(np.uint16))
        if e % 50000 == 0 or e == len(miss):
            log(f"  nearest {e}/{len(miss)}")
    if miss_ids:
        mapping[~exact_mask] = np.concatenate(miss_ids)
    log(f"palette exact={exact} nearest={nearest}")
    # rebuild dict path below expects mapping dict? we'll switch to arrays
    keys = uniq.astype(np.uint32)
    vals = mapping.astype(np.uint16)

    order = np.argsort(keys)
    keys, vals = keys[order], vals[order]

    dest_dir = OUT / "cache"
    dest_dir.mkdir(parents=True, exist_ok=True)
    tmp = dest_dir / "img_idx.tmp.npy"
    tok = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.uint16, shape=(n, 1024))
    CH = 4096
    unmapped_px = 0  # nearest still maps; count colours that needed nearest
    for s in range(0, n, CH):
        e = min(n, s + CH)
        blk = imgs[s:e]
        p = (blk[..., 0].astype(np.uint32) << 16) | (blk[..., 1].astype(np.uint32) << 8) | blk[..., 2].astype(np.uint32)
        a = blk[..., 3]
        # searchsorted
        idx = np.searchsorted(keys, p)
        idx = np.clip(idx, 0, len(keys) - 1) if len(keys) else idx
        hit = (len(keys) > 0) & (keys[idx] == p)
        out = np.zeros(p.shape, dtype=np.uint16)
        if len(keys):
            out[hit] = vals[idx[hit]]
        out[a < 128] = 0
        # any opaque miss (shouldn't happen if uniq was complete)
        miss = (a >= 128) & ~hit
        if miss.any():
            # shouldn't, but map 0,0,0-safe nearest one by one
            ys, xs, _wait = None, None, None
            coords = np.argwhere(miss)
            for cy, cx in coords:
                u = int(p[cy, cx])
                r, g, b = (u >> 16) & 255, (u >> 8) & 255, u & 255
                d = pal_rgb - np.array([r, g, b], dtype=np.int16)
                ss = np.sum(d * d, axis=1)
                ss[0] = 1 << 30
                out[cy, cx] = int(np.argmin(ss))
                unmapped_px += 1
        tok[s:e] = out.reshape(e - s, 1024)
        if e % 20000 == 0 or e == n:
            log(f"  cache {e}/{n}")
    tok.flush()
    del tok
    tmp.replace(dest_dir / "img_idx.npy")
    report.update({
        "unique_opaque_colours": int(len(uniq)),
        "palette_exact_colours": exact,
        "palette_nearest_colours": nearest,
        "opaque_pixels_fallback": unmapped_px,
        "token_cache_path": str(dest_dir / "img_idx.npy"),
        "seconds": round(time.time() - t0, 1),
    })
    (OUT / "cache_estimate.json").write_text(json.dumps(report, indent=2))
    (dest_dir / "cache_manifest.json").write_text(json.dumps(report, indent=2))
    log("cache done", report["token_cache_path"], "nearest_colours", nearest, "s", report["seconds"])


if __name__ == "__main__":
    main()
