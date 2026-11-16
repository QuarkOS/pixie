"""Shared helpers for Pixel AI reward-v7 (CLIP/SigLIP picker)."""
from __future__ import annotations

import math
from collections import defaultdict

import numpy as np
from PIL import Image

LIGHT_BG = (245, 245, 245)
DARK_BG = (34, 34, 34)
UPSCALE = 224


def rgba_from_indices(indices, palette) -> np.ndarray:
    """Index 0 is the transparent slot (alpha 0); all other indices are opaque."""
    flat = np.asarray(indices, dtype=np.int64).reshape(-1)
    assert flat.shape == (256,), flat.shape
    rgb = palette[flat]
    alpha = np.where(flat == 0, 0, 255).astype(np.uint8)
    return np.concatenate([rgb, alpha[:, None]], axis=1).reshape(16, 16, 4)


def composite_rgb(rgba: np.ndarray, bg) -> np.ndarray:
    """Composite RGBA uint8 onto an opaque RGB background. Transparent index-0
    becomes the background; opaque black (palette 2117) stays (0,0,0)."""
    a = rgba[..., 3:4].astype(np.float32) / 255.0
    rgb = rgba[..., :3].astype(np.float32)
    out = rgb * a + np.asarray(bg, dtype=np.float32) * (1.0 - a)
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


def upscale_nn(rgb: np.ndarray, size: int = UPSCALE) -> np.ndarray:
    return np.asarray(Image.fromarray(rgb, mode="RGB").resize((size, size), Image.NEAREST))


def render_sprite(indices, palette, bg, size: int = UPSCALE) -> np.ndarray:
    return upscale_nn(composite_rgb(rgba_from_indices(indices, palette), bg), size)


def fill_count(indices) -> int:
    a = np.asarray(indices).reshape(-1)
    return int((a != 0).sum())


def color_count(indices) -> int:
    a = np.asarray(indices).reshape(-1)
    return int(len(set(a.tolist()) - {0}))


def connected_components(indices, conn: int = 4) -> int:
    """Count 4-connected components of opaque (non-index-0) pixels on the 16x16 grid."""
    a = np.asarray(indices).reshape(16, 16)
    vis = a == 0
    n = 0
    neigh = ((-1, 0), (1, 0), (0, -1), (0, 1)) if conn == 4 else (
        (-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)
    )
    for y in range(16):
        for x in range(16):
            if vis[y, x]:
                continue
            n += 1
            stack = [(y, x)]
            vis[y, x] = True
            while stack:
                cy, cx = stack.pop()
                for dy, dx in neigh:
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < 16 and 0 <= nx < 16 and not vis[ny, nx]:
                        vis[ny, nx] = True
                        stack.append((ny, nx))
    return n


def auc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    return float(((pos[:, None] > neg[None, :]) + 0.5 * (pos[:, None] == neg[None, :])).mean())


def cluster_boot(rows, key, n_boot=4000, seed=0, filt=None):
    rows = [r for r in rows if r.get(key) is not None and (filt is None or filt(r))]
    byp = defaultdict(list)
    for r in rows:
        byp[r["prompt"]].append(r[key])
    ps = list(byp)
    if not ps:
        return dict(mean=float("nan"), lo=float("nan"), hi=float("nan"), n=0)
    rng = np.random.RandomState(seed)
    allv = np.array([v for p in ps for v in byp[p]])
    means = []
    for _ in range(n_boot):
        pick = rng.randint(0, len(ps), len(ps))
        v = np.concatenate([byp[ps[i]] for i in pick])
        means.append(v.mean())
    lo, hi = np.percentile(means, [2.5, 97.5])
    return dict(mean=float(allv.mean()), lo=float(lo), hi=float(hi), n=len(allv))


def paired_diff_ci(rows_a, rows_b, key, n_boot=4000, seed=0):
    """Prompt-cluster bootstrap CI of mean(a - b) aligned by round_id."""
    ba = {r["round_id"]: r for r in rows_a if r.get(key) is not None}
    bb = {r["round_id"]: r for r in rows_b if r.get(key) is not None}
    ids = sorted(set(ba) & set(bb))
    if not ids:
        return dict(mean=float("nan"), lo=float("nan"), hi=float("nan"), n=0)
    byp = defaultdict(list)
    for rid in ids:
        byp[ba[rid]["prompt"]].append(ba[rid][key] - bb[rid][key])
    ps = list(byp)
    rng = np.random.RandomState(seed)
    allv = np.array([v for p in ps for v in byp[p]])
    means = []
    for _ in range(n_boot):
        pick = rng.randint(0, len(ps), len(ps))
        v = np.concatenate([byp[ps[i]] for i in pick])
        means.append(v.mean())
    lo, hi = np.percentile(means, [2.5, 97.5])
    return dict(mean=float(allv.mean()), lo=float(lo), hi=float(hi), n=len(allv), n_prompts=len(ps))


def wilson(k, n, z=1.96):
    if n <= 0:
        return dict(mean=float("nan"), lo=float("nan"), hi=float("nan"), n=0, k=0)
    p = k / n
    den = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    half = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / den
    return dict(mean=float(p), lo=float(max(0.0, mid - half)), hi=float(min(1.0, mid + half)), n=int(n), k=int(k))


def ab_cluster(rows, n_boot=4000, seed=0):
    byp = defaultdict(list)
    for r in rows:
        byp[r["prompt"]].append(1.0 if r.get("correct") else 0.0)
    ps = list(byp)
    if not ps:
        return dict(mean=float("nan"), lo=float("nan"), hi=float("nan"), n=0)
    rng = np.random.RandomState(seed)
    allv = np.array([v for p in ps for v in byp[p]])
    means = []
    for _ in range(n_boot):
        pick = rng.randint(0, len(ps), len(ps))
        v = np.concatenate([byp[ps[i]] for i in pick])
        means.append(v.mean())
    lo, hi = np.percentile(means, [2.5, 97.5])
    return dict(mean=float(allv.mean()), lo=float(lo), hi=float(hi), n=len(allv), n_prompts=len(ps))


def ab_pack(rows):
    k = int(sum(1 for r in rows if r.get("correct")))
    n = len(rows)
    return dict(correct=k, n=n, wilson=wilson(k, n), cluster=ab_cluster(rows),
                chance=0.5, vs_chance=float(k / n - 0.5) if n else float("nan"))


def l2norm(x, axis=-1, eps=1e-8):
    n = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.clip(n, eps, None)


def fit_pca(X, k):
    X = np.asarray(X, dtype=np.float64)
    mu = X.mean(0)
    Z = X - mu
    # economy SVD; Vt is (min(n,d), d)
    _, _, Vt = np.linalg.svd(Z, full_matrices=False)
    k = int(min(k, Vt.shape[0]))
    W = Vt[:k].T
    return mu.astype(np.float32), W.astype(np.float32)


def apply_pca(X, mu, W):
    return (np.asarray(X, dtype=np.float32) - mu) @ W
