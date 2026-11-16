"""Pixie v3 s7 data helpers (Opus roadmap s7): caption table + trim, palette/style stats, balanced sampling.

Caption table: every distinct caption a training sample can see (main, synonym, typo copy, subject-only,
blank), trimmed to text_len CLIP positions (BOS + text_len-2 ids + EOS; longer captions are cut and get
EOS in the last slot). CLIP is causal, so for captions that fit, the trimmed features equal the first
text_len positions of the old 26-position features exactly. Rows are sorted unique (np.unique), so the
table is deterministic and shared by the trainer and the feature-cache builder (sha256 checked).
"""
import hashlib
import numpy as np
import torch

CLIP_BOS, CLIP_EOS = 49406, 49407


def content_len(t):
    """uint16/int [N,26] CLIP-format -> number of ids between BOS and first EOS."""
    a = np.asarray(t).astype(np.int64) & 0xFFFF
    e = a[:, 1:] == CLIP_EOS
    return np.where(e.any(1), e.argmax(1), a.shape[1] - 1)


def trim(t, text_len):
    """[N,26] uint16 -> [N,text_len] uint16 (cut; EOS forced into the last slot when the caption was longer)."""
    a = np.ascontiguousarray(np.asarray(t).view(np.uint16)[:, :text_len]).copy()
    long_ = content_len(t) > text_len - 2
    a[long_, text_len - 1] = CLIP_EOS
    return a


def blank_caption(text_len):
    a = np.full((1, text_len), CLIP_EOS, np.uint16); a[0, 0] = CLIP_BOS
    return a


def build_caption_table(tr, text_len):
    """tr: v3_train npz (with text_typo). Returns dict: table [C,TL] uint16, row_main/row_syn/row_typo int32 [N]
    (-1 where the variant does not exist), subj int32 [S], blank int, sha256, stats."""
    N = tr["text_main"].shape[0]
    parts = [trim(tr["text_main"], text_len), trim(tr["text_syn"], text_len)]
    has_typo = "text_typo" in tr.files
    if has_typo:
        parts.append(trim(tr["text_typo"], text_len))
    parts += [trim(tr["subj_text"], text_len), blank_caption(text_len)]
    allc = np.concatenate(parts)
    v = allc.view(np.dtype((np.void, 2 * text_len))).ravel()
    uniq, inv = np.unique(v, return_inverse=True)
    table = uniq.view(np.uint16).reshape(-1, text_len)
    inv = inv.astype(np.int32).ravel()
    o = 0
    row_main = inv[o:o + N]; o += N
    row_syn = np.where(tr["has_syn"], inv[o:o + N], -1).astype(np.int32); o += N
    if has_typo:
        row_typo = np.where(tr["has_typo"], inv[o:o + N], -1).astype(np.int32); o += N
    else:
        row_typo = np.full(N, -1, np.int32)
    S = tr["subj_text"].shape[0]
    subj = inv[o:o + S]; o += S
    blank = int(inv[o])
    used = np.zeros(len(table), bool)
    used[row_main] = True; used[row_syn[row_syn >= 0]] = True; used[row_typo[row_typo >= 0]] = True
    used[subj] = True; used[blank] = True
    sha = hashlib.sha256(table.tobytes()).hexdigest()
    fit = {k: float((content_len(tr[k][m]) <= text_len - 2).mean()) for k, m in
           (("text_main", slice(None)), ("text_syn", tr["has_syn"]),
            *((("text_typo", tr["has_typo"]),) if has_typo else ()), ("subj_text", slice(None)))}
    return dict(table=table, row_main=row_main, row_syn=row_syn, row_typo=row_typo, subj=subj, blank=blank,
                sha256=sha, n=len(table), all_used=bool(used.all()), fit=fit)


# ------------------------------------------------------------------ palette / style stats (GPU)
FILL_SMALL, FILL_MED, FILL_FULL = 1, 2, 3
EDGE_DARK, EDGE_SOFT = 1, 2
COL_LOW, COL_MID, COL_HIGH = 1, 2, 3
NO_PALETTE = 4097


@torch.no_grad()
def cond_stats(I, palette_rgb, chunk=8192):
    """I [N,256] palette ids (0 = transparent) on GPU, palette_rgb [4097,3] uint8.
    Returns top3 [N,3] long (most frequent non-transparent ids, ties -> lower id; NO_PALETTE if fewer
    colours) and style [N,3] long (fill, outline, colour-count codes; see constants)."""
    dev = I.device; N = I.shape[0]
    dark = torch.from_numpy(palette_rgb.astype(np.int64).sum(1) < 90).to(dev)
    top3 = torch.empty(N, 3, dtype=torch.long, device=dev); style = torch.empty(N, 3, dtype=torch.long, device=dev)
    for s in range(0, N, chunk):
        x = I[s:s + chunk].long(); b = x.shape[0]
        cnt = torch.zeros(b, 4097, dtype=torch.int32, device=dev)
        cnt.scatter_add_(1, x, torch.ones_like(x, dtype=torch.int32))
        key = (cnt[:, 1:] * 8192 + (4095 - torch.arange(4096, device=dev, dtype=torch.int32))).float()  # exact < 2^24
        v, ix = key.topk(3, 1)                                          # tie-break: lower id first
        top3[s:s + b] = torch.where(v >= 8192, ix + 1, torch.full_like(ix, NO_PALETTE))
        ncol = (cnt[:, 1:] > 0).sum(1); fill = 256 - cnt[:, 0]
        op = (x != 0).view(b, 16, 16)
        pad = torch.zeros(b, 18, 18, dtype=torch.bool, device=dev); pad[:, 1:17, 1:17] = op
        nb_tr = ~pad[:, :-2, 1:-1] | ~pad[:, 2:, 1:-1] | ~pad[:, 1:-1, :-2] | ~pad[:, 1:-1, 2:]  # edge counts as transparent
        bnd = op & nb_tr
        dk = dark[x].view(b, 16, 16) & bnd
        nb = bnd.flatten(1).sum(1); nd = dk.flatten(1).sum(1)
        edge = torch.where((nb > 0) & (2 * nd >= nb), EDGE_DARK, EDGE_SOFT)
        fc = torch.where(fill < 80, FILL_SMALL, torch.where(fill <= 180, FILL_MED, FILL_FULL))
        cc = torch.where(ncol <= 12, COL_LOW, torch.where(ncol <= 24, COL_MID, COL_HIGH))
        style[s:s + b] = torch.stack([fc, edge, cc], 1)
    return top3, style


def balanced_weights(subj_id, alpha=0.35, cap=6.0):
    """Per-row sampling weight p_i ~ n_s^-alpha, normalised to mean 1 (= views relative to uniform), capped."""
    sid = np.asarray(subj_id).astype(np.int64)
    n_s = np.bincount(sid)
    w = n_s[sid].astype(np.float64) ** (-alpha)
    w /= w.mean()
    for _ in range(50):
        if w.max() <= cap + 1e-9:
            break
        w = np.minimum(w, cap); w /= w.mean()
    w = np.minimum(w, cap)
    return w, n_s
