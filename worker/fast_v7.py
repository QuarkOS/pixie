"""Faster reward-v7 scoring: GPU nearest-neighbor renders, one image batch
per model, features and the linear head stay on CUDA.

Wraps the existing RewardV7 (does not edit reward-v7/src).
"""
from __future__ import annotations

from collections import OrderedDict
import numpy as np
import torch
import torch.nn.functional as F
from phase_timing import PhaseTimer

# Matches reward-v7/src/common.py. Imported lazily so this module can load
# before v7's src is on sys.path (the worker adds it in try_load_v7).
LIGHT_BG = (245, 245, 245)
DARK_BG = (34, 34, 34)


TEMPLATES = {
    "raw": "{prompt}",
    "pixel": "a 16x16 pixel art sprite of {prompt}",
}


def _unique_colors(idx_t):
    """idx_t [N, 256] long -> [N] number of distinct non-zero palette ids."""
    n, _ = idx_t.shape
    # 4097 palette slots; ignore index 0
    oh = torch.zeros(n, 4097, device=idx_t.device, dtype=torch.int16)
    oh.scatter_(1, idx_t.clamp(0, 4096), 1)
    oh[:, 0] = 0
    return oh.sum(1).to(torch.int32)


class FastRewardV7:
    def __init__(self, v7, palette, text_cache_size=0):
        self.v7 = v7
        self.device = v7.device
        pal = torch.as_tensor(np.asarray(palette), device=self.device)
        if pal.ndim == 2 and pal.shape[-1] >= 3:
            pal = pal[:, :3]
        self.pal = pal.float()
        self.mu = torch.as_tensor(v7.mu, device=self.device, dtype=torch.float32)
        self.sd = torch.as_tensor(v7.sd, device=self.device, dtype=torch.float32)
        self.w = torch.as_tensor(v7.w, device=self.device, dtype=torch.float32)
        self.pca = {
            k: (torch.as_tensor(mu, device=self.device, dtype=torch.float32),
                torch.as_tensor(W, device=self.device, dtype=torch.float32))
            for k, (mu, W) in ((k, v7.pca[k]) for k in v7.pca)
        }
        self.recipe = v7.recipe
        self.size = next(iter(v7.models.values()))["size"]
        self.norm = {}
        self._text_cache = OrderedDict()
        self.text_cache_size = text_cache_size
        self.text_cache_hits = self.text_cache_misses = 0
        self.profile_phases = False
        self.last_profile = None
        for key, m in v7.models.items():
            mean = torch.tensor(m["mean"], dtype=torch.float32, device=self.device).view(1, 3, 1, 1)
            std = torch.tensor(m["std"], dtype=torch.float32, device=self.device).view(1, 3, 1, 1)
            self.norm[key] = (mean, std)

    @property
    def text_cache_size(self):
        return self._text_cache_size

    @text_cache_size.setter
    def text_cache_size(self, size):
        if type(size) is not int or not 0 <= size <= 256:
            raise ValueError("reward text cache size must be an integer in 0..256")
        self._text_cache_size = size
        while len(self._text_cache) > size:
            self._text_cache.popitem(last=False)

    def clear_text_cache(self):
        self._text_cache.clear()

    def text_cache_info(self):
        return {"capacity": self.text_cache_size, "entries": len(self._text_cache),
                "hits": self.text_cache_hits, "misses": self.text_cache_misses,
                "tensor_bytes": sum(tensor.numel() * tensor.element_size()
                                    for entry in self._text_cache.values() for tensor in entry.values())}

    def _text_key(self, model, tokens):
        tensors = (*model.parameters(), *model.buffers())
        versions = tuple((id(p), p._version, p.device, p.dtype) for p in tensors)
        ids = tuple((name, tuple(tok.shape), str(tok.dtype), tuple(tok.reshape(-1).tolist()))
                    for name, tok in tokens.items())
        return (id(model), versions, self.device, ids,
                torch.are_deterministic_algorithms_enabled(),
                torch.backends.cuda.matmul.fp32_precision,
                torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
                torch.backends.cuda.matmul.allow_fp16_accumulation,
                torch.backends.cuda.flash_sdp_enabled(), torch.backends.cuda.math_sdp_enabled(),
                torch.backends.cuda.mem_efficient_sdp_enabled(), torch.backends.cuda.cudnn_sdp_enabled())

    def _render(self, idx, bg, size, mean, std):
        """idx [N,256] long on CUDA -> normalized [N,3,size,size] float32."""
        n = idx.shape[0]
        ids = idx.view(n, 16, 16)
        rgb = self.pal[ids]  # [N,16,16,3]
        a = (ids != 0).to(rgb.dtype).unsqueeze(-1)
        bg_t = rgb.new_tensor(bg)
        out = rgb * a + bg_t * (1 - a)
        out = out.permute(0, 3, 1, 2) / 255.0
        out = F.interpolate(out, size=(size, size), mode="nearest")
        return (out - mean) / std

    @torch.no_grad()
    def _encode_images(self, key, tensor):
        m = self.v7.models[key]
        use_amp = self.device.type == "cuda"
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            feat = m["model"].encode_image(tensor)
        feat = feat.float()
        return feat / feat.norm(dim=-1, keepdim=True).clamp_min(1e-8)

    @torch.no_grad()
    def _encode_text(self, key, prompt):
        m = self.v7.models[key]
        tokens = {name: m["tokenizer"]([template.format(prompt=prompt)])
                  for name, template in TEMPLATES.items()}
        cache_key = None
        if self.text_cache_size:
            if m["model"].training:
                raise ValueError("reward text caching requires eval-mode models")
            cache_key = self._text_key(m["model"], tokens)
            if cache_key in self._text_cache:
                self.text_cache_hits += 1
                self._text_cache.move_to_end(cache_key)
                return self._text_cache[cache_key]
            self.text_cache_misses += 1
        out = {}
        use_amp = self.device.type == "cuda"
        for tname, token in tokens.items():
            tok = token.to(self.device)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                feat = m["model"].encode_text(tok)
            feat = feat.float()
            feat = feat / feat.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            out[tname] = feat[0]
        if cache_key is not None:
            self._text_cache[cache_key] = out
            while len(self._text_cache) > self.text_cache_size:
                self._text_cache.popitem(last=False)
        return out

    @torch.no_grad()
    def score_indices(self, list_of_indices, prompt):
        timer = PhaseTimer(self.profile_phases,
                           lambda: torch.cuda.synchronize(self.device) if self.device.type == "cuda" else None)
        n = len(list_of_indices)
        # hand features: fill/color on GPU; connected components stays on CPU (64 x 16x16)
        with timer.phase("hand_features"):
            idx_np = np.stack([np.asarray(x, dtype=np.int64).reshape(-1) for x in list_of_indices], 0)
            idx = torch.as_tensor(idx_np, device=self.device, dtype=torch.long)
            fills = (idx != 0).sum(1).to(torch.float32)
            colors = _unique_colors(idx).to(torch.float32)
            from common import connected_components
            ccs = torch.tensor(
                [connected_components(r) for r in idx_np],
                device=self.device, dtype=torch.float32,
            )

        rec = self.recipe
        img = {}
        txt = {}
        for key in self.v7.models:
            with timer.phase("text_encode_or_cache"):
                txt[key] = self._encode_text(key, prompt)
            with timer.phase("render"):
                mean, std = self.norm[key]
                light = self._render(idx, LIGHT_BG, self.size, mean, std)
                dark = self._render(idx, DARK_BG, self.size, mean, std)
                rendered = torch.cat([light, dark], 0)
            with timer.phase("image_encode"):
                feat = self._encode_images(key, rendered)
                img[(key, "light")] = feat[:n]
                img[(key, "dark")] = feat[n:]

        with timer.phase("pca_linear_head"):
            cols = []
            for cname in rec.get("cosines", []):
                key, bg, tmpl = cname.split(":")
                cols.append((img[(key, bg)] * txt[key][tmpl]).sum(-1, keepdim=True))
            if rec.get("hand"):
                cols.append(torch.stack([fills, colors, ccs], 1))
            k = int(rec.get("pca_k", 0) or 0)
            if k > 0:
                for src in ("int", "cat"):
                    for cname in rec.get("pca_from", []):
                        pack_key = f"{src}:{cname}"
                        if pack_key not in self.pca:
                            continue
                        key, bg, tmpl = cname.split(":")
                        im, te = img[(key, bg)], txt[key][tmpl]
                        vec = (im * te) if src == "int" else torch.cat([im, te.expand(n, -1)], 1)
                        mu, W = self.pca[pack_key]
                        cols.append((vec - mu) @ W)
            X = torch.cat(cols, 1)
            Z = (X - self.mu) / self.sd
            scores = Z @ self.w[:-1] + self.w[-1]
        with timer.phase("score_readback"):
            result = scores.detach().cpu().numpy().astype(np.float32)
        self.last_profile = timer.finish()
        if self.last_profile is not None:
            self.last_profile["scope"] = "synchronized complete reward scoring; includes feature preparation and score readback; synchronization can perturb latency"
        return result
