"""Fast v3 P50 sampler: vectorized Gumbel pick, static KV cache, optional CUDA graph.

Does not modify a separate checkout. The live worker is not imported.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

import v3_model as M


def pick_rows_gumbel(lg, temperature, top_k, generator=None, uniforms=None):
    """One-shot GPU Gumbel-max over a [B, V] logit batch. Same categorical
    distribution as temperature + top-k + multinomial (not the same seed path)."""
    if temperature <= 0:
        return lg.argmax(-1)
    x = lg / temperature
    if top_k:
        thr = x.topk(int(top_k), dim=-1).values[:, -1:]
        x = x.masked_fill(x < thr, float("-inf"))
    if uniforms is None:
        u = torch.rand(x.shape, device=x.device, dtype=torch.float32, generator=generator)
    else:
        u = uniforms
    # (0, 1) open interval so -log(-log u) is finite
    u = u.clamp(min=torch.finfo(u.dtype).tiny, max=1.0 - 2e-7)
    return (x + (-torch.log(-torch.log(u)))).argmax(-1)


def pick_rows_legacy(lg, temperature, top_k, gens):
    """Exact copy of the live worker's per-row loop (for distribution checks)."""
    b = lg.shape[0]
    out = torch.empty(b, dtype=torch.long, device=lg.device)
    for j, g in enumerate(gens):
        row = lg[j]
        if temperature <= 0:
            out[j] = row.argmax()
            continue
        x = row / temperature
        if top_k:
            thr = x.topk(top_k).values[-1]
            x = x.masked_fill(x < thr, float("-inf"))
        out[j] = torch.multinomial(F.softmax(x, -1), 1, generator=g)
    return out


def _i64(u):
    """Interpret a uint64 constant as signed int64 (two's complement wrap)."""
    u = int(u) & ((1 << 64) - 1)
    if u >= (1 << 63):
        u -= 1 << 64
    return u


_C_GOLDEN = _i64(0x9E3779B97F4A7C15)
_C_MIX1 = _i64(0xBF58476D1CE4E5B9)
_C_MIX2 = _i64(0x94D049BB133111EB)


def uniforms_from_seeds(seeds, vocab, device):
    """Independent Uniform(0,1) [B, V] from per-row int seeds, no Python loop.

    Not the same stream as torch.Generator.manual_seed; used so a stored seed
    still uniquely determines a row without walking 64 CUDA generators.
    """
    s = seeds.to(device=device, dtype=torch.int64).view(-1, 1)
    cols = torch.arange(vocab, device=device, dtype=torch.int64)
    # Splitmix64-style mix in uint64 (two's complement wrap is fine).
    x = s * torch.tensor(_C_GOLDEN, dtype=torch.int64, device=device) + cols
    x = x ^ (x >> 30)
    x = x * torch.tensor(_C_MIX1, dtype=torch.int64, device=device)
    x = x ^ (x >> 27)
    x = x * torch.tensor(_C_MIX2, dtype=torch.int64, device=device)
    x = x ^ (x >> 31)
    # top 24 bits -> (0, 1); mask so arithmetic shift cannot sign-extend into u
    u = ((x >> 40) & 0xFFFFFF).to(torch.float32) * (1.0 / 16777216.0)
    return u.clamp(min=1.0 / 16777216.0, max=1.0 - 1.0 / 16777216.0)


def _maybe_cond(model, cond, batch, device):
    if model.NC == 0:
        return None
    if cond is None:
        return M.default_cond(model.config, batch, device)
    return cond


class FastSampler:
    """Cached P50 trunk + optional CUDA graph of the decode step."""

    def __init__(self, model, device, max_batch, cfg_scale=2.0, temperature=0.8, top_k=32,
                 use_graph=True, dtype=torch.bfloat16):
        self.model = model
        self.device = device
        self.max_batch = int(max_batch)  # CFG-expanded batch (e.g. 128 for n=64)
        self.cfg_scale = float(cfg_scale)
        self.temperature = float(temperature)
        self.top_k = int(top_k)
        self.use_graph = bool(use_graph)
        self.dtype = dtype
        c = model.config
        self.n_layer = c.n_layer
        self.n_head = c.n_head
        self.d = c.d_model
        self.dh = c.d_model // c.n_head
        self.OUT0 = int(model.OUT0)
        self.SEQ = int(model.SEQ)
        self.NP = int(model.NP)
        self.TL = int(model.TL)

        # Static KV cache and decode tables. Allocated once at max batch.
        self.k = torch.zeros(self.n_layer, self.max_batch, self.n_head, self.SEQ, self.dh,
                             device=device, dtype=dtype)
        self.v = torch.zeros_like(self.k)
        # For decode at absolute position p, keys 0..p are valid.
        ar = torch.arange(self.SEQ, device=device)
        self.decode_masks = torch.where(
            ar[None, :] <= ar[:, None],
            torch.zeros((), device=device, dtype=torch.float32),
            torch.full((), float("-inf"), device=device, dtype=torch.float32),
        ).view(self.SEQ, 1, 1, self.SEQ)

        # extra[m-1] = row_in(pr[m-1]) + col_in(pc[m-1]) + pos[OUT0+m] + tgt[m], m=1..63
        pr = model.pr
        pc = model.pc
        tgt = model.row_out(pr) + model.col_out(pc)
        pos = model.pos.weight
        extra = (model.row_in(pr[:-1]) + model.col_in(pc[:-1])
                 + pos[self.OUT0 + 1 : self.OUT0 + self.NP] + tgt[1:])
        self.extra_table = extra.to(device=device, dtype=dtype).contiguous()

        # Graph I/O buffers (filled by prepare()).
        self._graphs = {}  # batch -> dict
        self._prepared = False

    def prepare(self, batch_sizes):
        """Capture CUDA graphs for each CFG-expanded batch size. Must run
        before the first sample so capture cannot dirty a live prefill."""
        if not self.use_graph:
            return
        for b in batch_sizes:
            b = int(b)
            if b <= 0 or b > self.max_batch:
                raise ValueError(f"batch {b} not in 1..{self.max_batch}")
            if b not in self._graphs:
                self._ensure_graph(b)
        # Capture writes the cache; clear it so the next prefill starts clean.
        self.k.zero_()
        self.v.zero_()
        self._prepared = True

    def _pref_x(self, clip_h, cond):
        """Prefix + sep+tgt[0]: length OUT0+1, matches trunk positions 0..OUT0."""
        model = self.model
        b = clip_h.shape[0]
        d = self.d
        t = model.embed_text(clip_h)
        pre = [model.bos.expand(b, 1, d).to(t.dtype), t]
        if model.NC:
            pre.append(model.cond_emb(cond, b, clip_h.device).to(t.dtype))
        prefix = torch.cat(pre, 1)  # [B, OUT0, d]
        assert prefix.shape[1] == self.OUT0, (prefix.shape, self.OUT0)
        tgt0 = (model.row_out(model.pr[:1]) + model.col_out(model.pc[:1])).to(t.dtype)
        sep = model.sep.expand(b, 1, d).to(t.dtype) + tgt0
        x = torch.cat([prefix, sep], 1)
        x = x + model.pos.weight[None, : self.OUT0 + 1].to(x.dtype)
        return x

    def prefill(self, clip_h, cond, batch_slots):
        """Write cache[:, :OUT0+1] and return h[:, 0] for the used batch rows."""
        model = self.model
        x = self._pref_x(clip_h, cond).to(self.dtype)
        b, s, _ = x.shape
        assert b <= self.max_batch and s == self.OUT0 + 1
        # Zero only the used slice (cheap; keeps unused rows from leaking).
        self.k[:, :b].zero_()
        self.v[:, :b].zero_()
        for li, blk in enumerate(model.blocks):
            x = self._block_prefill(blk, x, li, b)
        return model.ln_f(x[:, -1])

    def _block_prefill(self, blk, x, layer, b):
        s = x.shape[1]
        q, k, v = blk.qkv(blk.ln1(x)).view(b, s, 3, self.n_head, self.dh).unbind(2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        self.k[layer, :b, :, :s].copy_(k)
        self.v[layer, :b, :, :s].copy_(v)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + blk.out(a.transpose(1, 2).reshape(b, s, self.d))
        return x + blk.fc2(F.gelu(blk.fc1(blk.ln2(x))))

    def decode_eager(self, prev_patch, m, batch):
        """One cached trunk step at patch m (1..63). prev_patch is patch m-1, [B,4]."""
        model = self.model
        pos = self.OUT0 + m
        im = sum(model.slot_emb[j](prev_patch[:, j]) for j in range(4)) + self.extra_table[m - 1]
        x = im.to(self.dtype)[:, None, :]
        bias = self.decode_masks[pos]
        for li, blk in enumerate(model.blocks):
            x = self._block_decode(blk, x, li, pos, bias, batch)
        return model.ln_f(x[:, 0])

    def _block_decode(self, blk, x, layer, pos, bias, b):
        q, k, v = blk.qkv(blk.ln1(x)).view(b, 1, 3, self.n_head, self.dh).unbind(2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        self.k[layer, :b, :, pos:pos + 1].copy_(k)
        self.v[layer, :b, :, pos:pos + 1].copy_(v)
        a = F.scaled_dot_product_attention(
            q, self.k[layer, :b], self.v[layer, :b], attn_mask=bias)
        x = x + blk.out(a.transpose(1, 2).reshape(b, 1, self.d))
        return x + blk.fc2(F.gelu(blk.fc1(blk.ln2(x))))

    def _ensure_graph(self, batch):
        if batch in self._graphs:
            return self._graphs[batch]
        model = self.model
        b = batch
        # Static graph I/O.
        # Embeddings are written into x_in *outside* the graph so capture
        # never sees a data-dependent gather of extra_table.
        x_in = torch.zeros(b, 1, self.d, device=self.device, dtype=self.dtype)
        pos_t = torch.zeros((), dtype=torch.long, device=self.device)    # OUT0+m
        attn_bias = torch.zeros(1, 1, 1, self.SEQ, device=self.device, dtype=torch.float32)
        h_out = torch.zeros(b, self.d, device=self.device, dtype=self.dtype)

        def body():
            x = x_in
            write = pos_t.view(1)
            for li, blk in enumerate(model.blocks):
                q, k, v = blk.qkv(blk.ln1(x)).view(b, 1, 3, self.n_head, self.dh).unbind(2)
                q = q.transpose(1, 2)
                k = k.transpose(1, 2)
                v = v.transpose(1, 2)
                self.k[li, :b].index_copy_(2, write, k)
                self.v[li, :b].index_copy_(2, write, v)
                a = F.scaled_dot_product_attention(
                    q, self.k[li, :b], self.v[li, :b], attn_mask=attn_bias)
                x = x + blk.out(a.transpose(1, 2).reshape(b, 1, self.d))
                x = x + blk.fc2(F.gelu(blk.fc1(blk.ln2(x))))
            h_out.copy_(model.ln_f(x[:, 0]))

        # Warmup + capture. Need concrete pos for warmup so kernels compile.
        def fill_step(m, prev_patch=None):
            if prev_patch is None:
                prev_patch = torch.zeros(b, 4, dtype=torch.long, device=self.device)
            im = (sum(model.slot_emb[j](prev_patch[:, j]) for j in range(4))
                  + self.extra_table[m - 1])
            x_in.copy_(im.to(self.dtype)[:, None, :])
            pos_t.fill_(self.OUT0 + m)
            attn_bias.copy_(self.decode_masks[self.OUT0 + m])

        # Warmup on the side stream as recommended by PyTorch.
        torch.cuda.synchronize()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            for m in (1, 2, 3):
                fill_step(m)
                body()
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()

        g = torch.cuda.CUDAGraph()
        fill_step(1)
        with torch.cuda.graph(g), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            body()
        pack = dict(graph=g, x_in=x_in, pos_t=pos_t,
                    attn_bias=attn_bias, h_out=h_out, fill_step=fill_step)
        self._graphs[batch] = pack
        return pack

    def decode_graph(self, prev_patch, m, batch):
        pack = self._ensure_graph(batch)
        pack["fill_step"](m, prev_patch)
        pack["graph"].replay()
        return pack["h_out"]

    def hidden_at(self, prev_patch, m, batch, use_graph=None):
        if m == 0:
            raise RuntimeError("patch 0 comes from prefill")
        if (self.use_graph if use_graph is None else use_graph):
            return self.decode_graph(prev_patch, m, batch)
        return self.decode_eager(prev_patch, m, batch)

    @torch.no_grad()
    def teacher_hidden(self, clip_h, cond, patches, use_graph=False):
        """Return h [B, 64, d] from the cached trunk, teacher-forced on `patches`."""
        b = clip_h.shape[0]
        cond = _maybe_cond(self.model, cond, b, clip_h.device)
        h0 = self.prefill(clip_h, cond, b)
        out = torch.empty(b, self.NP, self.d, device=self.device, dtype=h0.dtype)
        out[:, 0] = h0
        for m in range(1, self.NP):
            out[:, m] = self.hidden_at(patches[:, m - 1], m, b, use_graph=use_graph)
        return out

    @torch.no_grad()
    def sample(self, clip_h, clip_u, seeds=None, cond=None, temperature=None, top_k=None,
               cfg_scale=None, use_graph=None, pick="gumbel"):
        """Sample n = clip_h.shape[0] sprites. Returns palette indices [n, 256]."""
        model = self.model
        n = clip_h.shape[0]
        dev = self.device
        temp = self.temperature if temperature is None else temperature
        topk = self.top_k if top_k is None else top_k
        cfg = self.cfg_scale if cfg_scale is None else cfg_scale
        b = n * 2  # CFG
        if b > self.max_batch:
            raise RuntimeError(f"CFG batch {b} > max_batch {self.max_batch}")
        cond = _maybe_cond(model, cond, n, dev)
        hh = torch.cat([clip_h, clip_u], 0)
        cc = None if cond is None else torch.cat([cond, cond], 0)
        P = torch.zeros(n, model.NP, 4, dtype=torch.long, device=dev)

        gens = None
        if pick == "legacy":
            if seeds is None:
                raise ValueError("legacy pick needs seeds")
            gens = []
            for s in seeds:
                g = torch.Generator(device=dev)
                g.manual_seed(int(s))
                gens.append(g)

        job_gen = None
        if pick == "gumbel":
            job_gen = torch.Generator(device=dev)
            if seeds is not None:
                s0 = seeds[0]
                s0 = int(s0.item()) if torch.is_tensor(s0) else int(s0)
                job_gen.manual_seed(s0 & 0x7FFFFFFF)
            else:
                job_gen.manual_seed(int(torch.randint(0, 2**31 - 1, (1,)).item()))

        def pick_from(lg, slot):
            lg = lg[:, :M.PALETTE]
            if pick == "legacy":
                return pick_rows_legacy(lg, temp, topk, gens)
            # One batched Gumbel draw. A single CUDA generator is much cheaper
            # than building a [B, V] hash-uniform table at every pixel.
            return pick_rows_gumbel(lg, temp, topk, generator=job_gen)

        use_g = self.use_graph if use_graph is None else use_graph
        if use_g:
            self.prepare([b])
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            hid = self.prefill(hh, cc, b)
            for m in range(model.NP):
                if m > 0:
                    prev = torch.cat([P[:, m - 1], P[:, m - 1]], 0)
                    hid = self.hidden_at(prev, m, b, use_graph=use_g)
                prev_slots = []
                for k in range(4):
                    pp = [torch.cat([x, x], 0) for x in prev_slots]
                    lg = (model.head.step(hid, pp) + model.vmask).float()
                    lg = lg[n:] + cfg * (lg[:n] - lg[n:])
                    nxt = pick_from(lg, m * 4 + k)
                    P[:, m, k] = nxt
                    prev_slots.append(nxt)
        return M.from_patches(P)


@torch.no_grad()
def sample_uncached(model, clip_h, clip_u, seeds, device, cfg_scale, temperature, top_k, cond=None):
    """Exact live-worker loop: uncached trunk, per-row multinomial, batch is the
    full n (caller may chunk). Used as the correctness baseline."""
    n = clip_h.shape[0]
    gens = []
    for s in seeds:
        g = torch.Generator(device=device)
        g.manual_seed(int(s))
        gens.append(g)
    P = torch.zeros(n, model.NP, 4, dtype=torch.long, device=device)
    hh = torch.cat([clip_h, clip_u], 0)
    cc = None if cond is None else torch.cat([cond, cond], 0)
    for m in range(model.NP):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            hid = model.trunk(hh, torch.cat([P, P], 0), cc)[:, m]
            prev = []
            for k in range(4):
                pp = [torch.cat([x, x], 0) for x in prev]
                lg = (model.head.step(hid, pp) + model.vmask).float()
                lg = lg[n:] + cfg_scale * (lg[:n] - lg[n:])
                nxt = pick_rows_legacy(lg[:, :M.PALETTE], temperature, top_k, gens)
                P[:, m, k] = nxt
                prev.append(nxt)
    return M.from_patches(P)
