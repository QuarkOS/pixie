"""Pixel AI v2 SpriteTransformer (self-contained; the pixelgen package is not imported).

Sequence: [BOS] text(24 CLIP ids, 0 = PAD) [SEP] pixel_0 .. pixel_254   (281 inputs)
Output at index 25 + k predicts pixel k (k = 0..255) over the 4,097-way palette.

Text embedding = Linear(frozen CLIP-L/14 token embedding, 768 -> d) + a small
learned residual for CLIP tokens seen in training (zero-init). Unseen CLIP
tokens still get their projected CLIP vector. Image tokens get learned
palette embeddings. Positions: learned 1-D for all 281 inputs, plus 2-D axial
row/col embeddings of each input pixel and row/col embeddings of the pixel
being predicted.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

TEXT_LEN = 24
N_PIX = 256
PALETTE = 4097
SEQ_IN = 1 + TEXT_LEN + 1 + N_PIX - 1  # 281
OUT0 = 1 + TEXT_LEN  # index of SEP = first output position


@dataclass
class V2Config:
    d_model: int = 256
    n_layer: int = 12
    n_head: int = 8
    ffn_mult: int = 4
    dropout: float = 0.1
    clip_dim: int = 768
    resid_vocab: int = 7723
    palette: int = PALETTE

    def to_dict(self):
        return asdict(self)


class Block(nn.Module):  # same block as pixelgen.model.Block (pre-LN, SDPA causal, GELU MLP)
    def __init__(self, c: V2Config):
        super().__init__()
        self.n_head, self.d = c.n_head, c.d_model
        self.ln1 = nn.LayerNorm(c.d_model)
        self.qkv = nn.Linear(c.d_model, 3 * c.d_model)
        self.out = nn.Linear(c.d_model, c.d_model)
        self.ln2 = nn.LayerNorm(c.d_model)
        self.fc1 = nn.Linear(c.d_model, c.d_model * c.ffn_mult)
        self.fc2 = nn.Linear(c.d_model * c.ffn_mult, c.d_model)
        self.drop = nn.Dropout(c.dropout)
        self.p = c.dropout

    def forward(self, x):
        b, s, _ = x.shape
        h = self.d // self.n_head
        q, k, v = self.qkv(self.ln1(x)).view(b, s, 3, self.n_head, h).unbind(2)
        p = self.p if self.training else 0.0
        a = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                                           is_causal=True, dropout_p=p)
        x = x + self.drop(self.out(a.transpose(1, 2).reshape(b, s, self.d)))
        return x + self.drop(self.fc2(F.gelu(self.fc1(self.ln2(x)))))


class SpriteV2(nn.Module):
    def __init__(self, c: V2Config, clip_table: np.ndarray, clip2v2: np.ndarray):
        super().__init__()
        self.config = c
        d = c.d_model
        ct = torch.from_numpy(np.asarray(clip_table, dtype=np.float16))
        ct = torch.cat([torch.zeros(1, ct.shape[1], dtype=ct.dtype), ct], 0)  # row 0 = PAD
        assert ct.shape == (49409, c.clip_dim)
        self.register_buffer("clip_table", ct, persistent=False)  # frozen, reloaded from file
        self.register_buffer("clip2v2", torch.from_numpy(np.asarray(clip2v2, dtype=np.int64)), persistent=True)
        self.clip_ln = nn.LayerNorm(c.clip_dim)
        self.text_proj = nn.Linear(c.clip_dim, d)
        self.text_resid = nn.Embedding(c.resid_vocab, d)
        self.bos = nn.Parameter(torch.zeros(d))
        self.sep = nn.Parameter(torch.zeros(d))
        self.img_emb = nn.Embedding(c.palette, d)
        self.pos = nn.Embedding(SEQ_IN, d)
        self.row_in = nn.Embedding(16, d); self.col_in = nn.Embedding(16, d)
        self.row_out = nn.Embedding(16, d); self.col_out = nn.Embedding(16, d)
        self.blocks = nn.ModuleList(Block(c) for _ in range(c.n_layer))
        self.ln_f = nn.LayerNorm(d)
        self.head = nn.Linear(d, c.palette, bias=False)
        self.apply(_init)
        nn.init.zeros_(self.text_resid.weight)
        nn.init.normal_(self.bos, std=0.02); nn.init.normal_(self.sep, std=0.02)
        j = torch.arange(N_PIX)
        self.register_buffer("pr", j // 16, persistent=False)
        self.register_buffer("pc", j % 16, persistent=False)

    def embed_text(self, text):  # text: [B, 24] long, CLIP id + 1, 0 = PAD
        e = self.text_proj(self.clip_ln(self.clip_table[text].float()))
        return e + self.text_resid(self.clip2v2[text])

    def forward(self, text, img):
        """text [B,24] long, img [B,256] long palette ids -> logits [B,256,4097]."""
        b = text.shape[0]
        d = self.config.d_model
        t = self.embed_text(text)
        im = self.img_emb(img[:, :N_PIX - 1]) + self.row_in(self.pr[:N_PIX - 1]) + self.col_in(self.pc[:N_PIX - 1])
        x = torch.cat([self.bos.expand(b, 1, d), t, self.sep.expand(b, 1, d), im], 1)
        x = x + self.pos.weight[None]
        tgt = self.row_out(self.pr) + self.col_out(self.pc)  # [256, d], for output positions 25..280
        x = torch.cat([x[:, :OUT0], x[:, OUT0:] + tgt[None]], 1)
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.ln_f(x[:, OUT0:]))

    def n_params(self, trainable=True):
        return sum(p.numel() for p in self.parameters() if p.requires_grad or not trainable)


def _init(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Embedding):
        nn.init.normal_(m.weight, std=0.02)


def losses(logits, img, alpha_weight: float):
    """Exact 4097-way NLL (mean over pixels) and the auxiliary alpha NLL.

    alpha NLL = -log P(transparent) for transparent targets, -log(1 - P(transparent))
    for opaque ones, from the same softmax (index 0 = transparent). It is the
    coarse marginal of the same distribution, so it never leaks mass between
    transparent and opaque colours and its optimum coincides with exact NLL."""
    logp = F.log_softmax(logits.float(), dim=-1)
    nll = -logp.gather(-1, img.unsqueeze(-1)).squeeze(-1)
    lp0 = logp[..., 0]
    is_t = img == 0
    lop = torch.log(-torch.expm1(lp0.clamp(max=-1e-6)))
    a_nll = torch.where(is_t, -lp0, -lop)
    total = nll.mean() + alpha_weight * a_nll.mean()
    return total, nll, a_nll, lp0


@torch.no_grad()
def sample(model, text, *, temperature=0.8, top_k=32, cfg=1.0, generator=None):
    """Autoregressive sampling, no KV cache (causal: unknown future pixels are
    filled with 0 and never seen). cfg > 1 uses classifier-free guidance with an
    all-PAD caption as the unconditional branch."""
    model.eval()
    b = text.shape[0]
    dev = text.device
    img = torch.zeros(b, N_PIX, dtype=torch.long, device=dev)
    if cfg != 1.0:
        tt = torch.cat([text, torch.zeros_like(text)], 0)
    for k in range(N_PIX):
        with torch.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            if cfg != 1.0:
                lg = model(tt, torch.cat([img, img], 0))[:, k].float()
                lg = lg[b:] + cfg * (lg[:b] - lg[b:])
            else:
                lg = model(text, img)[:, k].float()
        if temperature <= 0:
            nxt = lg.argmax(-1)
        else:
            lg = lg / temperature
            if top_k:
                v, _ = lg.topk(top_k, -1)
                lg = lg.masked_fill(lg < v[:, -1:], float("-inf"))
            nxt = torch.multinomial(F.softmax(lg, -1), 1, generator=generator).squeeze(-1)
        img[:, k] = nxt
    return img
