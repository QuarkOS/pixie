"""CPU unit tests for Pixie v3 (run: python -m pytest -q test_v3.py, or python test_v3.py)."""
import itertools, math, os, sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
import v3_model as M
from train_v3 import to_v2_text, empty_text

torch.manual_seed(0)
CLIP_PT = os.environ.get("PIXIE_CLIP_PT", "")


def tiny(arch):
    c = M.V3Config(arch=arch, d_model=32, n_layer=2, n_head=4, dropout=0.0)
    return M.build(c).eval()


def rand_inputs(b=2):
    return torch.randn(b, M.TEXT_LEN, M.CLIP_D), torch.randint(0, M.PALETTE, (b, 256))


def test_patch_roundtrip_and_order():
    img = torch.arange(256)[None]
    p = M.to_patches(img)
    assert p.shape == (1, 64, 4)
    assert p[0, 0].tolist() == [0, 1, 16, 17]          # TL, TR, BL, BR of patch 0
    assert p[0, 1].tolist() == [2, 3, 18, 19]
    assert p[0, 8].tolist() == [32, 33, 48, 49]        # second patch row
    assert torch.equal(M.from_patches(p), img)
    s = M.slot_of_pixel("cpu")
    assert s[0] == 0 and s[1] == 1 and s[16] == 2 and s[17] == 3


def test_shapes():
    for arch in ("pix", "p50"):
        m = tiny(arch); h, img = rand_inputs()
        lg = m(h, img)
        assert lg.shape == (2, 256, M.VOCAB)
        assert torch.isinf(lg[..., M.PALETTE:]).all() and torch.isfinite(lg[..., :M.PALETTE]).all()


def test_head_slot_causality():
    """logits_k of a patch must not depend on p_j for j >= k, and must depend on j < k."""
    head = M.SlotCausalHead(16).eval()
    h = torch.randn(3, 5, 16); p = torch.randint(0, M.PALETTE, (3, 5, 4))
    base = head(h, p)
    for j in range(4):
        q = p.clone(); q[..., j] = (q[..., j] + 7) % M.PALETTE
        new = head(h, q)
        for k in range(4):
            same = torch.allclose(base[:, :, k], new[:, :, k], atol=1e-6)
            assert same == (k <= j), (j, k)
    # step() matches the teacher-forced pass
    for k in range(4):
        st = head.step(h[:, 2], [p[:, 2, j] for j in range(k)])
        assert torch.allclose(st, base[:, 2, k], atol=1e-5)


def test_p50_patch_causality():
    m = tiny("p50"); h, img = rand_inputs()
    base = m(h, img)
    P = M.to_patches(img)
    for mpatch in (0, 9, 37, 63):
        for slot in range(4):
            Q = P.clone(); Q[:, mpatch, slot] = (Q[:, mpatch, slot] + 11) % M.PALETTE
            new = m(h, M.from_patches(Q))
            lb = M.to_patches_logits(base) if hasattr(M, "to_patches_logits") else None
            b4 = base.view(2, 8, 2, 8, 2, -1).permute(0, 1, 3, 2, 4, 5).reshape(2, 64, 4, -1)
            n4 = new.view(2, 8, 2, 8, 2, -1).permute(0, 1, 3, 2, 4, 5).reshape(2, 64, 4, -1)
            assert torch.allclose(b4[:, :mpatch], n4[:, :mpatch], atol=1e-5)              # earlier patches
            assert torch.allclose(b4[:, mpatch, :slot + 1], n4[:, mpatch, :slot + 1], atol=1e-5)  # same patch, slots <= changed
            if slot < 3:
                assert not torch.allclose(b4[:, mpatch, slot + 1:], n4[:, mpatch, slot + 1:], atol=1e-6)
            if mpatch < 63:
                assert not torch.allclose(b4[:, mpatch + 1:], n4[:, mpatch + 1:], atol=1e-6)


def test_pix_causality():
    m = tiny("pix"); h, img = rand_inputs()
    base = m(h, img)
    for k in (0, 17, 100, 254, 255):
        q = img.clone(); q[:, k] = (q[:, k] + 5) % M.PALETTE
        new = m(h, q)
        assert torch.allclose(base[:, :k + 1], new[:, :k + 1], atol=1e-5)
        if k < 255:
            assert not torch.allclose(base[:, k + 1:], new[:, k + 1:], atol=1e-6)


def test_text_conditioning():
    for arch in ("pix", "p50"):
        m = tiny(arch); h, img = rand_inputs()
        assert not torch.allclose(m(h, img), m(h * 0, img))


def _restricted(m, K):
    m.vmask = m.vmask.clone(); m.vmask[K:] = float("-inf")
    return m


def _logp_prefix(m, h, prefix_pixels, order):
    """log P(first len(prefix) pixels in `order`), all later pixels 0."""
    img = torch.zeros(1, 256, dtype=torch.long)
    for pos, v in zip(order, prefix_pixels):
        img[0, pos] = v
    lg = m(h, img).float()
    lp = torch.log_softmax(lg, -1)
    return float(sum(lp[0, pos, v] for pos, v in zip(order, prefix_pixels)).detach())


def test_toy_normalisation_both_models():
    """Both factorisations define proper distributions: the probability of the first 6 pixels (in each
    model's own generation order) sums to 1 over all 3^6 colourings, so a mean over 256 pixel NLLs is an
    exact image NLL for both, and the numbers are directly comparable."""
    K = 3
    h = torch.randn(1, M.TEXT_LEN, M.CLIP_D)
    pix_order = list(range(6))
    patch_order = M.to_patches(torch.arange(256)[None])[0, :2].reshape(-1).tolist()[:6]  # patch0, then 2 slots of patch1
    for m, order in ((_restricted(tiny("pix"), K), pix_order), (_restricted(tiny("p50"), K), patch_order)):
        tot = 0.0
        for combo in itertools.product(range(K), repeat=6):
            tot += math.exp(_logp_prefix(m, h, combo, order))
        assert abs(tot - 1.0) < 1e-4, tot


def test_toy_nll_matches_between_models():
    """With uniform heads (zeroed output layers) on a K-colour toy palette, the patch and pixel models
    give the identical 256-pixel NLL: 256*log(K), i.e. log(K) per pixel, through the same loss code."""
    K = 5
    h, _ = rand_inputs(3)
    img = torch.randint(0, K, (3, 256))
    pix = _restricted(tiny("pix"), K); p50 = _restricted(tiny("p50"), K)
    torch.nn.init.zeros_(pix.head.weight); torch.nn.init.zeros_(p50.head.W_out.weight)
    n1 = M.losses(pix(h, img), img, 0.0)[1]; n2 = M.losses(p50(h, img), img, 0.0)[1]
    assert torch.allclose(n1, n2, atol=1e-5)
    assert abs(float(n1.sum(1).mean()) - 256 * math.log(K)) < 1e-3


def test_loss_equals_cross_entropy():
    m = tiny("p50"); h, img = rand_inputs()
    lg = m(h, img)
    _, nll, _, _ = M.losses(lg, img, 0.0)
    ce = torch.nn.functional.cross_entropy(lg[..., :M.PALETTE].reshape(-1, M.PALETTE), img.reshape(-1))
    assert abs(float(nll.mean()) - float(ce)) < 1e-5


def test_boundary_ratio():
    g = torch.Generator().manual_seed(0)
    noise = torch.randint(0, 50, (64, 256), generator=g)
    r, _, _ = M.boundary_ratio(noise)
    assert 0.95 < r < 1.05
    blocks = M.from_patches(torch.randint(0, 50, (64, 64, 1), generator=g).expand(64, 64, 4).contiguous())
    r2, _, inside = M.boundary_ratio(blocks)
    assert inside == 0.0 and r2 > 1e6


def test_to_v2_text_and_empty():
    t = torch.tensor([[49406, 736, 46055, 49407] + [49407] * 22,
                      [49406] + list(range(100, 124)) + [49407]])
    v = to_v2_text(t)
    assert v[0, :3].tolist() == [737, 46056, 0] and int(v[0].sum()) == 737 + 46056
    assert v[1].tolist() == list(range(101, 125))
    e = empty_text(2, "cpu")
    assert to_v2_text(e).abs().sum() == 0


def test_sampling_runs():
    for arch in ("pix", "p50"):
        m = tiny(arch)
        h = torch.randn(2, M.TEXT_LEN, M.CLIP_D)
        g = torch.Generator().manual_seed(0)
        s = m.sample(h, torch.zeros_like(h), cfg=2.0, generator=g)
        assert s.shape == (2, 256) and int(s.max()) < M.PALETTE
        g1 = torch.Generator().manual_seed(1); g2 = torch.Generator().manual_seed(1)
        assert torch.equal(m.sample(h, generator=g1, temperature=0), m.sample(h, generator=g2, temperature=0))


def test_greedy_matches_teacher_forced_argmax():
    m = tiny("p50"); h = torch.randn(1, M.TEXT_LEN, M.CLIP_D)
    s = m.sample(h, temperature=0)
    lg = m(h, s)
    assert torch.equal(lg[..., :M.PALETTE].argmax(-1), s)


def test_clip_penultimate_matches_hf():
    if not CLIP_PT or not Path(CLIP_PT).exists():
        print("skip clip test (set PIXIE_CLIP_PT)"); return
    from transformers import CLIPTextConfig, CLIPTextModel
    sd = torch.load(CLIP_PT, map_location="cpu")
    enc = M.ClipTextPenultimate(); enc.load_hf(sd); enc.eval()
    cfg = CLIPTextConfig(vocab_size=49408, hidden_size=768, intermediate_size=3072, num_hidden_layers=12,
                         num_attention_heads=12, max_position_embeddings=77, hidden_act="quick_gelu",
                         projection_dim=768, eos_token_id=49407, bos_token_id=49406)
    hf = CLIPTextModel(cfg).eval()
    hf_keys = set(hf.state_dict())
    sd2 = {(k if k in hf_keys else k[len("text_model."):]): v for k, v in sd.items()}  # transformers>=5 drops the prefix
    missing, unexpected = hf.load_state_dict(sd2, strict=False)
    assert not unexpected and all(".layers.11." in k or "final_layer_norm" in k or "position_ids" in k for k in missing)
    ids = torch.tensor([[49406, 736, 46055, 49407] + [49407] * 22, [49406, 1237, 2034, 6765, 49407] + [49407] * 21])
    with torch.no_grad():
        ref = hf(input_ids=ids, output_hidden_states=True).hidden_states[-2]
        got = enc(ids)
    err = float((ref - got).abs().max())
    print("clip max abs err", err)
    assert err < 1e-3


if __name__ == "__main__":
    fails = 0
    for k, f in list(globals().items()):
        if k.startswith("test_"):
            try:
                f(); print("PASS", k)
            except Exception as e:
                fails += 1; print("FAIL", k, type(e).__name__, e)
    print("failures:", fails); sys.exit(1 if fails else 0)
