"""Pixie v3 env check: flash SDPA + (in a separate venv) triton-windows / torch.compile. Writes JSON."""
import json, sys, time, traceback, platform
out = sys.argv[1]
r = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "python": sys.version, "executable": sys.executable, "platform": platform.platform()}
try:
    import torch, torch.nn.functional as F
    r["torch"] = torch.__version__; r["cuda"] = torch.version.cuda; r["cuda_available"] = torch.cuda.is_available()
    r["device"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    r["flash_sdp_enabled_flag"] = torch.backends.cuda.flash_sdp_enabled()
    r["mem_efficient_sdp_enabled_flag"] = torch.backends.cuda.mem_efficient_sdp_enabled()
    from torch.nn.attention import sdpa_kernel, SDPBackend
    q = torch.randn(8, 8, 283, 32, device="cuda", dtype=torch.bfloat16)
    for name, be in (("flash", SDPBackend.FLASH_ATTENTION), ("efficient", SDPBackend.EFFICIENT_ATTENTION), ("math", SDPBackend.MATH)):
        try:
            with sdpa_kernel(be):
                o = F.scaled_dot_product_attention(q, q, q, is_causal=True); torch.cuda.synchronize()
            t0 = time.time()
            with sdpa_kernel(be):
                for _ in range(50):
                    o = F.scaled_dot_product_attention(q, q, q, is_causal=True)
            torch.cuda.synchronize()
            r[f"sdpa_{name}_call"] = "ok"; r[f"sdpa_{name}_ms"] = round((time.time() - t0) * 20, 3)
        except Exception as e:
            r[f"sdpa_{name}_call"] = f"fail: {type(e).__name__}: {str(e)[:300]}"
    try:
        import triton
        r["triton"] = triton.__version__
    except Exception as e:
        r["triton"] = f"not importable: {type(e).__name__}: {str(e)[:200]}"
    if "--compile" in sys.argv:
        try:
            m = torch.nn.Sequential(torch.nn.Linear(256, 1024), torch.nn.GELU(), torch.nn.Linear(1024, 256)).cuda()
            x = torch.randn(4096, 256, device="cuda")
            cm = torch.compile(m)
            t0 = time.time(); y = cm(x); torch.cuda.synchronize(); r["compile_first_call_s"] = round(time.time() - t0, 2)
            r["compile_max_abs_err"] = float((y - m(x)).abs().max())
            r["compile"] = "ok"
        except Exception as e:
            r["compile"] = f"fail: {type(e).__name__}: {str(e)[:500]}"
except Exception as e:
    r["error"] = traceback.format_exc()[-2000:]
open(out, "w").write(json.dumps(r, indent=1)); print(json.dumps(r))
