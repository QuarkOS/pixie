"""Fetch openai/clip-vit-large-patch14 text_model.* tensors (layers 0..10 + embeddings; the last layer
and final LayerNorm are not needed for penultimate features) by HTTP range requests. Saves fp32 torch
state dict with HF names + a sha256 over the raw tensor bytes (same on every machine)."""
import hashlib, json, struct, sys, urllib.request
import numpy as np, torch

URL = "https://huggingface.co/openai/clip-vit-large-patch14/resolve/main/model.safetensors"


def rng(a, b):
    req = urllib.request.Request(URL, headers={"Range": f"bytes={a}-{b}"})
    return urllib.request.urlopen(req, timeout=120).read()


def wanted(k):
    if not k.startswith("text_model.") or k.endswith("position_ids"):
        return False
    if k.startswith("text_model.final_layer_norm") or ".layers.11." in k:
        return False
    return True


n = struct.unpack("<Q", rng(0, 7))[0]
hdr = json.loads(rng(8, 8 + n - 1))
keys = sorted(k for k in hdr if k != "__metadata__" and wanted(k))
items = sorted(((hdr[k]["data_offsets"][0], hdr[k]["data_offsets"][1], k) for k in keys))
lo, hi = items[0][0], max(e for _, e, _ in items)
h = hashlib.sha256(); sd = {}
# text tensors are contiguous-ish; fetch in 64 MB chunks covering [lo, hi)
buf = bytearray(); CH = 64 << 20
for s in range(lo, hi, CH):
    buf += rng(8 + n + s, 8 + n + min(hi, s + CH) - 1)
    print("fetched", len(buf) >> 20, "MB", flush=True)
for a, b, k in sorted(items, key=lambda x: x[2]):
    m = hdr[k]; assert m["dtype"] == "F32"
    raw = bytes(buf[a - lo:b - lo]); h.update(k.encode()); h.update(raw)
    sd[k] = torch.from_numpy(np.frombuffer(raw, dtype=np.float32).reshape(m["shape"]).copy())
torch.save(sd, sys.argv[1])
print(len(sd), "tensors", sum(v.numel() for v in sd.values()), "params sha256", h.hexdigest(), flush=True)
