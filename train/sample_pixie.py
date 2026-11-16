"""Sample sprites from a released Pixie checkpoint (model.safetensors).

Works for both releases: pixie-32 (P32, native 32x32, config arch "p32") and
pixie-16 (v3 / P50, 16x16, config arch "p50"). The release folder holds
model.safetensors, config.json and palette.npy. The text encoder is not
shipped: download openai/clip-vit-large-patch14 yourself and pass its
model.safetensors (or pytorch_model.bin) and tokenizer.json.

    python -m train.sample_pixie --model-dir pixie-32 \
        --clip clip-vit-large-patch14/model.safetensors \
        --tokenizer clip-vit-large-patch14/tokenizer.json \
        --prompt "red apple" --prompt "iron helmet" --out sprites.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from train.pixie32.build_data_p32 import encode
from train.pixie32.p32_model import P32, P32Config
from train.v3_model import ClipTextPenultimate, V3Config, build


def load_state(path: Path) -> dict:
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file
        return load_file(str(path))
    return torch.load(path, map_location="cpu", weights_only=True)


def _fields(cls, config: dict) -> dict:
    return {k: v for k, v in config.items() if k in cls.__dataclass_fields__}


def load_model(model_dir: Path, device: torch.device):
    """Return (model, palette, side) where side is 32 for P32 and 16 for v3."""
    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    if config.get("arch") == "p32":
        model, side = P32(P32Config(**_fields(P32Config, config))), 32
    else:
        model, side = build(V3Config(**_fields(V3Config, config))), 16
    model.load_state_dict(load_state(model_dir / "model.safetensors"), strict=True)
    palette = np.load(model_dir / config.get("palette_file", "palette.npy"))
    return model.to(device).eval(), palette, side


def load_clip(path: Path, device: torch.device) -> ClipTextPenultimate:
    clip = ClipTextPenultimate()
    clip.load_hf(load_state(path))
    return clip.to(device).eval()


def empty_text(batch: int, device: torch.device) -> torch.Tensor:
    text = torch.full((batch, 18), 49407, dtype=torch.long, device=device)
    text[:, 0] = 49406
    return text


def to_rgba(indices: np.ndarray, palette: np.ndarray) -> np.ndarray:
    rgb = palette[indices][..., :3].astype(np.uint8)
    alpha = ((indices != 0) * 255).astype(np.uint8)[..., None]
    return np.concatenate([rgb, alpha], axis=-1)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--clip", type=Path, required=True, help="CLIP ViT-L/14 weights with HF text_model.* names")
    p.add_argument("--tokenizer", type=Path, required=True, help="CLIP tokenizer.json")
    p.add_argument("--prompt", action="append", required=True)
    p.add_argument("--n", type=int, default=4, help="samples per prompt")
    p.add_argument("--cfg", type=float, default=2.0)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--scale", type=int, default=4)
    p.add_argument("--out", type=Path, default=Path("sprites.png"))
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args(argv)

    from PIL import Image
    from tokenizers import Tokenizer

    device = torch.device(a.device)
    model, palette, side = load_model(a.model_dir, device)
    clip = load_clip(a.clip, device)
    tokenizer = Tokenizer.from_file(str(a.tokenizer))
    generator = torch.Generator(device=device).manual_seed(a.seed)

    rows = []
    for prompt in a.prompt:
        ids, _ = encode(tokenizer, prompt)
        text = torch.as_tensor(np.stack([ids] * a.n).astype(np.int64), device=device)
        out = model.sample(clip(text), clip(empty_text(a.n, device)), cfg=a.cfg,
                           temperature=a.temperature, top_k=a.top_k, generator=generator)
        rows.append(np.concatenate([to_rgba(x, palette) for x in out.cpu().numpy().reshape(-1, side, side)], axis=1))
    grid = Image.fromarray(np.concatenate(rows, axis=0), "RGBA")
    grid = grid.resize((grid.width * a.scale, grid.height * a.scale), Image.Resampling.NEAREST)
    grid.save(a.out)
    print(json.dumps({"out": str(a.out), "prompts": a.prompt, "per_prompt": a.n, "seed": a.seed, "size": side}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
