#!/usr/bin/env python3
"""Export the 4097-entry palette JSON from a v3 checkpoint.

Usage:
  python3 export_palette4096.py --ckpt /path/to/checkpoint.pt --out web/public/palette4096.json
"""
import argparse
import json
import sys


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="Checkpoint or palette .pt file containing palette RGBs")
    p.add_argument("--out", default="palette4096.json", help="Output JSON path")
    args = p.parse_args()

    try:
        import torch
    except ImportError:
        sys.exit("PyTorch is required to read .pt checkpoints.")

    obj = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    pal = None
    for key in ("palette", "palette_rgb", "colors", "vocab_rgb"):
        if isinstance(obj, dict) and key in obj:
            pal = obj[key]
            break
    if pal is None and torch.is_tensor(obj):
        pal = obj
    if pal is None:
        sys.exit(f"Could not find palette tensor in {args.ckpt}; keys={list(obj.keys()) if isinstance(obj, dict) else type(obj)}")

    arr = pal.detach().cpu().numpy().tolist()
    if len(arr) == 4096:
        out = [None] + [[int(c[0]), int(c[1]), int(c[2])] for c in arr]
    elif len(arr) >= 4097:
        out = [None] + [[int(c[0]), int(c[1]), int(c[2])] for c in arr[1:4097]]
    else:
        sys.exit(f"Unexpected palette length: {len(arr)}")

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, separators=(",", ":"))
    print(f"Wrote {len(out)} entries to {args.out}")


if __name__ == "__main__":
    main()
