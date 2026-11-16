"""Build frozen CLIP features in a new folder; CUDA uses the trainer's BF16 path."""
import argparse
import os
from pathlib import Path

import torch

from train.pixie32.clip_cache import build_cache, idle_device
from train.pixie32.data import DEFAULT_BLIND50, DEFAULT_EVAL50, load_dataset, sha256
from train.pixie32.train_p32 import load_clip


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--clip-dir", type=Path, default=os.environ.get("CLIP_DIR"))
    parser.add_argument("--clip", type=Path)
    parser.add_argument("--blind50", type=Path, default=DEFAULT_BLIND50)
    parser.add_argument("--eval50", type=Path, default=DEFAULT_EVAL50)
    parser.add_argument("--device", required=True)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--clip-cache-max-gb", type=float, default=4)
    args = parser.parse_args()
    device = idle_device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    weights = args.clip or (args.clip_dir / "clip_l14_text_l11_f32.pt" if args.clip_dir else None)
    if weights is None or not weights.is_file():
        parser.error("provide --clip-dir, CLIP_DIR or --clip")
    dataset = load_dataset(args.data, args.blind50, args.eval50)
    result = build_cache(dataset, load_clip(weights, device), sha256(weights), args.out,
                         device, batch=args.batch, max_gb=args.clip_cache_max_gb)
    print(f"CLIP cache complete: unique_token_rows={result['rows']} compute={result['compute']}")


if __name__ == "__main__":
    main()
