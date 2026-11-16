"""Rank possible caption mismatches using explicitly local SigLIP assets."""
from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from train.pixie32.audit_r1 import contained, load_population, write_json
from train.pixie32.data import sha256


TEMPLATE = "a 32x32 pixel art game icon of {caption}"


def model_hash(directory):
    digest = hashlib.sha256()
    files = sorted(path for path in directory.rglob("*") if path.is_file())
    if not files:
        raise ValueError("Local model directory is empty")
    for path in files:
        digest.update(path.relative_to(directory).as_posix().encode("utf-8"))
        digest.update(bytes.fromhex(sha256(path)))
    return digest.hexdigest()


class LocalSiglip:
    def __init__(self, directory, device):
        if not directory.is_dir():
            raise ValueError("Supply an explicit local model directory")
        import torch
        from transformers import AutoModel, AutoProcessor
        self.device = device
        self.processor = AutoProcessor.from_pretrained(str(directory), local_files_only=True, trust_remote_code=False)
        self.model = AutoModel.from_pretrained(str(directory), local_files_only=True, trust_remote_code=False)
        if self.model.config.model_type not in ("siglip", "siglip2"):
            raise ValueError("Caption checker requires local SigLIP/SigLIP2 assets")
        self.model.to(device=device, dtype=torch.float32).eval()

    def score(self, images, texts):
        import torch
        inputs = self.processor(images=images, text=texts, padding="max_length", truncation=True, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.no_grad():
            output = self.model(**inputs)
        image = torch.nn.functional.normalize(output.image_embeds.float(), dim=-1)
        text = torch.nn.functional.normalize(output.text_embeds.float(), dim=-1)
        scores = (image @ text.T).cpu().numpy()
        if scores.shape != (len(images), len(texts)) or not np.isfinite(scores).all() or np.any(np.abs(scores) > 1.00001):
            raise ValueError("Semantic scorer returned invalid features")
        return scores


def negatives(rows, count):
    subjects = {}
    for row in rows:
        subjects.setdefault(row["subject"], row)
    names = sorted(subjects)
    if len(names) <= count:
        raise ValueError("Need more distinct subjects than negative captions")
    result = {}
    for row in rows:
        start = bisect.bisect_left(names, row["subject"])
        result[row["id"]] = [subjects[names[(start + step) % len(names)]] for step in range(1, count + 1)]
    return result


def score_rows(rows, root, scorer, *, batch, negative_count, margin_threshold=0.):
    comparison = negatives(rows, negative_count)
    results = []
    for start in range(0, len(rows), batch):
        chunk = rows[start:start + batch]
        images, texts = [], []
        for row in chunk:
            with Image.open(contained(root, row["file"])) as image:
                sprite = image.convert("RGBA")
            if sprite.size != (32, 32):
                raise ValueError("Caption checker requires native 32x32 sprites")
            background = Image.new("RGBA", sprite.size, (180, 180, 180, 255))
            background.alpha_composite(sprite)
            images.append(background.convert("RGB"))
            texts.extend(TEMPLATE.format(caption=entry["caption"]) for entry in [row, *comparison[row["id"]]])
        scores = np.asarray(scorer.score(images, texts), dtype=np.float64)
        if scores.shape != (len(chunk), len(texts)) or not np.isfinite(scores).all() or np.any(np.abs(scores) > 1.00001):
            raise ValueError("Scorer must return finite image-by-caption cosine scores")
        for offset, row in enumerate(chunk):
            columns = scores[offset, offset * (negative_count + 1):(offset + 1) * (negative_count + 1)]
            best = int(np.argmax(columns[1:]))
            own, other = float(columns[0]), float(columns[best + 1])
            path = contained(root, row["file"])
            results.append(dict(id=row["id"], file=row["file"], caption=row["caption"],
                                image_sha256=sha256(path), caption_sha256=sha256(path.with_suffix(".txt")),
                                caption_cosine=own, best_negative_cosine=other, margin=own - other,
                                review_flag=own - other < margin_threshold,
                                best_negative_id=comparison[row["id"]][best]["id"],
                                best_negative_caption=comparison[row["id"]][best]["caption"],
                                negative_ids=[entry["id"] for entry in comparison[row["id"]]]))
    return results


def prepare(args, scorer=None):
    if args.out.exists() or not 1 <= args.batch <= 32 or not 1 <= args.negatives <= 8 or not 0 <= args.review <= 400 or not math.isfinite(args.margin):
        raise ValueError("Choose a fresh folder, batch1..32, negatives1..8, review0..400 and finite margin")
    rows, policy, excluded = load_population(args)
    if args.expect_rows is not None and len(rows) != args.expect_rows:
        raise ValueError("Non-holdout population differs from expected rows")
    if args.device != "cpu":
        import torch
        from train.pixie32.clip_cache import idle_device
        device = torch.device(args.device)
        if device.type != "cuda" or device.index is None:
            raise ValueError("Select cpu or one explicit cuda index")
        idle_device(args.device)
        torch.cuda.set_device(device)
    if scorer is None:
        scorer = LocalSiglip(args.model_dir, args.device)
    results = score_rows(rows, args.data_root, scorer, batch=args.batch, negative_count=args.negatives, margin_threshold=args.margin)
    results.sort(key=lambda row: (row["margin"], row["id"]))
    args.out.mkdir(parents=True, exist_ok=False)
    write_json(args.out / "caption-scores.json", results)
    write_json(args.out / "review.json", results[:args.review])
    for start in range(0, min(args.review, len(results)), 40):
        page = Image.new("RGB", (8 * 144, 5 * 160), (180, 180, 180))
        draw = ImageDraw.Draw(page)
        for offset, row in enumerate(results[start:min(start + 40, args.review)]):
            x, y = offset % 8 * 144, offset // 8 * 160
            with Image.open(contained(args.data_root, row["file"])) as image:
                sprite = image.convert("RGBA").resize((128, 128), Image.Resampling.NEAREST)
            page.paste(sprite, (x, y), sprite)
            draw.text((x, y + 130), f"{start + offset}: {row['margin']:.3f}", fill="black")
        page.save(args.out / f"caption-review-{start // 40:03d}.png")
    report = dict(schema="pixie32-caption-review-v1", rows=len(rows), holdouts_excluded=excluded,
                  inputs={"metadata": sha256(args.data_root / "final/metadata.jsonl"),
                          "provenance": sha256(args.data_root / "manifest.jsonl"), "r1": sha256(args.r1),
                          "holdouts": policy.hashes, "model_directory_sha256": model_hash(args.model_dir)},
                  template=TEMPLATE, precision="float32", background_rgb=[180, 180, 180], negative_count=args.negatives,
                  negative_design="next distinct subjects in sorted normalized subject order; deterministic diagnostic alternatives, not exhaustive negatives",
                  processor_truncation=True, review_margin=args.margin, flagged=sum(row["review_flag"] for row in results),
                  status="uncalibrated diagnostic ranking; no rows rejected or downweighted; not a measured mismatch rate",
                  limitations="A negative winning can reflect ambiguity or a weak scorer. A positive margin does not prove caption correctness. Calibrate with human labels before filtering.")
    write_json(args.out / "caption-summary.json", report)
    return report


def parser():
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("data-root", "r1", "blind50", "eval50", "out", "model-dir"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--device", required=True, help="cpu or one explicit cuda index")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--negatives", type=int, default=3)
    p.add_argument("--review", type=int, default=200)
    p.add_argument("--margin", type=float, default=0., help="uncalibrated review threshold only; never changes training weights")
    p.add_argument("--expect-rows", type=int)
    return p


def main(argv=None):
    p = parser()
    try:
        result = prepare(p.parse_args(argv))
    except (OSError, ValueError, KeyError, TypeError, ImportError, RuntimeError) as error:
        p.error(f"caption check refused ({type(error).__name__}); requires explicit local SigLIP assets and optional project-venv transformers dependencies")
    print(json.dumps(result, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
