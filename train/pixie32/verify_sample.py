"""Read-only validation of the supplied loader sample, never training admission."""
import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw


def inspect_sample(root, output):
    if output.exists():
        raise ValueError("Sample report output must be new")
    output.mkdir(parents=True, exist_ok=False)
    paths = (root / "r1_list.txt").read_text(encoding="utf-8").splitlines()
    edge = set((root / "edge_px_ge32_list.txt").read_text(encoding="utf-8").splitlines())
    dimensions = set()
    alphas = set()
    pixel_hashes = set()
    caption_count = 0
    sheet = Image.new("RGB", (10 * 128, 20 * 144), "#e8edf2")
    draw = ImageDraw.Draw(sheet)
    for i, name in enumerate(paths):
        path = (root / name).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError("Sample path escapes its root")
        image = Image.open(path).convert("RGBA")
        dimensions.add(image.size)
        alphas.update(image.getchannel("A").tobytes())
        pixel_hashes.add(hashlib.sha256(image.tobytes()).hexdigest())
        if path.with_suffix(".txt").read_text(encoding="utf-8").strip():
            caption_count += 1
        thumb = image.resize((128, 128), Image.Resampling.NEAREST)
        sheet.paste(thumb, ((i % 10) * 128, (i // 10) * 144), thumb)
        draw.text(((i % 10) * 128 + 4, (i // 10) * 144 + 129), str(i), fill="#182434")
    missing = [name for name in ("manifest.jsonl", "final/metadata.jsonl") if not (root / name).is_file()]
    report = {
        "purpose": "Read-only loader sample validation; no training performed",
        "listed_images": len(paths), "unique_ids": len(set(paths)),
        "nonempty_captions": caption_count, "dimensions": sorted(dimensions),
        "alpha_values": sorted(alphas), "unique_pixel_hashes": len(pixel_hashes),
        "edge_list_intersection": len(set(paths) & edge),
        "missing_required_provenance_files": missing,
        "production_builder_ready": not missing,
    }
    assert dimensions == {(32, 32)} and alphas <= {0, 255}
    assert caption_count == len(paths) == len(set(paths))
    sheet.save(output / "sample-contact.png")
    (output / "sample-validation.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    inspect_sample(args.sample, args.out)
