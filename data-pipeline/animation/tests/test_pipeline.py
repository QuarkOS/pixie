"""Exercise the collection commands with generated CPU-only fixture assets."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from PIL import Image, ImageDraw

# Blind-50 / Eval-50 are not part of the public tree; tests that need the real
# files skip unless they are in <repo>/holdout/ or PIXIE_HOLDOUT_DIR.
import os as _os
import unittest as _unittest
from pathlib import Path as _Path
_HOLDOUT_DIR = _Path(_os.environ.get("PIXIE_HOLDOUT_DIR") or next(
    p for p in _Path(__file__).resolve().parents if (p / "train" / "pixie32").is_dir()) / "holdout")
needs_holdouts = _unittest.skipUnless(
    (_HOLDOUT_DIR / "blind-50.json").is_file() and (_HOLDOUT_DIR / "eval-50.json").is_file(),
    "Blind-50/Eval-50 holdout files are not shipped; set PIXIE_HOLDOUT_DIR")



PIPELINE = Path(__file__).resolve().parents[2]
REPO = PIPELINE.parent
FROZEN_HASHES = {
    "blind-50.json": "7524db9cabd2449f5ec6b48b454bc10e2fed38825ba8c8d1e749311f3b5afb0a",
    "eval-50.json": "a1d6893e8c3c45ab99a845c1726c55547dcde29ff6f7a94b022f067c27a7872f",
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]


def inventory(root):
    return {path.relative_to(root).as_posix(): digest(path)
            for path in root.rglob("*") if path.is_file()}


class PipelineCliTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.assertEqual(self.holdout_hashes(), FROZEN_HASHES)
        self.addCleanup(lambda: self.assertEqual(self.holdout_hashes(), FROZEN_HASHES))

    def holdout_hashes(self):
        return {name: digest(_HOLDOUT_DIR / name) for name in FROZEN_HASHES}

    def cli(self, module, *args, succeeds=True):
        result = subprocess.run(
            [sys.executable, "-m", "animation." + module, *map(str, args)],
            cwd=PIPELINE, text=True, capture_output=True, timeout=30,
        )
        if succeeds:
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        else:
            self.assertNotEqual(result.returncode, 0)
        return result

    def write_json(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
        return path

    def reviews(self, collection, path):
        rows = read_jsonl(collection / "sequences.jsonl")
        return self.write_json(path, {
            "collection_sha256": digest(collection / "manifest.json"),
            "reviews": [{"sequence_id": row["sequence_id"], "status": "accepted"} for row in rows],
        })

    def check_manifest_files(self, root):
        manifest = read_json(root / "manifest.json")
        for relative, expected in manifest["files"].items():
            self.assertEqual(digest(root / relative), expected, relative)
        self.assertEqual(set(manifest["files"]), set(inventory(root)) - {"manifest.json"})
        return manifest

    @needs_holdouts
    def test_manual_magenta_pack_crop_timing_and_64_sheet_roundtrip(self):
        pack = self.root / "manual-pack"
        pack.mkdir()
        entries = []
        for index in range(4):
            name = f"phase-{index}.png"
            image = Image.new("RGBA", (78, 58), (255, 0, 255, 255))
            ImageDraw.Draw(image).rectangle((30 + index, 24, 37 + index, 31),
                                           fill=(20, 40, 60, 255))
            image.save(pack / name)
            entries.append({"path": name})
        spec = self.write_json(pack / "ingest.json", {"sequences": [{
            "sequence_id": "fixture-copper-motion", "lineage_group": "fixture-copper-family",
            "source_url": "https://example.org/manual-animation-pack", "subject": "fixture copperform",
            "action": "slide", "direction": "right", "loop": False, "anchor_px": [34, 32],
            "format": "folder", "frames": entries,
        }]})
        import zipfile

        archive = self.root / "private-download-name.zip"
        with zipfile.ZipFile(archive, "w") as bundle:
            for path in sorted(pack.iterdir()):
                bundle.write(path, path.name)
        before = inventory(pack)
        download = self.root / "download-record"
        self.cli("download", "--local-file", archive,
                 "--source-url", "https://example.org/manual-animation-pack", "--output", download)
        record = read_json(download / "download.json")
        self.assertEqual(record["acquisition"], "manual_download")
        self.assertEqual(record["sha256"], digest(archive))
        self.assertEqual((download / "asset.bin").read_bytes(), archive.read_bytes())
        self.assertNotIn(archive.name, (download / "download.json").read_text())
        collection = self.root / "keyed-cropped-64"
        self.cli("ingest", "--input", pack, "--spec", spec, "--output", collection,
                 "--key-rgb", 255, 0, 255, "--crop-to-content", "--crop-padding", 1,
                 "--canvas", 64, 64, "--default-duration-ms", 100)
        row = read_jsonl(collection / "sequences.jsonl")[0]
        self.assertEqual(row["canvas_px"], [64, 64])
        self.assertEqual(row["normalization"]["original_canvas_px"], [78, 58])
        self.assertEqual(row["normalization"]["crop_box_px"], [29, 23, 42, 33])
        self.assertEqual(row["timing_origin"], "default")
        self.assertEqual([frame["duration_ms"] for frame in row["frames"]], [100] * 4)
        with Image.open(collection / row["original_masters"][0]["path"]) as master:
            self.assertEqual(master.size, (78, 58))
            self.assertEqual(master.getpixel((0, 0)), (255, 0, 255, 255))
        collection_before = inventory(collection)
        reviews = self.reviews(collection, self.root / "reviews-64.json")
        sheets = self.root / "sheets-64"
        self.cli("sheets", "--input", collection, "--reviews", reviews,
                 "--output", sheets, "--logical-px", 64, "--eval-fraction", 0)
        manifest = self.check_manifest_files(sheets)
        self.assertEqual(manifest["logical_px"], 64)
        self.assertEqual(manifest["counts"], {"train": 1, "eval": 0})
        sheet = manifest["records"][0]
        with Image.open(sheets / sheet["image"]) as image:
            self.assertEqual(image.size, (512, 512))
            self.assertEqual(image.getpixel((0, 0)), (255, 0, 255))
        scored = self.root / "review-64"
        self.cli("score", "--input", collection, "--output", scored)
        self.check_manifest_files(scored)
        score = read_jsonl(scored / "score.jsonl")[0]
        self.assertEqual(score["measurement_space"]["canvas_px"], [64, 64])
        self.assertTrue(score["measurement_space"]["source_resampled_during_ingest"])
        self.assertEqual(score["palette"]["visible_palette_union"], 1)
        self.assertEqual(inventory(pack), before)
        self.assertEqual(inventory(collection), collection_before)

    @needs_holdouts
    def test_artist_ingest_review_sheets_score_and_network_free_pilot(self):
        pack = self.root / "artist-pack"
        pack.mkdir()
        specs = []
        colors = {"alpha": (190, 70, 60, 255), "beta": (60, 160, 80, 255)}
        durations = [70, 90, 110, 130]
        for family, color in colors.items():
            frame_specs = []
            for index, duration_ms in enumerate(durations):
                name = f"{family}-{index}.png"
                image = Image.new("RGBA", (32, 32), (23, 41, 59, 0))
                ImageDraw.Draw(image).rectangle((6 + index * 2, 11, 9 + index * 2, 15), fill=color)
                image.save(pack / name)
                frame_specs.append({"path": name, "duration_ms": duration_ms})
            specs.append({
                "sequence_id": family + "-walk", "lineage_group": family,
                "source_url": "https://example.org/animation-fixture/" + family,
                "subject": "fixture " + family + "form", "action": "walk",
                "direction": "right", "loop": True, "anchor_px": [16, 24],
                "format": "folder", "frames": frame_specs,
            })
        spec = self.write_json(pack / "ingest.json", {"sequences": specs})
        pack_before = inventory(pack)
        collection = self.root / "artist-collection"
        self.cli("ingest", "--input", pack, "--spec", spec, "--output", collection)
        collection_before = inventory(collection)
        self.assertEqual(pack_before, inventory(pack))
        self.check_manifest_files(collection)
        sequences = read_jsonl(collection / "sequences.jsonl")
        self.assertEqual(len(sequences), 2)
        for row in sequences:
            self.assertEqual(row["review_status"], "unreviewed")
            self.assertEqual(row["frame_count"], 4)
            self.assertEqual(row["anchor_px"], [16, 24])
            self.assertEqual([frame["duration_ms"] for frame in row["frames"]], durations)
        review_file = self.reviews(collection, self.root / "artist-reviews.json")
        audit = self.write_json(self.root / "eval-subjects.json", ["fixture betaform"])
        sheets = self.root / "artist-sheets"
        self.cli("sheets", "--input", collection, "--reviews", review_file,
                 "--output", sheets, "--eval-fraction", 0, "--eval-subjects", audit)
        sheet_manifest = self.check_manifest_files(sheets)
        self.assertEqual(sheet_manifest["counts"], {"train": 1, "eval": 1})
        self.assertEqual(len(sheet_manifest["records"]), 2)
        for row in sheet_manifest["records"]:
            family = row["sequence_id"].split("-")[0]
            self.assertEqual(row["split"], "eval" if family == "beta" else "train")
            self.assertEqual(row["frame_order"], [0, 1, 2, 3])
            self.assertIn("4-frame animation sheet", (sheets / row["caption_file"]).read_text())
            with Image.open(sheets / row["image"]) as image:
                self.assertEqual(image.size, (512, 512))
                for index in range(4):
                    left, top = (index % 2) * 256, (index // 2) * 256
                    self.assertEqual(image.getpixel((left, top)), (255, 0, 255))
                    self.assertEqual(image.getpixel((left + (7 + index * 2) * 8, top + 12 * 8)), colors[family][:3])
        scored = self.root / "artist-review-export"
        self.cli("score", "--input", collection, "--output", scored)
        self.check_manifest_files(scored)
        scored_rows = read_jsonl(scored / "score.jsonl")
        self.assertEqual(len(scored_rows), 2)
        for row in scored_rows:
            self.assertEqual(row["status"], "UNCALIBRATED")
            self.assertEqual(row["identity"]["status"], "unavailable")
            self.assertFalse(row["motion"]["all_frames_identical"])
            artifacts = row["review_artifacts"]
            self.assertTrue((scored / artifacts["contact_sheet"]).is_file())
            with Image.open(scored / artifacts["gif"]) as gif:
                self.assertEqual(gif.n_frames, 4)
                self.assertEqual(gif.info["loop"], 0)
                actual_durations = []
                for index in range(gif.n_frames):
                    gif.seek(index)
                    actual_durations.append(gif.info["duration"])
                self.assertEqual(actual_durations, durations)
        inputs = self.root / "pilot-inputs"
        prompts = self.write_json(inputs / "audit-prompts.json", {
            "schema": "pixie-animation-baseline-prompts-v1",
            "audit_set": {"id": "fixture-animation-audit", "frozen": True},
            "heldout_subjects": ["fixture deltaform"],
            "prompts": [{"subject": "fixture deltaform", "action": "walk", "direction": "right", "loop": True}],
        })
        training = self.write_json(inputs / "training-subjects.json", {
            "frozen": True, "subjects": [row["subject"] for row in sheet_manifest["records"] if row["split"] == "train"],
        })
        pilot = self.root / "pilot-plan"
        self.cli("pilot", "--prompts", prompts, "--training-subjects", training,
                 "--output", pilot, "--n", 1, "--seed", 3,
                 "--model-sha256", "1" * 64, "--lora-sha256", "2" * 64)
        pilot_manifest = self.check_manifest_files(pilot)
        self.assertFalse(pilot_manifest["executed"])
        self.assertEqual(pilot_manifest["completed_count"], 0)
        self.assertEqual([row["status"] for row in read_jsonl(pilot / "requests.jsonl")], ["planned"])
        self.assertEqual(list((pilot / "sheets").iterdir()), [])
        self.assertEqual(collection_before, inventory(collection))
        self.assertEqual(pack_before, inventory(pack))

    @needs_holdouts
    def test_reviewed_generated_parent_procedural_motion_and_opt_in_sheets(self):
        pack = self.root / "generated-parent"
        pack.mkdir()
        image = Image.new("RGBA", (32, 32))
        ImageDraw.Draw(image).rectangle((12, 12, 19, 19), fill=(80, 140, 220, 255))
        image.save(pack / "parent.png")
        spec = self.write_json(pack / "ingest.json", {"sequences": [{
            "sequence_id": "generated-static", "lineage_group": "fixture-gamma-family", "kind": "generated",
            "source_url": "https://example.org/generated-fixture", "subject": "fixture gammaform",
            "action": "static", "direction": "unknown", "loop": False, "anchor_px": [16, 24],
            "format": "folder", "frames": [{"path": "parent.png", "duration_ms": 100}],
        }]})
        collection = self.root / "generated-static"
        self.cli("ingest", "--input", pack, "--spec", spec, "--output", collection)
        collection_before = inventory(collection)
        review_file = self.reviews(collection, self.root / "static-reviews.json")
        params = self.write_json(self.root / "bob-params.json", {"amplitude_px": 2, "phase": 0})
        motions = self.root / "bob-derivative"
        self.cli("motions", "--input", collection, "--reviews", review_file, "--output", motions,
                 "--motion", "bob", "--frames", 4, "--seed", 42, "--params", params)
        self.check_manifest_files(motions)
        row = read_jsonl(motions / "sequences.jsonl")[0]
        self.assertEqual(row["provenance"]["kind"], "procedural")
        self.assertEqual(row["lineage_group"], "fixture-gamma-family")
        self.assertEqual(row["anchor_px"], [16, 24])
        self.assertEqual(row["review_status"], "unreviewed")
        expected_boxes = [(12, 12, 20, 20), (12, 14, 20, 22), (12, 12, 20, 20), (12, 10, 20, 18)]
        for frame, box in zip(row["frames"], expected_boxes):
            with Image.open(motions / frame["path"]) as actual:
                self.assertEqual(actual.getchannel("A").getbbox(), box)
        source = read_jsonl(motions / "sources.jsonl")[0]
        self.assertEqual(source["kind"], "procedural")
        self.assertEqual(source["recipe"]["seed"], 42)
        self.assertEqual(source["recipe"]["parent_id"], "generated-static")
        self.assertEqual(source["recipe"]["params"]["amplitude_px"], 2)
        procedural_reviews = self.reviews(motions, self.root / "motion-reviews.json")
        refused = self.root / "procedural-not-enabled"
        self.cli("sheets", "--input", motions, "--reviews", procedural_reviews, "--output", refused,
                 "--eval-fraction", 0, succeeds=False)
        self.assertFalse(refused.exists())
        sheets = self.root / "procedural-sheets"
        self.cli("sheets", "--input", motions, "--reviews", procedural_reviews, "--output", sheets,
                 "--eval-fraction", 0, "--include-procedural")
        manifest = self.check_manifest_files(sheets)
        self.assertEqual(manifest["counts"], {"train": 1, "eval": 0})
        self.assertEqual(manifest["records"][0]["provenance"]["kind"], "procedural")
        self.assertEqual(manifest["records"][0]["lineage_group"], "fixture-gamma-family")
        self.assertEqual(collection_before, inventory(collection))


if __name__ == "__main__":
    unittest.main()
