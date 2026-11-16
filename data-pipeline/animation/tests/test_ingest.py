"""Imported source bytes, timing, layout and collection integrity are observable."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

from animation.common import DEFAULT_BLIND50, load_collection, relative_file, reviewed_sequences, sha256
from animation.ingest import ingest

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



class IngestTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.pack = self.root / "pack"
        self.pack.mkdir()
        self.spec = self.pack / "ingest.json"
        self.output = self.root / "collection"

    def run_import(self, specific, canvas=(8, 8), subject="a copper beetle", **options):
        entry = {"sequence_id": "beetle-walk", "source_url": "https://example.org/pack", "lineage_group": "beetle", "subject": subject, "action": "walk", "direction": "right", "loop": True, "anchor_px": [4, 7], **specific}
        self.spec.write_text(json.dumps({"sequences": [entry]}), encoding="utf-8")
        ingest(self.pack, self.spec, self.output, canvas, **options)
        return load_collection(self.output)

    def image(self, name, point, color=(255, 20, 30, 255), size=(8, 8)):
        image = Image.new("RGBA", size)
        image.putpixel(point, color)
        image.save(self.pack / name)
        return image

    @needs_holdouts
    def test_folder_order_timing_shared_anchor_and_original_bytes(self):
        second = self.image("z.png", (6, 2), (20, 40, 60, 128))
        first = self.image("a.png", (1, 2))
        original = (self.pack / "z.png").read_bytes()
        collection = self.run_import({"format": "folder", "frames": [{"path": "z.png", "duration_ms": 70}, {"path": "a.png", "duration_ms": 120}]})
        seq = collection.sequences[0]
        self.assertEqual(seq["anchor_px"], [4, 7])
        self.assertEqual(seq["frame_count"], 2)
        self.assertEqual(seq["alpha_mode"], "partial")
        self.assertEqual(seq["review_status"], "unreviewed")
        self.assertEqual([f["duration_ms"] for f in seq["frames"]], [70, 120])
        with Image.open(collection.root / seq["frames"][0]["path"]) as frame:
            self.assertEqual(frame.tobytes(), second.tobytes())
        with Image.open(collection.root / seq["frames"][1]["path"]) as frame:
            self.assertEqual(frame.tobytes(), first.tobytes())
        asset = collection.sources[0]["assets"][0]
        self.assertEqual((collection.root / asset["path"]).read_bytes(), original)
        self.assertEqual(asset["sha256"], hashlib.sha256(original).hexdigest())

    @needs_holdouts
    def test_common_fit_keeps_motion_and_source_size_masters(self):
        self.image("first.png", (2, 1), size=(16, 8))
        self.image("second.png", (12, 1), size=(16, 8))
        collection = self.run_import({"format": "folder", "frames": [{"path": "first.png", "duration_ms": 100}, {"path": "second.png", "duration_ms": 100}]})
        seq = collection.sequences[0]
        self.assertEqual(seq["normalization"]["resized_px"], [8, 4])
        self.assertEqual(seq["normalization"]["offset_px"], [0, 2])
        self.assertEqual(seq["anchor_px"], [2, 5.5])
        for master in seq["original_masters"]:
            with Image.open(collection.root / master["path"]) as image:
                self.assertEqual(image.size, (16, 8))

    @needs_holdouts
    def test_sheet_explicit_cell_order(self):
        sheet = Image.new("RGBA", (16, 16))
        for point, color in [((0, 0), "red"), ((8, 0), "green"), ((0, 8), "blue"), ((8, 8), "yellow")]:
            sheet.paste(Image.new("RGBA", (8, 8), color), point)
        sheet.save(self.pack / "sheet.png")
        collection = self.run_import({"format": "sheet", "path": "sheet.png", "cell_px": [8, 8], "order": [[1, 1], [0, 0], [1, 0], [0, 1]], "durations_ms": [10, 20, 30, 40]})
        frame = collection.sequences[0]["frames"][0]
        with Image.open(collection.root / frame["path"]) as image:
            self.assertEqual(image.getpixel((0, 0)), (255, 255, 0, 255))
        self.assertEqual(frame["duration_ms"], 10)

    @needs_holdouts
    def test_aseprite_trimmed_offsets_and_tag_order(self):
        sheet = Image.new("RGBA", (4, 2))
        sheet.paste(Image.new("RGBA", (2, 2), "red"), (0, 0))
        sheet.paste(Image.new("RGBA", (2, 2), "blue"), (2, 0))
        sheet.save(self.pack / "sheet.png")
        values = []
        for index, x in enumerate([1, 5]):
            values.append({"frame": {"x": index * 2, "y": 0, "w": 2, "h": 2}, "sourceSize": {"w": 8, "h": 8}, "spriteSourceSize": {"x": x, "y": 3, "w": 2, "h": 2}, "duration": 50 + index * 20, "trimmed": True})
        data = {"frames": values, "meta": {"image": "sheet.png", "frameTags": [{"name": "walk", "from": 0, "to": 1, "direction": "reverse"}]}}
        (self.pack / "sprite.json").write_text(json.dumps(data), encoding="utf-8")
        collection = self.run_import({"format": "aseprite", "path": "sprite.json", "tag": "walk"})
        frames = collection.sequences[0]["frames"]
        self.assertEqual([f["duration_ms"] for f in frames], [70, 50])
        with Image.open(collection.root / frames[0]["path"]) as image:
            self.assertEqual(image.getpixel((5, 3)), (0, 0, 255, 255))
            self.assertEqual(image.getpixel((1, 3)), (0, 0, 0, 0))

    @needs_holdouts
    def test_gif_composited_frames_and_source_durations(self):
        first = Image.new("RGBA", (8, 8), "red")
        second = Image.new("RGBA", (8, 8), "blue")
        first.save(self.pack / "animation.gif", save_all=True, append_images=[second], duration=[60, 140], loop=0, disposal=2)
        collection = self.run_import({"format": "gif", "path": "animation.gif"})
        seq = collection.sequences[0]
        self.assertEqual([f["duration_ms"] for f in seq["frames"]], [60, 140])
        with Image.open(collection.root / seq["frames"][1]["path"]) as image:
            self.assertEqual(image.getpixel((4, 4)), (0, 0, 255, 255))

    @needs_holdouts
    def test_holdout_normalized_match_refuses_before_output(self):
        self.image("one.png", (1, 1))
        repo_holdout = _HOLDOUT_DIR / "blind-50.json"
        phrase = json.loads(repo_holdout.read_text(encoding="utf-8"))[0]
        with self.assertRaisesRegex(ValueError, "holdout"):
            self.run_import({"format": "folder", "frames": [{"path": "one.png", "duration_ms": 100}]}, subject=phrase.replace(" ", "-"))
        self.assertFalse(self.output.exists())

    @needs_holdouts
    def test_missing_timing_and_escaping_paths_refuse_before_output(self):
        self.image("one.png", (1, 1))
        with self.assertRaisesRegex(ValueError, "durations"):
            self.run_import({"format": "folder", "frames": [{"path": "one.png"}]})
        self.assertFalse(self.output.exists())
        with self.assertRaises(ValueError):
            relative_file(self.pack, "../secret.png")
        with self.assertRaises(ValueError):
            relative_file(self.pack, "C:/not-allowed.png")

    @needs_holdouts
    def test_converted_or_reformatted_holdout_copy_is_refused_without_writes(self):
        original = DEFAULT_BLIND50.read_bytes()
        self.image("frame.png", (1, 2))
        converted = self.root / "blind-copy.json"
        for contents in (original.replace(b"\n", b"\r\n"), json.dumps(json.loads(original), indent=4).encode()):
            converted.write_bytes(contents)
            with patch("animation.common.DEFAULT_BLIND50", converted):
                with self.assertRaisesRegex(ValueError, "Frozen Blind-50/Eval-50 bytes differ"):
                    self.run_import({"format": "folder", "frames": [{"path": "frame.png", "duration_ms": 100}]})
            self.assertFalse(self.output.exists())
            self.assertEqual(converted.read_bytes(), contents)
            self.assertEqual(DEFAULT_BLIND50.read_bytes(), original)
    @needs_holdouts
    def test_existing_output_and_output_inside_pack_refused(self):
        self.image("one.png", (1, 1))
        details = {"format": "folder", "frames": [{"path": "one.png", "duration_ms": 100}]}
        self.output.mkdir()
        (self.output / "keep.txt").write_text("keep")
        with self.assertRaises(FileExistsError):
            self.run_import(details)
        self.assertEqual((self.output / "keep.txt").read_text(), "keep")
        self.output = self.pack / "nested-output"
        with self.assertRaisesRegex(ValueError, "outside"):
            self.run_import(details)

    @needs_holdouts
    def test_modified_file_and_record_tampering_are_detected(self):
        self.image("one.png", (1, 1))
        collection = self.run_import({"format": "folder", "frames": [{"path": "one.png", "duration_ms": 100}]})
        path = collection.root / collection.sequences[0]["frames"][0]["path"]
        path.write_bytes(path.read_bytes() + b"changed")
        with self.assertRaisesRegex(ValueError, "hash"):
            load_collection(self.output)

    @needs_holdouts
    def test_nested_manifest_filename_is_hash_bound(self):
        self.image("one.png", (1, 1))
        collection = self.run_import({"format": "folder", "frames": [{"path": "one.png", "duration_ms": 100}]})
        nested = collection.root / "masters" / "manifest.json"
        nested.write_text("{}")
        with self.assertRaisesRegex(ValueError, "hash"):
            load_collection(self.output)

    @needs_holdouts
    def test_review_overlay_is_bound_to_manifest_and_does_not_write_inputs(self):
        self.image("one.png", (1, 1))
        collection = self.run_import({"format": "folder", "frames": [{"path": "one.png", "duration_ms": 100}]})
        before = (collection.root / "sequences.jsonl").read_bytes()
        review_file = self.root / "reviews.json"
        record = {"collection_sha256": sha256(collection.root / "manifest.json"), "reviews": [{"sequence_id": "beetle-walk", "status": "accepted"}]}
        review_file.write_text(json.dumps(record))
        reviewed = reviewed_sequences(collection, review_file)
        self.assertEqual(reviewed[0]["review_status"], "accepted")
        self.assertEqual(collection.sequences[0]["review_status"], "unreviewed")
        self.assertEqual((collection.root / "sequences.jsonl").read_bytes(), before)
        for edits in [
            {"collection_sha256": "0" * 64},
            {"reviews": [{"sequence_id": "unknown", "status": "accepted"}]},
            {"reviews": [{"sequence_id": "beetle-walk", "status": "automatic"}]},
            {"reviews": record["reviews"] * 2},
        ]:
            review_file.write_text(json.dumps({**record, **edits}))
            with self.assertRaises(ValueError):
                reviewed_sequences(collection, review_file)

    @needs_holdouts
    def test_generated_source_kind_and_provenance_are_preserved(self):
        self.image("one.png", (1, 1))
        collection = self.run_import({"format": "folder", "frames": [{"path": "one.png", "duration_ms": 100}], "kind": "generated", "provenance": {"model_sha256": "a" * 64, "seed": 7}})
        self.assertEqual(collection.sources[0]["kind"], "generated")
        self.assertEqual(collection.sequences[0]["provenance"]["kind"], "generated")
        self.assertEqual(collection.sequences[0]["provenance"]["seed"], 7)

    @needs_holdouts
    def test_aseprite_hash_frame_order_and_rotation_refusal(self):
        sheet = Image.new("RGBA", (16, 8), "red")
        sheet.paste(Image.new("RGBA", (8, 8), "blue"), (8, 0))
        sheet.save(self.pack / "sheet.png")
        entries = {}
        for name, x in [("last", 8), ("first", 0)]:
            entries[name] = {"frame": {"x": x, "y": 0, "w": 8, "h": 8}, "sourceSize": {"w": 8, "h": 8}, "duration": 100}
        data = {"frames": entries, "meta": {"image": "sheet.png"}}
        (self.pack / "sprite.json").write_text(json.dumps(data))
        collection = self.run_import({"format": "aseprite", "path": "sprite.json", "frame_order": ["first", "last"]})
        with Image.open(collection.root / collection.sequences[0]["frames"][0]["path"]) as frame:
            self.assertEqual(frame.getpixel((0, 0)), (255, 0, 0, 255))
        self.output = self.root / "rotated"
        data["frames"]["first"]["rotated"] = True
        (self.pack / "sprite.json").write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, "rotated"):
            self.run_import({"format": "aseprite", "path": "sprite.json"})
        self.assertFalse(self.output.exists())

    @needs_holdouts
    def test_exact_key_preserves_pink_partial_alpha_and_unkeyed_masters(self):
        image = Image.new("RGBA", (8, 8), (255, 0, 255, 255))
        image.putpixel((2, 2), (254, 0, 255, 255))
        image.putpixel((3, 2), (80, 90, 100, 128))
        image.putpixel((4, 2), (255, 0, 255, 0))
        image.save(self.pack / "keyed.png")
        original = (self.pack / "keyed.png").read_bytes()
        collection = self.run_import({"format": "folder", "frames": [{"path": "keyed.png", "duration_ms": 100}]},
                                     key_rgb=(255, 0, 255))
        row = collection.sequences[0]
        with Image.open(collection.root / row["frames"][0]["path"]) as normalized:
            self.assertEqual(normalized.getpixel((0, 0)), (255, 0, 255, 0))
            self.assertEqual(normalized.getpixel((2, 2)), (254, 0, 255, 255))
            self.assertEqual(normalized.getpixel((3, 2)), (80, 90, 100, 128))
        self.assertEqual(row["normalization"]["key_rgb"], [255, 0, 255])
        self.assertEqual(row["normalization"]["keyed_pixels_per_frame"], [61])
        with Image.open(collection.root / row["original_masters"][0]["path"]) as master:
            self.assertEqual(master.tobytes(), image.tobytes())
        self.assertEqual((collection.root / collection.sources[0]["assets"][0]["path"]).read_bytes(), original)
        self.assertEqual((self.pack / "keyed.png").read_bytes(), original)

    @needs_holdouts
    def test_key_is_opt_in_and_sequence_can_override_global_key(self):
        Image.new("RGBA", (8, 8), (255, 0, 255, 255)).save(self.pack / "magenta.png")
        collection = self.run_import({"format": "folder", "frames": [{"path": "magenta.png", "duration_ms": 100}],
                                      "key_rgb": None}, key_rgb=(255, 0, 255))
        row = collection.sequences[0]
        with Image.open(collection.root / row["frames"][0]["path"]) as frame:
            self.assertEqual(frame.getpixel((0, 0))[3], 255)
        self.assertEqual(row["normalization"]["keyed_pixels_per_frame"], [0])

    @needs_holdouts
    def test_all_keyed_or_invalid_key_fails_before_output(self):
        Image.new("RGBA", (8, 8), (255, 0, 255, 255)).save(self.pack / "magenta.png")
        spec = {"format": "folder", "frames": [{"path": "magenta.png", "duration_ms": 100}]}
        for key, message in [((255, 0, 255), "no visible pixels"), ((256, 0, 0), "key_rgb"), ((1, 2), "key_rgb")]:
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, message):
                self.run_import(spec, key_rgb=key)
            self.assertFalse(self.output.exists())

    @needs_holdouts
    def test_sequence_union_crop_enlarges_big_cells_without_recentering(self):
        for name, left in [("first.png", 30), ("second.png", 34)]:
            image = Image.new("RGBA", (78, 58))
            image.paste(Image.new("RGBA", (8, 8), (20, 40, 60, 255)), (left, 24))
            image.save(self.pack / name)
        specific = {"format": "folder", "frames": [{"path": "first.png", "duration_ms": 100},
                     {"path": "second.png", "duration_ms": 100}], "anchor_px": [34, 32]}
        baseline = self.run_import(specific, canvas=(32, 32))
        with Image.open(baseline.root / baseline.sequences[0]["frames"][0]["path"]) as image:
            baseline_box = image.getchannel("A").getbbox()
        self.output = self.root / "cropped"
        cropped = self.run_import(specific, canvas=(32, 32), crop_to_content=True)
        row = cropped.sequences[0]
        self.assertEqual(row["normalization"]["crop_box_px"], [30, 24, 42, 32])
        self.assertEqual(row["normalization"]["original_canvas_px"], [78, 58])
        boxes = []
        for frame in row["frames"]:
            with Image.open(cropped.root / frame["path"]) as image:
                boxes.append(image.getchannel("A").getbbox())
        self.assertGreater(boxes[0][2] - boxes[0][0], baseline_box[2] - baseline_box[0])
        self.assertLess(boxes[0][0], boxes[1][0])
        self.assertEqual(boxes[0][1], boxes[1][1])
        self.assertAlmostEqual(row["anchor_px"][0], 4 * 32 / 12)
        self.assertAlmostEqual(row["anchor_px"][1], 26)
        with Image.open(cropped.root / row["original_masters"][0]["path"]) as image:
            self.assertEqual(image.size, (78, 58))
        self.output = self.root / "canvas64"
        bigger = self.run_import(specific, canvas=(64, 64))
        self.assertEqual(bigger.sequences[0]["canvas_px"], [64, 64])
        self.assertEqual(bigger.sequences[0]["normalization"]["crop_to_content"], False)

    @needs_holdouts
    def test_union_crop_includes_shared_anchor_and_padding(self):
        self.image("first.png", (3, 2))
        self.image("last.png", (5, 2))
        result = self.run_import({"format": "folder", "frames": [{"path": "first.png", "duration_ms": 100},
                                 {"path": "last.png", "duration_ms": 100}], "anchor_px": [4, 7]},
                                 crop_to_content=True, crop_padding=1)
        self.assertEqual(result.sequences[0]["normalization"]["crop_box_px"], [2, 1, 7, 8])
        self.assertEqual(result.sequences[0]["normalization"]["crop_padding"], 1)

    @needs_holdouts
    def test_explicit_default_duration_fills_untimed_pose_order_without_override(self):
        self.image("first.png", (1, 1))
        self.image("last.png", (2, 1))
        result = self.run_import({"format": "folder", "frames": [{"path": "last.png"},
                                 {"path": "first.png", "duration_ms": 70}]}, default_duration_ms=100)
        row = result.sequences[0]
        self.assertEqual([frame["duration_ms"] for frame in row["frames"]], [100, 70])
        self.assertEqual(row["timing_origin"], "default")
        self.assertEqual(row["normalization"]["default_duration_ms"], 100)
        self.output = self.root / "sequence-duration"
        result = self.run_import({"format": "folder", "duration_ms": 80,
                                 "frames": [{"path": "first.png"}]}, default_duration_ms=100)
        self.assertEqual(result.sequences[0]["frames"][0]["duration_ms"], 80)

    @needs_holdouts
    def test_source_gif_and_aseprite_timing_take_priority_over_defaults(self):
        first = Image.new("RGBA", (8, 8), "red")
        last = Image.new("RGBA", (8, 8), "blue")
        first.save(self.pack / "animation.gif", save_all=True, append_images=[last], duration=[60, 140], loop=0)
        result = self.run_import({"format": "gif", "path": "animation.gif", "duration_ms": 70}, default_duration_ms=100)
        self.assertEqual([frame["duration_ms"] for frame in result.sequences[0]["frames"]], [60, 140])
        self.assertEqual(result.sequences[0]["timing_origin"], "source_metadata")
        first.save(self.pack / "sheet.png")
        (self.pack / "sprite.json").write_text(json.dumps({"frames": [
            {"frame": {"x": 0, "y": 0, "w": 8, "h": 8}, "sourceSize": {"w": 8, "h": 8}, "duration": 55},
            {"frame": {"x": 0, "y": 0, "w": 8, "h": 8}, "sourceSize": {"w": 8, "h": 8}, "duration": 0}],
            "meta": {"image": "sheet.png"}}))
        self.output = self.root / "aseprite-default"
        result = self.run_import({"format": "aseprite", "path": "sprite.json"}, default_duration_ms=120)
        self.assertEqual([frame["duration_ms"] for frame in result.sequences[0]["frames"]], [55, 120])
        self.assertEqual(result.sequences[0]["timing_origin"], "default")

    @needs_holdouts
    def test_cli_accepts_key_crop_canvas_and_default_duration_flags(self):
        image = Image.new("RGBA", (8, 8), (255, 0, 255, 255))
        image.putpixel((4, 4), (20, 40, 60, 255))
        image.save(self.pack / "one.png")
        entry = {"sequence_id": "bird", "source_url": "https://example.org/bird", "lineage_group": "bird",
                 "subject": "a copper bird", "action": "flap", "direction": "right", "loop": True,
                 "anchor_px": [4, 7], "format": "sheet", "path": "one.png", "cell_px": [8, 8], "order": [[0, 0]]}
        self.spec.write_text(json.dumps({"sequences": [entry]}))
        from animation.ingest import main
        main(["--input", str(self.pack), "--spec", str(self.spec), "--output", str(self.output),
              "--key-rgb", "255", "0", "255", "--crop-to-content", "--crop-padding", "1",
              "--canvas", "64", "64", "--default-duration-ms", "100"])
        row = load_collection(self.output).sequences[0]
        self.assertEqual(row["canvas_px"], [64, 64])
        self.assertEqual(row["frames"][0]["duration_ms"], 100)
        self.assertEqual(row["normalization"]["key_rgb"], [255, 0, 255])


if __name__ == "__main__":
    unittest.main()
