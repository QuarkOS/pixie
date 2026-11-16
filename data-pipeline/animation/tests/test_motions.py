import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image, ImageDraw

from animation.common import load_collection, new_output, save_frames, sha256, write_collection
from animation.motions import MOTIONS, VERSION, MotionClippingError, derive, main, render_motion

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



def sprite():
    image = Image.new("RGBA", (32, 32))
    draw = ImageDraw.Draw(image)
    draw.rectangle((12, 12, 19, 19), fill=(80, 140, 220, 255))
    draw.rectangle((14, 13, 15, 15), fill=(10, 10, 10, 255))
    return image


def static_collection(path, accepted=True, subject="fixture traveler", parents=None):
    root = new_output(path)
    rows, sources = [], []
    for parent_id, image, anchor in parents or [("parent", sprite(), [16, 20])]:
        frames = save_frames(root, parent_id, [image], [100])
        source_id = "asset-" + parent_id
        sources.append({"source_id": source_id, "kind": "generated", "source_url": "https://example.org/fixture",
                        "sha256": frames[0]["file_sha256"], "assets": [{"path": frames[0]["path"], "sha256": frames[0]["file_sha256"]}]})
        rows.append({"sequence_id": parent_id, "source_id": source_id, "lineage_group": "one-family", "subject": subject,
                     "caption": subject, "action": "static", "direction": "unknown", "loop": False,
                     "canvas_px": list(image.size), "anchor_px": anchor, "frames": frames, "frame_count": 1,
                     "alpha_mode": "binary", "alpha_association": "straight", "color_space": "srgb",
                     "blend_mode": "alpha_over", "normalization": {}, "provenance": "generated",
                     "review_status": "accepted" if accepted else "unreviewed", "split": "validation"})
    write_collection(root, rows, sources)
    return root


class MotionTests(unittest.TestCase):
    def test_spin_reports_first_clipping_frame_when_foreground_leaves_scratch_canvas(self):
        image = Image.new("RGBA", (8, 64))
        ImageDraw.Draw(image).rectangle((2, 0, 5, 3), fill=(20, 40, 60, 255))
        with self.assertRaises(MotionClippingError) as raised:
            render_motion(image, "spin", 4, {}, 0, [4, 64])
        self.assertEqual(raised.exception.frame_index, 1)

    def test_all_motions_have_ordered_common_canvas_and_reproducible_pixels(self):
        blink = {"eye_boxes": [[14, 13, 16, 16]], "fill_rgba": [80, 140, 220, 255], "ink_rgba": [10, 10, 10, 255]}
        for motion in MOTIONS:
            with self.subTest(motion=motion):
                params = blink if motion == "blink" else {}
                a, effective = render_motion(sprite(), motion, 4, params, 21, [16, 20])
                b, same = render_motion(sprite(), motion, 4, params, 21, [16, 20])
                self.assertEqual(effective, same)
                self.assertEqual([im.tobytes() for im in a], [im.tobytes() for im in b])
                self.assertEqual([im.size for im in a], [(32, 32)] * 4)
                self.assertEqual([im.mode for im in a], ["RGBA"] * 4)
                self.assertGreater(len({im.tobytes() for im in a}), 1)

    def test_fade_changes_only_alpha_and_keeps_partial_values(self):
        image = sprite()
        frames, _ = render_motion(image, "fade", 4, {}, 0, [16, 20])
        for frame in frames:
            np.testing.assert_array_equal(np.asarray(frame)[..., :3], np.asarray(image)[..., :3])
        self.assertEqual(frames[0].getpixel((12, 12))[3], 255)
        self.assertEqual(frames[-1].getpixel((12, 12))[3], 0)
        self.assertTrue(0 < frames[1].getpixel((12, 12))[3] < 255)

    def test_blink_requires_annotated_eyes_and_modifies_only_that_box(self):
        with self.assertRaisesRegex(ValueError, "eye_boxes"):
            render_motion(sprite(), "blink", 4, {}, 0, [16, 20])
        params = {"eye_boxes": [[14, 13, 16, 16]], "fill_rgba": [80, 140, 220, 255],
                  "ink_rgba": [10, 10, 10, 255], "closed_frames": [1]}
        frames, _ = render_motion(sprite(), "blink", 4, params, 0, [16, 20])
        original = np.asarray(sprite())
        actual = np.asarray(frames[1])
        mask = np.ones((32, 32), dtype=bool)
        mask[13:16, 14:16] = False
        np.testing.assert_array_equal(actual[mask], original[mask])
        self.assertEqual(frames[0].tobytes(), sprite().tobytes())

    def test_clipping_invalid_parameters_and_missing_anchor_fail(self):
        full = Image.new("RGBA", (32, 32), (20, 30, 40, 255))
        with self.assertRaisesRegex(ValueError, "clip"):
            render_motion(full, "bob", 4, {"phase": 0}, 0, [16, 16])
        for params in ({"amplitude_px": float("nan")}, {"amplitude_px": -1}, {"extra": 2}):
            with self.assertRaises(ValueError):
                render_motion(sprite(), "bob", 4, params, 0, [16, 20])
        with self.assertRaisesRegex(ValueError, "anchor"):
            render_motion(sprite(), "bob", 4, {}, 0, None)
        with self.assertRaisesRegex(ValueError, "complete loop"):
            render_motion(sprite(), "spin", 4, {"turns": 0.5}, 0, [16, 20])

    @needs_holdouts
    def test_collection_preserves_lineage_split_source_and_parent_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            parent = static_collection(root / "parent")
            before = {p.relative_to(parent).as_posix(): sha256(p) for p in parent.rglob("*") if p.is_file()}
            derive(parent, root / "derived", motion="bob", params={"phase": 0}, seed=42)
            output = load_collection(root / "derived")
            row = output.sequences[0]
            self.assertEqual((row["lineage_group"], row["split"], row["provenance"]["kind"], row["review_status"]),
                             ("one-family", "validation", "procedural", "unreviewed"))
            self.assertEqual(row["anchor_px"], [16, 20])
            self.assertEqual([f["duration_ms"] for f in row["frames"]], [100] * 4)
            self.assertEqual(output.sources[0]["recipe"]["parent_id"], "parent")
            self.assertEqual(output.sources[0]["recipe"]["seed"], 42)
            self.assertEqual(output.sources[0]["source_url"], "https://example.org/fixture")
            copied = output.root / output.sources[0]["assets"][0]["path"]
            original = parent / "frames/parent/0000.png"
            self.assertEqual(copied.read_bytes(), original.read_bytes())
            self.assertEqual(before, {p.relative_to(parent).as_posix(): sha256(p) for p in parent.rglob("*") if p.is_file()})
            with self.assertRaises(FileExistsError):
                derive(parent, root / "derived", motion="bob")

    @needs_holdouts
    def test_unreviewed_and_holdout_parents_are_refused_before_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            parent = static_collection(root / "parent", accepted=False)
            with self.assertRaisesRegex(ValueError, "accepted"):
                derive(parent, root / "no-output", motion="bob")
            self.assertFalse((root / "no-output").exists())
            from animation.common import DEFAULT_BLIND50
            subject = json.loads(DEFAULT_BLIND50.read_text())[0].replace(" ", "-").upper()
            held = static_collection(root / "held", subject=subject)
            with self.assertRaisesRegex(ValueError, "holdout"):
                derive(held, root / "still-no-output", motion="bob")
            self.assertFalse((root / "still-no-output").exists())

    @needs_holdouts
    def test_nested_output_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            parent = static_collection(Path(tmp) / "parent")
            with self.assertRaisesRegex(ValueError, "separate"):
                derive(parent, parent / "child", motion="bob")

    @needs_holdouts
    def test_human_review_overlay_admits_parent_without_editing_collection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            parent = static_collection(root / "parent", accepted=False)
            reviews = root / "reviews.json"
            reviews.write_text(json.dumps({"collection_sha256": sha256(parent / "manifest.json"),
                                          "reviews": [{"sequence_id": "parent", "status": "accepted"}]}))
            derive(parent, root / "derived", motion="fade", reviews=reviews)
            self.assertEqual(load_collection(parent).sequences[0]["review_status"], "unreviewed")
            self.assertEqual(load_collection(root / "derived").manifest["review_sha256"], sha256(reviews))

    @needs_holdouts
    def test_clipping_parent_is_reported_and_later_parent_is_created(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            full = Image.new("RGBA", (32, 32), (20, 30, 40, 255))
            parent = static_collection(root / "parent", parents=[("clipping", full, [16, 16]),
                                                                 ("safe", sprite(), [16, 20])])
            before = {p.relative_to(parent).as_posix(): sha256(p) for p in parent.rglob("*") if p.is_file()}
            derive(parent, root / "derived", motion="bob", params={"phase": 0}, seed=42)
            output = load_collection(root / "derived")
            self.assertEqual([row["parent_id"] for row in output.sequences], ["safe"])
            self.assertEqual(output.manifest["status"], "complete_with_skips")
            self.assertEqual(output.manifest["counts"], {"requested": 2, "created": 1, "skipped": 1})
            skipped = output.manifest["skipped"][0]
            self.assertEqual(skipped["parent_id"], "clipping")
            self.assertEqual(skipped["reason"], "foreground_clipping")
            self.assertEqual(skipped["frame_index"], 1)
            self.assertEqual(skipped["recipe"]["version"], VERSION)
            self.assertEqual(skipped["recipe"]["seed"], 42)
            self.assertEqual(skipped["recipe"]["params"], {"phase": 0, "allow_clipping": False, "amplitude_px": 1})
            self.assertEqual(skipped["recipe"]["parent_sha256"], sha256(parent / "frames/clipping/0000.png"))
            self.assertEqual(before, {p.relative_to(parent).as_posix(): sha256(p) for p in parent.rglob("*") if p.is_file()})

    @needs_holdouts
    def test_foot_anchor_spin_is_skipped_without_cropping_and_all_skips_are_reported(self):
        image = Image.new("RGBA", (32, 32))
        ImageDraw.Draw(image).rectangle((12, 3, 19, 29), fill=(20, 30, 40, 255))
        with self.assertRaises(MotionClippingError) as raised:
            render_motion(image, "spin", 4, {}, 0, [16, 30])
        self.assertEqual(raised.exception.motion, "spin")
        self.assertEqual(raised.exception.frame_index, 1)
        with self.assertRaisesRegex(ValueError, "never permits clipping"):
            render_motion(image, "spin", 4, {"allow_clipping": True}, 0, [16, 30])
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            parent = static_collection(root / "parent", parents=[("foot-anchor", image, [16, 30])])
            result = main(["--input", str(parent), "--output", str(root / "derived"), "--motion", "spin"])
            self.assertEqual(result, 3)
            output = load_collection(root / "derived")
            self.assertEqual(output.sequences, [])
            self.assertEqual(output.manifest["status"], "all_skipped")
            self.assertEqual(output.manifest["counts"], {"requested": 1, "created": 0, "skipped": 1})
            self.assertEqual(output.manifest["skipped"][0]["parent_id"], "foot-anchor")
            self.assertFalse((root / "derived/frames").exists())

    @needs_holdouts
    def test_recipe_and_integrity_errors_are_not_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            full = Image.new("RGBA", (32, 32), (20, 30, 40, 255))
            parent = static_collection(root / "parent", parents=[("clipping", full, [16, 16]),
                                                                 ("safe", sprite(), [16, 20])])
            for params in ({"amplitude_px": float("nan")}, {"unknown": 1}, []):
                with self.assertRaises(ValueError):
                    derive(parent, root / "invalid", motion="bob", params=params)
                self.assertFalse((root / "invalid").exists())
            file = parent / "frames/safe/0000.png"
            file.write_bytes(file.read_bytes() + b"tampered")
            with self.assertRaisesRegex(ValueError, "hash"):
                derive(parent, root / "corrupt", motion="bob")
            self.assertFalse((root / "corrupt").exists())


if __name__ == "__main__":
    unittest.main()
