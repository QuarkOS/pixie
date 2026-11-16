from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from animation.common import DEFAULT_BLIND50, load_collection, new_output, save_frames, sha256, write_collection
from animation.sheets import build

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



def collection(path, specs):
    root = new_output(path)
    sequences, sources = [], []
    for index, spec in enumerate(specs):
        sid = spec.get("id", f"sequence-{index}")
        frames = []
        for f in range(spec.get("count", 4)):
            side = spec.get("canvas", 32)
            image = Image.new("RGBA", (side, side), (*spec.get("hidden", (0, 0, 0)), 0))
            if not spec.get("blank_end") or f != spec.get("count", 4) - 1:
                image.putpixel((8 + f % 4, 12), (20 + spec.get("color", index * 20), 30 + f, 40, spec.get("alpha", 255)))
            frames.append(image)
        metadata = save_frames(root, sid, frames, [70 + f * 10 for f in range(len(frames))])
        source_id = "source-" + sid
        kind = spec.get("kind", "artist")
        sources.append({"source_id": source_id, "kind": kind, "source_url": "https://example.org/fixture",
                        "sha256": metadata[0]["file_sha256"], "assets": [{"path": metadata[0]["path"], "sha256": metadata[0]["file_sha256"]}]})
        sequences.append({"sequence_id": sid, "source_id": source_id, "lineage_group": spec.get("lineage", sid),
                          "subject": spec.get("subject", "fixture subject " + sid), "caption": spec.get("subject", "fixture subject " + sid),
                          "action": spec.get("action", "walk"), "direction": "right", "loop": True,
                          "canvas_px": [spec.get("canvas", 32)] * 2, "anchor_px": [16, 20], "frames": metadata,
                          "frame_count": len(frames), "alpha_mode": "partial" if 0 < spec.get("alpha", 255) < 255 else "binary", "alpha_association": "straight",
                          "color_space": "srgb", "blend_mode": "alpha_over", "normalization": {},
                          "provenance": {"kind": kind}, "review_status": spec.get("review", "accepted"),
                          "split": spec.get("split", "unassigned")})
    write_collection(root, sequences, sources)
    return root


class SheetTests(unittest.TestCase):
    @needs_holdouts
    def test_row_major_order_margins_captions_and_timing(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = collection(root / "collection", [{}])
            before = {p.relative_to(source).as_posix(): sha256(p) for p in source.rglob("*") if p.is_file()}
            manifest = build([source], root / "sheets", cell_px=32, eval_fraction=0)
            row = manifest["records"][0]
            with Image.open(root / "sheets" / row["image"]) as sheet:
                self.assertEqual(sheet.size, (64, 64))
                self.assertEqual(sheet.getpixel((0, 0)), (255, 0, 255))
                for f in range(4):
                    self.assertEqual(sheet.getpixel(((f % 2) * 32 + 8 + f, (f // 2) * 32 + 12)), (20, 30 + f, 40))
            self.assertIn("4-frame animation sheet", row["caption"])
            self.assertIn("walk", row["caption"])
            self.assertEqual(row["anchor_px"], [16, 20])
            self.assertEqual([f["duration_ms"] for f in row["frames"]], [70, 80, 90, 100])
            self.assertEqual(before, {p.relative_to(source).as_posix(): sha256(p) for p in source.rglob("*") if p.is_file()})
            for name, digest in manifest["files"].items():
                self.assertEqual(sha256(root / "sheets" / name), digest)

    @needs_holdouts
    def test_larger_grids_require_real_matching_frame_counts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for grid in (3, 4):
                source = collection(root / f"collection-{grid}", [{"count": grid * grid}])
                manifest = build([source], root / f"sheet-{grid}", grid=grid, cell_px=32, eval_fraction=0)
                row = manifest["records"][0]
                self.assertEqual(len(row["frames"]), grid * grid)
                self.assertIn(f"{grid * grid}-frame", row["caption"])
                with Image.open(root / f"sheet-{grid}" / row["image"]) as image:
                    self.assertEqual(image.size, (grid * 32, grid * 32))
            source = collection(root / "short", [{}])
            with self.assertRaisesRegex(ValueError, "No accepted"):
                build([source], root / "no-padding", grid=3, eval_fraction=0)
            self.assertFalse((root / "no-padding").exists())

    @needs_holdouts
    def test_shared_lineage_and_subject_are_grouped_in_eval(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = collection(root / "collection", [
                {"id": "a", "lineage": "same", "subject": "fixture subject a", "split": "validation"},
                {"id": "b", "lineage": "same", "subject": "fixture subject b"},
                {"id": "c", "subject": "fixture-subject-b"},
                {"id": "d", "subject": "fixture other"}])
            manifest = build([source], root / "sheets", cell_px=32, eval_fraction=0)
            rows = {row["sequence_id"]: row for row in manifest["records"]}
            self.assertEqual({rows[key]["split"] for key in ("a", "b", "c")}, {"eval"})
            self.assertEqual(rows["d"]["split"], "train")
            self.assertEqual(len({rows[key]["split_group"] for key in ("a", "b", "c")}), 1)

    @needs_holdouts
    def test_cross_collection_visible_duplicates_ignore_hidden_rgb(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            a = collection(root / "a", [{"id": "a", "color": 1, "split": "validation"}, {"id": "independent", "color": 6}])
            b = collection(root / "b", [{"id": "b", "color": 1, "hidden": (88, 99, 77)}])
            manifest = build([a, b], root / "sheets", cell_px=32, eval_fraction=0)
            self.assertEqual(manifest["counts"], {"train": 1, "eval": 1})
            self.assertEqual([row["reason"] for row in manifest["excluded"]], ["exact_visible_sequence_duplicate"])

    @needs_holdouts
    def test_holdouts_audit_only_and_source_kinds_are_excluded(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            heldout = json.loads(DEFAULT_BLIND50.read_text())[0].upper().replace(" ", "-")
            source = collection(root / "collection", [
                {"id": "allowed"}, {"id": "heldout", "subject": heldout}, {"id": "test", "split": "test"},
                {"id": "procedural", "kind": "procedural"}, {"id": "generated", "kind": "generated"}])
            manifest = build([source], root / "sheets", cell_px=32, eval_fraction=0)
            self.assertEqual(len(manifest["records"]), 1)
            self.assertEqual({row["reason"] for row in manifest["excluded"]}, {"frozen_holdout", "audit_only", "source_kind_not_enabled"})
            mixed = build([source], root / "mixed", cell_px=32, eval_fraction=0, include_generated=True, include_procedural=True)
            self.assertEqual(len(mixed["records"]), 2)
            self.assertEqual(sum(row["provenance"]["kind"] == "artist" for row in mixed["records"]), 1)
            self.assertEqual(len([row for row in mixed["excluded"] if row["reason"] == "procedural_training_cap"]), 1)

    @needs_holdouts
    def test_review_overlay_and_frozen_eval_subjects(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = collection(root / "collection", [{"id": "a", "review": "unreviewed"}, {"id": "b", "review": "unreviewed"}])
            with self.assertRaisesRegex(ValueError, "No accepted"):
                build([source], root / "no-output", eval_fraction=0)
            reviews = root / "reviews.json"
            reviews.write_text(json.dumps({"collection_sha256": sha256(source / "manifest.json"), "reviews": [
                {"sequence_id": "a", "status": "accepted"}, {"sequence_id": "b", "status": "accepted"}]}))
            subjects = root / "eval.json"
            subjects.write_text(json.dumps(["fixture subject a"]))
            manifest = build([source], root / "sheets", cell_px=32, eval_fraction=0, reviews=[reviews], eval_subjects=subjects)
            self.assertEqual(manifest["counts"], {"train": 1, "eval": 1})
            self.assertEqual(manifest["eval_subjects_sha256"], sha256(subjects))
            self.assertTrue(all(row["review_status"] == "unreviewed" for row in load_collection(source).sequences))

    @needs_holdouts
    def test_blank_frames_do_not_join_unrelated_lineages(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = collection(root / "collection", [
                {"id": "a", "blank_end": True, "split": "validation"}, {"id": "b", "blank_end": True}])
            manifest = build([source], root / "sheets", cell_px=32, eval_fraction=0)
            self.assertEqual(manifest["counts"], {"train": 1, "eval": 1})
            self.assertEqual(len(manifest["lineage_splits"]), 2)

    @needs_holdouts
    def test_partial_alpha_is_retained_in_collection_but_not_sheet_targets(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = collection(root / "collection", [{"id": "opaque"}, {"id": "partial", "alpha": 128}])
            manifest = build([source], root / "sheets", cell_px=32, eval_fraction=0)
            self.assertEqual(len(manifest["records"]), 1)
            self.assertEqual(manifest["excluded"][0]["reason"], "requires_opaque_cutout")
            self.assertEqual(load_collection(source).sequences[1]["alpha_mode"], "partial")

    @needs_holdouts
    def test_64_canvas_requires_explicit_format_and_keeps_integer_scale(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = collection(root / "large", [{"canvas": 64}])
            with self.assertRaisesRegex(ValueError, "No accepted"):
                build([source], root / "default-refuses", eval_fraction=0)
            with self.assertRaisesRegex(ValueError, "multiple of 64"):
                build([source], root / "noninteger", logical_px=64, cell_px=96, eval_fraction=0)
            manifest = build([source], root / "sheets", logical_px=64, cell_px=256, eval_fraction=0)
            self.assertEqual(manifest["logical_px"], 64)
            self.assertEqual(manifest["records"][0]["canvas_px"], [64, 64])
            with Image.open(root / "sheets" / manifest["records"][0]["image"]) as image:
                self.assertEqual(image.size, (512, 512))
                self.assertEqual(image.getpixel((8 * 4, 12 * 4)), (20, 30, 40))

    @needs_holdouts
    def test_64_canvas_requires_two_eval_lineages_by_default(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = collection(root / "large", [{"id": "one", "canvas": 64}, {"id": "two", "canvas": 64}])
            with self.assertRaisesRegex(ValueError, "lineage groups"):
                build([source], root / "too-small", logical_px=64, cell_px=256, eval_fraction=0.5)
            manifest = build([source], root / "two-groups", logical_px=64, cell_px=256,
                             eval_fraction=0.5, eval_min_lineages=1)
            self.assertEqual(manifest["eval_min_lineages"], 1)

    @needs_holdouts
    def test_overwrite_nesting_invalid_scale_and_unsplittable_groups_refused(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = collection(root / "collection", [{}])
            build([source], root / "sheets", eval_fraction=0)
            with self.assertRaises(FileExistsError):
                build([source], root / "sheets", eval_fraction=0)
            with self.assertRaisesRegex(ValueError, "separate"):
                build([source], source / "nested", eval_fraction=0)
            with self.assertRaisesRegex(ValueError, "multiple"):
                build([source], root / "bad-scale", cell_px=200)
            with self.assertRaisesRegex(ValueError, "No evaluation|All lineage"):
                build([source], root / "no-groups", eval_fraction=0.5)


if __name__ == "__main__":
    unittest.main()
