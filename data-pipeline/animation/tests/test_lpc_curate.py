import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image, ImageDraw

from animation.common import load_collection
from animation.lpc_curate import curate

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



class LpcCurateTests(unittest.TestCase):
    @needs_holdouts
    def test_curates_explicit_sheet_and_records_archive_provenance(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pack = root / "pack"
            pack.mkdir()
            sheet = Image.new("RGBA", (128, 128))
            for index in range(4):
                ImageDraw.Draw(sheet).rectangle((index * 32 + 8, 3 * 32 + 8, index * 32 + 15, 3 * 32 + 24), fill=(20, 60 + index, 100, 255))
            sheet.save(pack / "fish.png")
            archive_hash = "a" * 64
            curation = root / "curation.json"
            curation.write_text(json.dumps({
                "schema": "pixie-lpc-curation-v1", "pack_id": "lpc-fish-v1",
                "source_url": "https://opengameart.org/content/lpc-fish",
                "archive_sha256": archive_hash,
                "sequences": [{"sequence_id": "trout-swim", "lineage_group": "trout",
                    "subject": "trout", "action": "swim", "direction": "south", "loop": True,
                    "anchor_px": [16, 28], "format": "sheet", "path": "fish.png",
                    "cell_px": [32, 32], "order": [[0, 3], [1, 3], [2, 3], [3, 3]],
                    "durations_ms": [150, 150, 150, 150]}]}, indent=2), encoding="utf-8")
            output = root / "collection"
            collection = load_collection(curate(pack, curation, output))
            row = collection.sequences[0]
            self.assertEqual(row["provenance"]["lpc_pack_id"], "lpc-fish-v1")
            self.assertEqual(row["provenance"]["archive_sha256"], archive_hash)
            self.assertEqual(row["provenance"]["curation_sha256"], hashlib.sha256(curation.read_bytes()).hexdigest())
            self.assertEqual(row["timing_origin"], "annotated")
            self.assertEqual(row["provenance"]["curation_schema"], "pixie-lpc-curation-v1")

    @needs_holdouts
    def test_default_duration_is_recorded_and_bad_pack_fails_before_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pack = root / "pack"
            pack.mkdir()
            for index in range(4):
                Image.new("RGBA", (8, 8), (20 + index, 40, 60, 255)).save(pack / f"{index}.png")
            curation = root / "curation.json"
            curation.write_text(json.dumps({"pack_id": "lpc", "source_url": "https://example.org/lpc",
                "archive_sha256": "b" * 64, "sequences": [{"sequence_id": "poses", "lineage_group": "poses",
                "subject": "poses", "action": "walk", "direction": "right", "loop": True,
                "anchor_px": [4, 7], "format": "folder", "frames": [{"path": f"{i}.png"} for i in range(4)]}]}), encoding="utf-8")
            row = load_collection(curate(pack, curation, root / "out", default_duration_ms=100)).sequences[0]
            self.assertEqual(row["timing_origin"], "default")
            bad = root / "bad.json"
            bad.write_text(json.dumps({"pack_id": "lpc", "source_url": "https://example.org/lpc",
                "archive_sha256": "not-hash", "sequences": []}), encoding="utf-8")
            with self.assertRaises(ValueError):
                curate(pack, bad, root / "bad-out")
            self.assertFalse((root / "bad-out").exists())

    def test_fish_row_order_and_horse_anchor_motion_conventions_are_checked(self):
        from animation.lpc_curate import load_curation
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "bad-fish.json"
            path.write_text(json.dumps({"pack_id": "lpc-fish-v1", "source_url": "https://example.org/fish",
                "archive_sha256": "d" * 64, "sequences": [{"sequence_id": "fish", "format": "sheet",
                "direction": "down", "order": [[0, 2]], "source_url": "https://example.org/fish"}]}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "up, left, right, down"):
                load_curation(path)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "bad-horse.json"
            path.write_text(json.dumps({"pack_id": "lpc-horses-v1", "source_url": "https://example.org/horses",
                "archive_sha256": "e" * 64, "sequences": [{"sequence_id": "horse", "format": "folder",
                "direction": "right", "action": "walk", "source_motion": "gallop", "anchor_px": [64, 120]}]}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "source_ground_center"):
                load_curation(path)

    def test_official_fish_row_order_passes_and_old_down_right_order_fails(self):
        from animation.lpc_curate import LPC_DIRECTION_ROWS, load_curation
        self.assertEqual(LPC_DIRECTION_ROWS, {"up": 0, "left": 1, "right": 2, "down": 3})
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "fish.json"
            sequences = [{"sequence_id": f"fish-{direction}", "format": "sheet", "direction": direction,
                "order": [[0, row], [1, row], [2, row], [3, row]]}
                for direction, row in (("up", 0), ("left", 1), ("right", 2), ("down", 3), ("south", 3), ("east", 2))]
            path.write_text(json.dumps({"pack_id": "lpc-fish-v1", "source_url": "https://example.org/fish",
                "archive_sha256": "d" * 64, "sequences": sequences}), encoding="utf-8")
            loaded = load_curation(path)
            self.assertEqual([row["direction"] for row in loaded["sequences"]],
                             ["up", "left", "right", "down", "south", "east"])
            for direction, row in (("down", 2), ("right", 3), ("south", 2), ("east", 3)):
                path.write_text(json.dumps({"pack_id": "lpc-fish-v1", "source_url": "https://example.org/fish",
                    "archive_sha256": "d" * 64, "sequences": [{"sequence_id": "fish", "format": "sheet",
                    "direction": direction, "order": [[0, row], [1, row], [2, row], [3, row]]}]}), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "up, left, right, down"):
                    load_curation(path)

    def test_horse_sheet_group_zero_is_gallop_and_group_one_is_walk(self):
        from animation.lpc_curate import LPC_HORSE_MOTION_GROUPS, load_curation
        self.assertEqual(LPC_HORSE_MOTION_GROUPS, {0: "gallop", 1: "walk"})

        def curation(sequences):
            return {"pack_id": "lpc-horses-v1", "source_url": "https://example.org/horses",
                "archive_sha256": "e" * 64, "sequences": sequences}

        def sequence(sequence_id, action, row, order=None):
            direction = ("up", "left", "down", "right")[row % 4]
            return {"sequence_id": sequence_id, "format": "sheet", "direction": direction, "action": action,
                "source_motion": action, "anchor_policy": "source_ground_center", "anchor_px": [64, 120],
                "order": order or [[0, row], [1, row], [2, row], [3, row]]}

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "horse.json"
            path.write_text(json.dumps(curation([
                sequence("gallop-up", "gallop", 0), sequence("gallop-right", "gallop", 3),
                sequence("walk-up", "walk", 4), sequence("walk-right", "walk", 7),
            ])), encoding="utf-8")
            loaded = load_curation(path)
            self.assertEqual([(row["sequence_id"], row["action"]) for row in loaded["sequences"]], [
                ("gallop-up", "gallop"), ("gallop-right", "gallop"),
                ("walk-up", "walk"), ("walk-right", "walk")])
            for action, row in (("walk", 0), ("walk", 3), ("gallop", 4), ("gallop", 7)):
                path.write_text(json.dumps(curation([sequence("swapped", action, row)])), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "group 0 is gallop and group 1 is walk"):
                    load_curation(path)
            path.write_text(json.dumps(curation([
                sequence("mixed", "gallop", 0, order=[[0, 0], [1, 4]])])), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "one motion group"):
                load_curation(path)

    @needs_holdouts
    def test_alpha_threshold_is_explicit_and_makes_soft_edges_sheet_eligible(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pack = root / "pack"
            pack.mkdir()
            for alpha in (80, 120, 128, 255):
                image = Image.new("RGBA", (8, 8), (20, 40, 60, alpha))
                image.save(pack / f"{alpha}.png")
            curation = root / "curation.json"
            curation.write_text(json.dumps({"pack_id": "lpc-animals", "source_url": "https://example.org/lpc-animals",
                "archive_sha256": "c" * 64, "sequences": [{"sequence_id": "mice-walk", "lineage_group": "mice",
                "subject": "mouse", "action": "walk", "direction": "down", "loop": True,
                "anchor_px": [4, 7], "format": "folder", "frames": [{"path": f"{alpha}.png", "duration_ms": 100} for alpha in (80, 120, 128, 255)]}]}), encoding="utf-8")
            collection = load_collection(curate(pack, curation, root / "out", alpha_threshold=128))
            row = collection.sequences[0]
            self.assertEqual(row["alpha_mode"], "binary")
            self.assertEqual(row["normalization"]["alpha_threshold"], 128)
            self.assertEqual(row["normalization"]["alpha_thresholded_pixels_per_frame"], [64, 64, 64, 0])

    @needs_holdouts
    def test_integer_source_anchor_stays_integer_after_128_to_64_normalization(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pack = root / "pack"
            pack.mkdir()
            for index in range(4):
                Image.new("RGBA", (128, 128), (20, 40, 60, 255)).save(pack / f"{index}.png")
            curation = root / "curation.json"
            curation.write_text(json.dumps({"pack_id": "lpc-horses-v1", "source_url": "https://example.org/horses",
                "archive_sha256": "f" * 64, "sequences": [{"sequence_id": "horse-walk", "lineage_group": "horse",
                "subject": "horse", "action": "walk", "source_motion": "walk", "anchor_policy": "source_ground_center",
                "direction": "right", "loop": True, "anchor_px": [64, 120], "format": "folder",
                "frames": [{"path": f"{i}.png", "duration_ms": 100} for i in range(4)]}]}), encoding="utf-8")
            row = load_collection(curate(pack, curation, root / "out", canvas=(64, 64))).sequences[0]
            self.assertEqual(row["anchor_px"], [32, 60])


if __name__ == "__main__":
    unittest.main()
