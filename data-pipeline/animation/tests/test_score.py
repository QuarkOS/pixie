from __future__ import annotations

import json
import hashlib
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from animation.score import _load_feature_sidecar, export_review, score_sequence, run, build_parser
from animation.common import save_frames, write_collection, new_output, sha256

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



def _write_frame(root: Path, name: str, pixels: list[tuple[int, int, int, int]], size=(8, 8)) -> str:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGBA", size, (0, 0, 0, 0))
    image.putdata(pixels)
    image.save(path)
    return name.replace("\\", "/")


def _sequence(root: Path, loop: bool = True, shift: int = 0, different: bool = False) -> dict:
    transparent = (15, 240, 15, 0)
    pixels_a = [transparent] * 64
    pixels_b = [transparent] * 64
    for x in range(2 + shift, 4 + shift):
        for y in range(2, 4):
            pixels_a[y * 8 + x] = (255, 0, 0, 255)
    for x in range(2 + shift, 4 + shift):
        for y in range(2, 4):
            pixels_b[y * 8 + x] = ((250, 0, 0, 255) if different else (255, 0, 0, 255))
    paths = [_write_frame(root, "frames/s1/0000.png", pixels_a), _write_frame(root, "frames/s1/0001.png", pixels_b)]
    return {
        "sequence_id": "s1",
        "source_id": "src",
        "lineage_group": "lineage",
        "subject": "red block",
        "action": "idle",
        "anchor_px": [4, 4],
        "loop": loop,
        "frames": [
            {"path": paths[0], "file_sha256": sha256(root / paths[0]), "duration_ms": 25},
            {"path": paths[1], "file_sha256": sha256(root / paths[1]), "duration_ms": 35},
        ],
    }


class ScoreTests(unittest.TestCase):
    def test_palette_measurement_space_distinguishes_native_and_resampled_frames(self):
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            sequence = _sequence(root)
            native = Image.new("RGBA", (8, 8), (255, 0, 0, 255))
            native.putpixel((0, 0), (0, 255, 0, 255))
            for frame in sequence["frames"]:
                native.save(root / frame["path"])
            sequence["normalization"] = {"source_canvas_px": [8, 8], "resized_px": [8, 8]}
            before = score_sequence(root, sequence)
            self.assertEqual(before["palette"]["visible_palette_union"], 2)
            self.assertFalse(before["measurement_space"]["source_resampled_during_ingest"])
            self.assertEqual(before["measurement_space"]["canvas_px"], [8, 8])
            for frame in sequence["frames"]:
                native.resize((2, 2), Image.Resampling.NEAREST).save(root / frame["path"])
            sequence["normalization"] = {"source_canvas_px": [8, 8], "resized_px": [2, 2]}
            after = score_sequence(root, sequence)
            self.assertEqual(after["palette"]["visible_palette_union"], 1)
            self.assertTrue(after["measurement_space"]["source_resampled_during_ingest"])
            self.assertEqual(after["measurement_space"]["canvas_px"], [2, 2])
            self.assertIn("matching", after["measurement_space"]["palette_comparability"])

    def test_score_reports_unavailable_identity_and_alpha_aware_palette(self) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            sequence = _sequence(root)
            metrics = score_sequence(root, sequence)
            self.assertEqual(metrics["status"], "UNCALIBRATED")
            self.assertEqual(metrics["identity"]["status"], "unavailable")
            self.assertEqual(metrics["palette"]["visible_palette_union"], 1)
            self.assertTrue(metrics["geometry"]["metric_note"].startswith("bbox jitter"))
            self.assertTrue(metrics["loop"]["applicable"])

    def test_identity_sidecar_is_exact_and_detects_changed_embedding(self) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            sequence = _sequence(root)
            features = {
                ("s1", "frames/s1/0000.png"): np.array([1.0, 0.0]),
                ("s1", "frames/s1/0001.png"): np.array([0.0, 1.0]),
            }
            metrics = score_sequence(root, sequence, features)
            self.assertEqual(metrics["identity"]["status"], "available")
            self.assertAlmostEqual(metrics["identity"]["embedding_similarity"]["adjacent_min"], 0.0)
            bad = dict(features)
            bad[("s1", "frames/s1/0001.png")] = np.zeros(2)
            self.assertEqual(score_sequence(root, sequence, bad)["identity"]["status"], "invalid_features")

    def test_nonloop_gif_plays_once_and_loop_preserves_duration(self) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            sequence = _sequence(root, loop=False)
            artifacts = export_review(root, sequence, root / "review")
            self.assertFalse(artifacts["gif_loop"])
            with Image.open(root / "review" / artifacts["gif"]) as gif:
                self.assertNotIn("loop", gif.info)
            self.assertTrue((root / "review" / artifacts["contact_sheet"]).is_file())
            loop_artifacts = export_review(root, _sequence(root, loop=True, different=True), root / "review-loop")
            self.assertEqual(loop_artifacts["gif_source_durations_ms"], [25, 35])
            self.assertEqual(loop_artifacts["gif_durations_ms"], [20, 30])
            with Image.open(root / "review-loop" / loop_artifacts["gif"]) as gif:
                self.assertEqual(gif.n_frames, 2)
                gif.seek(0)
                self.assertEqual(gif.info["duration"], 20)
                self.assertEqual(gif.info["loop"], 0)

    def test_bbox_shift_is_reported_and_transparent_rgb_does_not_change_palette(self) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            sequence = _sequence(root)
            moved = Image.new("RGBA", (8, 8))
            for x in range(4, 6):
                for y in range(2, 4):
                    moved.putpixel((x, y), (255, 0, 0, 255))
            moved.save(root / sequence["frames"][1]["path"])
            metrics = score_sequence(root, sequence)
            self.assertEqual(metrics["geometry"]["bbox_center_span_px"], 2.0)
            self.assertEqual(metrics["palette"]["visible_palette_union"], 1)

    def test_feature_sidecar_requires_exact_frame_hashes(self) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            sequence = _sequence(root)
            hashes = [hashlib.sha256((root / frame["path"]).read_bytes()).hexdigest() for frame in sequence["frames"]]
            sidecar = root / "features.npz"
            np.savez(
                sidecar,
                features=np.asarray([[1.0, 0.0], [0.0, 1.0]]),
                sequence_ids=np.asarray(["s1", "s1"]),
                frame_paths=np.asarray([frame["path"] for frame in sequence["frames"]]),
                frame_sha256=np.asarray(hashes),
            )
            loaded = _load_feature_sidecar(sidecar, root, [sequence])
            self.assertEqual(sorted(loaded), sorted(("s1", frame["path"]) for frame in sequence["frames"]))
            sidecar_bad = root / "features-bad.npz"
            np.savez(
                sidecar_bad,
                features=np.asarray([[1.0, 0.0], [0.0, 1.0]]),
                sequence_ids=np.asarray(["s1", "s1"]),
                frame_paths=np.asarray([frame["path"] for frame in sequence["frames"]]),
                frame_sha256=np.asarray(["0" * 64, hashes[1]]),
            )
            with self.assertRaisesRegex(ValueError, "does not match"):
                _load_feature_sidecar(sidecar_bad, root, [sequence])

    def test_hidden_rgb_does_not_create_motion_loop_or_palette_defects(self) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            sequence = _sequence(root)
            second = root / sequence["frames"][1]["path"]
            with Image.open(second) as original:
                pixels = np.asarray(original.convert("RGBA")).copy()
            pixels[pixels[..., 3] == 0, :3] = [2, 3, 4]
            Image.fromarray(pixels).save(second)
            metrics = score_sequence(root, sequence)
            self.assertTrue(metrics["motion"]["all_frames_identical"])
            self.assertEqual(metrics["motion"]["adjacent_changed_fraction"], [0.0])
            self.assertEqual(metrics["loop"]["rgba_mean_abs_error"], 0.0)
            self.assertEqual(metrics["palette"]["palette_consistency_jaccard"], 1.0)

    def test_sidecar_rejects_zero_nonfinite_missing_and_duplicate_rows(self) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            sequence = _sequence(root)
            keys = dict(sequence_ids=np.array(["s1", "s1"]),
                        frame_paths=np.array([frame["path"] for frame in sequence["frames"]]),
                        frame_sha256=np.array([frame["file_sha256"] for frame in sequence["frames"]]))
            for vectors in ([[0, 0], [1, 0]], [[float("nan"), 0], [1, 0]], [[float("inf"), 0], [1, 0]]):
                np.savez(root / "features.npz", features=np.asarray(vectors), **keys)
                with self.assertRaisesRegex(ValueError, "finite and nonzero"):
                    _load_feature_sidecar(root / "features.npz", root, [sequence])
            missing = {name: value[:1] for name, value in keys.items()}
            np.savez(root / "features.npz", features=np.array([[1, 0]]), **missing)
            with self.assertRaisesRegex(ValueError, "missing collection frames"):
                _load_feature_sidecar(root / "features.npz", root, [sequence])
            duplicate = {name: value[[0, 0]] for name, value in keys.items()}
            np.savez(root / "features.npz", features=np.array([[1, 0], [1, 0]]), **duplicate)
            with self.assertRaisesRegex(ValueError, "duplicate"):
                _load_feature_sidecar(root / "features.npz", root, [sequence])

    def test_visible_palette_change_and_nonloop_scoring(self) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            sequence = _sequence(root, different=True, loop=False)
            metrics = score_sequence(root, sequence)
            self.assertEqual(metrics["palette"]["palette_consistency_jaccard"], 0.0)
            self.assertFalse(metrics["loop"]["applicable"])
            self.assertEqual(metrics["loop"]["status"], "not_applicable")
            self.assertFalse(metrics["motion"]["all_frames_identical"])

    def test_path_escape_is_refused(self) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            sequence = _sequence(root)
            for bad in ("../outside.png", "/outside.png", "frames\\outside.png"):
                sequence["frames"][0]["path"] = bad
                with self.assertRaises(ValueError):
                    score_sequence(root, sequence)

    @needs_holdouts
    def test_real_collection_run_exports_bound_review_and_refuses_overwrite(self) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            collection = new_output(root / "collection")
            first = Image.new("RGBA", (8, 8))
            first.putpixel((2, 3), (255, 0, 0, 255))
            second = first.copy()
            second.putpixel((3, 3), (255, 0, 0, 255))
            frames = save_frames(collection, "s1", [first, second], [25, 35])
            source = {"source_id": "source", "kind": "artist", "source_url": "https://example.org/fixture",
                      "sha256": frames[0]["file_sha256"], "assets": [{"path": frames[0]["path"], "sha256": frames[0]["file_sha256"]}]}
            sequence = {"sequence_id": "s1", "source_id": "source", "lineage_group": "fixture-family",
                        "subject": "fixture", "caption": "fixture moving", "action": "moving", "direction": "right",
                        "loop": False, "canvas_px": [8, 8], "anchor_px": [4, 6], "frames": frames, "frame_count": 2,
                        "alpha_mode": "binary", "alpha_association": "straight", "color_space": "srgb",
                        "blend_mode": "alpha_over", "normalization": {}, "review_status": "unreviewed",
                        "provenance": "artist", "split": "unassigned"}
            write_collection(collection, [sequence], [source])
            before = {path.relative_to(collection).as_posix(): sha256(path) for path in collection.rglob("*") if path.is_file()}
            args = build_parser().parse_args(["--input", str(collection), "--output", str(root / "review")])
            self.assertEqual(run(args), 0)
            output = json.loads((root / "review/manifest.json").read_text())
            self.assertEqual(output["status"], "UNCALIBRATED")
            self.assertEqual(output["source_manifest_sha256"], before["manifest.json"])
            metrics = json.loads((root / "review/score.jsonl").read_text())
            for key in ("contact_sheet", "gif"):
                self.assertTrue((root / "review" / metrics["review_artifacts"][key]).is_file())
            self.assertEqual(before, {path.relative_to(collection).as_posix(): sha256(path) for path in collection.rglob("*") if path.is_file()})
            with self.assertRaises(FileExistsError):
                run(args)
            args.output = collection / "nested"
            with self.assertRaisesRegex(ValueError, "separate"):
                run(args)
            self.assertFalse(args.output.exists())

    def test_invalid_collection_does_not_reserve_output(self) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = build_parser().parse_args(["--input", str(root / "missing"), "--output", str(root / "review")])
            with self.assertRaises(FileNotFoundError):
                run(args)
            self.assertFalse(args.output.exists())

    def test_gif_records_actual_encoded_timing_when_static_frames_merge(self) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            sequence = _sequence(root)
            artifacts = export_review(root, sequence, root / "review")
            self.assertEqual(artifacts["gif_source_durations_ms"], [25, 35])
            self.assertEqual(artifacts["gif_requested_durations_ms"], [20, 30])
            with Image.open(root / "review" / artifacts["gif"]) as encoded:
                actual = []
                for index in range(encoded.n_frames):
                    encoded.seek(index)
                    actual.append(encoded.info["duration"])
            self.assertEqual(artifacts["gif_durations_ms"], actual)
            self.assertEqual(sum(actual), 50)


if __name__ == "__main__":
    unittest.main()
