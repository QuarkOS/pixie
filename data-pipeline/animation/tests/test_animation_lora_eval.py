import json
import tempfile
import unittest
from pathlib import Path

from animation_lora_eval import compare


class AnimationLoraEvalTests(unittest.TestCase):
    def test_compares_same_keys_and_requires_artist_target(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            baseline, candidate = root / "baseline", root / "candidate"
            baseline.mkdir(); candidate.mkdir()
            row = {"status": "UNCALIBRATED", "subject": "silver beetle", "action": "walk", "frame_count": 4,
                   "motion": {"all_frames_identical": False}, "geometry": {"bbox_jitter_px": 1.0},
                   "loop": {"applicable": True, "rgba_mean_abs_error": 3.0}, "identity": {"status": "unavailable"}}
            (baseline / "score.jsonl").write_text(json.dumps(row) + "\n")
            candidate_row = json.loads(json.dumps(row)); candidate_row["geometry"]["bbox_jitter_px"] = 0.5
            (candidate / "score.jsonl").write_text(json.dumps(candidate_row) + "\n")
            sheets = root / "sheets"; sheets.mkdir()
            (sheets / "manifest.json").write_text(json.dumps({"holdouts": {
                "blind50": "7524db9cabd2449f5ec6b48b454bc10e2fed38825ba8c8d1e749311f3b5afb0a",
                "eval50": "a1d6893e8c3c45ab99a845c1726c55547dcde29ff6f7a94b022f067c27a7872f"},
                "records": [{"split": "train", "provenance": {"kind": "artist"}}] * 400}), encoding="utf-8")
            output = root / "comparison.json"
            compare(baseline, candidate, sheets, output)
            result = json.loads(output.read_text())
            self.assertEqual(result["candidate_artist_train_sheets"], 400)
            self.assertEqual(result["candidate_minus_baseline"]["bbox_jitter_mean_px"], -0.5)

    def test_below_target_and_mismatched_keys_fail(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name, subject in (("baseline", "a"), ("candidate", "b")):
                path = root / name; path.mkdir()
                (path / "score.jsonl").write_text(json.dumps({"status": "UNCALIBRATED", "subject": subject,
                    "action": "walk", "frame_count": 4, "motion": {}, "geometry": {}, "loop": {}, "identity": {}}) + "\n")
            sheets = root / "sheets"; sheets.mkdir()
            (sheets / "manifest.json").write_text(json.dumps({"holdouts": {
                "blind50": "7524db9cabd2449f5ec6b48b454bc10e2fed38825ba8c8d1e749311f3b5afb0a",
                "eval50": "a1d6893e8c3c45ab99a845c1726c55547dcde29ff6f7a94b022f067c27a7872f"},
                "records": []}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "400"):
                compare(root / "baseline", root / "candidate", sheets, root / "out")
            with self.assertRaisesRegex(ValueError, "same subject"):
                compare(root / "baseline", root / "candidate", sheets, root / "out2", allow_below_target=True)


if __name__ == "__main__":
    unittest.main()
