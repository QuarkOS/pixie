import copy
import json
from pathlib import Path
import tempfile
import unittest
from contextlib import redirect_stdout
import io

from train.pixie32.report_cadence import main, report


class CadenceTests(unittest.TestCase):
    def setUp(self):
        self.summary = {"status": "smoke", "segment_images": 8000, "images_seen": 8000,
                        "segment_wall_seconds": 100.,
                        "phase_seconds": {"training": 27., "evaluation": 30., "sampling": 40., "checkpoint": 3.},
                        "phase_counts": {"training": 100, "evaluation": 1, "sampling": 1, "checkpoint": 3}}

    def project(self, summary=None, **kwargs):
        return report(summary or self.summary, remaining_images=12_000_000 - 8000,
                      eval_every=250_000, sample_every=500_000, **kwargs)

    def test_measured_short_share_is_not_extrapolated_as_long_share(self):
        result = self.project()
        self.assertAlmostEqual(result["measured_segment"]["reporting_wall_share"], .73)
        phases = result["estimated_remaining"]["phases"]
        self.assertEqual(phases["evaluation"]["events_including_final"], 48)
        self.assertEqual(phases["sampling"]["events_including_final"], 24)
        self.assertEqual(phases["evaluation"]["seconds"], 1440.)
        self.assertIsNone(phases["checkpoint"]["periodic_seconds"])
        self.assertNotIn("total_wall_seconds", result["estimated_remaining"])

    def test_final_events_and_measured_checkpoint_cycle(self):
        self.summary["images_seen"] = 8081
        phases = self.project(checkpoint_cycle_seconds=2.)["estimated_remaining"]["phases"]
        self.assertEqual(phases["evaluation"]["events_including_final"], 49)
        self.assertEqual(phases["checkpoint"]["periodic_seconds"], 240.)

    def test_empty_phase_is_unknown_and_invalid_times_fail(self):
        self.summary["phase_counts"]["sampling"] = 0
        self.summary["phase_seconds"]["sampling"] = 0.
        self.assertIsNone(self.project()["estimated_remaining"]["phases"]["sampling"]["seconds"])
        for bad in (float("nan"), -1., 200.):
            value = copy.deepcopy(self.summary)
            value["phase_seconds"]["training"] = bad
            with self.assertRaises(ValueError):
                self.project(value)

    def test_periodic_event_on_final_partial_batch_is_not_repeated(self):
        self.summary["images_seen"] = 249_921
        result = report(self.summary, remaining_images=80, eval_every=250_000,
                        sample_every=500_000, batch=80)
        self.assertEqual(result["estimated_remaining"]["phases"]["evaluation"]["events_including_final"], 1)
        self.assertEqual(result["estimated_remaining"]["phases"]["sampling"]["events_including_final"], 1)

    def test_steady_rate_excludes_compile_contaminated_segment_mean(self):
        result = self.project(steady_update_seconds=.221)["estimated_remaining"]
        self.assertEqual(result["updates"], 149900)
        self.assertAlmostEqual(result["training_seconds"], 149900 * .221)
        self.assertEqual(result["steady_update_seconds"], .221)
        self.assertEqual(self.project()["estimated_remaining"]["steady_update_seconds"], None)
        for invalid in (0., -1., float("nan"), True):
            with self.assertRaises(ValueError):
                self.project(steady_update_seconds=invalid)

    def test_partial_final_update_is_charged_at_full_cost(self):
        result = report(self.summary, remaining_images=81, eval_every=250000,
                        sample_every=500000, batch=80, steady_update_seconds=.221)
        self.assertAlmostEqual(result["estimated_remaining"]["training_seconds"], .442)

    def test_cli_reads_without_mutation_and_refuses_existing_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "paused.json"
            source.write_text(json.dumps(self.summary), encoding="utf-8")
            before = source.read_bytes()
            args = ["--summary", str(source), "--out", str(root / "analysis"), "--remaining-images", "100000"]
            with redirect_stdout(io.StringIO()):
                self.assertEqual(main(args), 0)
            self.assertEqual(source.read_bytes(), before)
            self.assertTrue((root / "analysis" / "cadence.json").is_file())
            with self.assertRaises(SystemExit):
                main(args)


if __name__ == "__main__":
    unittest.main()
