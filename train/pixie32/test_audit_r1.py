"""Coverage and annotation behavior on explicit synthetic inputs."""
from argparse import Namespace
import csv
import json
import random
import unittest

from train.pixie32.audit_r1 import contained, prepare, summarize, wilson
from train.pixie32 import test_data


class AuditTests(unittest.TestCase):
    def setUp(self):
        self.fixture = test_data.DataTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root = self.fixture.root
        self.args = Namespace(data_root=self.fixture.source, r1=self.root / "r1.txt",
                              blind50=self.root / "blind.json", eval50=self.root / "eval.json",
                              out=self.root / "audit", sample=8, seed=13)

    def test_full_audit_uniform_packet_reproducibility_and_no_overwrite(self):
        report = prepare(self.args)
        self.assertEqual(report["population_rows"], 48)
        self.assertEqual(report["source_counts"], {"3080ti": 48})
        self.assertEqual(report["subjects"]["unique"], 12)
        self.assertEqual(report["visible_pixel_duplicate_excess"], 0)
        self.assertEqual(report["nonbinary_alpha_images"], 0)
        self.assertEqual(report["weighting_scenario"]["effective_rows"], 48)
        sample = json.loads((self.args.out / "sample.json").read_text())
        ids = sorted(row["id"] for row in self.fixture.metadata)
        self.assertEqual([row["id"] for row in sample], random.Random(13).sample(ids, 8))
        self.assertEqual(len(list((self.args.out / "images").glob("*.png"))), 8)
        self.assertTrue((self.args.out / "contact-000.png").is_file())
        self.assertTrue((self.args.out / "coverage.csv").is_file())
        with self.assertRaisesRegex(ValueError, "exists"):
            prepare(self.args)
        self.args.out = self.root / "second"
        prepare(self.args)
        self.assertEqual((self.root / "audit/sample.json").read_bytes(), (self.root / "second/sample.json").read_bytes())

    def test_punctuation_holdout_is_excluded_from_packet(self):
        self.fixture.change_caption(0, "a HELD-subject sprite")
        self.fixture.write_metadata()
        report = prepare(self.args)
        self.assertEqual(report["holdouts_excluded"], 1)
        sample = json.loads((self.args.out / "sample.json").read_text())
        self.assertNotIn(self.fixture.metadata[0]["id"], [row["id"] for row in sample])
        self.assertEqual(report["population_rows"], 47)

    def test_actual_visible_pixel_duplicates_are_counted(self):
        first = self.fixture.source / self.fixture.paths[0]
        second = self.fixture.source / self.fixture.paths[1]
        second.write_bytes(first.read_bytes())
        report = prepare(self.args)
        self.assertEqual(report["visible_pixel_duplicate_excess"], 1)
        self.assertEqual(report["duplicate_groups"], 1)

    def test_expected_count_and_verified_training_weights(self):
        self.args.expect_rows = 133688
        with self.assertRaisesRegex(ValueError, "expected row count"):
            prepare(self.args)
        self.assertFalse(self.args.out.exists())
        self.args.expect_rows = 48
        test_data.build(self.fixture.args)
        self.args.training_cache = self.fixture.args.out
        report = prepare(self.args)
        self.assertIsNotNone(report["training_cache"])
        self.assertGreater(report["training_cache"]["edge_sampling_probability"], 0)
        self.assertLess(report["training_cache"]["edge_sampling_probability"], 1)
        original = self.args.blind50.read_bytes()
        self.args.blind50.write_text(json.dumps(json.loads(original), indent=2))
        self.args.out = self.root / "changed-holdout-audit"
        with self.assertRaisesRegex(ValueError, "byte hashes differ"):
            prepare(self.args)
        self.assertFalse(self.args.out.exists())

    def test_path_escape_and_missing_keep_row_are_rejected(self):
        with self.assertRaises(ValueError):
            contained(self.root, "../escape.png")
        with self.assertRaises(ValueError):
            contained(self.root, "C:/escape.png")
        self.args.r1.write_text("final/images/missing.png")
        with self.assertRaisesRegex(ValueError, "missing from metadata"):
            prepare(self.args)
        self.assertFalse(self.args.out.exists())

    def test_labels_require_complete_identity_and_do_not_claim_partial_population_rate(self):
        prepare(self.args)
        summary_args = Namespace(packet=self.args.out, labels=self.args.out / "labels.csv", out=self.root / "quality.json")
        result = summarize(summary_args)
        self.assertEqual(result["metrics"]["junk"]["missing"], 8)
        self.assertIsNone(result["metrics"]["junk"]["wilson_95_among_labeled"])
        self.assertFalse(result["metrics"]["junk"]["population_inference_valid"])
        with summary_args.labels.open(newline="") as stream:
            labels = list(csv.DictReader(stream))
        for index, row in enumerate(labels):
            row.update(junk="yes" if index < 2 else "no", caption_match="yes")
        with summary_args.labels.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["id", "junk", "caption_match", "defect"])
            writer.writeheader()
            writer.writerows(labels)
        summary_args.out = self.root / "complete.json"
        result = summarize(summary_args)
        self.assertEqual(result["metrics"]["junk"]["fraction_among_labeled"], .25)
        self.assertTrue(result["metrics"]["junk"]["population_inference_valid"])
        self.assertEqual(result["metrics"]["caption_mismatch"]["positive"], 0)
        with self.assertRaisesRegex(ValueError, "exists"):
            summarize(summary_args)
        labels[-1]["id"] = labels[0]["id"]
        with summary_args.labels.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["id", "junk", "caption_match", "defect"])
            writer.writeheader()
            writer.writerows(labels)
        with self.assertRaisesRegex(ValueError, "exactly once"):
            summarize(summary_args)

    def test_wilson_bounds_for_zero_and_all_positive(self):
        self.assertAlmostEqual(wilson(0, 100)[1], .0369934982)
        self.assertAlmostEqual(wilson(100, 100)[0], .9630065018)
        self.assertEqual(wilson(0, 100)[0], 0.)


if __name__ == "__main__":
    unittest.main()
