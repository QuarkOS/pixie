from argparse import Namespace
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

from train.pixie32 import caption_quality as quality
from train.pixie32 import test_data as fixtures
from train.pixie32.audit_r1 import load_population


class FakeScores:
    def score(self, images, texts):
        values = np.full((len(images), len(texts)), .1)
        width = len(texts) // len(images)
        for i in range(len(images)):
            values[i, i * width] = .9
        values[0, 0], values[0, 1] = .2, .8
        return values


class CaptionQualityTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.DataTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        f = self.fixture
        model_dir = f.root / "local-model"
        model_dir.mkdir()
        (model_dir / "config.json").write_text("{}")
        self.args = Namespace(data_root=f.source, r1=f.args.r1, blind50=f.args.blind50,
                              eval50=f.args.eval50, out=f.root / "caption-check", model_dir=model_dir,
                              device="cpu", batch=8, negatives=3, review=40, margin=0., expect_rows=48)

    def test_different_subject_negatives_and_mismatch_ranking(self):
        rows, _, _ = load_population(self.args)
        selected = quality.negatives(rows, 3)
        by_id = {row["id"]: row for row in rows}
        for row in rows:
            self.assertTrue(all(other["subject"] != row["subject"] for other in selected[row["id"]]))
        scores = quality.score_rows(rows, self.args.data_root, FakeScores(), batch=8, negative_count=3)
        self.assertEqual(sum(row["review_flag"] for row in scores), 6)
        self.assertAlmostEqual(scores[0]["margin"], -.6)
        self.assertTrue(scores[0]["review_flag"])
        self.assertEqual(scores[0]["best_negative_caption"], by_id[scores[0]["best_negative_id"]]["caption"])

    def test_actual_packet_has_hashes_review_grid_and_no_training_actions(self):
        before = {path: path.read_bytes() for path in (self.args.r1, self.args.blind50, self.args.eval50)}
        result = quality.prepare(self.args, FakeScores())
        self.assertEqual((result["rows"], result["flagged"]), (48, 6))
        self.assertEqual(result["status"].split(";")[0], "uncalibrated diagnostic ranking")
        self.assertTrue((self.args.out / "caption-review-000.png").is_file())
        self.assertFalse((self.args.out / "sample_weights.json").exists())
        scored = json.loads((self.args.out / "caption-scores.json").read_text())
        self.assertTrue(all(len(row["image_sha256"]) == len(row["caption_sha256"]) == 64 for row in scored))
        self.assertEqual(before, {path: path.read_bytes() for path in before})
        self.assertNotIn(str(self.fixture.root), (self.args.out / "caption-summary.json").read_text())
        with self.assertRaises(ValueError):
            quality.prepare(self.args, FakeScores())

    def test_nonfinite_wrong_shape_and_noncosine_scores_fail(self):
        rows, _, _ = load_population(self.args)
        for invalid in (np.ones((1, 1)), np.full((8, 32), float("nan")), np.full((8, 32), 2.)):
            with self.assertRaises(ValueError):
                quality.score_rows(rows, self.args.data_root, SimpleNamespace(score=lambda a, b: invalid), batch=8, negative_count=3)

    def test_local_model_api_disables_downloads_and_normalizes_embeddings(self):
        calls = []
        class Processor:
            @staticmethod
            def from_pretrained(directory, **kwargs):
                calls.append(kwargs)
                return Processor()
            def __call__(self, **kwargs):
                self.options = kwargs
                return {"pixel_values": torch.ones(len(kwargs["images"]), 3), "input_ids": torch.ones(len(kwargs["text"]), 2)}
        class Model:
            config = SimpleNamespace(model_type="siglip")
            @staticmethod
            def from_pretrained(directory, **kwargs):
                calls.append(kwargs)
                return Model()
            def to(self, **kwargs):
                return self
            def eval(self):
                return self
            def __call__(self, **kwargs):
                return SimpleNamespace(image_embeds=torch.tensor([[3., 4.]]), text_embeds=torch.tensor([[3., 4.], [-3., -4.]]))
        fake_transformers = SimpleNamespace(AutoProcessor=Processor, AutoModel=Model)
        with patch.dict(sys.modules, {"transformers": fake_transformers}):
            scorer = quality.LocalSiglip(self.args.model_dir, "cpu")
            values = scorer.score([Image.new("RGB", (32, 32))], ["a", "b"])
        np.testing.assert_allclose(values, [[1., -1.]], atol=1e-6)
        self.assertTrue(all(call == {"local_files_only": True, "trust_remote_code": False} for call in calls))
        self.assertEqual(scorer.processor.options["padding"], "max_length")
        self.assertTrue(scorer.processor.options["truncation"])


if __name__ == "__main__":
    unittest.main()
