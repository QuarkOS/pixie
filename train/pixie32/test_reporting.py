"""Reporting uses each trained state once and exposes segment overhead."""
from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from train.pixie32 import train_p32 as trainer
from train.pixie32 import bench_reporting
from train.pixie32.test_resume import FixtureModel, FixtureClip


class ReportingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        (self.root / "tokenizer.json").write_text("{}")
        torch.save({}, self.root / "clip.pt")
        torch.save({}, self.root / "v3.pt")
        arrays = {"tokens": np.ones((5, 1024), dtype=np.uint16),
                  "text": np.tile(np.array([49406] + [34] * 16 + [49407], dtype=np.uint16), (5, 1)),
                  "subj_text": np.tile(np.array([49406] + [55] * 16 + [49407], dtype=np.uint16), (5, 1)),
                  "noflip": np.zeros(5, dtype=bool), "cond": np.tile([1, 2, 3, 1, 1, 1], (5, 1)),
                  "split": np.array([0, 0, 0, 1, 2]), "weight": np.ones(5)}
        self.dataset = SimpleNamespace(arrays=arrays, fingerprint="fixture",
            manifest={"inputs": {"tokenizer": hashlib.sha256(b"{}").hexdigest()}},
            holdouts=None, palette=np.zeros((4097, 3), dtype=np.uint8))
        self.common = ["--data", str(self.root), "--clip", str(self.root / "clip.pt"),
            "--tokenizer", str(self.root / "tokenizer.json"), "--device", "cpu", "--batch", "4",
            "--images", "24", "--warmup", "0", "--decay-images", "0", "--seed", "17"]
        self.stack = []
        for target, value in (("load_dataset", self.dataset), ("P32", FixtureModel),
                              ("ClipTextPenultimate", FixtureClip),
                              ("load_sample_prompts", [("fixture", arrays["text"][0])])):
            replacement = patch.object(trainer, target, return_value=value) if target in ("load_dataset", "load_sample_prompts") else patch.object(trainer, target, value)
            replacement.start()
            self.stack.append(replacement)
        replacement = patch("tokenizers.Tokenizer.from_file")
        replacement.start()
        self.stack.append(replacement)

    def tearDown(self):
        for replacement in reversed(self.stack):
            replacement.stop()
        self.temporary.cleanup()

    def run_training(self, name, extra):
        args = trainer.parser().parse_args(self.common + ["--out", str(self.root / name)] + extra)
        with redirect_stdout(io.StringIO()):
            return trainer.run(args)

    def test_final_boundary_does_not_repeat_evaluation_or_sample(self):
        source = ["--v3-ckpt", str(self.root / "v3.pt")]
        with patch.object(trainer, "sample_grid") as render, patch.object(trainer, "evaluate", wraps=trainer.evaluate) as evaluate:
            boundary = self.run_training("boundary", source + ["--images", "8", "--eval-every-images", "8", "--sample-every-images", "8"])
            self.assertEqual((evaluate.call_count, render.call_count), (1, 1))
        with patch.object(trainer, "sample_grid") as render, patch.object(trainer, "evaluate", wraps=trainer.evaluate) as evaluate:
            final = self.run_training("final", source + ["--images", "8"])
            self.assertEqual((evaluate.call_count, render.call_count), (1, 1))
        for name in ("boundary", "final"):
            state = torch.load(self.root / name / "last.pt", weights_only=False)
            self.assertEqual((state["step"], state["images_seen"]), (2, 8))
        before = torch.load(self.root / "boundary" / "last.pt", weights_only=False)
        after = torch.load(self.root / "final" / "last.pt", weights_only=False)
        for name in before["model"]:
            self.assertTrue(torch.equal(before["model"][name], after["model"][name]))
        self.assertTrue(torch.equal(before["rng"]["torch"], after["rng"]["torch"]))
        self.assertEqual(boundary["iid_nll"], final["iid_nll"])

    def test_phase_times_are_disjoint_and_count_segment_images(self):
        source = ["--v3-ckpt", str(self.root / "v3.pt")]
        with patch.object(trainer, "sample_grid"):
            result = self.run_training("part", source + ["--smoke", "--smoke-steps", "2"])
            resumed = self.run_training("resumed", ["--resume", str(self.root / "part" / "last.pt"), "--smoke", "--smoke-steps", "1"])
        for value in (result, resumed):
            self.assertEqual(value["phase_counts"]["training"], value["measured_steps"])
            self.assertEqual(value["phase_counts"]["evaluation"], 1)
            self.assertEqual(value["phase_counts"]["sampling"], 1)
            self.assertEqual(value["phase_counts"]["checkpoint"], 3)
            self.assertGreaterEqual(value["unclassified_seconds"], 0)
            self.assertLessEqual(sum(value["phase_seconds"].values()), value["segment_wall_seconds"])
            self.assertEqual(value["training_seconds"], value["phase_seconds"]["training"])
            self.assertAlmostEqual(value["wall_images_per_second"], value["segment_images"] / value["segment_wall_seconds"])
        self.assertEqual((resumed["images_seen"], resumed["segment_images"]), (12, 4))

    def test_benchmark_collects_five_fresh_segments_without_paths(self):
        arguments = ["benchmark", "--out", str(self.root / "bench"), "--steps", "1", "--trials", "5", "--",
                     *self.common, "--resume", str(self.root / "source.pt")]
        def child(command, **kwargs):
            out = Path(command[command.index("--out") + 1])
            out.mkdir()
            value = {"status": "smoke", "measured_steps": 1, "segment_images": 4,
                     "training_seconds": 1., "segment_wall_seconds": 2., "images_per_second": 4.,
                     "wall_images_per_second": 2., "phase_seconds": {"training": 1.},
                     "phase_counts": {"training": 1}, "unclassified_seconds": 1.}
            (out / "paused.json").write_text(json.dumps(value))
            return SimpleNamespace(returncode=0)
        with patch("sys.argv", arguments), patch.object(bench_reporting, "idle_device"), \
             patch.object(bench_reporting.subprocess, "run", side_effect=child) as launch, redirect_stdout(io.StringIO()):
            self.assertEqual(bench_reporting.main(), 0)
        self.assertEqual(launch.call_count, 5)
        encoded = (self.root / "bench" / "benchmark.json").read_text()
        self.assertNotIn(str(self.root), encoded)
        self.assertTrue(json.loads(encoded)["comparison_valid"])


if __name__ == "__main__":
    unittest.main()
