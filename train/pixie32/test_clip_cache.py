"""Synthetic CPU coverage for cache identity, caption choices and exact resume."""
from contextlib import redirect_stdout
from dataclasses import replace
import hashlib
import io
import json
from pathlib import Path
import tempfile
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from train.pixie32.clip_cache import FeatureCache, build_cache, idle_device
from train.pixie32.data import sha256
from train.pixie32 import train_p32 as trainer
from train.pixie32.test_resume import FixtureModel


class FrozenClip(torch.nn.Module):
    def forward(self, text):
        return (text.float() % 128).unsqueeze(-1).expand(-1, -1, 768)


class CacheTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.device = torch.device("cpu")
        text = np.tile(np.array([49406] + [34] * 16 + [49407], dtype=np.uint16), (5, 1))
        subject = text.copy()
        subject[:, 1] = 55
        self.dataset = SimpleNamespace(
            arrays={"text": text, "subj_text": subject, "tokens": np.zeros((5, 1024), dtype=np.uint16),
                    "cond": np.tile([1, 2, 3, 1, 1, 1], (5, 1)), "noflip": np.zeros(5, dtype=bool),
                    "weight": np.ones(5), "split": np.array([0, 0, 0, 1, 2])},
            ids=[str(i) for i in range(5)], manifest={"inputs": {"tokenizer": hashlib.sha256(b"{}").hexdigest()}},
            fingerprint="synthetic-dataset", holdouts=None, palette=np.zeros((4097, 3), dtype=np.uint8))
        self.cache_path = self.root / "cache"
        build_cache(self.dataset, FrozenClip(), "weights", self.cache_path, self.device, batch=2)

    def tearDown(self):
        self.temporary.cleanup()

    def manifest(self, update=None):
        path = self.cache_path / "manifest.json"
        value = json.loads(path.read_text())
        if update:
            update(value)
            path.write_text(json.dumps(value))
        return value

    def test_row_keys_survive_reordering_and_cover_caption_augmentation(self):
        for name in ("tokens.npy", "features.npy"):
            np.save(self.cache_path / name, np.load(self.cache_path / name)[::-1].copy())
        self.manifest(lambda m: m["files"].update({n: sha256(self.cache_path / n) for n in m["files"]}))
        cache = FeatureCache(self.cache_path, self.dataset, "weights")
        recipe = trainer.Recipe(3, 1, "dataset", {}, 3, 0, 0, .001, .0001)
        plan = {"rows": torch.tensor([0, 1, 2]), "random": torch.tensor([[.01, 1, 1, 1, 0], [.15, 1, 1, 1, 0], [.8, 1, 1, 1, 0]])}
        before = torch.get_rng_state().clone()
        text, image, cond = trainer.prepared_batch(self.dataset.arrays, plan, 0, 3, recipe, self.device, text_on_cpu=True)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        self.assertEqual(text[:, 1].tolist(), [49407, 55, 34])
        torch.testing.assert_close(cache(text, self.device), FrozenClip()(text), atol=0, rtol=0)
        expected = trainer.prepared_batch(self.dataset.arrays, plan, 0, 3, recipe, self.device)
        for actual, original in zip((text, image, cond), expected):
            self.assertTrue(torch.equal(actual, original))

    def test_identity_budget_corruption_and_missing_keys_are_refused(self):
        for data, weights, budget, pattern in (
            (self.dataset, "changed", 4, "identity"),
            (SimpleNamespace(**{**vars(self.dataset), "ids": list(reversed(self.dataset.ids))}), "weights", 4, "identity"),
            (self.dataset, "weights", 1e-8, "RAM budget")):
            with self.subTest(pattern=pattern), self.assertRaisesRegex(ValueError, pattern):
                FeatureCache(self.cache_path, data, weights, budget)
        cache = FeatureCache(self.cache_path, self.dataset, "weights")
        with self.assertRaisesRegex(ValueError, "missing"):
            cache(torch.zeros((1, 18), dtype=torch.long), self.device)
        with (self.cache_path / "features.npy").open("ab") as handle:
            handle.write(b"corrupt")
        with self.assertRaisesRegex(ValueError, "digest"):
            FeatureCache(self.cache_path, self.dataset, "weights")

    def test_invalid_keys_features_and_overwrite_are_refused(self):
        with self.assertRaisesRegex(ValueError, "already exists"):
            build_cache(self.dataset, FrozenClip(), "weights", self.cache_path, self.device)
        np.save(self.cache_path / "features.npy", np.full((3, 18, 768), np.nan, dtype=np.float16))
        self.manifest(lambda m: m["files"].update({"features.npy": sha256(self.cache_path / "features.npy")}))
        with self.assertRaisesRegex(ValueError, "non-finite"):
            FeatureCache(self.cache_path, self.dataset, "weights")

    def test_cached_mode_resume_is_exact_and_online_switch_is_refused(self):
        tokenizer, weights, source = (self.root / n for n in ("tokenizer.json", "clip.pt", "source.pt"))
        tokenizer.write_text("{}")
        torch.save({}, weights)
        torch.save({}, source)
        self.manifest(lambda m: m["identity"].update({"clip": sha256(weights)}))
        common = ["--data", str(self.root), "--clip", str(weights), "--tokenizer", str(tokenizer),
                  "--clip-cache", str(self.cache_path), "--device", "cpu", "--batch", "2", "--images", "8", "--decay-images", "0"]
        def run(name, extra):
            return trainer.run(trainer.parser().parse_args(common + ["--out", str(self.root / name)] + extra))
        with patch.object(trainer, "load_dataset", return_value=self.dataset), \
             patch.object(trainer, "P32", FixtureModel), \
             patch.object(trainer, "load_clip", return_value=FrozenClip()), \
             patch.object(trainer, "sample_grid"), \
             patch.object(trainer, "load_sample_prompts", return_value=[("fixture", self.dataset.arrays["text"][0])]), \
             patch("tokenizers.Tokenizer.from_file"), redirect_stdout(io.StringIO()):
            run("full", ["--v3-ckpt", str(source)])
            run("part", ["--v3-ckpt", str(source), "--smoke", "--smoke-steps", "2"])
            run("resumed", ["--resume", str(self.root / "part" / "last.pt")])
        full = torch.load(self.root / "full" / "last.pt", weights_only=False)
        resumed = torch.load(self.root / "resumed" / "last.pt", weights_only=False)
        for name in full["model"]:
            self.assertTrue(torch.equal(full["model"][name], resumed["model"][name]), name)
        for key in ("step", "images_seen", "epoch", "offset"):
            self.assertEqual(full[key], resumed[key])
        for parameter, state in full["optimizer"]["state"].items():
            for name, value in state.items():
                self.assertTrue(torch.equal(value, resumed["optimizer"]["state"][parameter][name]))
        self.assertTrue(torch.equal(full["rng"]["torch"], resumed["rng"]["torch"]))
        with self.assertRaisesRegex(ValueError, "clip_cache_sha256"):
            recipe = trainer.Recipe(2, 1, "dataset", {}, 8, 0, 0, .001, .0001)
            trainer.check_resume(trainer.fingerprint(recipe), trainer.fingerprint(replace(recipe, clip_cache_sha256="cache")))

    def test_busy_device_is_refused_before_cuda_initialization(self):
        with patch.dict("os.environ", {}, clear=True), patch("subprocess.check_output", return_value="12345\n"), \
             patch("torch.cuda.set_device") as cuda:
            with self.assertRaisesRegex(ValueError, "busy"):
                idle_device("cuda:0")
            cuda.assert_not_called()

    def test_benchmark_completes_alternating_trials_and_writes_only_results(self):
        from train.pixie32 import bench_clip_cache as bench
        from train.pixie32.p32_model import P32Config
        weights, checkpoint = self.root / "clip.pt", self.root / "checkpoint.pt"
        torch.save({}, weights)
        config = P32Config(d_model=32, n_layer=1, n_head=4)
        torch.save({"model": FixtureModel(config).state_dict(), "config": config.to_dict()}, checkpoint)
        self.manifest(lambda m: m["identity"].update({"clip": sha256(weights)}))
        args = ["bench", "--data", str(self.root), "--clip", str(weights), "--checkpoint", str(checkpoint),
                "--clip-cache", str(self.cache_path), "--out", str(self.root / "bench"), "--device", "cpu",
                "--batch", "2", "--steps", "1", "--warmup", "1", "--trials", "5"]
        with patch.object(sys, "argv", args), patch.object(bench, "load_dataset", return_value=self.dataset), \
             patch.object(bench, "P32", FixtureModel), patch.object(bench, "load_clip", return_value=FrozenClip()), \
             redirect_stdout(io.StringIO()):
            bench.main()
        report = json.loads((self.root / "bench" / "benchmark.json").read_text())
        self.assertEqual([t["cached"] for t in report["trials"]], [False, True, True, False, False, True, True, False, False, True])
        self.assertEqual(sum(t["images"] for t in report["trials"]), 20)
        self.assertEqual(report["feature_max_abs_error"], 0)
        self.assertEqual(len(list((self.root / "bench").iterdir())), 1)


if __name__ == "__main__":
    unittest.main()
