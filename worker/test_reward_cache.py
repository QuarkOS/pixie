"""Reward cache preserves exact scoring; no GPU performance is asserted."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from fast_v7 import FastRewardV7
from benchmark_reward_cache import compare, main as benchmark_main


class Encoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([.5, .8, .2, .1]))
        self.text_calls = self.image_calls = 0

    def encode_text(self, tokens):
        self.text_calls += 1
        return tokens.float().sum(1, keepdim=True) * self.weight + self.weight.flip(0)

    def encode_image(self, images):
        self.image_calls += 1
        return images.mean((1, 2, 3), keepdim=False)[:, None] * self.weight + self.weight.flip(0)


class RewardCacheTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.model = Encoder().eval()
        self.tokenizer = lambda texts: torch.tensor([[len(text), sum(map(ord, text)) % 1024] for text in texts])
        self.recipe = {"cosines": ["fake:light:raw", "fake:dark:pixel"], "hand": True,
                       "pca_k": 1, "pca_from": ["fake:light:raw"]}
        self.v7 = SimpleNamespace(device=torch.device("cpu"),
            models={"fake": {"model": self.model, "tokenizer": self.tokenizer, "size": 16,
                             "mean": [0., 0., 0.], "std": [1., 1., 1.]}}, recipe=self.recipe,
            pca={"int:fake:light:raw": (np.zeros(4), np.ones((4, 1))),
                 "cat:fake:light:raw": (np.zeros(8), np.ones((8, 1)))},
            mu=np.zeros(7), sd=np.ones(7), w=np.arange(8) / 10.)
        self.palette = np.tile(np.array([[30, 80, 120]], dtype=np.uint8), (4097, 1))
        self.rows = np.random.default_rng(12).integers(0, 17, (64, 256))
        self.common = patch.dict(sys.modules, {"common": SimpleNamespace(connected_components=lambda row: int(np.any(row)))})
        self.common.start()

    def tearDown(self):
        self.common.stop()

    def test_repeat_is_exact_and_image_work_is_not_cached(self):
        baseline = FastRewardV7(self.v7, self.palette)
        candidate = FastRewardV7(self.v7, self.palette, 4)
        expected = baseline.score_indices(self.rows, "copper golem")
        cold = candidate.score_indices(self.rows, "copper golem")
        warm = candidate.score_indices(self.rows, "copper golem")
        self.assertTrue(np.array_equal(expected, cold))
        self.assertTrue(np.array_equal(expected, warm))
        self.assertEqual(self.model.text_calls, 4)
        self.assertEqual(self.model.image_calls, 3)
        self.assertEqual((candidate.text_cache_hits, candidate.text_cache_misses), (1, 1))
        info = candidate.text_cache_info()
        self.assertEqual(info["entries"], 1)
        self.assertEqual(info["tensor_bytes"], 2 * 4 * 4)

    def test_weight_and_token_changes_invalidate(self):
        candidate = FastRewardV7(self.v7, self.palette, 4)
        candidate._encode_text("fake", "fixture")
        with torch.no_grad():
            self.model.weight.add_(.1)
        candidate._encode_text("fake", "fixture")
        self.v7.models["fake"]["tokenizer"] = lambda texts: torch.ones(len(texts), 2, dtype=torch.long)
        candidate._encode_text("fake", "fixture")
        self.assertEqual(self.model.text_calls, 6)
        self.assertEqual(candidate.text_cache_hits, 0)

    def test_math_mode_change_invalidates(self):
        candidate = FastRewardV7(self.v7, self.palette, 4)
        candidate._encode_text("fake", "fixture")
        previous = torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction
        try:
            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = not previous
            candidate._encode_text("fake", "fixture")
        finally:
            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = previous
        self.assertEqual(candidate.text_cache_misses, 2)

    def test_fp16_accumulation_change_invalidates(self):
        candidate = FastRewardV7(self.v7, self.palette, 4)
        candidate._encode_text("fake", "fixture")
        previous = torch.backends.cuda.matmul.allow_fp16_accumulation
        try:
            torch.backends.cuda.matmul.allow_fp16_accumulation = not previous
            candidate._encode_text("fake", "fixture")
        finally:
            torch.backends.cuda.matmul.allow_fp16_accumulation = previous
        self.assertEqual(candidate.text_cache_misses, 2)

    def test_model_replacement_invalidation_and_lru_eviction(self):
        candidate = FastRewardV7(self.v7, self.palette, 1)
        candidate._encode_text("fake", "first")
        candidate._encode_text("fake", "second")
        candidate._encode_text("fake", "first")
        self.assertEqual(candidate.text_cache_misses, 3)
        replacement = Encoder().eval()
        self.v7.models["fake"]["model"] = replacement
        candidate._encode_text("fake", "first")
        self.assertEqual(replacement.text_calls, 2)
        candidate.text_cache_size = 0
        self.assertEqual(len(candidate._text_cache), 0)

    def test_invalid_sizes_and_training_cache_are_refused(self):
        for invalid in (-1, True, 257, "4"):
            with self.assertRaises(ValueError):
                FastRewardV7(self.v7, self.palette, invalid)
        candidate = FastRewardV7(self.v7, self.palette, 4)
        self.model.train()
        with self.assertRaises(ValueError):
            candidate._encode_text("fake", "fixture")

    def test_profiles_actual_scorer_without_changing_scores(self):
        candidate = FastRewardV7(self.v7, self.palette, 4)
        expected = candidate.score_indices(self.rows, "fixture")
        self.assertIsNone(candidate.last_profile)
        candidate.profile_phases = True
        actual = candidate.score_indices(self.rows, "fixture")
        self.assertTrue(np.array_equal(expected, actual))
        phases = candidate.last_profile["phase_ms"]
        self.assertEqual(set(phases), {"hand_features", "text_encode_or_cache", "render", "image_encode", "pca_linear_head", "score_readback"})
        self.assertLessEqual(sum(phases.values()), candidate.last_profile["total_ms"])

    def make_generator(self, mismatch=False, fallback=False):
        generator = SimpleNamespace(v7=FastRewardV7(self.v7, self.palette), device=torch.device("cpu"),
                                    palette=self.palette, scorer_name="reward_v7", last_candidates=[],
                                    last_sampling={"scheme": "fixture"}, calls=[])
        def run(prompt, n, show, candidates, seed=None):
            self.assertEqual((n, show, candidates), (64, 4, 20))
            generator.calls.append(seed)
            scores = generator.v7.score_indices(self.rows, prompt)
            marker = bool(mismatch and generator.v7.scorer.text_cache_size)
            output = [{"png": "changed" if marker else "same", "score": float(scores[0])}]
            generator.last_candidates = output
            return output, [{"score": float(score)} for score in scores], 1., 1., "feat_linear" if fallback else "reward_v7"
        generator.run = run
        return generator

    def test_benchmark_checks_scores_outputs_and_real_cache_work(self):
        generator = self.make_generator()
        with patch("torch.cuda.synchronize"), patch("torch.cuda.reset_peak_memory_stats"), \
             patch("torch.cuda.max_memory_allocated", return_value=0), patch("torch.cuda.max_memory_reserved", return_value=0):
            result = compare(SimpleNamespace(), generator, prompts=["fixture"], trials=5, warmup=2,
                             batch=64, show=4, candidates=20)
        self.assertTrue(result["comparison_valid"])
        self.assertEqual(len(generator.calls), 21)
        self.assertEqual(len(result["rows"]), 5)
        self.assertTrue(all(row["sides"]["warm"]["hits"] == 1 for row in result["rows"]))
        self.assertTrue(all(row["sides"]["cold"]["misses"] == 1 for row in result["rows"]))
        self.assertEqual(len(result["separate_reward_profiles"]), 2)

    def test_benchmark_mismatch_and_scorer_fallback_are_invalid(self):
        with patch("torch.cuda.synchronize"), patch("torch.cuda.reset_peak_memory_stats"), \
             patch("torch.cuda.max_memory_allocated", return_value=0), patch("torch.cuda.max_memory_reserved", return_value=0):
            result = compare(SimpleNamespace(), self.make_generator(mismatch=True), prompts=["fixture"],
                             trials=5, warmup=2, batch=64, show=4, candidates=20)
            self.assertFalse(result["comparison_valid"])
            with self.assertRaises(ValueError):
                compare(SimpleNamespace(), self.make_generator(fallback=True), prompts=["fixture"],
                        trials=5, warmup=2, batch=64, show=4, candidates=20)

    def test_cli_load_failure_writes_invalid_result_without_private_text(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = root / "config.json"
            config.write_text(json.dumps({"scorer": "reward_v7"}))
            def fail():
                raise SystemExit("private operator path")
            with patch.dict("os.environ"), patch("benchmark_reward_cache.require_idle_gpu"), \
                 patch("torch.cuda.set_device"), patch("benchmark_reward_cache.importlib.import_module", return_value=SimpleNamespace(Gen=fail)), \
                 redirect_stdout(io.StringIO()):
                code = benchmark_main(["--config", str(config), "--device", "cuda:0", "--out", str(root / "out")])
            self.assertEqual(code, 1)
            encoded = (root / "out" / "reward-cache-benchmark.json").read_text()
            self.assertFalse(json.loads(encoded)["comparison_valid"])
            self.assertNotIn("private operator path", encoded)


if __name__ == "__main__":
    unittest.main()
