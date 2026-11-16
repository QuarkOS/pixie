"""Seed-contract behavior without checkpoints, a worker key, or network calls."""
import ast
import base64
from contextlib import redirect_stdout
import io
from pathlib import Path
from types import SimpleNamespace
import time
import tempfile
import unittest
from unittest.mock import patch
import warnings

import numpy as np
from PIL import Image
import torch

import v3_model as M
from fast_sample import FastSampler
from benchmark_blank_cache import write_report
from sampling_provenance import SCHEME, candidate_budget, job_seed
from phase_timing import PhaseTimer
from png_encoding import encode_png


class SeedTests(unittest.TestCase):
    def test_failed_hardware_replay_invalidates_report(self):
        report = dict(finite_features=True, features_exact=True, exact_sprite_match_fraction=1.,
                      seed_replay=dict(exact_replay_matches=2, ignored_tail_matches=1, sprites_checked=2))
        with tempfile.TemporaryDirectory() as temporary, redirect_stdout(io.StringIO()):
            self.assertEqual(write_report(report, Path(temporary) / "report.json"), 1)
        self.assertFalse(report["comparison_valid"])

    def test_seed_validation_and_fresh_default(self):
        for invalid in (-1, 2**31, True, "123", 1.5):
            with self.assertRaises(ValueError):
                job_seed(invalid)
        for valid in (0, 101, 2**31 - 1):
            self.assertEqual(job_seed(valid), valid)
        with patch("sampling_provenance.secrets.randbelow", return_value=17) as fresh:
            self.assertEqual(job_seed(), 17)
            fresh.assert_called_once_with(2**31)

    def test_requested_comparison_budget_and_chat_budget(self):
        self.assertEqual(candidate_budget({"source": "admin_grid", "n": 4}, True, 64), 4)
        self.assertEqual(candidate_budget({"n": 20}, True, 64), 64)
        self.assertEqual(candidate_budget({"n": 20}, False, 64), 20)
        for invalid in (0, 65, True, "4"):
            with self.assertRaises(ValueError):
                candidate_budget({"n": invalid}, True, 64)

    def test_real_sampler_replays_complete_batch_and_ignores_seed_tail(self):
        torch.set_num_threads(4)
        torch.manual_seed(9)
        config = M.V3Config(arch="p50", d_model=32, n_layer=1, n_head=4,
                            dropout=0, text_len=4, n_pal=3, style=True)
        sampler = FastSampler(M.build(config).eval(), torch.device("cpu"),
                              max_batch=4, use_graph=False, dtype=torch.float32)
        h, u = torch.randn(2, 4, M.CLIP_D), torch.randn(2, 4, M.CLIP_D)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="User provided device_type of 'cuda'")
            expected = sampler.sample(h, u, seeds=[101, 202])
            replay = sampler.sample(h, u, seeds=[101, 101])
            tail = sampler.sample(h, u, seeds=[101, 999])
        self.assertTrue(torch.equal(expected, replay))
        self.assertTrue(torch.equal(expected, tail))

    def test_actual_run_keeps_original_indices_after_ranking(self):
        # Execute the real method without worker import-time key/asset side effects.
        source = ast.parse((Path(__file__).parent / "worker_v3.py").read_text(encoding="utf-8"))
        cls = next(n for n in source.body if isinstance(n, ast.ClassDef) and n.name == "Gen")
        run = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "run")
        captured = []
        sampler = SimpleNamespace(use_graph=False, sample=lambda h, u, seeds, cond: captured.append(seeds.tolist()) or torch.zeros(3, 256, dtype=torch.long))
        fake = SimpleNamespace(tok=None, TL=4, blank_features=None, device="cpu", sampler=sampler,
                               palette=None, _encode=lambda rows: rows, cond_tensor=lambda c, n: None,
                               _scores=lambda p, ids: ([.1, .9, .5], [1]*3, [1]*3, "fake"))
        env = dict(torch=torch, np=np, time=time, Image=Image, io=io, base64=base64,
                   M=M, SCHEME=SCHEME, job_seed=job_seed, CFG_SCALE=2., TEMP=.8, TOPK=32,
                   CACHE_BLANK=False, SCALE=1, PROFILE_PHASES=False, PhaseTimer=PhaseTimer,
                   PNG_MODE="legacy", encode_png=encode_png,
                   t26=lambda t, p: [1, 2, 2, 2], trim=lambda row, n: row,
                   rgba_from_indices=lambda ids, pal: np.zeros((16, 16, 4), dtype=np.uint8))
        exec(compile(ast.Module(body=[run], type_ignores=[]), "worker_v3.py", "exec"), env)
        shown, ranked, _, _, _ = env["run"](fake, "copper golem", 3, 2, 3, seed=123)
        self.assertEqual(captured, [[123, 123, 123]])
        self.assertEqual([r["idx"] for r in shown], [1, 2])
        self.assertEqual([r["idx"] for r in ranked], [1, 2, 0])
        self.assertEqual([r["idx"] for r in fake.last_candidates], [1, 2, 0])
        self.assertTrue(all(r["seed"] == 123 for r in ranked))
        self.assertEqual(fake.last_sampling["job_seed"], 123)
        self.assertEqual(fake.last_sampling["batch_size"], 3)
        self.assertIsNone(fake.last_profile)
        fake.profile_phases = True
        profiled = env["run"](fake, "copper golem", 3, 2, 3, seed=123)
        self.assertEqual(profiled[:2], (shown, ranked))
        self.assertEqual(set(fake.last_profile["phase_ms"]),
                         {"prepare", "clip", "sample", "readback_rgba", "score_rank", "png"})
        self.assertLessEqual(sum(fake.last_profile["phase_ms"].values()), fake.last_profile["total_ms"])


if __name__ == "__main__":
    unittest.main()
