"""CPU behavior checks; GPU parity must pass the hardware benchmark."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import warnings

import torch

import v3_model as M
from blank_features import BlankFeatures
from benchmark_blank_cache import compare_outputs, require_idle_gpu, write_report
from fast_sample import FastSampler


class CountingClip(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(64, M.CLIP_D)
        self.rows = []

    def forward(self, ids):
        self.rows.append(tuple(ids.shape))
        offsets = torch.arange(len(ids), device=ids.device)[:, None, None]
        return self.embedding(ids) + offsets


class BlankFeaturesTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        torch.set_num_threads(4)
        self.clip = CountingClip().eval()
        self.features = BlankFeatures(self.clip)
        self.row, self.blank = [1, 3, 4, 2], [1, 2, 2, 2]

    def test_full_batch_rows_are_preserved_and_only_blank_is_reused(self):
        expected = self.features.encode_rows([self.blank] * 64)
        self.clip.rows.clear()
        h, u = self.features.batch(self.row, self.blank, 64)
        h2, u2 = self.features.batch([1, 5, 6, 2], self.blank, 64)
        self.assertTrue(torch.equal(u, expected))
        self.assertTrue(torch.equal(u2, expected))
        self.assertNotEqual(u[0, 0, 0], u[1, 0, 0])
        self.assertFalse(torch.equal(h, h2))
        self.assertEqual(u.data_ptr(), u2.data_ptr())
        self.assertEqual(self.clip.rows, [(64, 4), (64, 4), (64, 4)])
        self.assertFalse(h.requires_grad)

    def test_cache_invalidates_for_batch_tokens_dtype_weights_and_encoder(self):
        self.features.batch(self.row, self.blank, 2)
        self.features.batch(self.row, self.blank, 3)
        self.features.batch(self.row, [1, 2, 3, 2], 3)
        self.clip.to(torch.float64)
        _, u = self.features.batch(self.row, self.blank, 3)
        self.assertEqual(u.dtype, torch.float64)
        state = {name: value.clone() + .1 for name, value in self.clip.state_dict().items()}
        self.clip.load_state_dict(state)
        _, reloaded = self.features.batch(self.row, self.blank, 3)
        self.assertFalse(torch.equal(u, reloaded))
        self.assertEqual(len(self.clip.rows), 10)
        other = CountingClip().eval()
        self.features.encoder = other
        self.features.batch(self.row, self.blank, 3)
        self.assertEqual(other.rows, [(3, 4), (3, 4)])
        self.features.clear()
        self.features.batch(self.row, self.blank, 3)
        self.assertEqual(len(other.rows), 4)

    def test_eval_and_token_shape_guards(self):
        for row, blank, n in ((self.row, self.blank, 0), (self.row, [], 2)):
            with self.assertRaises(ValueError):
                self.features.batch(row, blank, n)
        self.clip.train()
        with self.assertRaises(ValueError):
            self.features.batch(self.row, self.blank, 2)

    def test_real_clip_layer_and_seeded_sprites_match_exactly(self):
        clip = M.ClipTextPenultimate(n_layers=1).eval()
        features = BlankFeatures(clip)
        c = M.V3Config(arch="p50", d_model=32, n_layer=1, n_head=4,
                       dropout=0, text_len=4, n_pal=3, style=True)
        model = M.build(c).eval()
        sampler = FastSampler(model, torch.device("cpu"), max_batch=4,
                              use_graph=False, dtype=torch.float32)
        seeds = torch.tensor([101, 202])
        baseline = (features.encode_rows([self.row] * 2), features.encode_rows([self.blank] * 2))
        features.batch(self.row, self.blank, 2)
        reused = features.batch(self.row, self.blank, 2)
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(baseline, reused)))
        cond = torch.tensor([[1, 2, M.NO_PALETTE, 3, 1, 2]]).expand(2, -1)
        with warnings.catch_warnings(), torch.no_grad():
            warnings.filterwarnings("ignore", message="User provided device_type of 'cuda'")
            for ct in (None, cond):
                expected = sampler.sample(*baseline, seeds=seeds, cond=ct)
                actual = sampler.sample(*reused, seeds=seeds, cond=ct)
                self.assertTrue(torch.equal(expected, actual))

    def test_exact_mismatch_writes_invalid_report_and_returns_failure(self):
        a = torch.zeros(2, 4, 768)
        b = a.clone()
        b[0, 0, 0] = torch.finfo(b.dtype).eps
        sprites = torch.zeros(2, 256, dtype=torch.long)
        parity = compare_outputs((a, a, sprites), (a, b, sprites))
        self.assertFalse(parity["features_exact"])
        self.assertEqual(parity["matched_sprites"], 2)
        with tempfile.TemporaryDirectory() as temporary, redirect_stdout(io.StringIO()):
            out = Path(temporary) / "report.json"
            report = {"finite_features": True, "features_exact": False, "exact_sprite_match_fraction": 1.0}
            self.assertEqual(write_report(report, out), 1)
            self.assertFalse(json.loads(out.read_text())["comparison_valid"])
            with self.assertRaises(FileExistsError):
                write_report(report, out)
        changed = sprites.clone()
        changed[0, 1] = 1
        self.assertEqual(compare_outputs((a, a, sprites), (a, a, changed))["matched_sprites"], 1)

    def test_busy_or_unknown_gpu_is_refused(self):
        with patch.dict("os.environ", {}, clear=True), patch("subprocess.run") as run:
            run.return_value.stdout = ""
            require_idle_gpu(0)
            for value in ("12345\n", "N/A\n"):
                run.return_value.stdout = value
                with self.assertRaises(RuntimeError):
                    require_idle_gpu(0)


if __name__ == "__main__":
    unittest.main()
