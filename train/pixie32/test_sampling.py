"""Configurable grids preserve native pixels, prompt order and training RNG."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image
from tokenizers import Tokenizer, models, pre_tokenizers
import torch
from torch import nn

from train.pixie32.data import Holdouts
from train.pixie32 import train_p32 as trainer

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



class GridModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.batches = []

    def sample(self, text, unconditional, *, cfg, generator):
        self.batches.append(len(text))
        torch.rand(1, generator=generator)
        pixels = text[:, 1, None].expand(-1, 1024).clone()
        pixels[:, 0] = 0
        return pixels


class SamplingTests(unittest.TestCase):
    def test_configured_grid_dimensions_order_alpha_and_rng(self):
        samples = [(f"fixture {i}", np.array([49406, i] + [49407] * 16, dtype=np.uint16)) for i in range(1, 6)]
        palette = np.zeros((4097, 3), dtype=np.uint8)
        palette[1:6, 0] = np.arange(1, 6) * 40
        model = GridModel().train()
        before = torch.get_rng_state().clone()
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary)
            trainer.sample_grid(model, lambda text: text, samples, palette, torch.device("cpu"),
                                out, 12, 42, scale=3, columns=2, batch=2)
            with Image.open(out / "samples_00000012.png") as grid:
                self.assertEqual(grid.size, (192, 288))
                for i in range(5):
                    x, y = (i % 2) * 96, (i // 2) * 96
                    self.assertEqual(grid.getpixel((x, y))[3], 0)
                    self.assertEqual(grid.getpixel((x + 3, y)), ((i + 1) * 40, 0, 0, 255))
                    self.assertEqual(grid.getpixel((x + 5, y + 2)), grid.getpixel((x + 3, y)))
            metadata = json.loads((out / "samples_00000012.json").read_text())
        self.assertEqual(metadata["prompts"], [prompt for prompt, _ in samples])
        self.assertEqual((metadata["scale"], metadata["columns"], metadata["batch"], metadata["seed"]), (3, 2, 2, 54))
        self.assertEqual(model.batches, [2, 2, 1])
        self.assertTrue(model.training)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))

    @needs_holdouts
    def test_prompt_file_excludes_punctuated_holdouts_and_refuses_empty(self):
        tokenizer = Tokenizer(models.WordLevel({"[UNK]": 0, "safe": 1, "sprite": 2}, unk_token="[UNK]"))
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        with tempfile.TemporaryDirectory() as temporary, redirect_stdout(io.StringIO()):
            path = Path(temporary) / "prompts.txt"
            path.write_text("green-poison-flask\n\nsafe sprite\n", encoding="utf-8")
            safe = trainer.load_sample_prompts(path, tokenizer, Holdouts())
            self.assertEqual([prompt for prompt, _ in safe], ["safe sprite"])
            path.write_text("green-poison-flask\n", encoding="utf-8")
            with self.assertRaisesRegex(trainer.InputError, "no safe sample prompts"):
                trainer.load_sample_prompts(path, tokenizer, Holdouts())
            path.write_text("\n", encoding="utf-8")
            with self.assertRaises(trainer.InputError):
                trainer.load_sample_prompts(path, tokenizer, Holdouts())

    def test_grid_failure_restores_eval_mode(self):
        model = GridModel().eval()
        def failed_clip(text):
            raise RuntimeError("fixture failure")
        with tempfile.TemporaryDirectory() as temporary, self.assertRaisesRegex(RuntimeError, "fixture failure"):
            trainer.sample_grid(model, failed_clip, [("fixture", np.zeros(18, dtype=np.uint16))],
                                np.zeros((4097, 3), dtype=np.uint8), torch.device("cpu"), Path(temporary), 0, 0)
        self.assertFalse(model.training)


if __name__ == "__main__":
    unittest.main()
