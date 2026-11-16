"""Operator input resolution and full-shape synthetic CLIP loading on CPU."""
import os
from contextlib import redirect_stderr
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from train.v3_model import ClipTextPenultimate
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



def clip_hf_fixture(model):
    state = model.state_dict()
    result = {
        "text_model.embeddings.token_embedding.weight": state["token_embedding.weight"],
        "text_model.embeddings.position_embedding.weight": state["position_embedding.weight"],
    }
    names = {"q_proj": "self_attn.q_proj", "k_proj": "self_attn.k_proj",
             "v_proj": "self_attn.v_proj", "out_proj": "self_attn.out_proj",
             "fc1": "mlp.fc1", "fc2": "mlp.fc2",
             "layer_norm1": "layer_norm1", "layer_norm2": "layer_norm2"}
    for layer in range(11):
        for own, external in names.items():
            for suffix in ("weight", "bias"):
                result[f"text_model.encoder.layers.{layer}.{external}.{suffix}"] = state[f"layers.{layer}.{own}.{suffix}"]
    return result


class InputTests(unittest.TestCase):
    def test_cli_preserves_oom_traceback_and_message(self):
        stderr = io.StringIO()
        with patch.object(trainer, "run", side_effect=torch.OutOfMemoryError("fixture allocation failed")), \
             patch("sys.argv", ["train_p32", "--data", "fixture", "--out", "new", "--device", "cpu"]), \
             redirect_stderr(stderr):
            self.assertEqual(trainer.main(), 1)
        message = stderr.getvalue()
        self.assertIn("Traceback (most recent call last)", message)
        self.assertIn("OutOfMemoryError: fixture allocation failed", message)
        self.assertNotIn("verify supplied inputs", message)

    def args(self, extra=()):
        return trainer.parser().parse_args(["--data", "fixture-data", "--out", "fixture-run",
                                           "--device", "cpu", *extra])

    @needs_holdouts
    def test_environment_inputs_and_holdout_defaults(self):
        with patch.dict(os.environ, {"V3_CKPT": "env-v3.pt", "CLIP_DIR": "env-clip",
                                     "TOKENIZER": "env-tokenizer.json",
                                     **{k: v for k, v in os.environ.items() if k == "PIXIE_HOLDOUT_DIR"}},
                        clear=True):
            args = trainer.resolve_inputs(self.args())
        self.assertEqual(args.warmstart, Path("env-v3.pt"))
        self.assertEqual(args.clip, Path("env-clip") / "clip_l14_text_l11_f32.pt")
        self.assertEqual(args.tokenizer, Path("env-tokenizer.json"))
        self.assertEqual(args.blind50.name, "blind-50.json")
        self.assertEqual(args.eval50.name, "eval-50.json")
        self.assertTrue(args.blind50.is_file())
        self.assertTrue(args.eval50.is_file())

    def test_flags_override_environment(self):
        with patch.dict(os.environ, {"V3_CKPT": "env-v3.pt", "CLIP_DIR": "env-clip",
                                     "TOKENIZER": "env-tokenizer.json",
                                     **{k: v for k, v in os.environ.items() if k == "PIXIE_HOLDOUT_DIR"}},
                        clear=True):
            args = trainer.resolve_inputs(self.args(["--v3-ckpt", "flag-v3.pt", "--clip-dir", "flag-clip",
                                                     "--tokenizer", "flag-tokenizer.json",
                                                     "--blind50", "flag-blind.json", "--eval50", "flag-eval.json"]))
        self.assertEqual(args.warmstart, Path("flag-v3.pt"))
        self.assertEqual(args.clip, Path("flag-clip") / "clip_l14_text_l11_f32.pt")
        self.assertEqual(args.tokenizer, Path("flag-tokenizer.json"))
        self.assertEqual(args.blind50, Path("flag-blind.json"))
        self.assertEqual(args.eval50, Path("flag-eval.json"))

    def test_resume_does_not_require_or_use_v3_checkpoint(self):
        with patch.dict(os.environ, {"V3_CKPT": "ignored.pt", "CLIP_DIR": "env-clip",
                                     "TOKENIZER": "env-tokenizer.json",
                                     **{k: v for k, v in os.environ.items() if k == "PIXIE_HOLDOUT_DIR"}},
                        clear=True):
            args = trainer.resolve_inputs(self.args(["--resume", "last.pt"]))
        self.assertIsNone(args.warmstart)
        self.assertEqual(args.resume, Path("last.pt"))

    def test_missing_environment_is_clear(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(trainer.InputError, "--v3-ckpt or V3_CKPT"):
                trainer.resolve_inputs(self.args())
            with self.assertRaisesRegex(trainer.InputError, "--tokenizer or TOKENIZER"):
                trainer.resolve_inputs(self.args(["--v3-ckpt", "fixture.pt"]))
            with self.assertRaisesRegex(trainer.InputError, "--clip-dir or CLIP_DIR"):
                trainer.resolve_inputs(self.args(["--v3-ckpt", "fixture.pt", "--tokenizer", "fixture.json"]))

    def test_full_shape_clip_hf_load_and_frozen_forward(self):
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(min(previous_threads, 4))
        try:
            source = ClipTextPenultimate()
            fixture = clip_hf_fixture(source)
            self.assertEqual(len(fixture), 178)
            self.assertEqual(sum(value.numel() for value in fixture.values()), 115_971_072)
            self.assertTrue(all(value.dtype == torch.float32 for value in fixture.values()))
            with tempfile.TemporaryDirectory() as temporary:
                path = Path(temporary) / "clip_l14_text_l11_f32.pt"
                torch.save(fixture, path)
                del source, fixture
                loaded = trainer.load_clip(path, torch.device("cpu"))
                ids = trainer.empty_text(1, 18, torch.device("cpu"))
                ids[0, 1:3] = torch.tensor([736, 46055])
                features = loaded(ids)
                self.assertEqual(tuple(features.shape), (1, 18, 768))
                self.assertTrue(torch.isfinite(features).all())
                self.assertFalse(features.requires_grad)
                self.assertFalse(loaded.training)
                self.assertTrue(all(not parameter.requires_grad for parameter in loaded.parameters()))
        finally:
            torch.set_num_threads(previous_threads)


if __name__ == "__main__":
    unittest.main()
