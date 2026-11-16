"""CPU behavioral tests. Tiny model/CLIP fixtures are never a training mode."""
from contextlib import redirect_stdout
from dataclasses import replace
import hashlib
import io
import signal
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from train.pixie32 import train_p32 as trainer


class FixtureModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layers = nn.Sequential(nn.Linear(1, 8), nn.Dropout(0.3), nn.Linear(8, 1))

    def load_v3_warmstart(self, path):
        return {"copied": [], "remapped": [], "new": []}

    def loss_fwd(self, features, image, alpha_weight, cond=None, chunks=0):
        prediction = self.layers(features.mean((1, 2)).unsqueeze(1) / 50_000)
        loss = (prediction - image.float().mean(1, keepdim=True) / 100).square().mean()
        return loss, loss.detach()


class FixtureClip(nn.Module):
    def load_hf(self, weights):
        pass

    def forward(self, text):
        return text.float().unsqueeze(-1)


class ResumeTests(unittest.TestCase):
    def recipe(self):
        return trainer.Recipe(4, 17, "dataset-digest", {"d_model": 8}, 24, 2, 8, 1e-4, 1e-5)

    def test_changed_recipe_is_refused(self):
        recipe = self.recipe()
        for changed in (replace(recipe, batch=8), replace(recipe, seed=18),
                        replace(recipe, dataset="different"), replace(recipe, images=28),
                        replace(recipe, ce_chunks=8), replace(recipe, device_type="cuda")):
            with self.subTest(changed=changed), self.assertRaisesRegex(ValueError, "fingerprint mismatch"):
                trainer.check_resume(trainer.fingerprint(recipe), trainer.fingerprint(changed))

    def test_epoch_plans_do_not_consume_training_rng(self):
        torch.manual_seed(12)
        before = torch.get_rng_state().clone()
        args = (torch.tensor([2, 5, 7]), torch.tensor([1., 0.5, 2.]), 4, 3, 8)
        first, second = trainer.epoch_plan(*args), trainer.epoch_plan(*args)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        self.assertTrue(torch.equal(first["rows"], second["rows"]))
        self.assertTrue(torch.equal(first["random"], second["random"]))

    def test_wsd_warmup_plateau_and_decay(self):
        recipe = self.recipe()
        self.assertEqual(trainer.lr_at(0, 0, recipe), recipe.lr / 2)
        self.assertEqual(trainer.lr_at(2, 8, recipe), recipe.lr)
        self.assertEqual(trainer.lr_at(4, 16, recipe), recipe.lr)
        self.assertAlmostEqual(trainer.lr_at(5, 20, recipe), (recipe.lr + recipe.lr_min) / 2)
        self.assertAlmostEqual(trainer.lr_at(6, 24, recipe), recipe.lr_min)

    def test_real_training_loop_resume_matches_uninterrupted(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tokenizer = root / "tokenizer.json"
            tokenizer.write_text("{}", encoding="utf-8")
            clip, warmstart = root / "clip.pt", root / "warmstart.pt"
            torch.save({}, clip)
            torch.save({}, warmstart)
            arrays = {
                "tokens": np.arange(5 * 1024, dtype=np.uint16).reshape(5, 1024) % 12,
                "text": np.tile(np.array([49406] + [34] * 16 + [49407], dtype=np.uint16), (5, 1)),
                "subj_text": np.tile(np.array([49406] + [55] * 16 + [49407], dtype=np.uint16), (5, 1)),
                "noflip": np.array([False, True, False, False, False]),
                "cond": np.tile(np.array([1, 2, 3, 1, 1, 1]), (5, 1)),
                "split": np.array([0, 0, 0, 1, 2]), "weight": np.array([1., .5, 2., 1., 1.]),
            }
            dataset = SimpleNamespace(arrays=arrays, fingerprint="fixture",
                                      manifest={"inputs": {"tokenizer": hashlib.sha256(b"{}").hexdigest()}},
                                      holdouts=None, palette=np.zeros((4097, 3), dtype=np.uint8))
            common = ["--data", str(root), "--blind50", str(root / "blind.json"),
                      "--eval50", str(root / "eval.json"), "--clip", str(clip),
                      "--tokenizer", str(tokenizer), "--device", "cpu", "--batch", "4",
                      "--images", "24", "--warmup", "2", "--decay-images", "8", "--seed", "17"]
            def run(name, extra):
                return trainer.run(trainer.parser().parse_args(common + ["--out", str(root / name)] + extra))
            with patch.object(trainer, "load_dataset", return_value=dataset), \
                 patch.object(trainer, "P32", FixtureModel), \
                 patch.object(trainer, "ClipTextPenultimate", FixtureClip), \
                 patch.object(trainer, "sample_grid"), \
                 patch.object(trainer, "load_sample_prompts", return_value=[("fixture", arrays["text"][0])]), \
                 patch("tokenizers.Tokenizer.from_file"), redirect_stdout(io.StringIO()):
                run("full", ["--warmstart", str(warmstart)])
                paused = run("part", ["--warmstart", str(warmstart), "--smoke", "--smoke-steps", "2"])
                self.assertEqual(paused["status"], "smoke")
                self.assertFalse((root / "part" / "train_summary.json").exists())
                self.assertTrue((root / "part" / "paused.json").exists())
                run("resumed", ["--resume", str(root / "part" / "last.pt")])
                with self.assertRaisesRegex(ValueError, "already exists"):
                    run("full", ["--warmstart", str(warmstart)])
                with self.assertRaisesRegex(ValueError, "fingerprint mismatch"):
                    run("changed_batch", ["--resume", str(root / "part" / "last.pt"), "--batch", "8"])
                self.assertFalse((root / "changed_batch").exists())
                timed = run("time_pause", ["--warmstart", str(warmstart), "--max-hours", "1e-9"])
                self.assertEqual(timed["status"], "max_hours")
                self.assertFalse((root / "time_pause" / "train_summary.json").exists())
                prepare = trainer.prepared_batch
                def interrupt_batch(*args, **kwargs):
                    signal.raise_signal(signal.SIGINT)
                    return prepare(*args, **kwargs)
                with patch.object(trainer, "prepared_batch", side_effect=interrupt_batch):
                    stopped = run("interrupt", ["--warmstart", str(warmstart)])
                self.assertEqual((stopped["status"], stopped["step"]), ("interrupt", 1))
                self.assertTrue((root / "interrupt" / "last.pt").exists())
                self.assertTrue((root / "interrupt" / "paused.json").exists())
                self.assertFalse((root / "interrupt" / "train_summary.json").exists())
                self.assertFalse((root / "interrupt" / "worker.pid").exists())
                run("interrupt_resumed", ["--resume", str(root / "interrupt" / "last.pt")])
                real_compile = torch.compile
                with patch.object(torch, "compile", side_effect=lambda function: real_compile(function, backend="eager")):
                    run("compiled_full", ["--warmstart", str(warmstart), "--compile"])
                    run("compiled_part", ["--warmstart", str(warmstart), "--compile", "--smoke", "--smoke-steps", "2"])
                    run("compiled_resumed", ["--resume", str(root / "compiled_part" / "last.pt"), "--compile"])
                with self.assertRaisesRegex(ValueError, "compile"):
                    run("changed_compile", ["--resume", str(root / "compiled_part" / "last.pt")])
                inherited = torch.load(root / "part" / "last.pt", weights_only=False)
                inherited["best_iid_nll"] = -1.0
                torch.save(inherited, root / "earlier_best.pt")
                local = run("local_best", ["--resume", str(root / "earlier_best.pt")])
                self.assertEqual(local["global_best_iid_nll"], -1.0)
                self.assertGreaterEqual(local["segment_best_iid_nll"], 0.0)
                self.assertTrue((root / "local_best" / "best.pt").exists())
            full = torch.load(root / "full" / "last.pt", weights_only=False)
            for name in ("resumed", "interrupt_resumed"):
                resumed = torch.load(root / name / "last.pt", weights_only=False)
                self.assertEqual((resumed["step"], resumed["images_seen"]), (6, 24))
                self.assertEqual((full["epoch"], full["offset"]), (resumed["epoch"], resumed["offset"]))
                for key in full["model"]:
                    self.assertTrue(torch.equal(full["model"][key], resumed["model"][key]), key)
                for number, state in full["optimizer"]["state"].items():
                    for key, value in state.items():
                        self.assertTrue(torch.equal(value, resumed["optimizer"]["state"][number][key]), key)
                self.assertTrue(torch.equal(full["rng"]["torch"], resumed["rng"]["torch"]))
            compiled = torch.load(root / "compiled_full" / "last.pt", weights_only=False)
            resumed = torch.load(root / "compiled_resumed" / "last.pt", weights_only=False)
            self.assertEqual(set(compiled["model"]), set(full["model"]))
            for key, value in compiled["model"].items():
                self.assertTrue(torch.equal(value, resumed["model"][key]), key)
            for number, state in compiled["optimizer"]["state"].items():
                for key, value in state.items():
                    self.assertTrue(torch.equal(value, resumed["optimizer"]["state"][number][key]), key)
            self.assertTrue(torch.equal(compiled["rng"]["torch"], resumed["rng"]["torch"]))


if __name__ == "__main__":
    unittest.main()
