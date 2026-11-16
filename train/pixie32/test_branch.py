"""Full-state P32 branches change batching without reinterpreting progress."""
from contextlib import ExitStack, redirect_stdout
import copy
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
from train.pixie32.clip_cache import build_cache
from train.pixie32.data import sha256
from train.pixie32.test_clip_cache import FrozenClip
from train.pixie32.test_resume import FixtureModel


class BranchTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.tokenizer, self.clip, self.source = [self.root / name for name in ("tokenizer.json", "clip.pt", "v3.pt")]
        self.tokenizer.write_text("{}")
        torch.save({}, self.clip)
        torch.save({}, self.source)
        text = np.tile(np.array([49406] + [34] * 16 + [49407], dtype=np.uint16), (7, 1))
        self.dataset = SimpleNamespace(
            arrays={"text": text, "subj_text": text.copy(),
                    "tokens": np.arange(7 * 1024, dtype=np.uint16).reshape(7, 1024) % 12,
                    "cond": np.tile([1, 2, 3, 1, 1, 1], (7, 1)), "noflip": np.zeros(7, dtype=bool),
                    "weight": np.ones(7), "split": np.array([0, 0, 0, 0, 0, 1, 2])},
            ids=[str(i) for i in range(7)], manifest={"inputs": {"tokenizer": hashlib.sha256(b"{}").hexdigest()}},
            fingerprint="synthetic-dataset", holdouts=None, palette=np.zeros((4097, 3), dtype=np.uint8))
        self.cache = self.root / "clip-cache"
        build_cache(self.dataset, FrozenClip(), sha256(self.clip), self.cache, torch.device("cpu"), batch=2)
        self.common = ["--data", str(self.root), "--clip", str(self.clip), "--tokenizer", str(self.tokenizer),
                       "--device", "cpu", "--batch", "4", "--images", "28", "--warmup", "2",
                       "--decay-images", "8", "--seed", "17"]
        stack = self.enterContext(ExitStack())
        for name, value in (("load_dataset", self.dataset), ("load_clip", FrozenClip()),
                            ("load_sample_prompts", [("fixture", text[0])])):
            stack.enter_context(patch.object(trainer, name, return_value=value))
        stack.enter_context(patch.object(trainer, "P32", FixtureModel))
        stack.enter_context(patch.object(trainer, "sample_grid"))
        stack.enter_context(patch("tokenizers.Tokenizer.from_file"))
        stack.enter_context(redirect_stdout(io.StringIO()))
        self.run_training("parent", ["--v3-ckpt", str(self.source), "--smoke", "--smoke-steps", "2"])
        self.parent = self.root / "parent" / "last.pt"
        self.saved = torch.load(self.parent, weights_only=False)

    def run_training(self, name, extra):
        return trainer.run(trainer.parser().parse_args(self.common + ["--out", str(self.root / name)] + extra))

    def branch_args(self, source=None):
        return ["--branch-from", str(source or self.parent), "--batch", "5", "--clip-cache", str(self.cache),
                "--save-every-images", "7", "--eval-every-images", "7", "--sample-every-images", "7"]

    def test_branch_carries_optimizer_progress_and_new_batch_plan(self):
        before = sha256(self.parent)
        prepare = trainer.prepared_batch
        calls = []
        def capture(arrays, plan, start, count, *args, **kwargs):
            calls.append((plan, start, count, kwargs))
            return prepare(arrays, plan, start, count, *args, **kwargs)
        with patch.object(trainer, "prepared_batch", side_effect=capture):
            result = self.run_training("branch", self.branch_args() + ["--smoke", "--smoke-steps", "1"])
        checkpoint = torch.load(self.root / "branch" / "last.pt", weights_only=False)
        self.assertEqual((result["step"], result["images_seen"]), (3, 13))
        self.assertNotEqual(result["images_seen"], result["step"] * 5)
        self.assertEqual((checkpoint["epoch"], checkpoint["offset"]), (0, 5))
        self.assertEqual((checkpoint["next_save"], checkpoint["next_eval"], checkpoint["next_sample"]), (14, 14, 14))
        self.assertEqual((calls[0][1], calls[0][2], calls[0][3]), (0, 5, {"text_on_cpu": True}))
        expected = trainer.epoch_plan(torch.arange(5), torch.ones(5), 5, 17, 0)
        for key in expected:
            self.assertTrue(torch.equal(calls[0][0][key], expected[key]))
        for state in checkpoint["optimizer"]["state"].values():
            self.assertEqual(state["step"].item(), 3)
        self.assertEqual(checkpoint["optimizer"]["param_groups"][0]["lr"], 1e-4)
        provenance = json.loads((self.root / "branch" / "branch.json").read_text())
        self.assertEqual(provenance["parent_sha256"], before)
        self.assertEqual(provenance["parent_images_seen"], 8)
        self.assertEqual(set(provenance["changes"]), {"batch", "clip_cache_sha256", "save_every_images", "eval_every_images", "sample_every_images"})
        self.assertNotIn(str(self.root), json.dumps(provenance))
        self.assertEqual(before, sha256(self.parent))
        self.assertIn("--branch-from $P32_CKPT", (self.root / "branch" / "launch.txt").read_text())

    def test_branch_can_resume_exactly_with_lineage(self):
        self.run_training("whole", self.branch_args())
        self.run_training("part", self.branch_args() + ["--smoke", "--smoke-steps", "2"])
        resume = self.branch_args()[2:] + ["--resume", str(self.root / "part" / "last.pt")]
        self.run_training("resumed", resume)
        whole = torch.load(self.root / "whole" / "last.pt", weights_only=False)
        resumed = torch.load(self.root / "resumed" / "last.pt", weights_only=False)
        for field in ("step", "images_seen", "epoch", "offset", "branch"):
            self.assertEqual(whole[field], resumed[field])
        for key in whole["model"]:
            self.assertTrue(torch.equal(whole["model"][key], resumed["model"][key]), key)
        for number, values in whole["optimizer"]["state"].items():
            for key, value in values.items():
                self.assertTrue(torch.equal(value, resumed["optimizer"]["state"][number][key]))
        self.assertTrue(torch.equal(whole["rng"]["torch"], resumed["rng"]["torch"]))
        self.assertEqual(json.loads((self.root / "part" / "branch.json").read_text()),
                         json.loads((self.root / "resumed" / "branch.json").read_text()))
        with self.assertRaisesRegex(ValueError, "fingerprint mismatch"):
            self.run_training("bad-resume", resume + ["--batch", "6"])
        self.assertFalse((self.root / "bad-resume").exists())

    def test_branch_refuses_changed_immutable_recipe_or_existing_output(self):
        for flag, value in (("--images", "30"), ("--lr", "0.0002"), ("--seed", "18"), ("--warmup", "0")):
            with self.subTest(flag=flag), self.assertRaisesRegex(ValueError, "branch-from cannot change"):
                self.run_training("invalid", self.branch_args() + [flag, value])
            self.assertFalse((self.root / "invalid").exists())
        with patch.object(trainer, "load_dataset", return_value=SimpleNamespace(**{**vars(self.dataset), "fingerprint": "changed"})):
            with self.assertRaisesRegex(ValueError, "identity|cannot change"):
                self.run_training("invalid-data", self.branch_args())
        with self.assertRaisesRegex(ValueError, "already exists"):
            self.run_training("parent", self.branch_args())

    def test_branch_refuses_incomplete_progress_config_digest_and_optimizer(self):
        mutations = [lambda x: x.pop("optimizer"), lambda x: x.update(format="model-only"),
                     lambda x: x.update(step=-1), lambda x: x.update(images_seen=28),
                     lambda x: x.update(images_seen=0), lambda x: x.update(config={}),
                     lambda x: x["fingerprint"].update(sha256="broken"),
                     lambda x: x["optimizer"]["param_groups"][0].update(weight_decay=.1),
                     lambda x: x["optimizer"]["param_groups"][0]["params"].reverse(),
                     lambda x: x["optimizer"]["state"][0].update(step=torch.tensor(100.))]
        for index, mutate in enumerate(mutations):
            with self.subTest(index=index):
                saved = copy.deepcopy(self.saved)
                mutate(saved)
                source = self.root / f"invalid-{index}.pt"
                torch.save(saved, source)
                with self.assertRaises(ValueError):
                    self.run_training("invalid", self.branch_args(source))
                self.assertFalse((self.root / "invalid").exists())

    def test_branch_source_flags_are_mutually_exclusive(self):
        for flag in ("--resume", "--v3-ckpt"):
            with self.subTest(flag=flag), self.assertRaises(SystemExit):
                with patch("sys.stderr", io.StringIO()):
                    trainer.parser().parse_args(self.common + ["--out", "new", "--branch-from", "source", flag, "other"])


if __name__ == "__main__":
    unittest.main()
