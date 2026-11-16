from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from train.pixie32.bench_checkpoint import main, on_device, same_tensors


class CheckpointTests(unittest.TestCase):
    def fixture(self):
        model = torch.nn.Linear(2, 2)
        optimizer = torch.optim.AdamW(model.parameters())
        model(torch.ones(1, 2)).sum().backward()
        optimizer.step()
        return {"format": "pixie32-full-v1", "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "config": {}, "step": 1, "images_seen": 80, "rng": {"torch": torch.get_rng_state()}}

    def test_real_write_readback_five_pairs_without_source_changes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.pt"
            saved = self.fixture()
            torch.save(saved, source)
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            args = ["--checkpoint", str(source), "--device", "cpu", "--out", str(root / "bench")]
            with redirect_stdout(io.StringIO()):
                self.assertEqual(main(args), 0)
            report = json.loads((root / "bench/checkpoint-benchmark.json").read_text())
            self.assertTrue(report["comparison_valid"])
            self.assertEqual(len(report["trials"]), 5)
            self.assertEqual(report["checkpoint_sha256"], digest)
            self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), digest)
            self.assertEqual(len(list((root / "bench").glob("trial-*/*.pt"))), 10)
            self.assertTrue(all(row["roundtrip_exact"] for row in report["trials"]))
            self.assertNotIn(str(root), json.dumps(report))
            with self.assertRaises(SystemExit):
                main(args)

    def test_transfer_isolated_optimizer_state_and_cpu_rng(self):
        saved = self.fixture()
        candidate = on_device(saved, torch.device("cpu"))
        self.assertTrue(same_tensors(saved["optimizer"], candidate["optimizer"]))
        next(iter(candidate["optimizer"]["state"].values()))["exp_avg"].zero_()
        self.assertFalse(same_tensors(saved["optimizer"], candidate["optimizer"]))
        self.assertEqual(candidate["rng"]["torch"].device.type, "cpu")

    def test_write_error_returns_invalid_report(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.pt"
            torch.save(self.fixture(), source)
            with patch("train.pixie32.bench_checkpoint.atomic_save", side_effect=OSError("private path")), redirect_stdout(io.StringIO()):
                self.assertEqual(main(["--checkpoint", str(source), "--device", "cpu", "--out", str(root / "bench")]), 1)
            raw = (root / "bench/checkpoint-benchmark.json").read_text()
            self.assertFalse(json.loads(raw)["comparison_valid"])
            self.assertNotIn("private path", raw)


if __name__ == "__main__":
    unittest.main()
