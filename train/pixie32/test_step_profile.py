"""The profile counts real completed updates and preserves training state."""
from contextlib import nullcontext
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from train.pixie32.step_profile import PHASES, StepProfile
from train.pixie32 import test_reporting
from train.pixie32 import profile_steps


class StepProfileTests(unittest.TestCase):
    def test_cpu_bounds_and_completed_work(self):
        profile = StepProfile(torch.device("cpu"), 2, 1)
        for step in range(5):
            profile.begin(step, 3)
            for _ in PHASES:
                profile.mark()
            profile.finish()
        report = profile.report()
        self.assertEqual((report["completed_profiled_updates"], report["completed_profiled_images"]), (2, 6))
        self.assertEqual([row["segment_step"] for row in report["rows"]], [1, 2])
        for row in report["rows"]:
            self.assertIsNone(row["cuda_stream_phase_ms"])
            self.assertEqual(set(row["host_phase_seconds"]), set(PHASES))
            self.assertLessEqual(sum(row["host_phase_seconds"].values()), row["completed_update_seconds"])

    def test_cuda_events_use_selected_stream_without_synchronizing(self):
        events = []
        class Event:
            def __init__(self, **kwargs):
                self.index = len(events)
                events.append(self)
            def record(self, stream):
                self.stream = stream
            def elapsed_time(self, other):
                return (other.index - self.index) * 2.0
        device = torch.device("cuda:3")
        with patch("torch.cuda.device", return_value=nullcontext()) as selected, \
             patch("torch.cuda.current_stream", return_value="selected-stream") as stream, \
             patch("torch.cuda.Event", Event), patch("torch.cuda.synchronize") as sync:
            profile = StepProfile(device, 1, 0)
            profile.begin(0, 80)
            for _ in PHASES:
                profile.mark()
            profile.finish()
        self.assertEqual(profile.report()["rows"][0]["cuda_stream_phase_ms"], dict.fromkeys(PHASES, 2.0))
        self.assertEqual(len(events), len(PHASES) + 1)
        self.assertTrue(all(event.stream == "selected-stream" for event in events))
        stream.assert_called_with(device)
        selected.assert_called_with(device)
        sync.assert_not_called()

    def test_real_trainer_profile_preserves_model_optimizer_rng_and_fingerprint(self):
        fixture = test_reporting.ReportingTests()
        fixture.setUp()
        try:
            from train.pixie32 import train_p32
            with patch.object(train_p32, "sample_grid"):
                source = ["--v3-ckpt", str(fixture.root / "v3.pt"), "--smoke", "--smoke-steps", "3"]
                fixture.run_training("baseline", source)
                fixture.run_training("profiled", source + ["--profile-steps", "1", "--profile-warmup", "1"])
            def load(name):
                return torch.load(fixture.root / name / "last.pt", weights_only=False)
            baseline, actual = load("baseline"), load("profiled")
            self.assertEqual(baseline["fingerprint"], actual["fingerprint"])
            for key in baseline["model"]:
                self.assertTrue(torch.equal(baseline["model"][key], actual["model"][key]))
            for number, state in baseline["optimizer"]["state"].items():
                for key, value in state.items():
                    self.assertTrue(torch.equal(value, actual["optimizer"]["state"][number][key]))
            self.assertTrue(torch.equal(baseline["rng"]["torch"], actual["rng"]["torch"]))
            report = json.loads((fixture.root / "profiled" / "step-profile.json").read_text())
            self.assertEqual((report["completed_profiled_updates"], report["completed_profiled_images"]), (1, 4))
            self.assertFalse((fixture.root / "baseline" / "step-profile.json").exists())
            self.assertNotIn(str(fixture.root), json.dumps(report))
        finally:
            fixture.tearDown()

    def test_wrapper_requires_checkpoint_and_forwards_bounded_profile(self):
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary) / "new"
            argv = ["profile", "--out", str(out), "--steps", "2", "--warmup", "1", "--",
                    "--data", "dataset", "--device", "cpu", "--resume", "full.pt"]
            with patch("sys.argv", argv), patch.object(profile_steps, "idle_device") as idle, \
                 patch.object(profile_steps.train_p32, "run") as run:
                profile_steps.main()
            config = run.call_args.args[0]
            self.assertEqual((config.profile_steps, config.profile_warmup, config.smoke_steps), (2, 1, 3))
            idle.assert_called_once_with("cpu")
            with patch("sys.argv", argv[:-2]), patch("sys.stderr", io.StringIO()), self.assertRaises(SystemExit):
                profile_steps.main()


if __name__ == "__main__":
    unittest.main()
