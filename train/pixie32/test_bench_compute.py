"""Compute comparisons count completed work and reject invalid parity."""
from contextlib import redirect_stdout
import copy
from dataclasses import asdict
import io
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import torch

from train.pixie32 import bench_compute as bench
from train.pixie32 import bench_compute_suite as suite
from train.pixie32.p32_model import P32, P32Config


class ComputeBenchmarkTests(unittest.TestCase):
    def test_each_experiment_changes_one_setting(self):
        for name in bench.EXPERIMENTS:
            before, after = map(asdict, bench.variants(name))
            self.assertEqual(sum(before[key] != after[key] for key in before), 1, name)

    def test_actual_p32_parity_restores_weights_and_moments(self):
        torch.set_num_threads(2)
        torch.manual_seed(71)
        model = P32(P32Config(d_model=32, n_layer=1, n_head=4, dropout=.1))
        inputs = torch.randn(1, 18, 768), torch.randint(0, 4097, (1, 1024)), None
        inputs[1][:, ::5] = 0
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        model.eval()
        loss, _ = model.loss_fwd(*inputs[:2], chunks=16)
        loss.backward()
        optimizer.step()
        saved = {"model": copy.deepcopy(model.state_dict()), "optimizer": copy.deepcopy(optimizer.state_dict())}
        moments_before = copy.deepcopy(saved["optimizer"])
        report = bench.parity(model, saved, (bench.Compute(), bench.Compute(fused_adamw=True)),
                              torch.device("cpu"), inputs, 16, .25)
        self.assertTrue(report["valid"], report)
        self.assertEqual(len(report["comparisons"]), 1)
        for identifier, state in moments_before["state"].items():
            for key, value in state.items():
                self.assertTrue(torch.equal(value, saved["optimizer"]["state"][identifier][key]))
        a = SimpleNamespace(warmup=1, steps=2, batch=1, ce_chunks=16)
        result = bench.run_updates(model, saved, bench.Compute(), torch.device("cpu"), lambda step: inputs, a, .25)
        self.assertEqual((result["completed_updates"], result["completed_images"]), (2, 2))
        self.assertEqual(len(result["training_losses"]), 2)
        self.assertTrue(model.training)
        self.assertGreater(result["seconds"], 0)

    def test_nonfinite_or_wrong_gradients_fail_parity(self):
        for candidate in (torch.tensor([float("nan")]), torch.tensor([1.1])):
            result = bench.differences({"gradient": torch.tensor([1.])}, {"gradient": candidate}, atol=1e-5, rtol=.01)
            self.assertFalse(result["valid"])

    def test_parameter_rounding_uses_global_relative_l2_not_max_abs(self):
        reference = {"weight": torch.ones(1000), "zero_bias": torch.zeros(1)}
        candidate = {"weight": torch.ones(1000), "zero_bias": torch.tensor([3.1e-5])}
        report = bench.differences(reference, candidate, atol=0., rtol=1e-5, absolute_gate=False)
        self.assertTrue(report["valid"], report)
        self.assertGreater(report["max_abs_error"], 2e-6)
        self.assertIsNone(report["atol"])
        candidate["weight"][0] += .1
        self.assertFalse(bench.differences(reference, candidate, atol=0., rtol=1e-5, absolute_gate=False)["valid"])
        candidate["weight"][0] = float("inf")
        self.assertFalse(bench.differences(reference, candidate, atol=0., rtol=1e-5, absolute_gate=False)["valid"])

    def trajectory(self, offset=0., updates=40):
        return {"valid": True, "completed_updates": updates,
                "training_losses": [1. + step * .001 + offset for step in range(updates)],
                "evaluation_nll": [{"update": step, "nll": {"iid": 1., "probe": 1.1}}
                                   for step in range(0, updates + 1, 20)]}

    def test_multi_step_curve_rejects_drift_incomplete_and_nonfinite(self):
        reference = self.trajectory()
        self.assertTrue(bench.curve_comparison(reference, self.trajectory(1e-5), steps=40)["valid"])
        self.assertFalse(bench.curve_comparison(reference, self.trajectory(.05), steps=40)["valid"])
        self.assertFalse(bench.curve_comparison(reference, self.trajectory(updates=20), steps=40)["valid"])
        candidate = self.trajectory()
        candidate["training_losses"][0] = float("nan")
        self.assertFalse(bench.curve_comparison(reference, candidate, steps=40)["valid"])
        candidate = self.trajectory()
        candidate["evaluation_nll"][-1]["nll"]["probe"] += .01
        self.assertFalse(bench.curve_comparison(reference, candidate, steps=40)["valid"])

    def test_curve_noise_allowance_reports_actual_paired_variance(self):
        reference, candidate = self.trajectory(), self.trajectory()
        candidate["training_losses"] = [value + (.01 if step % 2 else -.01) for step, value in enumerate(candidate["training_losses"])]
        result = bench.curve_comparison(reference, candidate, steps=40)
        self.assertTrue(result["valid"])
        self.assertTrue(all(row["paired_standard_error"] > 0 for row in result["loss_blocks"]))

    def test_actual_p32_short_curve_retains_dropout_and_uses_evaluation_only(self):
        torch.set_num_threads(2)
        torch.manual_seed(71)
        model = P32(P32Config(d_model=32, n_layer=1, n_head=4, dropout=.1))
        inputs = torch.randn(1, 18, 768), torch.randint(0, 4097, (1, 1024)), None
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        saved = {"model": copy.deepcopy(model.state_dict()), "optimizer": optimizer.state_dict()}
        args = SimpleNamespace(validate_steps=2, ce_chunks=16)
        trace = []
        def evaluation():
            trace.append(model.training)
            return {"iid": inputs, "probe": inputs}
        result = bench.run_curve(model, saved, bench.Compute(), torch.device("cpu"), lambda step: inputs,
                                 evaluation, args, .25)
        self.assertEqual(result["completed_updates"], 2)
        self.assertEqual([row["update"] for row in result["evaluation_nll"]], [0, 2])
        self.assertEqual(trace, [False, False])
        self.assertTrue(model.training)

    def test_failed_trial_cannot_publish_a_gain(self):
        rows = [{"side": side, "valid": True, "images_per_second": 300. + 20 * side}
                for _ in range(5) for side in (0, 1)]
        valid = bench.summarize(rows, True, 5)
        self.assertTrue(valid["comparison_valid"])
        self.assertAlmostEqual(valid["measured_throughput_change_percent"], 20 / 300 * 100)
        rows[-1] = {"side": 1, "valid": False, "error_type": "OutOfMemoryError"}
        failed = bench.summarize(rows, True, 5)
        self.assertFalse(failed["comparison_valid"])
        self.assertNotIn("measured_throughput_change_percent", failed)
        self.assertFalse(bench.summarize(rows[:2], True, 5)["comparison_valid"])
        self.assertFalse(bench.summarize(rows[:-1], False, 5)["comparison_valid"])

    def test_overlapping_timing_ranges_are_not_a_speed_winner(self):
        rows = [{"side": side, "valid": True, "images_per_second": 300. + trial * 10 + side}
                for trial in range(5) for side in (0, 1)]
        result = bench.summarize(rows, True, 5)
        self.assertTrue(result["comparison_valid"])
        self.assertGreater(result["measured_throughput_change_percent"], 0)
        self.assertFalse(result["candidate_range_above_baseline"])

    def test_cli_alternates_five_fresh_processes_per_side(self):
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary) / "new"
            argv = ["--data", "data", "--clip-cache", "cache", "--checkpoint", "full.pt", "--clip", "clip.pt",
                    "--device", "cuda:0", "--experiment", "fused-adamw", "--out", str(out), "--steps", "1", "--warmup", "1"]
            order = []

            def run(command, **kwargs):
                worker = command[command.index("--worker") + 1]
                order.append(worker)
                path = Path(command[command.index("--result") + 1])
                value = {"valid": True} if worker == "parity" else {"valid": True, "images_per_second": 300., "completed_updates": 1}
                path.write_text(json.dumps(value))
                return SimpleNamespace(returncode=0)

            with patch.object(bench, "idle_device"), patch.object(bench.subprocess, "run", side_effect=run), redirect_stdout(io.StringIO()):
                self.assertEqual(bench.main(argv), 0)
            self.assertEqual(order, ["parity", "0", "1", "1", "0", "0", "1", "1", "0", "0", "1"])
            report = (out / "benchmark.json").read_text()
            self.assertNotIn(str(out), report)
            self.assertTrue(json.loads(report)["comparison_valid"])

    def test_cli_stops_when_parity_fails(self):
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary) / "new"
            argv = ["--data", "data", "--clip-cache", "cache", "--checkpoint", "full.pt", "--clip", "clip.pt",
                    "--device", "cuda:0", "--experiment", "sdpa-flash", "--out", str(out)]

            def run(command, **kwargs):
                Path(command[command.index("--result") + 1]).write_text('{"valid": false}')
                return SimpleNamespace(returncode=1)

            with patch.object(bench, "idle_device"), patch.object(bench.subprocess, "run", side_effect=run) as child, redirect_stdout(io.StringIO()):
                self.assertEqual(bench.main(argv), 1)
            self.assertEqual(child.call_count, 1)
            self.assertFalse(json.loads((out / "benchmark.json").read_text())["comparison_valid"])

    def test_positive_timing_result_runs_multi_step_gate(self):
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary) / "new"
            argv = ["--data", "data", "--clip-cache", "cache", "--checkpoint", "full.pt", "--clip", "clip.pt",
                    "--device", "cuda:0", "--experiment", "fused-adamw", "--out", str(out), "--validate-steps", "40"]
            workers = []
            def run(command, **kwargs):
                worker = command[command.index("--worker") + 1]
                workers.append(worker)
                if worker.startswith("curve"):
                    value = self.trajectory(.1 if worker == "curve-1" else 0.)
                elif worker == "parity":
                    value = {"valid": True}
                else:
                    value = {"valid": True, "images_per_second": 320. if worker == "1" else 300.}
                Path(command[command.index("--result") + 1]).write_text(json.dumps(value))
                return SimpleNamespace(returncode=0)
            with patch.object(bench, "idle_device"), patch.object(bench.subprocess, "run", side_effect=run), redirect_stdout(io.StringIO()):
                self.assertEqual(bench.main(argv), 1)
            self.assertEqual(workers[-2:], ["curve-0", "curve-1"])
            report = json.loads((out / "benchmark.json").read_text())
            self.assertTrue(report["comparison_valid"])
            self.assertFalse(report["adoption_gate_passed"])
            self.assertFalse(report["multi_step_parity"]["valid"])

    def test_suite_runs_six_independent_b80_five_trial_comparisons(self):
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary) / "suite"
            experiments = []
            def run(command, **kwargs):
                experiment = command[command.index("--experiment") + 1]
                experiments.append(experiment)
                self.assertEqual(command[command.index("--batch") + 1], "80")
                self.assertEqual(command[command.index("--trials") + 1], "5")
                folder = Path(command[command.index("--out") + 1])
                folder.mkdir()
                folder.joinpath("benchmark.json").write_text(json.dumps({"comparison_valid": True}))
                return SimpleNamespace(returncode=1 if experiment == "sdpa-efficient" else 0)
            with patch.object(suite.subprocess, "run", side_effect=run):
                self.assertEqual(suite.main(["--out", str(out), "--", "--data", "data"]), 1)
            self.assertEqual(tuple(experiments), suite.SUITE)
            self.assertEqual(len(json.loads((out / "suite.json").read_text())["experiments"]), 6)
            with self.assertRaises(SystemExit):
                suite.main(["--out", str(out), "--", "--data", "data"])


if __name__ == "__main__":
    unittest.main()
