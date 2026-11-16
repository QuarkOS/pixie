import unittest
import ast
from contextlib import redirect_stdout
import io
from types import SimpleNamespace
import json
import os
from pathlib import Path
import runpy
import tempfile
from unittest.mock import patch
from phase_timing import PhaseTimer
import time
from benchmark_worker import benchmark, main as benchmark_main
from sampling_provenance import candidate_budget


class TimerTests(unittest.TestCase):
    def test_exclusive_phases_and_unclassified_time(self):
        ticks = iter([0., 1., 3., 4., 7., 10.])
        synchronized = []
        timer = PhaseTimer(True, lambda: synchronized.append(1), lambda: next(ticks))
        with timer.phase("clip"):
            pass
        with timer.phase("sample"):
            pass
        result = timer.finish()
        self.assertEqual(result["phase_ms"], {"clip": 2000., "sample": 3000.})
        self.assertEqual(result["unclassified_ms"], 5000.)
        self.assertEqual(result["total_ms"], 10000.)
        self.assertEqual(len(synchronized), 6)

    def test_default_does_not_synchronize_or_read_clock(self):
        def forbidden():
            raise AssertionError("disabled timer did work")
        timer = PhaseTimer(False, forbidden, forbidden)
        with timer.phase("clip"):
            pass
        self.assertIsNone(timer.finish())

    def test_repeated_phases_add_and_nesting_fails(self):
        ticks = iter(range(20))
        timer = PhaseTimer(True, clock=lambda: next(ticks))
        for _ in range(2):
            with timer.phase("clip"):
                pass
        with timer.phase("sample"):
            with self.assertRaises(RuntimeError):
                with timer.phase("clip"):
                    pass
        self.assertEqual(timer.finish()["phase_ms"]["clip"], 2000.)


class WorkerBenchmarkTests(unittest.TestCase):
    def test_checkpoint_validation_exit_writes_invalid_cli_result(self):
        with tempfile.TemporaryDirectory() as temporary, patch.dict(os.environ):
            root = Path(temporary)
            config = root / "worker.json"
            config.write_text(json.dumps({"device": "cuda"}))
            def failed_load():
                raise SystemExit("private diagnostic must not be reported")
            module = SimpleNamespace(Gen=failed_load)
            with patch("benchmark_worker.require_idle_gpu"), patch("torch.cuda.set_device"), \
                 patch("benchmark_worker.importlib.import_module", return_value=module), redirect_stdout(io.StringIO()):
                code = benchmark_main(["--config", str(config), "--device", "cuda:0", "--out", str(root / "new")])
            self.assertEqual(code, 1)
            raw = (root / "new" / "worker-benchmark.json").read_text()
            self.assertFalse(json.loads(raw)["comparison_valid"])
            self.assertEqual(json.loads(raw)["error_type"], "SystemExit")
            self.assertNotIn("private diagnostic", raw)

    def test_actual_queue_loop_profiles_ack_without_exposing_payload(self):
        source = ast.parse((Path(__file__).parent / "worker_v3.py").read_text(encoding="utf-8"))
        main = next(node for node in source.body if isinstance(node, ast.FunctionDef) and node.name == "main")
        generator = SimpleNamespace(info={}, device="cpu", scorer_name="feat_linear", last_candidates=[],
                                    last_sampling={}, last_profile={"phase_ms": {"png": 1.}})
        generator.run = lambda *args, **kwargs: ([], [], 1., 1., "feat_linear")
        captured = []
        posted = []
        def post(path, body, timeout):
            if path.endswith("next"):
                if posted:
                    raise KeyboardInterrupt
                return {"job": {"id": "fixture", "prompt": "fixture", "n": 3}}
            posted.append(body)
            return {}
        fake_socket = SimpleNamespace(socket=lambda *args: SimpleNamespace(bind=lambda address: None),
                                      AF_INET=0, SOCK_STREAM=0, gethostname=lambda: "")
        env = {"_load_worker_key": lambda: "offline-fixture", "socket": fake_socket, "CFG": {},
               "Gen": lambda: generator, "log": lambda *args: captured.append(args), "URLS": [], "HOLD": None,
               "USE_GRAPH": False, "N_SAMPLES_V7": 64, "N_SHOW_DEFAULT": 4, "PROFILE_PHASES": True,
               "time": time, "post": post, "candidate_budget": candidate_budget, "json": json}
        exec(compile(ast.Module(body=[main], type_ignores=[]), "worker_v3.py", "exec"), env)
        with self.assertRaises(KeyboardInterrupt):
            env["main"]()
        profile = json.loads(next(row[1] for row in captured if row[0] == "WORKER_PROFILE"))
        self.assertTrue(profile["result_acknowledged"])
        self.assertEqual(profile["post_attempts"], 1)
        self.assertGreaterEqual(profile["received_to_result_ack_ms"], 0.)
        for private_key in ("id", "prompt", "host", "key"):
            self.assertNotIn(private_key, profile)

    def test_offline_import_needs_no_key_and_cannot_post(self):
        with tempfile.TemporaryDirectory() as temporary:
            config = Path(temporary) / "worker.json"
            config.write_text(json.dumps({"app_urls": [], "ckpt": "checkpoint.pt", "v3_code": ".", "v7_root": "."}))
            with patch.dict(os.environ, {"PIXEL_WORKER_CONFIG": str(config)}), patch("urllib.request.urlopen") as request:
                module = runpy.run_path(str(Path(__file__).parent / "worker_v3.py"))
                self.assertIsNone(module["KEY"])
                with self.assertRaises(RuntimeError):
                    module["post"]("/unused", {}, 1)
                request.assert_not_called()

    def make_worker(self, drift=False, fallback=False):
        cuda = SimpleNamespace(synchronize=lambda device: None, reset_peak_memory_stats=lambda device: None,
                               max_memory_allocated=lambda device: 100, max_memory_reserved=lambda device: 200)
        module = SimpleNamespace(torch=SimpleNamespace(cuda=cuda), SCORER_CFG="reward_v7",
                                 CACHE_BLANK=True, USE_GRAPH=True)
        worker = SimpleNamespace(device="cuda:0", calls=[])
        def run(prompt, n, show, ncand, seed=None):
            worker.calls.append((worker.profile_phases, seed, n, show, ncand))
            row = {"png": "changed" if drift and worker.profile_phases else "unchanged", "seed": seed, "score": 1.}
            worker.last_sampling = {"job_seed": seed, "batch_size": n}
            worker.last_candidates = [row.copy()]
            worker.last_profile = {"phase_ms": {"clip": 1.}, "total_ms": 2.} if worker.profile_phases else None
            return [row.copy()], [{"seed": seed, "score": 1.}], 2., 1., "feat_linear" if fallback else "reward_v7"
        worker.run = run
        return module, worker

    def test_five_alternating_pairs_keep_actual_budget_and_seed(self):
        module, worker = self.make_worker()
        result = benchmark(module, worker, prompts=["fixture"], trials=5, warmup=2,
                           batch=64, show=4, candidates=20)
        self.assertTrue(result["comparison_valid"])
        self.assertEqual(len(result["rows"]), 5)
        self.assertEqual(len(worker.calls), 12)
        for trial in range(5):
            calls = worker.calls[2 + trial * 2:4 + trial * 2]
            self.assertEqual([x[0] for x in calls], [False, True] if trial % 2 == 0 else [True, False])
            self.assertEqual(calls[0][1:], calls[1][1:])
            self.assertEqual(calls[0][2:], (64, 4, 20))

    def test_output_drift_is_invalid(self):
        module, worker = self.make_worker(drift=True)
        result = benchmark(module, worker, prompts=["fixture"], trials=5, warmup=2,
                           batch=4, show=1, candidates=2)
        self.assertFalse(result["comparison_valid"])

    def test_scorer_fallback_cannot_be_measured_as_configured_worker(self):
        module, worker = self.make_worker(fallback=True)
        with self.assertRaises(ValueError):
            benchmark(module, worker, prompts=["fixture"], trials=5, warmup=2,
                      batch=4, show=1, candidates=2)


if __name__ == "__main__":
    unittest.main()
