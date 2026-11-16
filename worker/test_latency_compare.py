from contextlib import redirect_stdout
import copy
import io
import json
from pathlib import Path
import tempfile
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from compare_latency import PHASES, compare, main
from latency_identity import PROTOCOL, json_hash, source_hash, state_hash, workload_identity


def source(gpu="GPU reference", factor=1.):
    identity = {"protocol": PROTOCOL, "ordered_prompt_sha256": ["d" * 64], "seed_policy": "fixture",
                "batch": 64, "show": 4, "stored_candidates": 20, "trials": 5, "warmup_calls": 5,
                "checkpoint_sha256": "a" * 64, "clip_weights_sha256": "b" * 64,
                "tokenizer_sha256": "c" * 64, "palette_state_sha256": "e" * 64,
                "implementation_sha256": {"worker_v3.py": "f" * 64},
                "loaded_model_implementation_sha256": "f" * 64,
                "loaded_runtime_implementation_sha256": {"FastSampler": "f" * 64},
                "sampling_settings": {"dtype": "bfloat16", "cfg_scale": 2.}, "png_scale": 16, "png_mode": "legacy",
                "reward": {"head": "f" * 64}, "runtime": {"torch": "fixture"}, "math": {"tf32": False}}
    metric = {"median": 10. * factor, "min": 10. * factor, "max": 10. * factor}
    phase_metric = {"median": factor, "min": factor, "max": factor}
    return {"comparison_valid": True, "workload_identity": identity, "workload_sha256": json_hash(identity),
            "gpu_model": gpu, "batch": 64, "show": 4, "stored_candidates": 20, "trials": 5, "warmup_calls": 5,
            "unprofiled_ms": metric.copy(), "profiled_ms": metric.copy(), "phase_ms": {key: phase_metric.copy() for key in PHASES},
            "rows": [{"trial": i, "prompt_index": 0, "exact_outputs": True,
                      "unprofiled_ms": 10. * factor, "profiled_ms": 10. * factor,
                      "profile": {"phase_ms": {key: factor for key in PHASES}, "total_ms": 10. * factor,
                                  "unclassified_ms": 4. * factor}} for i in range(5)]}


class ComparisonTests(unittest.TestCase):
    def test_matching_evidence_reports_separate_ratios(self):
        value = compare(source(), source("GPU candidate", 2.))
        self.assertTrue(value["comparison_valid"])
        self.assertEqual(value["clean_total"]["candidate_over_reference_median"], 2.)
        self.assertEqual(value["phase_ratios"]["sample"]["candidate_over_reference_median"], 2.)
        self.assertNotIn("sum_phase_ms", value)

    def test_missing_or_different_identity_is_inconclusive(self):
        for field in ("workload_identity", "workload_sha256", "rows", "gpu_model"):
            value = source()
            del value[field]
            result = compare(source(), value)
            self.assertFalse(result["comparison_valid"])
            self.assertNotIn("clean_total", result)
        for field, replacement in (("sampling_settings", {"dtype": "float32"}), ("png_mode", "fast"),
                                   ("runtime", {"torch": "different"}), ("ordered_prompt_sha256", ["z" * 64])):
            value = source()
            value["workload_identity"][field] = replacement
            value["workload_sha256"] = json_hash(value["workload_identity"])
            self.assertFalse(compare(source(), value)["comparison_valid"])

    def test_nonfinite_incomplete_or_malformed_evidence_is_invalid(self):
        for field in ("unprofiled_ms", "profiled_ms"):
            for bad in (float("nan"), float("inf"), -1., True):
                value = source()
                value[field]["median"] = bad
                self.assertFalse(compare(source(), value)["comparison_valid"])
        value = source()
        value["rows"].pop()
        self.assertFalse(compare(source(), value)["comparison_valid"])
        value = source()
        value["rows"][1] = copy.deepcopy(value["rows"][0])
        self.assertFalse(compare(source(), value)["comparison_valid"])
        self.assertFalse(compare([], source())["comparison_valid"])

    def test_tampered_aggregate_or_missing_phase_is_invalid(self):
        value = source()
        value["unprofiled_ms"]["median"] = 10.5
        value["unprofiled_ms"]["max"] = 11.
        self.assertFalse(compare(source(), value)["comparison_valid"])
        value = source()
        del value["phase_ms"]["png"]
        self.assertFalse(compare(source(), value)["comparison_valid"])
        value = source()
        value["rows"][0]["profile"]["total_ms"] = 20.
        value["rows"][0]["profile"]["unclassified_ms"] = 14.
        self.assertFalse(compare(source(), value)["comparison_valid"])

    def test_cli_reads_without_mutation_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            left, right = root / "left.json", root / "right.json"
            left.write_text(json.dumps(source()))
            right.write_text(json.dumps(source("GPU candidate", 2.)))
            before = left.read_bytes(), right.read_bytes()
            args = ["--reference", str(left), "--candidate", str(right), "--out", str(root / "out")]
            with redirect_stdout(io.StringIO()):
                self.assertEqual(main(args), 0)
            self.assertEqual(before, (left.read_bytes(), right.read_bytes()))
            result = (root / "out/latency-comparison.json").read_text()
            self.assertNotIn(str(root), result)
            self.assertTrue(json.loads(result)["comparison_valid"])
            with self.assertRaises(SystemExit):
                main(args)


class RewardFixture(SimpleNamespace):
    pass


class IdentityTests(unittest.TestCase):
    def test_source_hash_normalizes_only_python_line_endings(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            a, b = root / "a.py", root / "b.py"
            a.write_bytes(b"x = 1\n")
            b.write_bytes(b"x = 1\r\n")
            self.assertEqual(source_hash(a), source_hash(b))
            b.write_bytes(b"x = 2\r\n")
            self.assertNotEqual(source_hash(a), source_hash(b))
            raw = root / "asset.bin"
            raw.write_bytes(b"x = 1\r\n")
            self.assertNotEqual(source_hash(a), source_hash(raw))

    def test_idle_helper_import_does_not_preload_configurable_model(self):
        script = "import sys; sys.path.insert(0, sys.argv[1]); import benchmark_blank_cache; import benchmark_worker; assert 'v3_model' not in sys.modules; assert 'fast_sample' not in sys.modules"
        result = subprocess.run([sys.executable, "-c", script, str(Path(__file__).parent)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_state_hash_tracks_bytes_dtype_shape_and_name(self):
        initial = state_hash({"weight": torch.arange(4, dtype=torch.bfloat16)})
        self.assertEqual(initial, state_hash({"weight": torch.arange(4, dtype=torch.bfloat16)}))
        for changed in ({"weight": torch.arange(4, dtype=torch.float32)},
                        {"weight": torch.arange(4, dtype=torch.bfloat16).reshape(2, 2)},
                        {"other": torch.arange(4, dtype=torch.bfloat16)},
                        {"weight": torch.arange(4, dtype=torch.bfloat16) + 1}):
            self.assertNotEqual(initial, state_hash(changed))

    def test_actual_identity_tracks_prompts_settings_palette_and_reward_weights(self):
        model = torch.nn.Linear(2, 2).eval()
        scorer = RewardFixture(mu=np.zeros(1), sd=np.ones(1), w=np.ones(2), pca={}, recipe={"hand": True},
            models={"fixture": {"model": model, "size": 16, "mean": [0., 0., 0.], "std": [1., 1., 1.],
                                "tokenizer": lambda texts: torch.tensor([[len(text)] for text in texts])}})
        module = SimpleNamespace(torch=torch, CFG={"clip_pt": "unused", "clip_tokenizer": "unused"}, SCALE=16,
                                 sha256=lambda value: "f" * 64, M=SimpleNamespace(__file__=__file__),
                                 FastSampler=RewardFixture, FastRewardV7=RewardFixture, BlankFeatures=RewardFixture,
                                 PhaseTimer=RewardFixture)
        worker = SimpleNamespace(v7=SimpleNamespace(v7=scorer, size=16), palette=np.zeros((4, 3), dtype=np.uint8),
                                 scorer_name="reward_v7", info={"sha": "e" * 64},
                                 last_sampling={"job_seed": 101, "dtype": "bfloat16"})
        def identity(prompts=["fixture"]):
            with patch.dict("sys.modules", {"common": SimpleNamespace(__file__=__file__)}):
                return workload_identity(module, worker, prompts, batch=64, show=4, candidates=20, trials=5, warmup=5)
        original = identity()
        self.assertNotIn("job_seed", original["sampling_settings"])
        self.assertNotIn("fixture", original["ordered_prompt_sha256"])
        self.assertNotEqual(original, identity(["changed"]))
        with torch.no_grad():
            model.weight.add_(1)
        self.assertNotEqual(original["reward"]["models"], identity()["reward"]["models"])
        worker.palette[0, 0] = 1
        self.assertNotEqual(original["palette_state_sha256"], identity()["palette_state_sha256"])


if __name__ == "__main__":
    unittest.main()
