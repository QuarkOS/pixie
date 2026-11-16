import base64
import hashlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from animation import pilot

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



def digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


class PilotTests(unittest.TestCase):
    def fixture(self, root: Path):
        input_root = root / "inputs"
        input_root.mkdir()
        prompts = {
            "schema": pilot.PROMPT_SCHEMA,
            "audit_set": {"id": "animation-audit-v1", "frozen": True},
            "heldout_subjects": ["silver beetle", "red fox"],
            "prompts": [
                {"subject": "silver beetle", "action": "walk", "direction": "right", "loop": True},
                {"subject": "red fox", "action": "blink", "direction": "left", "loop": False},
            ],
        }
        prompt_path = input_root / "prompts.json"
        prompt_path.write_text(json.dumps(prompts, indent=2) + "\n", encoding="utf-8")
        training_path = input_root / "training.json"
        training_path.write_text(json.dumps({"frozen": True, "subjects": ["blue slime"]}) + "\n", encoding="utf-8")
        return prompt_path, training_path

    @staticmethod
    def png_b64():
        image = Image.new("RGBA", (512, 512), (255, 0, 255, 255))
        stream = io.BytesIO()
        image.save(stream, format="PNG")
        return base64.b64encode(stream.getvalue()).decode(), stream.getvalue()

    @needs_holdouts
    def test_dry_run_writes_plan_without_network_and_preserves_prompt_bytes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            prompts, training = self.fixture(root)
            raw = prompts.read_bytes()
            manifest = pilot.run(
                prompts_path=prompts,
                training_subjects_path=training,
                output_path=root / "run",
                count=2,
                seed=100,
                model_sha256=digest("model"),
                lora_sha256=digest("lora"),
            )
            self.assertFalse(manifest["executed"])
            self.assertEqual((root / "run/prompts.json").read_bytes(), raw)
            rows = [json.loads(line) for line in (root / "run/requests.jsonl").read_text().splitlines()]
            self.assertEqual([row["requested_seed"] for row in rows], [100, 101])
            self.assertTrue(all(row["status"] == "planned" for row in rows))
            self.assertEqual(manifest["completed_count"], 0)

    @needs_holdouts
    def test_execute_sends_four_frame_request_and_records_png_hash(self):
        calls = []
        image = Image.new("RGBA", (512, 512), (255, 0, 255, 255))
        stream = io.BytesIO()
        image.save(stream, format="PNG")
        encoded = base64.b64encode(stream.getvalue()).decode()

        def transport(url, payload):
            self.assertTrue((root / "run/plan.json").is_file())
            calls.append((url, payload))
            if payload is not None:
                return 202, {"poll_url": "/sdcpp/v1/jobs/test"}
            return 200, {"status": "completed", "seed": 808, "result": {"images": [{"b64_json": encoded}]}}

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            prompts, training = self.fixture(root)
            manifest = pilot.run(
                prompts_path=prompts,
                training_subjects_path=training,
                output_path=root / "run",
                count=1,
                seed=100,
                model_sha256=digest("model"),
                lora_sha256=digest("lora"),
                server="http://127.0.0.1:17890",
                model="model-id",
                lora="lora-id",
                execute=True,
                transport=transport,
            )
            self.assertEqual(len(calls), 2)
            body = calls[0][1]
            self.assertEqual(body["batch_count"], 1)
            self.assertIn("2x2 grid, 4 frames", body["prompt"])
            self.assertEqual(body["seed"], 100)
            self.assertEqual(body["sample_params"]["custom_sigmas"], [1.0, 0.92664734, 0.82570166, 0.67798672, 0.44119975, 0.0])
            self.assertEqual(body["vae_tiling_params"], {"enabled": False})
            self.assertEqual(manifest["completed_count"], 1)
            row = json.loads((root / "run/requests.jsonl").read_text())
            self.assertEqual(row["actual_seed"], 808)
            self.assertEqual(row["seed_semantics"], "server_returned")
            self.assertEqual(row["file_sha256"], hashlib.sha256(stream.getvalue()).hexdigest())

    @needs_holdouts
    def test_interrupted_request_keeps_full_plan_without_claiming_completion(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prompts, training = self.fixture(root)

            def interrupted(url, payload):
                raise KeyboardInterrupt()

            with self.assertRaises(KeyboardInterrupt):
                pilot.run(prompts_path=prompts, training_subjects_path=training,
                          output_path=root / "run", count=2, seed=40,
                          model_sha256=digest("model"), lora_sha256=digest("lora"),
                          server="http://127.0.0.1:17890", model="model-id", lora="lora-id",
                          execute=True, transport=interrupted)
            plan = json.loads((root / "run/plan.json").read_text())
            self.assertEqual([row["requested_seed"] for row in plan["requests"]], [40, 41])
            self.assertTrue(all(row["status"] == "planned" for row in plan["requests"]))
            self.assertIn("not_verified", plan["weight_identity_verification"])
            self.assertFalse((root / "run/manifest.json").exists())

    @needs_holdouts
    def test_holdout_is_rejected_before_output_or_transport(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            prompts, training = self.fixture(root)
            value = json.loads(prompts.read_text())
            blind_phrase = json.loads(Path(pilot.DEFAULT_BLIND50).read_text(encoding="utf-8"))[0]
            value["heldout_subjects"] = [blind_phrase + "!", "red fox"]
            value["prompts"][0]["subject"] = blind_phrase + "!"
            prompts.write_text(json.dumps(value), encoding="utf-8")
            called = []
            with self.assertRaisesRegex(ValueError, "Blind-50 or Eval-50"):
                pilot.run(
                    prompts_path=prompts,
                    training_subjects_path=training,
                    output_path=root / "run",
                    count=1,
                    seed=1,
                    model_sha256=digest("model"),
                    lora_sha256=digest("lora"),
                    execute=True,
                    server="http://127.0.0.1:17890",
                    model="m",
                    lora="l",
                    transport=lambda *_: called.append(True),
                )
            self.assertFalse((root / "run").exists())
            self.assertEqual(called, [])

    @needs_holdouts
    def test_real_holdout_policy_rejects_punctuation_normalized_subject(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            input_root = root / "inputs"
            input_root.mkdir()
            prompts = input_root / "prompts.json"
            prompts.write_text(
                json.dumps(
                    {
                        "schema": pilot.PROMPT_SCHEMA,
                        "heldout_subjects": ["green, poison flask"],
                        "audit_set": {"id": "real-policy-test", "frozen": True},
                        "prompts": [{"subject": "green, poison flask", "action": "walk", "direction": "right", "loop": True}],
                    }
                ),
                encoding="utf-8",
            )
            training = input_root / "training.json"
            training.write_text(json.dumps(["blue slime"]), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Blind-50 or Eval-50"):
                pilot.run(
                    prompts_path=prompts,
                    training_subjects_path=training,
                    output_path=root / "run",
                    count=1,
                    seed=1,
                    model_sha256=digest("model"),
                    lora_sha256=digest("lora"),
                )
            self.assertFalse((root / "run").exists())

    @needs_holdouts
    def test_training_subject_overlap_and_existing_output_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            prompts, training = self.fixture(root)
            training.write_text(json.dumps(["silver beetle"]), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "overlap"):
                pilot.run(
                    prompts_path=prompts,
                    training_subjects_path=training,
                    output_path=root / "run",
                    count=1,
                    seed=1,
                    model_sha256=digest("model"),
                    lora_sha256=digest("lora"),
                )
            training.write_text(json.dumps(["blue slime"]), encoding="utf-8")
            (root / "run").mkdir()
            with self.assertRaisesRegex(OSError, "output"):
                pilot.run(
                    prompts_path=prompts,
                    training_subjects_path=training,
                    output_path=root / "run",
                    count=1,
                    seed=1,
                    model_sha256=digest("model"),
                    lora_sha256=digest("lora"),
                )

    def test_non_turbo_step_count_is_rejected_before_output(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            prompts, training = self.fixture(root)
            with self.assertRaisesRegex(ValueError, "exactly 5 Turbo steps"):
                pilot.run(
                    prompts_path=prompts,
                    training_subjects_path=training,
                    output_path=root / "run",
                    count=1,
                    seed=1,
                    model_sha256=digest("model"),
                    lora_sha256=digest("lora"),
                    steps=6,
                )
            self.assertFalse((root / "run").exists())

    @needs_holdouts
    def test_malformed_server_failure_is_recorded_without_host_or_path(self):
        def transport(url, payload):
            if payload is not None:
                return 202, {"poll_url": "/sdcpp/v1/jobs/test"}
            raise RuntimeError("server http://127.0.0.1:17890 failed at C:\\private\\secret")

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            prompts, training = self.fixture(root)
            manifest = pilot.run(
                prompts_path=prompts,
                training_subjects_path=training,
                output_path=root / "run",
                count=1,
                seed=100,
                model_sha256=digest("model"),
                lora_sha256=digest("lora"),
                server="http://127.0.0.1:17890",
                model="model-id",
                lora="lora-id",
                execute=True,
                transport=transport,
            )
            self.assertEqual(manifest["failed_count"], 1)
            record = json.loads((root / "run/requests.jsonl").read_text())
            self.assertEqual(record["status"], "failed")
            self.assertNotIn("127.0.0.1", record["error"])
            self.assertNotIn("private", record["error"])

    @needs_holdouts
    def test_external_poll_url_is_rejected_and_not_recorded(self):
        def transport(url, payload):
            if payload is not None:
                return 202, {"poll_url": "https://secret.example/jobs/test"}
            self.fail("external poll URL must not be requested")

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            prompts, training = self.fixture(root)
            manifest = pilot.run(
                prompts_path=prompts,
                training_subjects_path=training,
                output_path=root / "run",
                count=1,
                seed=100,
                model_sha256=digest("model"),
                lora_sha256=digest("lora"),
                server="http://127.0.0.1:17890",
                model="model-id",
                lora="lora-id",
                execute=True,
                transport=transport,
            )
            self.assertEqual(manifest["failed_count"], 1)
            record = json.loads((root / "run/requests.jsonl").read_text())
            self.assertNotIn("secret.example", record["error"])

    @needs_holdouts
    def test_timeout_and_malformed_completed_response_are_recorded(self):
        def pending(url, payload):
            if payload is not None:
                return 202, {"poll_url": "/sdcpp/v1/jobs/pending"}
            return 200, {"status": "running"}

        def malformed(url, payload):
            if payload is not None:
                return 202, {"poll_url": "/sdcpp/v1/jobs/malformed"}
            return 200, {"status": "completed", "result": {"images": []}}

        for transport, expected in ((pending, "TimeoutError"), (malformed, "RuntimeError")):
            with self.subTest(expected=expected), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                prompts, training = self.fixture(root)
                manifest = pilot.run(
                    prompts_path=prompts,
                    training_subjects_path=training,
                    output_path=root / "run",
                    count=1,
                    seed=100,
                    model_sha256=digest("model"),
                    lora_sha256=digest("lora"),
                    server="http://127.0.0.1:17890",
                    model="model-id",
                    lora="lora-id",
                    execute=True,
                    transport=transport,
                    timeout=0.001 if expected == "TimeoutError" else 1,
                )
                self.assertEqual(manifest["failed_count"], 1)
                record = json.loads((root / "run/requests.jsonl").read_text())
                self.assertEqual(record["error_type"], expected)

    @needs_holdouts
    def test_missing_server_seed_is_marked_not_replayable(self):
        encoded, _ = self.png_b64()

        def transport(url, payload):
            if payload is not None:
                return 202, {"poll_url": "/sdcpp/v1/jobs/no-seed"}
            return 200, {"status": "completed", "result": {"images": [{"b64_json": encoded}]}}

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            prompts, training = self.fixture(root)
            manifest = pilot.run(
                prompts_path=prompts,
                training_subjects_path=training,
                output_path=root / "run",
                count=1,
                seed=100,
                model_sha256=digest("model"),
                lora_sha256=digest("lora"),
                server="http://127.0.0.1:17890",
                model="model-id",
                lora="lora-id",
                execute=True,
                transport=transport,
            )
            self.assertEqual(manifest["completed_count"], 1)
            record = json.loads((root / "run/requests.jsonl").read_text())
            self.assertIsNone(record["actual_seed"])
            self.assertEqual(record["seed_semantics"], "requested_only")


if __name__ == "__main__":
    unittest.main()
