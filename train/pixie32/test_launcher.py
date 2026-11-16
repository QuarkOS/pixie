"""Launcher behavior with a fake GPU query and a non-training Python command."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


class LauncherTests(unittest.TestCase):
    def setUp(self):
        candidate = Path(os.environ.get("ProgramFiles", "")) / "Git/bin/bash.exe"
        self.bash = str(candidate) if os.name == "nt" and candidate.is_file() else shutil.which("bash")
        if not self.bash:
            self.skipTest("Bash is unavailable")
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.launcher = Path(__file__).with_name("run_p32.sh").resolve()
        for name, body in {
            "nvidia-smi": 'printf "%s\\n" "$*" >> "$QUERY_LOG"\n'
                          'if [[ "$*" == *query-gpu* ]]; then echo GPU-fixture; '
                          'elif [[ "${MOCK_BUSY:-0}" == 1 ]]; then echo 123; fi\n',
            "fake-python": 'printf "%s\\n" "$@" > "$CAPTURE"\n',
        }.items():
            script = self.root / name
            script.write_text("#!/usr/bin/env bash\n" + body, encoding="utf-8")
            script.chmod(0o755)
        self.env = dict(os.environ, PY=(self.root / "fake-python").as_posix(),
                        CLIP_DIR="fixture-clip", TOKENIZER="fixture-tokenizer.json", V3_CKPT="fixture-v3.pt",
                        DATASET_OUT="fixture-dataset", RUNS=(self.root / "runs").as_posix(),
                        RUN_NAME="fixture-new", GPU="1", QUERY_LOG=(self.root / "queries.txt").as_posix(),
                        CAPTURE=(self.root / "args.txt").as_posix())
        self.env["PATH"] = str(self.root) + os.pathsep + os.environ.get("PATH", "")
        self.env.pop("RESUME", None)
        self.env.pop("P32_CKPT", None)

    def launch(self, *args):
        return subprocess.run([self.bash, str(self.launcher), *args], env=self.env,
                              capture_output=True, text=True, timeout=20)

    def test_idle_selected_gpu_and_asset_forwarding(self):
        result = self.launch("--smoke", "--smoke-steps", "2")
        self.assertEqual(result.returncode, 0, result.stderr)
        queries = (self.root / "queries.txt").read_text()
        self.assertIn("-i 1 --query-gpu=uuid", queries)
        self.assertIn("-i GPU-fixture --query-compute-apps=pid", queries)
        args = (self.root / "args.txt").read_text().splitlines()
        for flag, value in (("--device", "cuda:0"), ("--v3-ckpt", "fixture-v3.pt"),
                            ("--clip-dir", "fixture-clip"), ("--tokenizer", "fixture-tokenizer.json")):
            self.assertEqual(args[args.index(flag) + 1], value)

    def test_busy_selected_gpu_refuses_before_python(self):
        self.env["MOCK_BUSY"] = "1"
        self.assertNotEqual(self.launch().returncode, 0)
        self.assertFalse((self.root / "args.txt").exists())

    def test_existing_output_and_device_override_refused(self):
        self.assertNotEqual(self.launch("--device=cuda:1").returncode, 0)
        (self.root / "runs/fixture-new").mkdir(parents=True)
        self.assertNotEqual(self.launch().returncode, 0)
        self.assertFalse((self.root / "args.txt").exists())

    def test_branch_source_forwarding_and_ambiguous_sources_refused(self):
        self.env["P32_CKPT"] = "fixture-p32.pt"
        result = self.launch()
        self.assertEqual(result.returncode, 0, result.stderr)
        args = (self.root / "args.txt").read_text().splitlines()
        self.assertEqual(args[args.index("--branch-from") + 1], "fixture-p32.pt")
        self.assertNotIn("--v3-ckpt", args)
        self.env["RESUME"] = "fixture-resume.pt"
        self.assertNotEqual(self.launch().returncode, 0)
        self.assertNotEqual(self.launch("--branch-from=other.pt").returncode, 0)

    def test_branch_smoke_wrapper_is_bounded_and_uses_guarded_launcher(self):
        self.launcher = self.launcher.with_name("smoke_p32_branch.sh")
        self.env.update(P32_CKPT="fixture-p32.pt", CLIP_CACHE="fixture-cache")
        result = self.launch("--lr", "0.0001", "--smoke-steps", "300")
        self.assertEqual(result.returncode, 0, result.stderr)
        args = (self.root / "args.txt").read_text().splitlines()
        for flag, value in (("--branch-from", "fixture-p32.pt"), ("--clip-cache", "fixture-cache"), ("--batch", "80")):
            self.assertEqual(args[args.index(flag) + 1], value)
        self.assertIn("--no-gradient-checkpointing", args)
        self.assertIn("--compile", args)
        self.assertEqual(args[-3:], ["--smoke", "--smoke-steps", "20"])
        self.assertIn("-i GPU-fixture --query-compute-apps=pid", (self.root / "queries.txt").read_text())


if __name__ == "__main__":
    unittest.main()
