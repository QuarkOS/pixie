"""JSON-equivalent holdouts are not byte-identical cache inputs."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest

from train.pixie32.check_holdouts import main
from train.pixie32.data import Holdouts, verify_holdout_bytes


class HoldoutByteTests(unittest.TestCase):
    def test_original_bytes_pass_but_reformatting_and_newlines_fail_without_writes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            blind, evaluation = root / "blind.json", root / "eval.json"
            values = [f"frozen subject {i}" for i in range(50)]
            original = (json.dumps(values, indent=1) + "\r\n").encode("utf-8")
            blind.write_bytes(original)
            evaluation.write_bytes(json.dumps([f"evaluation subject {i}" for i in range(50)]).encode())
            manifest = {"schema": "pixie32-v1", "holdouts": Holdouts(blind, evaluation).hashes}
            manifest_file = root / "manifest.json"
            manifest_file.write_bytes(json.dumps(manifest).encode())
            args = ["--data", str(root), "--blind50", str(blind), "--eval50", str(evaluation)]
            frozen_manifest = manifest_file.read_bytes()
            output = io.StringIO()
            with redirect_stdout(output):
                self.assertEqual(main(args), 0)
            self.assertTrue(json.loads(output.getvalue())["byte_identity"])
            for replacement in (json.dumps(values, indent=2).encode(), original.replace(b"\r\n", b"\n")):
                blind.write_bytes(replacement)
                with self.assertRaisesRegex(ValueError, "original Blind-50"):
                    verify_holdout_bytes(manifest, Holdouts(blind, evaluation))
                output = io.StringIO()
                with redirect_stdout(output):
                    self.assertEqual(main(args), 1)
                self.assertNotIn(str(root), output.getvalue())
                self.assertEqual(blind.read_bytes(), replacement)
                self.assertEqual(manifest_file.read_bytes(), frozen_manifest)
            blind.write_bytes(original)
            with redirect_stdout(io.StringIO()):
                self.assertEqual(main(args), 0)

    def test_missing_inputs_fail_without_disclosing_paths(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = ["--data", temporary, "--blind50", temporary, "--eval50", temporary]
            output = io.StringIO()
            with redirect_stdout(output):
                self.assertEqual(main(args), 1)
            self.assertNotIn(temporary, output.getvalue())

    def test_malformed_manifest_structures_fail_without_tracebacks(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in ("blind.json", "eval.json"):
                (root / name).write_text(json.dumps([f"reserved {name} {i}" for i in range(50)]))
            args = ["--data", temporary, "--blind50", str(root / "blind.json"), "--eval50", str(root / "eval.json")]
            for malformed in ([], {"schema": "pixie32-v1", "holdouts": []}):
                (root / "manifest.json").write_text(json.dumps(malformed))
                output = io.StringIO()
                with redirect_stdout(output):
                    self.assertEqual(main(args), 1)
                self.assertFalse(json.loads(output.getvalue())["byte_identity"])


if __name__ == "__main__":
    unittest.main()
