import base64
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from benchmark_png import compare_signatures, main, run_pairs
from png_encoding import encode_png, pixels


class PngTests(unittest.TestCase):
    def fixture(self):
        rgba = np.zeros((16, 16, 4), dtype=np.uint8)
        rgba[2:14, 4:12] = (20, 180, 100, 255)
        rgba[6:8, 2:14] = (200, 50, 40, 255)
        return rgba

    def test_legacy_bytes_and_fast_pixels_match(self):
        for sprite in (self.fixture(), np.zeros((16, 16, 4), dtype=np.uint8),
                       np.random.default_rng(17).integers(0, 256, (16, 16, 4), dtype=np.uint8)):
            for scale in (1, 16):
                buffer = io.BytesIO()
                Image.fromarray(sprite).resize((16 * scale, 16 * scale), Image.Resampling.NEAREST).save(buffer, "PNG", optimize=True)
                legacy = encode_png(sprite, scale, "legacy")
                self.assertEqual(base64.b64decode(legacy.split(",", 1)[1]), buffer.getvalue())
                self.assertTrue(np.array_equal(np.asarray(pixels(legacy)), np.asarray(pixels(encode_png(sprite, scale, "fast")))))

    def test_output_comparison_rejects_pixels_or_metadata_drift(self):
        legacy = {"results": [{"png": encode_png(self.fixture(), 1), "score": .8}], "sampling": {"seed": 17}}
        fast = {"results": [{"png": encode_png(self.fixture(), 1, "fast"), "score": .8}], "sampling": {"seed": 17}}
        self.assertTrue(compare_signatures(legacy, fast))
        fast["results"][0]["score"] = .7
        self.assertFalse(compare_signatures(legacy, fast))
        fast["results"][0]["score"] = .8
        fast["results"][0]["png"] = encode_png(np.zeros((16, 16, 4), dtype=np.uint8), 1)
        self.assertFalse(compare_signatures(legacy, fast))

    def test_pairs_preserve_order_seed_and_report_bytes(self):
        calls = []
        def call(mode, prompt, seed):
            calls.append((mode, seed))
            return 1., [{"png": encode_png(self.fixture(), 1, mode)}]
        report = run_pairs(call, 5, ["fixture"])
        self.assertTrue(report["comparison_valid"])
        for trial in range(5):
            a, b = calls[trial * 2:trial * 2 + 2]
            self.assertEqual(a[1], b[1])
            self.assertEqual((a[0], b[0]), ("legacy", "fast") if trial % 2 == 0 else ("fast", "legacy"))
        self.assertTrue(all(row["fast_payload_bytes"] > 0 for row in report["rows"]))

    def test_cpu_cli_new_output_and_pixels_verified(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sprites = root / "sprites"
            sprites.mkdir()
            Image.fromarray(self.fixture()).save(sprites / "fixture.png")
            before = (sprites / "fixture.png").read_bytes()
            args = ["--sprites", str(sprites), "--count", "1", "--out", str(root / "bench")]
            with redirect_stdout(io.StringIO()):
                self.assertEqual(main(args), 0)
            report = json.loads((root / "bench/png-benchmark.json").read_text())
            self.assertTrue(report["comparison_valid"])
            self.assertEqual(len(report["rows"]), 5)
            self.assertEqual((sprites / "fixture.png").read_bytes(), before)
            self.assertNotIn(str(root), json.dumps(report))
            with self.assertRaises(SystemExit):
                main(args)

    def test_invalid_mode_is_not_silent(self):
        with self.assertRaises(ValueError):
            encode_png(self.fixture(), 16, "turbo")


if __name__ == "__main__":
    unittest.main()
