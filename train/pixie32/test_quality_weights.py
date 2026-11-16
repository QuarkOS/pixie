from argparse import Namespace
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from train.pixie32 import quality_weights as quality
from train.pixie32.build_data_p32 import build
from train.pixie32.data import Holdouts, load_dataset, sha256
from train.pixie32 import test_data as fixtures


class QualityWeightTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.DataTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        f = self.fixture
        self.args = Namespace(data_root=f.source, r1=f.args.r1, blind50=f.args.blind50,
                              eval50=f.args.eval50, out=f.root / "quality", simple_weight=.25, expect_rows=48)

    def test_plain_rectangle_flags_but_texture_and_icon_do_not(self):
        square = np.zeros((32, 32, 4), np.uint8)
        square[5:27, 5:27] = [80, 160, 90, 255]
        self.assertTrue(quality.simple_tile(square)["simple_tile"])
        outlined = square.copy()
        outlined[5:7, 5:27, :3] = 0
        outlined[25:27, 5:27, :3] = 0
        outlined[5:27, 5:7, :3] = 0
        outlined[5:27, 25:27, :3] = 0
        self.assertTrue(quality.simple_tile(outlined)["simple_tile"])
        checker = square.copy()
        checker[5:27:2, 5:27, :3] = [40, 70, 80]
        self.assertFalse(quality.simple_tile(checker)["simple_tile"])
        icon = square.copy()
        icon[5:16, 5:16, 3] = 0
        self.assertFalse(quality.simple_tile(icon)["simple_tile"])
        self.assertFalse(quality.simple_tile(np.zeros_like(square))["simple_tile"])

    def test_weights_are_consumed_and_splits_unchanged(self):
        f = self.fixture
        # Leave one geometrically complex row to make relative sampling change observable.
        image_path = f.source / f.paths[0]
        with Image.open(image_path) as image:
            pixels = np.array(image)
        pixels[8:16, 8:16, 3] = 0
        Image.fromarray(pixels).save(image_path)
        quality.prepare(self.args)
        baseline_out = f.root / "baseline"
        f.args.out = baseline_out
        build(f.args)
        baseline = load_dataset(baseline_out, f.args.blind50, f.args.eval50)
        f.args.out = f.root / "weighted"
        f.args.sample_weights = self.args.out / "sample_weights.json"
        manifest = build(f.args)
        weighted = load_dataset(f.args.out, f.args.blind50, f.args.eval50)
        self.assertEqual(baseline.ids, weighted.ids)
        np.testing.assert_array_equal(baseline.arrays["split"], weighted.arrays["split"])
        ratio_before = baseline.arrays["weight"][0] / baseline.arrays["weight"][1]
        ratio_after = weighted.arrays["weight"][0] / weighted.arrays["weight"][1]
        self.assertAlmostEqual(float(ratio_after / ratio_before), 4.)
        self.assertEqual(manifest["inputs"]["sample_weights"], sha256(f.args.sample_weights))

    def test_hash_boundaries_reject_changed_image_caption_or_holdout_bytes(self):
        f = self.fixture
        quality.prepare(self.args)
        path = self.args.out / "sample_weights.json"
        original = json.loads(path.read_text())
        hashes = Holdouts(f.args.blind50, f.args.eval50).hashes
        for field in ("image_sha256", "caption_sha256"):
            changed = json.loads(json.dumps(original))
            changed["rows"][0][field] = "0" * 64
            path.write_text(json.dumps(changed))
            with self.assertRaises(ValueError):
                quality.load_sample_weights(path, f.source, f.args.r1, hashes)
        path.write_text(json.dumps(original))
        with self.assertRaises(ValueError):
            quality.load_sample_weights(path, f.source, f.args.r1, {"blind50": "changed"})

    def test_invalid_multiplier_and_existing_output_fail(self):
        for invalid in (0., -1., float("nan"), 2.):
            self.args.simple_weight = invalid
            with self.assertRaises(ValueError):
                quality.prepare(self.args)
        self.args.simple_weight = .25
        result = quality.prepare(self.args)
        self.assertEqual(result["simple_tiles"], 48)
        with self.assertRaises(ValueError):
            quality.prepare(self.args)

    def test_actual_cli_output_has_no_private_root_and_original_files_unchanged(self):
        f = self.fixture
        files = (f.args.blind50, f.args.eval50, f.args.r1, f.source / "manifest.jsonl")
        before = [path.read_bytes() for path in files]
        arguments = ["--data-root", str(f.source), "--r1", str(f.args.r1), "--blind50", str(f.args.blind50),
                     "--eval50", str(f.args.eval50), "--out", str(self.args.out), "--expect-rows", "48"]
        with redirect_stdout(io.StringIO()) as output:
            self.assertEqual(quality.main(arguments), 0)
        self.assertNotIn(str(f.root), output.getvalue())
        self.assertEqual(before, [path.read_bytes() for path in files])


if __name__ == "__main__":
    unittest.main()
