import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from train import sample_pixie as sample_p32
from train.v3_model import V3Config, build
from train.pixie32.p32_model import P32, P32Config

try:
    from safetensors.torch import save_file
except ImportError:  # pragma: no cover
    save_file = None


@unittest.skipIf(save_file is None, "safetensors is not installed")
class SampleP32Tests(unittest.TestCase):
    def test_release_folder_round_trips_and_samples_rgba(self):
        config = P32Config(d_model=32, n_layer=1, n_head=4, dropout=0.0)
        model = P32(config)
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            save_file({k: v.contiguous() for k, v in model.state_dict().items()}, str(folder / "model.safetensors"))
            (folder / "config.json").write_text(json.dumps({**config.to_dict(), "palette_file": "palette.npy",
                                                            "image_size": 32}))
            palette = np.random.default_rng(0).integers(0, 256, (4097, 3), dtype=np.uint8)
            np.save(folder / "palette.npy", palette)
            loaded, loaded_palette, side = sample_p32.load_model(folder, torch.device("cpu"))
        for key, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, loaded.state_dict()[key]), key)
        np.testing.assert_array_equal(loaded_palette, palette)
        self.assertEqual(side, 32)
        features = torch.randn(2, 18, 768)
        indices = loaded.sample(features, torch.randn(2, 18, 768), cfg=2.0,
                                generator=torch.Generator().manual_seed(0)).numpy().reshape(-1, 32, 32)
        rgba = sample_p32.to_rgba(indices[0], palette)
        self.assertEqual(rgba.shape, (32, 32, 4))
        self.assertTrue(set(np.unique(rgba[..., 3])) <= {0, 255})
        np.testing.assert_array_equal(rgba[..., 3] == 0, indices[0] == 0)

    def test_v3_release_folder_round_trips_and_samples_16(self):
        config = V3Config(arch="p50", d_model=32, n_layer=1, n_head=4, dropout=0.0, text_len=18, n_pal=3, style=True)
        model = build(config)
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            save_file({k: v.contiguous() for k, v in model.state_dict().items()}, str(folder / "model.safetensors"))
            (folder / "config.json").write_text(json.dumps({**config.to_dict(), "image_size": 16}))
            np.save(folder / "palette.npy", np.zeros((4097, 3), dtype=np.uint8))
            loaded, _, side = sample_p32.load_model(folder, torch.device("cpu"))
        self.assertEqual(side, 16)
        for key, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, loaded.state_dict()[key]), key)
        out = loaded.sample(torch.randn(1, 18, 768), torch.randn(1, 18, 768), cfg=2.0,
                            generator=torch.Generator().manual_seed(0))
        self.assertEqual(tuple(out.shape), (1, 256))


if __name__ == "__main__":
    unittest.main()
