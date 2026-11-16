"""CPU behavioral checks for the native 32x32 model."""
import tempfile
import unittest
from pathlib import Path

import torch

from train import v3_model as M
from train.pixie32.p32_model import P32, P32Config, apply_2d_rope, from_patches, to_patches


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(3)
        self.config = P32Config(d_model=32, n_layer=2, n_head=4, dropout=0)
        self.model = P32(self.config).eval()
        self.text = torch.randn(1, 18, 768)
        self.image = torch.randint(0, M.PALETTE, (1, 1024))

    def test_patchify_roundtrip_and_slot_order(self):
        image = torch.arange(1024).reshape(1, -1)
        patches = to_patches(image)
        self.assertEqual(patches.shape, (1, 256, 4))
        self.assertEqual(patches[0, 0].tolist(), [0, 1, 32, 33])
        self.assertTrue(torch.equal(from_patches(patches), image))

    def test_trunk_causality_and_truncated_decode(self):
        patches = to_patches(self.image)
        changed = patches.clone()
        changed[:, 37:] = (changed[:, 37:] + 1) % M.PALETTE
        with torch.no_grad():
            before = self.model.trunk(self.text, patches)
            after = self.model.trunk(self.text, changed)
            truncated = self.model.trunk(self.text, patches, n_patches=38)
        torch.testing.assert_close(before[:, :38], after[:, :38], atol=0, rtol=0)
        torch.testing.assert_close(before[:, :38], truncated, atol=2e-6, rtol=2e-6)
        self.assertFalse(torch.allclose(before[:, 38:], after[:, 38:]))

    def test_slot_head_causality(self):
        hidden = torch.randn(1, 1, 32)
        patches = torch.tensor([[[1, 2, 3, 4]]])
        with torch.no_grad():
            before = self.model.head(hidden, patches)
            for slot in range(4):
                changed = patches.clone()
                changed[..., slot:] += 10
                after = self.model.head(hidden, changed)
                torch.testing.assert_close(before[..., :slot + 1, :], after[..., :slot + 1, :], atol=0, rtol=0)
            for slot in range(4):
                step = self.model.head.step(hidden[:, 0], [patches[:, 0, j] for j in range(slot)])
                torch.testing.assert_close(before[:, 0, slot], step)

    def test_rope_prefix_identity_and_target_coordinates(self):
        values = torch.randn(1, 4, 279, 8)
        rotated = apply_2d_rope(values, self.model.coordinates, 4)
        self.assertTrue(torch.equal(rotated[..., :24, :], values[..., :24, :]))
        self.assertTrue(torch.equal(rotated[..., 4:], values[..., 4:]))
        self.assertFalse(torch.equal(rotated[..., 24:, :4], values[..., 24:, :4]))
        self.assertEqual(self.model.coordinates[23].tolist(), [0, 0])
        self.assertEqual(self.model.coordinates[24].tolist(), [0, 1])
        self.assertEqual(self.model.coordinates[39].tolist(), [1, 0])

    def test_warmstart_remaps_only_positions(self):
        config = M.V3Config(**{**self.config.to_dict(), "arch": "p50"})
        v3 = M.P50(config)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "v3.pt"
            torch.save({"model": v3.state_dict(), "config": config.to_dict(), "arch": "p50"}, path)
            report = self.model.load_v3_warmstart(path)
            self.assertEqual(len(report["remapped"]), 5)
            self.assertEqual(report["new"], [])
            for key in report["copied"]:
                self.assertTrue(torch.equal(self.model.state_dict()[key], v3.state_dict()[key]), key)
            torch.testing.assert_close(self.model.pos.weight[:23], v3.pos.weight[:23])
            expected = v3.pos.weight[23:].reshape(8, 8, 32).repeat_interleave(2, 0).repeat_interleave(2, 1).reshape(256, 32)
            torch.testing.assert_close(self.model.pos.weight[23:], expected)
            torch.testing.assert_close(self.model.row_in.weight, v3.row_in.weight.repeat_interleave(2, 0))
            bad = config.to_dict()
            bad["text_len"] = 26
            torch.save({"model": v3.state_dict(), "config": bad}, path)
            with self.assertRaisesRegex(ValueError, "text_len"):
                self.model.load_v3_warmstart(path)

    def test_chunked_loss_and_gradients_match_raster(self):
        self.image[:, ::11] = 0
        logits = self.model(self.text, self.image)
        expected = M.losses(logits, self.image, 0.25)
        expected[0].backward()
        gradients = {name: p.grad.clone() for name, p in self.model.named_parameters()}
        self.model.zero_grad(set_to_none=True)
        self.model.gradient_checkpointing = True
        self.model.train()
        actual = self.model.loss_fwd(self.text, self.image, chunks=7)
        torch.testing.assert_close(actual[0], expected[0])
        torch.testing.assert_close(actual[1], expected[1].mean())
        actual[0].backward()
        for name, parameter in self.model.named_parameters():
            torch.testing.assert_close(parameter.grad, gradients[name], atol=2e-6, rtol=2e-4, msg=name)

    def test_warmstart_rejects_nonpositional_shape_mismatch(self):
        config = M.V3Config(**{**self.config.to_dict(), "arch": "p50"})
        state = M.P50(config).state_dict()
        state["head.W_out.weight"] = state["head.W_out.weight"][:-1]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "v3.pt"
            torch.save({"model": state, "config": config.to_dict()}, path)
            with self.assertRaisesRegex(ValueError, "head.W_out.weight"):
                self.model.load_v3_warmstart(path)

    def test_full_shape_model_only_checkpoint_format(self):
        config = M.V3Config(arch="p50", d_model=512, n_layer=16, n_head=8,
                            text_len=18, n_pal=3, style=True)
        source = M.P50(config)
        state = source.state_dict()
        self.assertEqual(len(state), 224)
        self.assertEqual(sum(value.numel() for value in state.values()), 66_309_632)
        self.assertTrue(all(value.dtype == torch.float32 for value in state.values()))
        fixture = dict(model=state, step=187500, images=24_000_000,
                       base_images=40_000_000, train_secs=0.0, next_eval=0.0,
                       next_save=0.0, best=0.0, arch="p50", config=config.to_dict(),
                       args=dict(text_len=18, cond=True, batch=128, alpha_weight=0.25),
                       epoch=0, cursor=0, rng=dict(torch=torch.get_rng_state(), cuda=None))
        self.assertNotIn("optimizer", fixture)
        target = P32(P32Config())
        self.assertEqual(sum(value.numel() for value in target.state_dict().values()), 66_424_320)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "v3-model-only.pt"
            torch.save(fixture, path)
            report = target.load_v3_warmstart(path)
        self.assertEqual(len(report["copied"]), 219)
        self.assertEqual(len(report["remapped"]), 5)
        self.assertEqual(report["new"], [])
        for key in report["copied"]:
            self.assertTrue(torch.equal(target.state_dict()[key], state[key]), key)
        self.assertEqual(tuple(target.pos.weight.shape), (279, 512))
        self.assertEqual(tuple(target.row_out.weight.shape), (16, 512))

    def test_sampling_cfg_and_generator(self):
        model = P32(P32Config(d_model=32, n_layer=1, n_head=4, dropout=0)).eval()
        generator = torch.Generator().manual_seed(9)
        state = generator.get_state()
        image = model.sample(self.text, torch.zeros_like(self.text), cfg=2, generator=generator)
        generator.set_state(state)
        repeated = model.sample(self.text, torch.zeros_like(self.text), cfg=2, generator=generator)
        self.assertEqual(image.shape, (1, 1024))
        self.assertTrue(torch.equal(image, repeated))
        self.assertTrue(bool(((image >= 0) & (image < M.PALETTE)).all()))
        with self.assertRaisesRegex(ValueError, "CFG"):
            model.sample(self.text, cfg=2)


if __name__ == "__main__":
    unittest.main()
