"""Exercise Dynamo on a tiny real P32 loss without a CUDA/Inductor benchmark."""
from dataclasses import asdict, replace
import hashlib
import json
import unittest

import torch

from train.pixie32.p32_model import P32, P32Config
from train.pixie32.train_p32 import Recipe, check_resume, fingerprint


class CompileTests(unittest.TestCase):
    def test_default_fingerprint_matches_launched_recipe_format(self):
        recipe = Recipe(4, 17, "fixture", {"d_model": 32}, 24, 2, 8, 1e-4, 1e-5)
        legacy = asdict(recipe)
        legacy.pop("compile")
        legacy.pop("clip_cache_sha256")
        digest = hashlib.sha256(json.dumps(legacy, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        self.assertEqual(fingerprint(recipe), {"fields": legacy, "sha256": digest})
        with self.assertRaisesRegex(ValueError, "compile"):
            check_resume(fingerprint(recipe), fingerprint(replace(recipe, compile=True)))

    def test_compiled_checkpointed_loss_matches_eager_and_preserves_keys(self):
        torch.set_num_threads(2)
        torch.manual_seed(42)
        model = P32(P32Config(d_model=32, n_layer=1, n_head=4, dropout=0)).train()
        model.gradient_checkpointing = True
        original_keys = set(model.state_dict())
        text = torch.randn(1, 18, 768)
        image = torch.randint(0, 4097, (1, 1024))
        image[:, ::11] = 0
        eager, _ = model.loss_fwd(text, image, chunks=16)
        eager.backward()
        gradients = {name: parameter.grad.clone() for name, parameter in model.named_parameters()}
        model.zero_grad(set_to_none=True)
        compiled_loss = torch.compile(model.loss_fwd, backend="eager", fullgraph=True)
        compiled, _ = compiled_loss(text, image, chunks=16)
        compiled.backward()
        torch.testing.assert_close(compiled, eager)
        for name, parameter in model.named_parameters():
            torch.testing.assert_close(parameter.grad, gradients[name], atol=2e-6, rtol=2e-4, msg=name)
        self.assertEqual(set(model.state_dict()), original_keys)
        self.assertTrue(all(not key.startswith("_orig_mod") for key in original_keys))


if __name__ == "__main__":
    unittest.main()
