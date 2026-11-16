"""One-entry cache of the worker's complete blank CLIP batch."""
from __future__ import annotations

import torch


class BlankFeatures:
    def __init__(self, encoder):
        self.encoder = encoder
        self.clear()

    def clear(self):
        self._key = None
        self._blank = None

    @torch.no_grad()
    def encode_rows(self, rows):
        parameter = next(self.encoder.parameters())
        ids = torch.tensor(rows, dtype=torch.long, device=parameter.device)
        with torch.autocast(device_type=parameter.device.type, dtype=torch.bfloat16,
                            enabled=parameter.device.type == "cuda"):
            return self.encoder(ids)

    def batch(self, row, blank, count):
        if self.encoder.training:
            raise ValueError("blank feature reuse requires an eval-mode encoder")
        if count < 1 or not row or len(row) != len(blank):
            raise ValueError("positive batch and matching nonempty token rows required")
        tensors = (*self.encoder.parameters(), *self.encoder.buffers())
        versions = tuple((id(p), p._version, p.device, p.dtype) for p in tensors)
        key = (id(self.encoder), versions, count, tuple(blank),
               torch.backends.cuda.matmul.fp32_precision, torch.backends.cuda.flash_sdp_enabled(),
               torch.backends.cuda.math_sdp_enabled(), torch.backends.cuda.mem_efficient_sdp_enabled(),
               torch.backends.cuda.cudnn_sdp_enabled())
        h = self.encode_rows([row] * count)
        if key != self._key:
            # Preserve the original batch shape and every row's output.
            self._blank = self.encode_rows([blank] * count)
            self._key = key
        return h, self._blank
