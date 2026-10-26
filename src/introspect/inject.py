"""Concept injection: add strength * vector to the residual stream at one layer.

Each row of a batch gets its own vector, strength and start position, so one generate() call can run
50 different concepts (and no-injection controls with strength 0) side by side. Injection covers every
position from the row's start to the end of the prompt, and every generated token, as in the paper.
"""

from __future__ import annotations

import contextlib

import numpy as np
import torch

from .model import decoder_layers


@contextlib.contextmanager
def inject(model, layer: int, vectors: np.ndarray, strengths: np.ndarray, starts: np.ndarray):
    """layer: hidden_states index (1..L-1); the hook sits on decoder layer `layer - 1`.
    vectors: (B, D); strengths: (B,); starts: (B,) token positions in the padded batch."""
    module = decoder_layers(model)[layer - 1]
    p = next(model.parameters())
    add = torch.tensor(vectors * strengths[:, None], dtype=torch.float32, device=p.device)
    starts_t = torch.tensor(starts, dtype=torch.long, device=p.device)
    state = {"seen": 0}  # number of positions processed so far (prefill, then one per decode step)

    def hook(_module, _inputs, output):
        h = output[0] if isinstance(output, tuple) else output
        T = h.shape[1]
        # a model split across GPUs puts this layer on its own device
        pos = torch.arange(state["seen"], state["seen"] + T, device=h.device)
        mask = (pos[None, :] >= starts_t.to(h.device)[:, None]).to(h.dtype)          # (B, T)
        h = h + mask[:, :, None] * add.to(device=h.device, dtype=h.dtype)[:, None, :]
        state["seen"] += T
        if isinstance(output, tuple):
            return (h,) + tuple(output[1:])
        return h

    handle = module.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()
