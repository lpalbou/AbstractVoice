# Derived from mlx-lm (https://github.com/ml-explore/mlx-lm, MIT, Copyright © 2023-2024
# Apple Inc.), file mlx_lm/models/cache.py, as redistributed by mlx-audio 0.6.1
# (mlx_audio/lm/models/cache.py). Local modifications, listed exhaustively:
#   1. Only ``KVCache`` is kept (no rotating/batched caches, no state (de)serialisation).
#   2. ``reset()`` added: it rewinds ``offset`` and KEEPS the buffers, so the code predictor's
#      cache (reset once per frame, never longer than 17 positions) is allocated once per
#      utterance instead of zero-filled ten times per frame.
"""Growable key/value cache for one attention layer."""

from __future__ import annotations

import mlx.core as mx


class KVCache:
    """Keys/values for one layer, grown in ``step``-sized blocks along the time axis."""

    step = 256

    def __init__(self) -> None:
        self.keys = None
        self.values = None
        self.offset = 0

    def reset(self) -> None:
        """Start over, keeping the allocation (stale slots are never read: fetch slices ``[:offset]``)."""
        self.offset = 0

    def update_and_fetch(self, keys: mx.array, values: mx.array):
        prev = self.offset
        if self.keys is None or (prev + keys.shape[2]) > self.keys.shape[2]:
            B, n_kv_heads, _, k_head_dim = keys.shape
            v_head_dim = values.shape[3]
            n_steps = (self.step + keys.shape[2] - 1) // self.step
            k_shape = (B, n_kv_heads, n_steps * self.step, k_head_dim)
            v_shape = (B, n_kv_heads, n_steps * self.step, v_head_dim)
            new_k = mx.zeros(k_shape, keys.dtype)
            new_v = mx.zeros(v_shape, values.dtype)
            if self.keys is not None:
                if prev % self.step != 0:
                    self.keys = self.keys[..., :prev, :]
                    self.values = self.values[..., :prev, :]
                self.keys = mx.concatenate([self.keys, new_k], axis=2)
                self.values = mx.concatenate([self.values, new_v], axis=2)
            else:
                self.keys, self.values = new_k, new_v

        self.offset += keys.shape[2]
        self.keys[..., prev : self.offset, :] = keys
        self.values[..., prev : self.offset, :] = values
        return self.keys[..., : self.offset, :], self.values[..., : self.offset, :]

    @property
    def size(self) -> int:
        return self.offset
