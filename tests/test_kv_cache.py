# file: tests/test_kv_cache.py
# description: Integration tests for TurboQuantLayerCache: update/attention correctness,
#              buffer overflow / flush behaviour, and reset.
# reference: 0xSero/turboquant/test_turboquant.py

from __future__ import annotations

import math

import mlx.core as mx
import numpy as np
import pytest

from turboquant_mlx.kv_cache import TurboQuantLayerCache
from turboquant_mlx.quantizer import TurboQuantMSE, TurboQuantProd


HEAD_DIM = 64


def _make_cache(buffer_size: int = 16, flush_batch: int = 8) -> TurboQuantLayerCache:
    kq = TurboQuantMSE(head_dim=HEAD_DIM, bits=3)
    vq = TurboQuantProd(head_dim=HEAD_DIM, bits=2)
    return TurboQuantLayerCache(
        key_quantizer=kq,
        value_quantizer=vq,
        buffer_size=buffer_size,
        flush_batch=flush_batch,
    )


def _rand(shape, seed=0) -> mx.array:
    rng = np.random.default_rng(seed)
    return mx.array(rng.standard_normal(shape).astype(np.float16))


class TestTurboQuantLayerCache:
    def test_update_increments_n_tokens(self):
        cache = _make_cache(buffer_size=32, flush_batch=16)
        keys   = _rand((10, HEAD_DIM))
        values = _rand((10, HEAD_DIM))
        cache.update(keys, values)
        assert cache.n_tokens == 10

    def test_flush_triggers_on_overflow(self):
        cache = _make_cache(buffer_size=8, flush_batch=8)
        keys   = _rand((16, HEAD_DIM))
        values = _rand((16, HEAD_DIM))
        cache.update(keys, values)
        assert len(cache.compressed_keys) >= 1, "flush should have occurred"

    def test_attention_output_shape(self):
        cache = _make_cache(buffer_size=32, flush_batch=16)
        T_ctx = 20
        cache.update(_rand((T_ctx, HEAD_DIM)), _rand((T_ctx, HEAD_DIM)))

        query = _rand((1, HEAD_DIM))
        out = cache.attention(query)
        assert out.shape == (1, HEAD_DIM), f"unexpected output shape: {out.shape}"

    def test_reset_clears_state(self):
        cache = _make_cache()
        cache.update(_rand((10, HEAD_DIM)), _rand((10, HEAD_DIM)))
        assert cache.n_tokens > 0
        cache.reset()
        assert cache.n_tokens == 0
        assert len(cache.compressed_keys) == 0
