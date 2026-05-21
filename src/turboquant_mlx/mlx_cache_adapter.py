from __future__ import annotations

import mlx.core as mx
from mlx_lm.models.cache import create_attention_mask

from .kv_cache import TurboQuantLayerCache


class TurboQuantKVCache:
    """
    Drop-in replacement for mlx_lm.models.cache.KVCache.

    Recent tokens stay uncompressed inside TurboQuantLayerCache. Older tokens are
    compressed lazily by the sidecar and materialized only when mlx-lm asks for
    the full key/value tensors. Native mlx-lm attention, RoPE offsets, and masks
    continue to own the attention math.
    """

    def __init__(self, tq: TurboQuantLayerCache, n_kv_heads: int, head_dim: int):
        self._tq = tq
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.offset = 0

    @property
    def tq_cache(self) -> TurboQuantLayerCache:
        return self._tq

    @property
    def keys(self) -> mx.array | None:
        return self._materialize_keys()

    @property
    def values(self) -> mx.array | None:
        return self._materialize_values()

    def update_and_fetch(self, keys: mx.array, values: mx.array):
        if keys.shape[0] != 1:
            raise NotImplementedError("TurboQuantKVCache currently supports batch size 1")

        _, n_kv_heads, t_new, head_dim = keys.shape
        if int(n_kv_heads) != self.n_kv_heads or int(head_dim) != self.head_dim:
            raise ValueError(
                f"KV geometry mismatch: got n_kv_heads={n_kv_heads} head_dim={head_dim}, "
                f"expected n_kv_heads={self.n_kv_heads} head_dim={self.head_dim}"
            )

        k_flat = keys[0].transpose(1, 0, 2).reshape(t_new, -1).astype(mx.float16)
        v_flat = values[0].transpose(1, 0, 2).reshape(t_new, -1).astype(mx.float16)
        self._tq.update(k_flat, v_flat)
        self.offset += int(t_new)
        return self.keys, self.values

    def size(self):
        return self.offset

    @property
    def state(self):
        return self.keys, self.values

    @state.setter
    def state(self, v):
        self._tq.reset()
        self.offset = 0
        if v is None:
            return
        keys, values = v
        if keys is None:
            return
        self.update_and_fetch(keys, values)

    @property
    def meta_state(self):
        return str(self.offset)

    @meta_state.setter
    def meta_state(self, v):
        self.offset = int(v) if v else self.offset

    def is_trimmable(self):
        return False

    def make_mask(self, *args, **kwargs):
        return create_attention_mask(*args, offset=self.offset, **kwargs)

    def empty(self):
        return self.offset == 0

    @property
    def nbytes(self):
        total = 0
        for ck in self._tq.compressed_keys:
            total += ck.packed.nbytes + ck.norms.nbytes
            if ck.qjl_signs is not None:
                total += ck.qjl_signs.nbytes
            if ck.residual_norms is not None:
                total += ck.residual_norms.nbytes
        for cv in self._tq.compressed_values:
            total += cv.packed.nbytes + cv.scales.nbytes + cv.zeros.nbytes
        for chunk in self._tq._key_chunks:
            total += chunk.nbytes
        for chunk in self._tq._value_chunks:
            total += chunk.nbytes
        return total

    def _materialize_keys(self) -> mx.array | None:
        if self._tq.n_tokens == 0:
            return None
        keys, _ = self._tq._materialise_kv()
        return keys.transpose(1, 0, 2)[None]

    def _materialize_values(self) -> mx.array | None:
        if self._tq.n_tokens == 0:
            return None
        _, values = self._tq._materialise_kv()
        return values.transpose(1, 0, 2)[None]
