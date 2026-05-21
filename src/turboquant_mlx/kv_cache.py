# file: src/turboquant_mlx/kv_cache.py
# description: TurboQuant KV cache manager for MLX-LM on Apple Silicon.
#              Replaces the vLLM paged-cache + Triton-kernel path from
#              0xSero/turboquant with an MLX-native ring-buffer design.
#              Keys: TurboQuantMSE (rotation + Lloyd-Max + QJL).
#              Values: TurboQuantProd (group quantisation).
#              Recent `buffer_size` tokens are kept uncompressed for quality.
#              GQA-aware: each layer cache stores n_kv_heads * head_dim per token
#              and expands to n_query_heads during attention.
# reference: 0xSero/turboquant/turboquant/kv_cache.py; yzamari/mlx-turboquant

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import mlx.core as mx

from .quantizer import (
    CompressedKey,
    CompressedValue,
    TurboQuantMSE,
    TurboQuantProd,
)
from .codebook import get_codebook


# ---------------------------------------------------------------------------
# Per-layer TurboQuant KV cache
# ---------------------------------------------------------------------------

@dataclass
class TurboQuantLayerCache:
    """
    KV cache for a single attention layer, GQA-aware.

    Each buffer/compressed entry stores one real token's full KV as a flat
    (n_kv_heads * head_dim,) vector.  The quantizer operates on individual
    head slices of shape (head_dim,); _flush reshapes accordingly.

    Attributes
    ----------
    key_quantizer   : TurboQuantMSE  – sized to head_dim (per single KV head)
    value_quantizer : TurboQuantProd – sized to head_dim
    buffer_size     : int  – max uncompressed tokens
    flush_batch     : int  – tokens accumulated before a compression flush
    n_kv_heads      : int  – number of KV heads (GQA; 1 = multi-head)
    n_query_heads   : int  – number of query heads
    """

    key_quantizer: TurboQuantMSE
    value_quantizer: TurboQuantProd
    buffer_size: int = 128
    flush_batch: int = 128
    n_kv_heads: int = 1
    n_query_heads: int = 1
    # Softmax scale. None ⇒ default 1/sqrt(head_dim).  Gemma 4 attention sets
    # scale=1.0 because Q/K are RMSNormed; using the wrong scale silently
    # flattens the softmax and produces gibberish.
    scale: float | None = None

    # internal state (not passed to __init__)
    compressed_keys: list[CompressedKey] = field(default_factory=list)
    compressed_values: list[CompressedValue] = field(default_factory=list)
    # Per-token chunks of (T_chunk, K*D); stacked on demand.  Using a Python
    # list of small chunks plus a single ``mx.concatenate`` at use-time avoids
    # the O(B²) reallocation of growing a single array by concatenation on
    # every decode step.
    _key_chunks: list[mx.array] = field(default_factory=list)
    _value_chunks: list[mx.array] = field(default_factory=list)
    _buffer_len: int = 0

    @property
    def buffer_len(self) -> int:
        return self._buffer_len

    @property
    def key_buffer(self) -> mx.array | None:
        if not self._key_chunks:
            return None
        if len(self._key_chunks) == 1:
            return self._key_chunks[0]
        merged = mx.concatenate(self._key_chunks, axis=0)
        self._key_chunks = [merged]
        return merged

    @property
    def value_buffer(self) -> mx.array | None:
        if not self._value_chunks:
            return None
        if len(self._value_chunks) == 1:
            return self._value_chunks[0]
        merged = mx.concatenate(self._value_chunks, axis=0)
        self._value_chunks = [merged]
        return merged

    @property
    def n_tokens(self) -> int:
        # compressed_keys[i].packed.shape[0] == n_real * n_kv_heads
        compressed = sum(
            int(ck.packed.shape[0]) // self.n_kv_heads
            for ck in self.compressed_keys
        )
        return compressed + self.buffer_len

    # ------------------------------------------------------------------

    def update(self, keys: mx.array, values: mx.array) -> None:
        """
        Append new key/value pairs to the cache.

        Parameters
        ----------
        keys   : (T_new, n_kv_heads * head_dim) float16
        values : (T_new, n_kv_heads * head_dim) float16
        """
        T_new = int(keys.shape[0])
        self._key_chunks.append(keys)
        self._value_chunks.append(values)
        self._buffer_len += T_new

        overflow = self._buffer_len - self.buffer_size
        if overflow >= self.flush_batch:
            self._flush(overflow)

    def _flush(self, n: int) -> None:
        """Compress the oldest n real tokens from the uncompressed buffer."""
        K = self.n_kv_heads
        D = self.key_quantizer.head_dim

        k_buf = self.key_buffer    # forces stack into a single array
        v_buf = self.value_buffer
        assert k_buf is not None and v_buf is not None

        k_batch = k_buf[:n]
        v_batch = v_buf[:n]

        # Treat every (token, kv_head) slot as an independent vector for the
        # quantizer.  Layout is token-major, KV-head-minor within each token:
        # [tok0_kv0, tok0_kv1, …, tok0_kv(K-1), tok1_kv0, …]
        self.compressed_keys.append(
            self.key_quantizer.compress(k_batch.reshape(n * K, D))
        )
        self.compressed_values.append(
            self.value_quantizer.compress(v_batch.reshape(n * K, D))
        )

        self._key_chunks   = [k_buf[n:]]
        self._value_chunks = [v_buf[n:]]
        self._buffer_len  -= n

    # ------------------------------------------------------------------

    def attention(self, query: mx.array) -> mx.array:
        """
        Compute full-context attention for a single query token.

        Strategy on Apple Silicon (no custom Metal kernel for score-on-packed-
        bits): decompress all compressed K and V into float16 once per step,
        concatenate with the uncompressed buffer, and call the fused
        ``mx.fast.scaled_dot_product_attention`` kernel.  We give up the QJL
        residual correction for compressed tokens (Lloyd-Max scalar
        quantisation alone provides the dominant fidelity), but in exchange
        recover the same fused-attention path that baseline mlx-lm uses.

        The compressed K/V cache stays small for memory-headroom benefit; the
        decompressed reconstruction is a transient per-step allocation.

        Parameters
        ----------
        query : (n_query_heads, head_dim) float16

        Returns
        -------
        output : (n_query_heads, head_dim) float16
        """
        H, D = query.shape
        K = self.n_kv_heads
        scale = self.scale if self.scale is not None else D ** -0.5

        keys_full, values_full = self._materialise_kv()  # (T_total, K, D) each

        # (1, K, T_total, D)
        k = keys_full.transpose(1, 0, 2)[None]
        v = values_full.transpose(1, 0, 2)[None]
        q = query.reshape(1, H, 1, D)

        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale)
        return out.reshape(H, D).astype(mx.float16)

    def _materialise_kv(self) -> tuple[mx.array, mx.array]:
        """Reconstruct full (T_total, K, D) K and V tensors for this step.

        Compressed batches are decoded back to approximate float16 keys via
        the codebook + rotation inverse, and approximate float16 values via
        per-group dequantisation; both are then concatenated with the
        uncompressed buffer.  Allocated fresh each call — no persistent
        decoded cache — so peak resident memory stays close to compressed
        size between calls.
        """
        K = self.n_kv_heads
        D = self.key_quantizer.head_dim

        keys_chunks: list[mx.array] = []
        values_chunks: list[mx.array] = []

        for ck in self.compressed_keys:
            keys_chunks.append(self.key_quantizer.decompress(ck).reshape(-1, K, D))
        for cv in self.compressed_values:
            values_chunks.append(self.value_quantizer.decompress(cv).reshape(-1, K, D))

        if self.buffer_len:
            keys_chunks.append(self.key_buffer.reshape(-1, K, D))
            values_chunks.append(self.value_buffer.reshape(-1, K, D))

        if not keys_chunks:
            raise RuntimeError("attention() called on empty TurboQuantLayerCache")
        if len(keys_chunks) == 1:
            return keys_chunks[0], values_chunks[0]
        return (
            mx.concatenate(keys_chunks, axis=0),
            mx.concatenate(values_chunks, axis=0),
        )

    def reset(self) -> None:
        """Clear all cached state."""
        self.compressed_keys.clear()
        self.compressed_values.clear()
        self._key_chunks.clear()
        self._value_chunks.clear()
        self._buffer_len = 0


# ---------------------------------------------------------------------------
# Multi-layer cache factory
# ---------------------------------------------------------------------------

def make_turboquant_cache(
    model,
    key_bits: int = 3,
    value_bits: int = 2,
    use_qjl: bool = False,
    buffer_size: int = 128,
    flush_batch: int = 128,
    group_size: int = 32,
    scale: float | None = None,
) -> list[TurboQuantLayerCache]:
    """
    Build a TurboQuant layer cache for every patchable attention layer.

    Parameters
    ----------
    model       : an mlx_lm model (must have .layers attribute)
    key_bits    : bits for key quantisation
    value_bits  : bits for value quantisation
    use_qjl     : enable experimental QJL residual correction for keys
    buffer_size : uncompressed token buffer per layer
    flush_batch : tokens per compression batch
    group_size  : value group size for per-group quantisation
    scale       : softmax scale override; pass 1.0 for Gemma 4 whose Q/K are
                  RMSNormed (wrong scale silently flattens softmax → gibberish)
    """
    layers = _attention_layers(model)
    codebook_dims = sorted({_read_head_dim_strict(_attn_module(layer)) for layer in layers})
    for d in codebook_dims:
        get_codebook(d, key_bits)

    caches = []
    for layer in layers:
        head_dim      = _read_head_dim_strict(_attn_module(layer))
        n_kv_heads    = _layer_n_kv_heads(layer)
        n_query_heads = _layer_n_query_heads(layer)

        kq = TurboQuantMSE(head_dim=head_dim, bits=key_bits, use_qjl=use_qjl)
        vq = TurboQuantProd(head_dim=head_dim, bits=value_bits, group_size=group_size)
        caches.append(
            TurboQuantLayerCache(
                key_quantizer=kq,
                value_quantizer=vq,
                buffer_size=buffer_size,
                flush_batch=flush_batch,
                n_kv_heads=n_kv_heads,
                n_query_heads=n_query_heads,
                scale=scale,
            )
        )
    return caches


# ---------------------------------------------------------------------------
# Layer introspection helpers
# ---------------------------------------------------------------------------

def _attention_layers(model) -> list:
    """Return model layers that have their own KV projections (i.e. are patchable)."""
    layers = []
    for layer in getattr(model, "layers", []):
        attn = _attn_module(layer)
        if attn is None:
            continue
        if not getattr(attn, "has_kv", True):
            continue
        layers.append(layer)
    return layers


def _attn_module(layer):
    return (
        getattr(layer, "self_attn", None)
        or getattr(layer, "attention", None)
        or getattr(layer, "attn", None)
    )


def _read_head_dim_strict(attn) -> int:
    hd = getattr(attn, "head_dim", None)
    if hd is None:
        raise RuntimeError(
            f"Attention layer {type(attn).__name__} has no .head_dim attribute. "
            f"Inspect the layer manually with dir(attn) and update _read_head_dim_strict. "
            f"Do NOT fall back to hidden_size / n_heads -- Gemma decouples these."
        )
    return int(hd)


def _layer_n_kv_heads(layer, fallback: int = 1) -> int:
    attn = _attn_module(layer)
    if attn is not None:
        n = (
            getattr(attn, "n_kv_heads", None)
            or getattr(attn, "num_key_value_heads", None)
        )
        if n:
            return int(n)
    return fallback


def _layer_n_query_heads(layer, fallback: int = 1) -> int:
    attn = _attn_module(layer)
    if attn is not None:
        n = (
            getattr(attn, "n_heads", None)
            or getattr(attn, "num_heads", None)
            or getattr(attn, "num_attention_heads", None)
        )
        if n:
            return int(n)
    return fallback
