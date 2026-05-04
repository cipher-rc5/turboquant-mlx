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
    # buffer of recent uncompressed tokens, stored as a single (B, K*D) array.
    # `None` until the first update; growing by `mx.concatenate` is O(B), but
    # B is bounded by buffer_size + flush_batch so it never blows up.
    key_buffer: mx.array | None = None
    value_buffer: mx.array | None = None

    @property
    def buffer_len(self) -> int:
        return 0 if self.key_buffer is None else int(self.key_buffer.shape[0])

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
        if self.key_buffer is None:
            self.key_buffer = keys
            self.value_buffer = values
        else:
            self.key_buffer   = mx.concatenate([self.key_buffer,   keys],   axis=0)
            self.value_buffer = mx.concatenate([self.value_buffer, values], axis=0)

        overflow = self.buffer_len - self.buffer_size
        if overflow >= self.flush_batch:
            self._flush(overflow)

    def _flush(self, n: int) -> None:
        """Compress the oldest n real tokens from the uncompressed buffer."""
        K = self.n_kv_heads
        D = self.key_quantizer.head_dim

        k_batch = self.key_buffer[:n]      # (n, K*D)
        v_batch = self.value_buffer[:n]

        # Treat every (token, kv_head) slot as an independent vector for the
        # quantizer.  Layout is token-major, KV-head-minor within each token:
        # [tok0_kv0, tok0_kv1, …, tok0_kv(K-1), tok1_kv0, …]
        self.compressed_keys.append(
            self.key_quantizer.compress(k_batch.reshape(n * K, D))
        )
        self.compressed_values.append(
            self.value_quantizer.compress(v_batch.reshape(n * K, D))
        )

        self.key_buffer   = self.key_buffer[n:]
        self.value_buffer = self.value_buffer[n:]

    # ------------------------------------------------------------------

    def attention(self, query: mx.array) -> mx.array:
        """
        Compute full-context attention for a single query token.

        Parameters
        ----------
        query : (n_query_heads, head_dim) float16

        Returns
        -------
        output : (n_query_heads, head_dim) float16
        """
        H, D = query.shape
        K = self.n_kv_heads
        G = H // K       # query groups per KV head  (G=1 for MHA, G>1 for GQA)
        scale = self.scale if self.scale is not None else D ** -0.5

        # ---- fast path: no compressed batches yet -------------------------
        # Until the buffer first overflows, the cache holds the full history
        # exactly.  Use the fused attention kernel — same path as baseline
        # mlx-lm — instead of hand-rolled matmul + softmax + einsum.
        if not self.compressed_keys and self.buffer_len:
            B = self.buffer_len
            # (1, K, B, D)
            k = self.key_buffer.reshape(B, K, D).transpose(1, 0, 2)[None]
            v = self.value_buffer.reshape(B, K, D).transpose(1, 0, 2)[None]
            # (1, H, 1, D)
            q = query.reshape(1, H, 1, D)
            out = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale)
            return out.reshape(H, D).astype(mx.float16)

        all_scores: list[mx.array] = []

        # ---- compressed key scores ----------------------------------------
        for ck in self.compressed_keys:
            n_virt = int(ck.packed.shape[0])   # n_real * K
            n_real = n_virt // K

            # raw: (H, n_virt)  — scores vs every (token, kv_head) virtual slot
            raw = self.key_quantizer.attention_scores(query, ck)

            if K == 1:
                all_scores.append(raw)                           # (H, n_real)
            else:
                # Layout of raw columns: token-major, kv-head-minor
                # raw[h, t*K + k] = approx score of query head h vs token t, kv head k
                # Reshape → (K, G, n_real, K); diagonal over the two K dims gives
                # (K, G, n_real) = (H, n_real): each query group attends its own KV head
                raw4 = raw.reshape(K, G, n_real, K)
                diag  = mx.eye(K, dtype=raw.dtype)               # (K, K)
                scores_gqa = (raw4 * diag[:, None, None, :]).sum(-1)  # (K, G, n_real)
                all_scores.append(scores_gqa.reshape(H, n_real))

        # ---- uncompressed buffer scores -----------------------------------
        if self.buffer_len:
            B   = self.buffer_len
            k_buf = self.key_buffer                              # (B, K*D)

            if K == 1:
                all_scores.append(query @ k_buf.T)               # (H, B)
            else:
                # (K, B, D) — KV heads as leading dim for batched matmul
                k_buf3 = k_buf.reshape(B, K, D).transpose(1, 0, 2)
                q3     = query.reshape(K, G, D)                  # (K, G, D)
                # (K, G, D) @ (K, D, B) → (K, G, B) → (H, B)
                all_scores.append(
                    (q3 @ k_buf3.transpose(0, 2, 1)).reshape(H, B)
                )

        # ---- softmax over all tokens --------------------------------------
        all_scores_cat = mx.concatenate(all_scores, axis=-1) * scale  # (H, T_total)
        weights = mx.softmax(all_scores_cat, axis=-1)                  # (H, T_total)

        # ---- weighted sum of values ---------------------------------------
        output = mx.zeros(query.shape, dtype=mx.float16)
        offset = 0

        for i, cv in enumerate(self.compressed_values):
            n_virt = int(self.compressed_keys[i].packed.shape[0])
            n_real = n_virt // K
            v_hat  = self.value_quantizer.decompress(cv)          # (n_virt, D)

            if K == 1:
                w = weights[:, offset : offset + n_real]          # (H, n_real)
                output = output + w @ v_hat
            else:
                v3 = v_hat.reshape(n_real, K, D)                  # (n_real, K, D)
                w  = weights[:, offset : offset + n_real].reshape(K, G, n_real)
                output = output + mx.einsum("kgn,nkd->kgd", w, v3).reshape(H, D)

            offset += n_real

        if self.buffer_len:
            B     = self.buffer_len
            v_buf = self.value_buffer                              # (B, K*D)
            w_buf = weights[:, offset:]                            # (H, B)

            if K == 1:
                output = output + w_buf @ v_buf
            else:
                v_buf3 = v_buf.reshape(B, K, D)                   # (B, K, D)
                w_buf3 = w_buf.reshape(K, G, B)                   # (K, G, B)
                output = output + mx.einsum("kgb,bkd->kgd", w_buf3, v_buf3).reshape(H, D)

        return output

    def reset(self) -> None:
        """Clear all cached state."""
        self.compressed_keys.clear()
        self.compressed_values.clear()
        self.key_buffer = None
        self.value_buffer = None


# ---------------------------------------------------------------------------
# Multi-layer cache factory
# ---------------------------------------------------------------------------

def make_turboquant_cache(
    model,
    key_bits: int = 3,
    value_bits: int = 2,
    buffer_size: int = 128,
    flush_batch: int = 128,
    group_size: int = 32,
) -> list[TurboQuantLayerCache]:
    """
    Build a TurboQuant layer cache for every patchable attention layer.

    Parameters
    ----------
    model       : an mlx_lm model (must have .layers attribute)
    key_bits    : bits for key quantisation (2-4)
    value_bits  : bits for value quantisation (2 or 4)
    buffer_size : uncompressed token buffer per layer
    flush_batch : tokens per compression batch
    group_size  : value group size for per-group quantisation
    """
    fallback_head_dim = _infer_head_dim(model)

    caches = []
    for layer in _attention_layers(model):
        head_dim     = _layer_head_dim(layer, fallback_head_dim)
        n_kv_heads   = _layer_n_kv_heads(layer)
        n_query_heads = _layer_n_query_heads(layer)

        kq = TurboQuantMSE(head_dim=head_dim, bits=key_bits)
        vq = TurboQuantProd(head_dim=head_dim, bits=value_bits, group_size=group_size)
        caches.append(
            TurboQuantLayerCache(
                key_quantizer=kq,
                value_quantizer=vq,
                buffer_size=buffer_size,
                flush_batch=flush_batch,
                n_kv_heads=n_kv_heads,
                n_query_heads=n_query_heads,
            )
        )
    return caches


# ---------------------------------------------------------------------------
# Layer introspection helpers
# ---------------------------------------------------------------------------

def _attention_layers(model) -> list:
    """
    Return model layers that have their own KV projections (i.e. are patchable).

    Gemma 4 uses hybrid attention: sliding-window (head_dim=256) and global
    (head_dim=512) layers alternate.  Layers whose attention module carries no
    k_proj (KV-shared layers) are excluded because they have nothing to compress.
    """
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


def _layer_head_dim(layer, fallback: int) -> int:
    """Read head_dim directly from the layer's attention sub-module."""
    attn = _attn_module(layer)
    if attn is not None:
        hd = getattr(attn, "head_dim", None)
        if hd:
            return int(hd)
    return fallback


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


def _infer_head_dim(model) -> int:
    """
    Fallback head_dim from the model-level config.

    Gemma 4 31B text: head_dim=256 (sliding) / global_head_dim=512.
    We return the sliding-window value as the safe default; per-layer
    _layer_head_dim will override it for global-attention layers.
    """
    cfg = getattr(model, "config", None) or getattr(model, "args", None)
    # unwrap nested text_config (gemma4 multimodal wrapper)
    text_cfg = getattr(cfg, "text_config", None) or cfg
    if text_cfg is not None:
        hd = getattr(text_cfg, "head_dim", None)
        if hd:
            return int(hd)
        n_heads = (
            getattr(text_cfg, "num_attention_heads", None)
            or getattr(text_cfg, "n_heads", None)
        )
        hidden = (
            getattr(text_cfg, "hidden_size", None)
            or getattr(text_cfg, "model_dim", None)
            or getattr(text_cfg, "d_model", None)
        )
        if n_heads and hidden:
            return int(hidden) // int(n_heads)
    # Gemma 4 31B sliding-window default
    return 256
