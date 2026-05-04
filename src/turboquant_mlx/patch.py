# file: src/turboquant_mlx/patch.py
# description: mlx-lm model monkey-patcher that injects TurboQuant KV cache into
#              every attention layer of a Gemma 4 model at runtime (text-only path).
#              mlx-lm >= 0.31 is required for the gemma4_text model type.
#              Prefill runs standard mlx-lm attention at full speed; TurboQuant
#              activates during decode when compressed cache reduces memory bandwidth.
#              Gemma 4 31B: 60 layers, sliding head_dim=256, global head_dim=512.
# reference: 0xSero/turboquant/turboquant/vllm_attn_backend.py
#            https://huggingface.co/mlx-community/gemma-4-31b-it-4bit

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from .kv_cache import TurboQuantLayerCache, make_turboquant_cache


# ---------------------------------------------------------------------------
# Attention layer wrapper
# ---------------------------------------------------------------------------

class TurboQuantAttention(nn.Module):
    """
    Wraps an existing mlx-lm attention layer and replaces its KV cache
    with TurboQuant compressed storage during decode.

    Prefill (prompt processing) passes through to the original implementation
    unchanged.  Decode steps use the TurboQuant cache for approximate attention
    over the compressed token history.

    The wrapper matches the gemma4_text Attention call convention:
        __call__(x, mask, cache, shared_kv, offset)
        -> (output, (keys, values), offset)
    """

    def __init__(self, original_attn: nn.Module, tq_cache: TurboQuantLayerCache) -> None:
        super().__init__()
        self._attn = original_attn
        self._tq_cache = tq_cache
        self._is_prefill = True
        # how many tokens we've already pulled into _tq_cache; used to slice off
        # only the new tail when the underlying KVCache returns the full history
        self._ingested = 0

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | None = None,
        cache=None,
        shared_kv=None,
        offset=None,
    ):
        # Auto-flip prefill→decode the first time we see a single-token forward
        # pass after at least one prefill call has populated the TQ buffer.  This
        # is what real callers (mlx_lm.generate) want: the prompt is processed in
        # one shot (T>1), then decode steps come in with T=1.
        if self._is_prefill and x.shape[1] == 1 and self._ingested > 0:
            self._is_prefill = False

        if self._is_prefill:
            return self._prefill_forward(x, mask, cache, shared_kv, offset)
        return self._decode_forward(x, mask, cache, shared_kv, offset)

    # ------------------------------------------------------------------

    def _prefill_forward(self, x, mask, cache, shared_kv, offset):
        """Standard attention; populate TurboQuant buffer with the *new* tokens only.

        Some mlx-lm attention modules return (output, (full_keys, full_values), offset)
        where the K/V tensors include the entire accumulated cache.  Re-ingesting
        the full cache every call is O(T_total) per layer and gives quadratic
        end-to-end cost; instead we slice off the last `T_new` rows since the
        KVCache append is left-to-right.
        """
        result = self._attn(x, mask=mask, cache=cache, shared_kv=shared_kv, offset=offset)

        # gemma4_text Attention returns (output, (keys, values), offset)
        if isinstance(result, tuple) and len(result) == 3:
            out, kv_pair, offset_out = result
        else:
            # fallback for simpler attention modules
            out, kv_pair, offset_out = result, None, offset

        keys = values = None
        if kv_pair is not None:
            keys, values = kv_pair
        elif cache is not None and getattr(cache, "keys", None) is not None:
            keys, values = cache.keys, cache.values

        if keys is not None and values is not None:
            # (batch, n_kv_heads, T_total, head_dim)
            T_total = int(keys.shape[-2])
            T_new = T_total - self._ingested
            if T_new > 0:
                k_new = keys[0, :, -T_new:, :].transpose(1, 0, 2).reshape(T_new, -1).astype(mx.float16)
                v_new = values[0, :, -T_new:, :].transpose(1, 0, 2).reshape(T_new, -1).astype(mx.float16)
                self._tq_cache.update(k_new, v_new)
                self._ingested = T_total

        return out, kv_pair, offset_out

    def _decode_forward(self, x, mask, cache, shared_kv, offset):
        """Single-token decode using TurboQuant compressed KV history.

        Mirrors gemma4_text.Attention's projection/norm/RoPE pipeline so that
        the new query and key are in the same coordinate system as the keys
        already in the TQ buffer (which were RoPE'd by the original attention
        during prefill).
        """
        attn = self._attn
        B, T, _ = x.shape   # T == 1 during decode

        if not hasattr(attn, "q_proj"):
            raise NotImplementedError(
                "unsupported attention variant — layer has no q_proj"
            )

        n_heads   = getattr(attn, "n_heads", None)   or getattr(attn, "num_heads", 1)
        n_kv_heads = getattr(attn, "n_kv_heads", None) or n_heads
        head_dim  = getattr(attn, "head_dim", None)
        if head_dim is None:
            head_dim = attn.q_proj.weight.shape[0] // n_heads

        # Position offset for RoPE: number of preceding real tokens.  We track
        # this locally via _ingested rather than reading cache.offset because
        # we no longer call self._attn() (which would advance the underlying
        # cache).  After prompt prefill, _ingested == T_prompt; on decode step
        # k (0-indexed), the new token sits at position T_prompt + k.
        pos = mx.array(self._ingested) if offset is None else offset

        # ---- query projection + norm + RoPE -----------------------------
        q = attn.q_proj(x).reshape(B, T, n_heads, head_dim)
        if hasattr(attn, "q_norm"):
            q = attn.q_norm(q)
        q = q.transpose(0, 2, 1, 3)             # (B, n_heads, T, head_dim)
        if hasattr(attn, "rope"):
            q = attn.rope(q, offset=pos)
        q_squeezed = q[0, :, 0, :]              # (n_heads, head_dim)

        # ---- key / value projection + norm + RoPE -----------------------
        new_kv = None
        if shared_kv is not None:
            keys, values = shared_kv
        elif hasattr(attn, "k_proj"):
            raw_k = attn.k_proj(x).reshape(B, T, n_kv_heads, head_dim)

            keys = raw_k
            if hasattr(attn, "k_norm"):
                keys = attn.k_norm(keys)
            keys = keys.transpose(0, 2, 1, 3)   # (B, n_kv_heads, T, head_dim)
            if hasattr(attn, "rope"):
                keys = attn.rope(keys, offset=pos)

            # for k_eq_v layers (global attention): values derive from same raw_k
            # with v_norm applied (different from k_norm), without RoPE
            use_k_eq_v = getattr(attn, "use_k_eq_v", False)
            if use_k_eq_v:
                values = raw_k
            else:
                values = attn.v_proj(x).reshape(B, T, n_kv_heads, head_dim)
            if hasattr(attn, "v_norm"):
                values = attn.v_norm(values)
            values = values.transpose(0, 2, 1, 3)  # (B, n_kv_heads, T, head_dim)

            new_kv = (keys, values)
            k_flat = keys[0].transpose(1, 0, 2).reshape(T, -1).astype(mx.float16)
            v_flat = values[0].transpose(1, 0, 2).reshape(T, -1).astype(mx.float16)
            self._tq_cache.update(k_flat, v_flat)
            self._ingested += T

        # keep the underlying mlx-lm cache offset in sync so siblings (e.g.
        # KV-shared layers) that read cache.offset still see a sensible value
        if cache is not None and hasattr(cache, "offset"):
            try:
                cache.offset = self._ingested
            except (AttributeError, TypeError):
                pass

        # ---- TurboQuant attention over full history ---------------------
        out_heads = self._tq_cache.attention(q_squeezed)   # (n_heads, head_dim)
        out_flat = out_heads.reshape(1, 1, -1)             # (B=1, T=1, embed_dim)

        if hasattr(attn, "o_proj"):
            out_flat = attn.o_proj(out_flat)

        return out_flat, new_kv, offset

    # ------------------------------------------------------------------

    def set_decode_mode(self) -> None:
        self._is_prefill = False

    def set_prefill_mode(self) -> None:
        self._is_prefill = True
        self._tq_cache.reset()
        self._ingested = 0


# ---------------------------------------------------------------------------
# Public patch / unpatch API
# ---------------------------------------------------------------------------

def patch_model(
    model,
    key_bits: int = 3,
    value_bits: int = 2,
    buffer_size: int = 128,
    flush_batch: int = 128,
    group_size: int = 32,
) -> list[TurboQuantAttention]:
    """
    Inject TurboQuant KV caches into all attention layers of an mlx-lm Gemma 4 model.

    Parameters
    ----------
    model       : mlx-lm model (Gemma 4 31B has 60 hybrid attention layers)
    key_bits    : 2, 3, or 4
    value_bits  : 2 or 4
    buffer_size : uncompressed recent token buffer per layer
    flush_batch : compression batch size
    group_size  : value quantisation group size

    Returns
    -------
    list of TurboQuantAttention wrappers
    """
    caches = make_turboquant_cache(
        model,
        key_bits=key_bits,
        value_bits=value_bits,
        buffer_size=buffer_size,
        flush_batch=flush_batch,
        group_size=group_size,
    )

    wrappers: list[TurboQuantAttention] = []
    cache_idx = 0
    for layer in getattr(model, "layers", []):
        attn_attr = _attn_attr_name(layer)
        if attn_attr is None:
            continue
        attn = getattr(layer, attn_attr)
        # skip KV-shared layers (no k_proj — nothing for TQ to do)
        if not getattr(attn, "has_kv", True):
            continue
        cache = caches[cache_idx]
        # Inherit the attention layer's softmax scale (gemma4 uses 1.0 because
        # Q/K are RMSNormed; default 1/sqrt(D) would crush the softmax).
        attn_scale = getattr(attn, "scale", None)
        if attn_scale is not None:
            cache.scale = float(attn_scale)
        wrapper = TurboQuantAttention(attn, cache)
        setattr(layer, attn_attr, wrapper)
        wrappers.append(wrapper)
        cache_idx += 1

    return wrappers


def set_decode_mode(wrappers: list[TurboQuantAttention]) -> None:
    """Switch all wrapped layers to decode (TurboQuant) mode."""
    for w in wrappers:
        w.set_decode_mode()


def set_prefill_mode(wrappers: list[TurboQuantAttention]) -> None:
    """Switch all wrapped layers back to prefill (standard) mode and reset caches."""
    for w in wrappers:
        w.set_prefill_mode()


def _attn_attr_name(layer) -> str | None:
    for name in ("self_attn", "attention", "attn"):
        if hasattr(layer, name):
            return name
    return None
