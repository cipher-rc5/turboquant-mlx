from __future__ import annotations

from .kv_cache import (
    _attention_layers,
    _attn_module,
    _layer_n_kv_heads,
    _layer_n_query_heads,
    _read_head_dim_strict,
    make_turboquant_cache,
)
from .mlx_cache_adapter import TurboQuantKVCache


def patch_model(
    model,
    key_bits: int = 3,
    value_bits: int = 2,
    use_qjl: bool = False,
    buffer_size: int = 128,
    flush_batch: int = 128,
    group_size: int = 32,
) -> list[TurboQuantKVCache]:
    """Inject TurboQuant by replacing the model cache factory, not attention math."""
    print("TurboQuant: discovering model geometry...")
    attention_layers = _attention_layers(model)
    for i, layer in enumerate(attention_layers[:3]):
        attn = _attn_module(layer)
        hd = _read_head_dim_strict(attn)
        nkv = _layer_n_kv_heads(layer)
        nq = _layer_n_query_heads(layer)
        print(f"  layer[{i}]: head_dim={hd} n_kv_heads={nkv} n_query_heads={nq}")

    def make_tq_prompt_cache() -> list[TurboQuantKVCache]:
        layer_caches = make_turboquant_cache(
            model,
            key_bits=key_bits,
            value_bits=value_bits,
            use_qjl=use_qjl,
            buffer_size=buffer_size,
            flush_batch=flush_batch,
            group_size=group_size,
        )
        adapters: list[TurboQuantKVCache] = []
        for layer, tq_cache in zip(attention_layers, layer_caches):
            attn = _attn_module(layer)
            adapters.append(
                TurboQuantKVCache(
                    tq=tq_cache,
                    n_kv_heads=_layer_n_kv_heads(layer),
                    head_dim=_read_head_dim_strict(attn),
                )
            )
        model._turboquant_last_cache = adapters
        return adapters

    model._turboquant_original_make_cache = getattr(model, "make_cache", None)
    model.make_cache = make_tq_prompt_cache
    return make_tq_prompt_cache()


def set_decode_mode(wrappers) -> None:
    """Retained for older callers; cache-swap mode has no decode switch."""
    return None


def set_prefill_mode(wrappers) -> None:
    """Retained for older callers; fresh generation creates a fresh cache."""
    for cache in wrappers or []:
        if hasattr(cache, "tq_cache"):
            cache.tq_cache.reset()
            cache.offset = 0
