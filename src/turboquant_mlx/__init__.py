# file: src/turboquant_mlx/__init__.py
# description: Public API for turboquant-mlx. Mirrors the API surface of
#              0xSero/turboquant but all CUDA/vLLM dependencies replaced with MLX.
# reference: 0xSero/turboquant/turboquant/__init__.py

from .kv_cache import TurboQuantLayerCache, make_turboquant_cache
from .patch import patch_model, set_decode_mode, set_prefill_mode
from .quantizer import CompressedKey, CompressedValue, TurboQuantMSE, TurboQuantProd

__all__ = [
    "TurboQuantMSE",
    "TurboQuantProd",
    "CompressedKey",
    "CompressedValue",
    "TurboQuantLayerCache",
    "make_turboquant_cache",
    "patch_model",
    "set_decode_mode",
    "set_prefill_mode",
]
