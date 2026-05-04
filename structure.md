# structure.md
# description: File structure reference for turboquant-mlx
# reference: https://huggingface.co/unsloth/gemma-4-31B-it; arXiv:2504.19874

## Overview

turboquant-mlx is a uv-managed Python project that ports the TurboQuant KV cache
compression algorithm (Zandieh et al., ICLR 2026) to Apple Silicon via mlx-lm.
It targets Gemma 4 31B text-only inference. All CUDA, Triton, and vLLM dependencies
from the upstream 0xSero/turboquant implementation are replaced with MLX primitives
running on Metal.

```
turboquant-mlx/
├── structure.md                          # this file
├── README.md                             # setup, usage, and architecture overview
├── pyproject.toml                        # uv project manifest (mlx-lm >= 0.31)
├── proof.py                              # A/B benchmark: baseline vs TurboQuant
├── .gitignore
│
├── src/
│   └── turboquant_mlx/                   # installable package
│       ├── __init__.py                   # public API surface
│       ├── codebook.py                   # Lloyd-Max quantizer + bit-packing
│       ├── codebooks/                    # cached JSON codebook files (auto-generated)
│       │   └── .gitkeep
│       ├── rotation.py                   # random orthogonal rotation + QJL projection
│       ├── quantizer.py                  # TurboQuantMSE + TurboQuantProd pipelines
│       ├── kv_cache.py                   # per-layer KV cache manager
│       ├── patch.py                      # mlx-lm model monkey-patcher
│       └── cli/
│           ├── __init__.py
│           ├── generate.py               # tq-generate entrypoint
│           └── chat.py                   # tq-chat entrypoint
│
└── tests/
    ├── __init__.py
    ├── test_quantizer.py                 # TurboQuantMSE + TurboQuantProd unit tests
    └── test_kv_cache.py                  # KV cache integration tests
```

---

## Root

### `pyproject.toml`
uv project manifest. Declares `mlx-lm >= 0.31` as the sole model-loading dependency
(mlx-lm 0.31 is the first release to include the `gemma4` model type and tool parser).
Defines four script entrypoints: `tq-generate`, `tq-chat`, `tq-codebook`, and the
optional `tq-server` under the `[server]` extra. Uses `hatchling` as the build backend
with the package root at `src/turboquant_mlx`.

### `proof.py`
Standalone A/B benchmark. Loads the model once, runs a fixed prompt under baseline
mlx-lm attention, then patches the model with TurboQuant and runs the same prompt again.
Prints a comparison table of tok/s and wall-clock time. Applies Gemma 4 official
sampling parameters: `temp=1.0`, `top_p=0.95`, `top_k=64`.

### `README.md`
User-facing documentation covering setup, checkpoint selection, codebook
pre-generation, CLI usage, the Python API, and architecture notes specific to
Gemma 4 31B (60-layer hybrid attention, head_dim=160, 256K context window).

---

## `src/turboquant_mlx/`

### `__init__.py`
Exports the public API:
- `TurboQuantMSE`, `TurboQuantProd` — quantizer classes
- `CompressedKey`, `CompressedValue` — storage dataclasses
- `TurboQuantLayerCache` — per-layer cache manager
- `make_turboquant_cache` — cache factory
- `patch_model`, `set_decode_mode`, `set_prefill_mode` — model patching

### `codebook.py`
Implements the Lloyd-Max optimal scalar quantizer for the Beta(0.5, 0.5) distribution.
This is the theoretical foundation of TurboQuant: after random orthogonal rotation,
each coordinate of a unit-normed key vector follows Beta(0.5, 0.5), and the Lloyd-Max
codebook is the MSE-optimal quantizer for that distribution.

Key functions:
- `lloyd_max(n_bins, n_iter, n_grid)` — runs the Lloyd-Max iteration on a numerical
  grid approximation of the Beta(0.5, 0.5) PDF
- `get_codebook(head_dim, bits)` — returns (boundaries, centroids) as mlx float16
  arrays, loading from disk cache or generating on first call
- `quantize_with_codebook(x, boundaries)` — maps float16 values to uint8 bin indices
- `dequantize_with_codebook(indices, centroids)` — maps indices back to centroid values
- `pack_bits(indices, bits)` — packs uint8 indices into bytes (4 per byte at 2-bit,
  2 per byte at 4-bit)
- `unpack_bits(packed, bits, original_dim)` — inverse of pack_bits
- `cli_main()` — entrypoint for `tq-codebook` CLI command

Codebook files are written to `src/turboquant_mlx/codebooks/d{dim}_b{bits}.json`.
For Gemma 4 31B, pre-generate with `--dims 160 --bits 2 3 4`.

### `rotation.py`
Implements the two randomised linear maps used by TurboQuant.

Key functions:
- `random_orthogonal(dim, seed)` — generates a (dim, dim) orthogonal matrix via QR
  decomposition of a Gaussian matrix; deterministic from seed for encode/decode
  consistency
- `rotate(x, Q)` — applies x @ Q^T; after rotation, unit-normed key coordinates
  follow Beta(0.5, 0.5) marginally
- `rotate_query(q, Q)` — rotates a query into the same basis as the compressed keys,
  enabling score computation without decompressing keys
- `random_qjl_matrix(dim, n_sketch, seed)` — generates the QJL sketch matrix S of
  shape (dim, n_sketch)
- `qjl_encode(residual, S)` — computes sign(S^T residual); stores 1 bit per sketch
  dimension as uint8
- `qjl_decode_correction(q_rot, signs, S)` — computes the unbiased inner-product
  correction from stored sign bits; adds to the codebook-based score estimate

### `quantizer.py`
Implements the two compression pipelines from the paper.

#### `TurboQuantMSE` (Algorithm 1 -- keys)
Compress path: normalise -> rotate -> map to [0,1] -> Lloyd-Max quantise -> QJL residual.
Decode path: rotate query once -> centroid lookup -> QJL correction -> attention scores.
The rotation is shared between encode and decode; the query is rotated forward rather
than decompressing keys.

#### `TurboQuantProd` (Algorithm 2 -- values)
Compress path: per-group min-max normalisation -> uniform quantise to n_bins -> pack bits.
Decompress path: unpack -> rescale by per-group scale and zero. No rotation is applied
to values since they are summed with softmax weights (inner-product preservation is not
needed).

Storage dataclasses:
- `CompressedKey` — packed uint8 indices, per-token L2 norms, optional QJL sign bits
- `CompressedValue` — packed uint8 indices, per-group scales and zeros

### `kv_cache.py`
Per-layer KV cache manager. Maintains a ring buffer of uncompressed recent tokens
(`buffer_size`, default 128) and a list of compressed batches flushed when the buffer
overflows by `flush_batch` tokens.

#### `TurboQuantLayerCache`
- `update(keys, values)` — appends new tokens; triggers a flush when buffer overflows
- `_flush(n)` — compresses the oldest n tokens from the buffer into
  `CompressedKey` / `CompressedValue` and appends to the compressed list
- `attention(query)` — computes full-context attention: approximate scores over all
  compressed batches (via `TurboQuantMSE.attention_scores`), exact scores over the
  uncompressed buffer, softmax over all, then weighted sum of decompressed values
  plus exact buffer values
- `reset()` — clears all state (called on prefill reset)

#### `make_turboquant_cache(model, ...)`
Factory that creates one `TurboQuantLayerCache` per attention layer. Infers `head_dim`
from the model config (Gemma 4 31B: 160). Helper `_attention_layers` filters to layers
with a `self_attn`, `attention`, or `attn` attribute.

### `patch.py`
Runtime monkey-patcher for mlx-lm models. Wraps each attention layer with
`TurboQuantAttention`, which switches between two modes:

- **Prefill mode** (default): calls the original attention layer unchanged, then
  populates the TurboQuant buffer from the standard KV cache as a side-channel
- **Decode mode**: projects Q/K/V via the original layer's projection matrices,
  updates the TurboQuant cache with the new token's K/V, and computes attention
  entirely from the compressed cache

Public functions:
- `patch_model(model, key_bits, value_bits, buffer_size, flush_batch, group_size)`
- `set_decode_mode(wrappers)` — switch all layers to TurboQuant decode
- `set_prefill_mode(wrappers)` — switch back to standard prefill and reset caches

---

## `src/turboquant_mlx/cli/`

### `generate.py` — `tq-generate`
Single-prompt generation CLI. Accepts `--model`, `--prompt`, `--max-tokens`,
`--key-bits`, `--value-bits`, `--buffer-size`, `--enable-thinking`, and
`--no-turboquant`. Applies Gemma 4 chat template via
`tokenizer.apply_chat_template(..., enable_thinking=...)`.

### `chat.py` — `tq-chat`
Interactive REPL. Maintains a conversation history list with the system prompt pinned
at index 0. On `/reset`, history is cleared back to the system prompt only. After each
model turn, the raw response is displayed but `_strip_thinking()` removes
`<|channel>thought...<channel|>` blocks before the response is appended to history,
per Gemma 4 multi-turn best practice.

---

## `tests/`

### `test_quantizer.py`
Unit tests for `TurboQuantMSE` and `TurboQuantProd`:
- Correct output shapes after compression
- Attention score output shape
- Inner-product estimator approximate unbiasedness (averaged over 20 random rotation
  seeds, within 15% relative error)
- Compression ratio >= 3.5x for 3-bit keys vs float16
- Value round-trip shape correctness
- Value round-trip MSE < 0.05 for 2-bit group quantisation

### `test_kv_cache.py`
Integration tests for `TurboQuantLayerCache`:
- `update` correctly increments `n_tokens`
- Flush triggers when buffer overflows
- `attention` returns correct output shape
- `reset` clears all state

---

## Dependency rationale

| Package          | Version   | Role                                                      |
|------------------|-----------|-----------------------------------------------------------|
| `mlx`            | >= 0.22   | core array framework; Metal backend on Apple Silicon      |
| `mlx-lm`         | >= 0.31   | Gemma 4 model loader, tokenizer, generation loop          |
| `numpy`          | >= 1.26   | Lloyd-Max iteration, bit-packing (CPU; no GPU required)   |
| `pytest`         | >= 8.0    | test runner (dev extra)                                   |
| `fastapi`        | >= 0.115  | OpenAI-compatible server (server extra, optional)         |
| `uvicorn`        | >= 0.30   | ASGI server for fastapi (server extra, optional)          |

`mlx-vlm` is intentionally excluded. It is only required for image, video, or audio
modalities. Gemma 4 31B text-only inference runs correctly under `mlx-lm`.

---

## Gemma 4 31B architecture reference

| Property                  | Value                                      |
|---------------------------|--------------------------------------------|
| Total parameters          | 30.7B                                      |
| Decoder layers            | 60                                         |
| Attention type            | hybrid: sliding-window (1024) + global     |
| hidden_size               | 5120                                       |
| num_attention_heads       | 32                                         |
| head_dim                  | 160 (5120 / 32)                            |
| Context window            | 256K tokens                                |
| Vocabulary size           | 262K                                       |
| Supported modalities      | Text, Image (audio on E2B/E4B only)        |
| Codebook dims to generate | 160                                        |
| Recommended checkpoint    | mlx-community/gemma-4-31b-it-4bit (~20 GB) |
