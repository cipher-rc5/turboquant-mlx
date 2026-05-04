# turboquant-mlx

> **Personal learning project.** This codebase exists solely for my own study of KV cache
> compression techniques. It is not intended for production use, external deployment, or
> any real-world application.

TurboQuant KV cache compression for **Gemma 4 31B text-only inference** on Apple Silicon.
Port of [0xSero/turboquant](https://github.com/0xSero/turboquant) with all CUDA/Triton/vLLM
dependencies removed and replaced with `mlx-lm`. Managed with `uv`.

Model: [`mlx-community/gemma-4-31b-it`](https://huggingface.co/mlx-community/gemma-4-31b-it-4bit) (Apache 2.0)
Paper: [arXiv:2504.19874](https://arxiv.org/abs/2504.19874) (Zandieh et al., ICLR 2026)

## mlx-lm vs mlx-vlm for Gemma 4

`mlx-vlm` is only required when you need image, video, or audio input. For text-only
use, `mlx-lm >= 0.31` is the correct and simpler dependency -- it includes the `gemma4`
model type, the Gemma 4 chat template, and the tool parser for function calling.
This codebase uses `mlx-lm` exclusively.

## Requirements

- Apple Silicon Mac (M1 or later; M2 Ultra or M4 Max for the 31B BF16 checkpoint)
- Python 3.12
- [uv](https://docs.astral.sh/uv/) >= 0.4

## Checkpoints

| Checkpoint                                  | VRAM    | Notes                  |
|---------------------------------------------|---------|------------------------|
| `mlx-community/gemma-4-31b-it-4bit`         | ~20 GB  | recommended starting point |
| `mlx-community/gemma-4-31b-it-bf16`         | ~34 GB  | full precision         |

## Setup

```sh
uv sync
```

## Pre-generate codebooks

```sh
# Gemma 4 31B: hidden_size=5120, num_attention_heads=32 -> head_dim=160
uv run tq-codebook --dims 160 --bits 2 3 4
```

Codebooks are cached to `src/turboquant_mlx/codebooks/` and auto-generated on first
use if absent (~5s per codebook).

## Run the A/B benchmark

```sh
uv run python proof.py --model mlx-community/gemma-4-31b-it-4bit
```

## Single-prompt generation

```sh
uv run tq-generate \
  --model mlx-community/gemma-4-31b-it-4bit \
  --prompt "Explain KV cache quantisation" \
  --key-bits 3 --max-tokens 512

# with thinking mode
uv run tq-generate \
  --model mlx-community/gemma-4-31b-it-4bit \
  --prompt "Solve: x^2 - 5x + 6 = 0" \
  --enable-thinking
```

## Interactive chat

```sh
uv run tq-chat --model mlx-community/gemma-4-31b-it-4bit
```

Commands: `/reset` clears history, `/quit` exits.

## Python API

```python
import mlx_lm
from turboquant_mlx import patch_model

model, tokenizer = mlx_lm.load("mlx-community/gemma-4-31b-it-4bit")

patch_model(model, key_bits=3, value_bits=2, buffer_size=128)

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user",   "content": "Hello!"},
]
prompt = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
output = mlx_lm.generate(model, tokenizer, prompt=prompt, temp=1.0, top_p=0.95, top_k=64)
print(output)
```

## Run tests

```sh
uv run pytest
```

## Gemma 4 sampling parameters (official recommendation)

`temperature=1.0`, `top_p=0.95`, `top_k=64`. All CLI entrypoints apply these as defaults.

## Architecture

```
turboquant_mlx/
  codebook.py      # Lloyd-Max codebook for Beta(0.5,0.5) + bit-packing
  codebooks/       # cached JSON codebook files (d=160 for Gemma 4 31B)
  rotation.py      # random orthogonal rotation + QJL projection
  quantizer.py     # TurboQuantMSE + TurboQuantProd pipelines
  kv_cache.py      # per-layer KV cache manager (ring buffer + flush)
  patch.py         # mlx-lm model monkey-patcher (Gemma 4 60-layer hybrid attention)
  cli/
    generate.py    # tq-generate entrypoint
    chat.py        # tq-chat entrypoint
proof.py           # A/B benchmark (baseline vs TurboQuant)
pyproject.toml     # uv project manifest (mlx-lm >= 0.31)
```

## Gemma 4 31B architecture notes

- 60 decoder layers, hybrid sliding-window (1024 tokens) + global attention
- head_dim = 160 (hidden_size=5120 / num_attention_heads=32)
- 256K token context window
- No audio encoder at 31B size (audio is E2B/E4B only per processor_config.json)
- TurboQuant patches all 60 attention layers; global layers benefit most since they
  accumulate the longest KV histories

## Limitations

- Bit-packing and codebook lookups go through numpy on CPU. Correct and complete;
  a native Metal kernel would add further decode speedup at very long contexts.
- Prefill uses standard mlx-lm attention; TurboQuant activates at decode time.
- Lossy compression. Increase `buffer_size` to keep more recent tokens uncompressed
  if exact retrieval over deep context is required.

## License

MIT
