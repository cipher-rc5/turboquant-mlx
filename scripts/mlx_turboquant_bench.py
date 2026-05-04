import sys
import mlx.core as mx
from mlx_lm import load, generate
from turboquant_mlx import patch_model

MODEL = "/Users/excalibur/.lmstudio/models/mlx-community/gemma-4-31b-it-4bit"

# Long-context prompt designed to exercise the compressed K/V path.
#
# Why this shape:
#   - TurboQuant's wins (memory + bandwidth) only show up once the cache
#     is dominated by *compressed* tokens, not the uncompressed buffer.
#     With buffer_size=128 + flush_batch=128, the buffer holds 128 tokens
#     in float16 and everything older is in 3-bit (key) / 2-bit (value)
#     packed form.
#   - We need (prompt_len + decode_len) >> buffer_size for that ratio to
#     be meaningful.  Aim for ~4-8K total context: at 4K context, ~97% of
#     the cache is compressed → 5-7× smaller K cache, 8× smaller V cache.
#   - Decode is what actually streams the cache through GPU memory each
#     step, so the generation length is what determines whether bandwidth
#     savings manifest as throughput.  Push max_tokens to 2K+.
#   - The prompt asks for a long, structured response so the model is
#     unlikely to early-terminate via EOS before we hit the regime.
PROMPT = (
    "You are writing a comprehensive technical reference document. "
    "Produce a self-contained, well-structured deep dive on transformer "
    "inference systems engineering. Cover, in order, with detailed "
    "subsections and worked numerical examples for each:\n"
    "  1. The decode-step memory bandwidth bottleneck — derive the "
    "arithmetic intensity of attention for a 30B-parameter model at "
    "context lengths 1K, 4K, 16K, and 64K.\n"
    "  2. KV cache layout strategies — paged vs. contiguous vs. ring "
    "buffers — and their interaction with prefix caching.\n"
    "  3. Quantization of the KV cache — per-tensor, per-channel, "
    "per-group, per-token; trade-offs between scalar codebooks "
    "(Lloyd-Max), uniform min-max, and rotation-based methods like "
    "QuaRot and TurboQuant.\n"
    "  4. Approximate attention — sparse, low-rank, sketching (JL / QJL), "
    "and their unbiased-estimator guarantees.\n"
    "  5. GQA and MQA — head sharing, expressivity loss, and "
    "implications for cache compression schemes.\n"
    "  6. RoPE and its variants — partial rotation, scaling laws, "
    "implications for long-context extrapolation.\n"
    "  7. Speculative decoding and its composition with KV compression.\n"
    "  8. End-to-end engineering checklist for a production decode "
    "service targeting 100K context on Apple Silicon.\n"
    "Be concrete. Include formulas, byte counts, and example calculations."
)

MAX_TOKENS = 2048
KEY_BITS = 3
VALUE_BITS = 2
BUFFER_SIZE = 128


def main() -> None:
    model_path  = sys.argv[1] if len(sys.argv) > 1 else MODEL
    prompt      = sys.argv[2] if len(sys.argv) > 2 else PROMPT
    max_tokens  = int(sys.argv[3]) if len(sys.argv) > 3 else MAX_TOKENS

    print(f"device    : {mx.default_device()}")
    print(f"model     : {model_path}")
    print(f"max_tokens: {max_tokens}")
    print(f"turboquant: k{KEY_BITS}v{VALUE_BITS} buf{BUFFER_SIZE}")
    print()

    model, tokenizer = load(model_path)  # ty:ignore[invalid-assignment]

    patch_model(model, key_bits=KEY_BITS, value_bits=VALUE_BITS, buffer_size=BUFFER_SIZE)

    messages = [{"role": "user", "content": prompt}]
    formatted = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    prompt_tokens = len(tokenizer.encode(formatted))
    print(f"prompt tokens : {prompt_tokens}")
    print(f"target context: {prompt_tokens + max_tokens} "
          f"(buffer holds last {BUFFER_SIZE}, "
          f"compressed = {max(0, prompt_tokens + max_tokens - BUFFER_SIZE)})")
    print()

    mx.reset_peak_memory()
    generate(model, tokenizer, prompt=formatted, max_tokens=max_tokens, verbose=True)
    peak_gb = mx.get_peak_memory() / 1e9
    print(f"Peak memory: {peak_gb:.3f} GB")


if __name__ == "__main__":
    main()
