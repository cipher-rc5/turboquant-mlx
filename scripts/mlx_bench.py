import os
import sys
import mlx.core as mx
from mlx_lm import load, generate

MODEL = os.environ.get(
    "TQ_MODEL",
    "/Users/excalibur/.lmstudio/models/mlx-community/gemma-4-31b-it-4bit",
)

# Kept in sync with scripts/mlx_turboquant_bench.py so baseline vs TurboQuant
# numbers are an apples-to-apples comparison.  When changing the prompt or
# max_tokens, update both files together.
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


def main() -> None:
    model_path = sys.argv[1] if len(sys.argv) > 1 else MODEL
    prompt = sys.argv[2] if len(sys.argv) > 2 else PROMPT
    max_tokens = int(sys.argv[3]) if len(sys.argv) > 3 else MAX_TOKENS

    print(f"device    : {mx.default_device()}")
    print(f"model     : {model_path}")
    print(f"max_tokens: {max_tokens}")
    print()

    model, tokenizer = load(model_path)  # ty:ignore[invalid-assignment]

    messages = [{"role": "user", "content": prompt}]
    formatted = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    prompt_tokens = len(tokenizer.encode(formatted))
    print(f"prompt tokens : {prompt_tokens}")
    print(f"target context: {prompt_tokens + max_tokens}")
    print()

    mx.reset_peak_memory()
    generate(
        model,
        tokenizer,
        prompt=formatted,
        max_tokens=max_tokens,
        verbose=True,
    )
    peak_gb = mx.get_peak_memory() / 1e9
    print(f"Peak memory: {peak_gb:.3f} GB")


if __name__ == "__main__":
    main()
