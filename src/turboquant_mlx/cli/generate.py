# file: src/turboquant_mlx/cli/generate.py
# description: Single-prompt CLI for Gemma 4 31B text-only inference via mlx-lm
#              with TurboQuant KV cache compression.
#              mlx-lm >= 0.31 includes the gemma4 model type and tool parser.
#              Sampling follows Gemma 4 official recommendation: temp=1.0, top_p=0.95, top_k=64.
# reference: https://huggingface.co/unsloth/gemma-4-31B-it (Best Practices)
#            https://github.com/ml-explore/mlx-lm

from __future__ import annotations

import argparse
import time

import mlx_lm
from mlx_lm.sample_utils import make_sampler

from turboquant_mlx import patch_model
from turboquant_mlx.config import DEFAULT_MODEL


def main() -> None:
    parser = argparse.ArgumentParser(
        description="TurboQuant + Gemma 4 31B text-only generation (mlx-lm)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help="mlx-community model ID or local path",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Explain quantum computing in simple terms.",
    )
    parser.add_argument("--max-tokens",  type=int, default=512)
    parser.add_argument("--key-bits",    type=int, default=3, choices=[2, 3, 4])
    parser.add_argument("--value-bits",  type=int, default=2, choices=[2, 4])
    parser.add_argument("--buffer-size", type=int, default=128)
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        help="enable Gemma 4 thinking mode via chat template",
    )
    parser.add_argument("--no-turboquant", action="store_true")
    args = parser.parse_args()

    print(f"loading {args.model} ...")
    model, tokenizer = mlx_lm.load(args.model)  # ty:ignore[invalid-assignment]

    if not args.no_turboquant:
        print(
            f"patching model: key_bits={args.key_bits} "
            f"value_bits={args.value_bits} buffer={args.buffer_size}"
        )
        patch_model(
            model,
            key_bits=args.key_bits,
            value_bits=args.value_bits,
            buffer_size=args.buffer_size,
        )
        mode_label = f"turboquant-k{args.key_bits}v{args.value_bits}"
    else:
        mode_label = "baseline"

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user",   "content": args.prompt},
    ]

    # apply_chat_template handles Gemma 4 thinking control tokens when
    # enable_thinking is passed; the gemma4 chat template prepends <|think|> accordingly.
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=args.enable_thinking,
    )

    t0 = time.perf_counter()
    response = mlx_lm.generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=args.max_tokens,
        verbose=True,
        sampler=make_sampler(temp=1.0, top_p=0.95, top_k=64),
    )
    elapsed = time.perf_counter() - t0

    print(f"\n--- {mode_label} | {elapsed:.2f}s ---")
    print(response)


if __name__ == "__main__":
    main()
