# file: src/turboquant_mlx/cli/chat.py
# description: Interactive REPL chat for Gemma 4 31B text-only inference via mlx-lm
#              with TurboQuant KV cache compression.
#              Applies Gemma 4 sampling defaults (temp=1.0, top_p=0.95, top_k=64).
#              Strips thinking blocks from history between turns per Gemma 4 multi-turn
#              best practice: thoughts must not appear before the next user turn.
# reference: https://huggingface.co/unsloth/gemma-4-31B-it (Best Practices sections 2 and 3)
#            https://github.com/ml-explore/mlx-lm

from __future__ import annotations

import argparse
import re
import time

import mlx_lm
from mlx_lm.sample_utils import make_sampler

from turboquant_mlx import patch_model
from turboquant_mlx.config import DEFAULT_MODEL


def _strip_thinking(text: str) -> str:
    """
    Remove <|channel>thought\\n ... <channel|> blocks before storing in history.
    Per Gemma 4 multi-turn best practice, thoughts from previous model turns
    must not be included before the next user turn begins
    """
    return re.sub(r"<\|channel>thought\n.*?<channel\|>", "", text, flags=re.DOTALL).strip()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="TurboQuant interactive chat with Gemma 4 31B text-only (mlx-lm)"
    )
    parser.add_argument("--model",       type=str, default=DEFAULT_MODEL)
    parser.add_argument("--key-bits",    type=int, default=3, choices=[2, 3, 4])
    parser.add_argument("--value-bits",  type=int, default=2, choices=[2, 4])
    parser.add_argument("--buffer-size", type=int, default=128)
    parser.add_argument("--max-tokens",  type=int, default=1024)
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        help="enable Gemma 4 thinking mode (<|think|> token prepended)",
    )
    parser.add_argument("--no-turboquant", action="store_true")
    args = parser.parse_args()

    print(f"loading {args.model} ...")
    model, tokenizer = mlx_lm.load(args.model)  # ty:ignore[invalid-assignment]

    if not args.no_turboquant:
        patch_model(
            model,
            key_bits=args.key_bits,
            value_bits=args.value_bits,
            buffer_size=args.buffer_size,
        )
        print(
            f"TurboQuant active: {args.key_bits}-bit keys / "
            f"{args.value_bits}-bit values / buffer {args.buffer_size}"
        )
    else:
        print("TurboQuant disabled (baseline mode)")

    if args.enable_thinking:
        print("thinking mode: enabled")

    print("type /reset to clear history, /quit to exit\n")

    history: list[dict] = [
        {"role": "system", "content": "You are a helpful assistant."},
    ]

    while True:
        try:
            user_input = input("user> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nexiting.")
            break

        if user_input == "/quit":
            break
        if user_input == "/reset":
            history = [history[0]]  # keep system prompt
            print("[history cleared]")
            continue
        if not user_input:
            continue

        history.append({"role": "user", "content": user_input})

        prompt = tokenizer.apply_chat_template(
            history,
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
            verbose=False,
            sampler=make_sampler(temp=1.0, top_p=0.95, top_k=64),
        )
        elapsed = time.perf_counter() - t0

        print(f"assistant> {response}")
        print(f"[{elapsed:.2f}s]\n")

        # strip thinking content before appending to history (Gemma 4 best practice)
        history.append({"role": "assistant", "content": _strip_thinking(response)})


if __name__ == "__main__":
    main()
