# file: proof.py
# description: A/B benchmark — baseline mlx-lm vs TurboQuant KV cache on Gemma 4 31B.
#              Reports prefill t/s and generation t/s separately to match llama-cli output.
#              Sampling follows Gemma 4 official recommendation: temp=1.0, top_p=0.95, top_k=64.
# reference: 0xSero/turboquant/proof.py (original CUDA/vLLM version)
#            https://huggingface.co/mlx-community/gemma-4-31b-it-4bit

from __future__ import annotations

import argparse

import mlx_lm
from mlx_lm.generate import stream_generate
from mlx_lm.sample_utils import make_sampler

from turboquant_mlx.config import DEFAULT_MODEL

PROMPT = (
    "Explain the difference between MoE and dense transformer architectures in detail."
)


def _run(model, tokenizer, label: str, max_tokens: int) -> dict:
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user",   "content": PROMPT},
    ]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    last = None
    text = ""
    for response in stream_generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=max_tokens,
        sampler=make_sampler(temp=1.0, top_p=0.95, top_k=64),
    ):
        text += response.text
        last = response

    return {
        "mode":         label,
        "prompt_tps":   round(last.prompt_tps, 1) if last else 0,
        "gen_tps":      round(last.generation_tps, 1) if last else 0,
        "gen_tokens":   last.generation_tokens if last else 0,
        "preview":      text[:160].replace("\n", " "),
    }


def print_table(results: list[dict], llama_ref: dict | None = None) -> None:
    header = f"{'mode':<44} {'prompt t/s':>12} {'gen t/s':>10} {'tokens':>8}"
    sep = "-" * len(header)
    print("\n" + "=" * len(header))
    print(header)
    print(sep)
    if llama_ref:
        print(
            f"{'[llama-cli ref] gemma-4-26B-A4B Q8_0':<44} "
            f"{llama_ref['prompt_tps']:>12.1f} "
            f"{llama_ref['gen_tps']:>10.1f} "
            f"{'256':>8}"
        )
        print(sep)
    for r in results:
        print(
            f"{r['mode']:<44} "
            f"{r['prompt_tps']:>12.1f} "
            f"{r['gen_tps']:>10.1f} "
            f"{r['gen_tokens']:>8}"
        )
    print("=" * len(header))
    print()
    for r in results:
        print(f"[{r['mode']}]\n  {r['preview']}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="TurboQuant + Gemma 4 31B text-only A/B benchmark (mlx-lm)"
    )
    parser.add_argument("--model",       type=str, default=DEFAULT_MODEL)
    parser.add_argument("--max-tokens",  type=int, default=256)
    parser.add_argument("--key-bits",    type=int, default=3)
    parser.add_argument("--value-bits",  type=int, default=2)
    parser.add_argument("--buffer-size", type=int, default=128)
    args = parser.parse_args()

    # llama-cli reference numbers from your test run
    llama_ref = {"prompt_tps": 62.3, "gen_tps": 85.0}

    print(f"loading {args.model} ...")
    model, tokenizer = mlx_lm.load(args.model)  # ty:ignore[invalid-assignment]

    results = []

    print("running baseline ...")
    results.append(_run(model, tokenizer, "baseline (mlx, no TQ)", args.max_tokens))

    print("patching model for TurboQuant ...")
    from turboquant_mlx import patch_model
    patch_model(
        model,
        key_bits=args.key_bits,
        value_bits=args.value_bits,
        buffer_size=args.buffer_size,
    )
    label = f"turboquant k{args.key_bits}v{args.value_bits} buf{args.buffer_size} (mlx)"
    print("running turboquant ...")
    results.append(_run(model, tokenizer, label, args.max_tokens))

    print_table(results, llama_ref=llama_ref)


if __name__ == "__main__":
    main()
