# file: proof.py
# description: A/B benchmark — baseline mlx-lm vs TurboQuant KV cache on Gemma 4 31B.
#              Reports prefill t/s and generation t/s separately to match llama-cli output.
#              Sampling follows Gemma 4 official recommendation: temp=1.0, top_p=0.95, top_k=64.
# reference: 0xSero/turboquant/proof.py (original CUDA/vLLM version)
#            https://huggingface.co/mlx-community/gemma-4-31b-it-4bit

from __future__ import annotations

import argparse
import math
import subprocess

import mlx.core as mx
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
        "prompt":       prompt,
        "text":         text,
        "ppl":          None,
        "hidden_cos":   None,
        "preview":      text[:160].replace("\n", " "),
    }


def print_table(results: list[dict], llama_ref: dict | None = None) -> None:
    header = f"{'mode':<44} {'prompt t/s':>12} {'gen t/s':>10} {'tokens':>8} {'ppl':>10} {'hcos':>8}"
    sep = "-" * len(header)
    print("\n" + "=" * len(header))
    print(header)
    print(sep)
    if llama_ref:
        print(
            f"{'[llama-cli ref] gemma-4-26B-A4B Q8_0':<44} "
            f"{llama_ref['prompt_tps']:>12.1f} "
            f"{llama_ref['gen_tps']:>10.1f} "
            f"{'256':>8} "
            f"{'-':>10} "
            f"{'-':>8}"
        )
        print(sep)
    for r in results:
        print(
            f"{r['mode']:<44} "
            f"{r['prompt_tps']:>12.1f} "
            f"{r['gen_tps']:>10.1f} "
            f"{r['gen_tokens']:>8} "
            f"{_fmt_metric(r['ppl']):>10} "
            f"{_fmt_metric(r['hidden_cos']):>8}"
        )
    print("=" * len(header))
    print()
    for r in results:
        print(f"[{r['mode']}]\n  {r['preview']}\n")


def _fmt_metric(value) -> str:
    if value is None:
        return "-"
    return f"{value:.3f}"


def _perplexity(model, tokenizer, prompt: str, generated: str) -> float:
    prompt_tokens = tokenizer.encode(prompt)
    gen_tokens = tokenizer.encode(generated)
    if not gen_tokens:
        return float("inf")

    input_tokens = prompt_tokens + gen_tokens[:-1]
    logits = model(mx.array([input_tokens]))
    start = len(prompt_tokens) - 1
    end = start + len(gen_tokens)
    logits = logits[:, start:end, :]
    log_probs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    targets = mx.array(gen_tokens, dtype=mx.int32)[None, :, None]
    token_log_probs = mx.take_along_axis(log_probs, targets, axis=-1).squeeze(-1)
    return math.exp(float((-token_log_probs.mean()).item()))


def _hidden_cosine(model, tokenizer, prompt: str, generated: str, max_steps: int = 16) -> float | None:
    hidden_model = _hidden_model(model)
    if hidden_model is None or not hasattr(model, "_turboquant_original_make_cache"):
        return None
    original_make_cache = model._turboquant_original_make_cache
    if original_make_cache is None:
        return None

    prompt_tokens = tokenizer.encode(prompt)
    gen_tokens = tokenizer.encode(generated)
    if not prompt_tokens or not gen_tokens:
        return None

    baseline_cache = original_make_cache()
    tq_cache = model.make_cache()
    prompt_arr = mx.array([prompt_tokens])

    base_h = hidden_model(prompt_arr, cache=baseline_cache)
    tq_h = hidden_model(prompt_arr, cache=tq_cache)
    cosines = [_cosine(base_h[:, -1, :], tq_h[:, -1, :])]

    for token in gen_tokens[: max_steps - 1]:
        token_arr = mx.array([[token]])
        base_h = hidden_model(token_arr, cache=baseline_cache)
        tq_h = hidden_model(token_arr, cache=tq_cache)
        cosines.append(_cosine(base_h[:, -1, :], tq_h[:, -1, :]))

    return float(sum(cosines) / len(cosines))


def _hidden_model(model):
    language_model = model["language_model"] if "language_model" in model else model
    if "model" not in language_model:
        return None
    return language_model["model"]


def _cosine(a: mx.array, b: mx.array) -> float:
    a = a.reshape(-1).astype(mx.float32)
    b = b.reshape(-1).astype(mx.float32)
    denom = mx.linalg.norm(a) * mx.linalg.norm(b)
    return float((mx.sum(a * b) / denom).item())


def main() -> None:
    result = subprocess.run(
        ["uv", "run", "pytest", "tests/test_lossless_parity.py", "-v"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print("ABORT: lossless parity test failed. Fix integration before benchmarking.")
        print(result.stdout)
        print(result.stderr)
        raise SystemExit(1)

    parser = argparse.ArgumentParser(
        description="TurboQuant + Gemma 4 31B text-only A/B benchmark (mlx-lm)"
    )
    parser.add_argument("--model",       type=str, default=DEFAULT_MODEL)
    parser.add_argument("--max-tokens",  type=int, default=256)
    parser.add_argument("--key-bits",    type=int, default=4)
    parser.add_argument("--value-bits",  type=int, default=4)
    parser.add_argument("--use-qjl",     action="store_true")
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
        use_qjl=args.use_qjl,
        buffer_size=args.buffer_size,
    )
    qjl_label = " qjl" if args.use_qjl else " no-qjl"
    label = f"turboquant k{args.key_bits}v{args.value_bits} buf{args.buffer_size}{qjl_label} (mlx)"
    print("running turboquant ...")
    results.append(_run(model, tokenizer, label, args.max_tokens))

    print("computing quality metrics ...")
    for r in results:
        r["ppl"] = _perplexity(model, tokenizer, r["prompt"], r["text"])
    results[0]["hidden_cos"] = 1.0
    results[1]["hidden_cos"] = _hidden_cosine(
        model,
        tokenizer,
        results[1]["prompt"],
        results[1]["text"],
    )

    print_table(results, llama_ref=llama_ref)


if __name__ == "__main__":
    main()
