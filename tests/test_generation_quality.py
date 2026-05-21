import mlx_lm
from mlx_lm.generate import generate
from mlx_lm.sample_utils import make_sampler

from turboquant_mlx import patch_model
from turboquant_mlx.kv_cache import _attention_layers, _attn_module


MODEL_ID = "mlx-community/gemma-4-31b-it-4bit"


def _max_run_length(tokens: list[int]) -> int:
    max_run = 0
    current = 0
    previous = None
    for token in tokens:
        if token == previous:
            current += 1
        else:
            previous = token
            current = 1
        max_run = max(max_run, current)
    return max_run


def test_4bit_generation_is_coherent_smoke():
    model, tok = mlx_lm.load(MODEL_ID)
    patch_model(model, key_bits=4, value_bits=4, buffer_size=128, use_qjl=False)
    prompt = tok.apply_chat_template(
        [{"role": "user", "content": "Write one concise paragraph about why Paris is historically important."}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    output = generate(
        model,
        tok,
        prompt=prompt,
        max_tokens=64,
        sampler=make_sampler(temp=0.0),
    )
    tokens = tok.encode(tok.decode(tok.encode(output)))

    assert len(set(tokens)) >= 10, f"too few distinct tokens: {output!r}"
    assert _max_run_length(tokens) <= 5, f"token repetition run too long: {output!r}"


def test_patch_cache_head_dim_matches_attention_layers():
    model, _ = mlx_lm.load(MODEL_ID)
    caches = patch_model(model, key_bits=3, value_bits=2, buffer_size=128, use_qjl=False)
    layers = _attention_layers(model)

    assert len(caches) == len(layers)
    for cache, layer in zip(caches, layers):
        attn = _attn_module(layer)
        assert cache.tq_cache.key_quantizer.head_dim == attn.head_dim
