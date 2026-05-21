import mlx_lm
from mlx_lm.generate import generate
from mlx_lm.sample_utils import make_sampler

from turboquant_mlx import patch_model


PROMPT = "The capital of France is"


def test_high_bit_no_qjl_matches_baseline():
    """With no flushing, TurboQuant cache injection should leave greedy decode unchanged."""
    model, tok = mlx_lm.load("mlx-community/gemma-4-31b-it-4bit")
    sampler = make_sampler(temp=0.0)

    baseline = generate(model, tok, prompt=PROMPT, max_tokens=32, sampler=sampler)

    model2, tok2 = mlx_lm.load("mlx-community/gemma-4-31b-it-4bit")
    patch_model(model2, key_bits=8, value_bits=8, buffer_size=4096)
    tq_out = generate(model2, tok2, prompt=PROMPT, max_tokens=32, sampler=sampler)

    assert baseline == tq_out, f"baseline={baseline!r}\ntq={tq_out!r}"
