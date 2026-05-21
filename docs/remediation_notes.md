# Remediation Notes

TurboQuant-MLX originally generated gibberish because several independent issues all pushed attention away from the baseline `mlx-lm` behavior.

## Root Causes

1. The largest mathematical issue was the codebook domain. Rotated unit-vector coordinates were mapped from `[-1, 1]` to `[0, 1]` and quantized with a Beta codebook, even though each coordinate is concentrated around `0` at scale `1/sqrt(d)`. The fix is to quantize `sqrt(d) * coord` with a Gaussian Lloyd-Max codebook and decode without renormalizing.
2. The largest architectural issue was reimplementing Gemma attention in Python. That path had to mirror sliding masks, RoPE offsets, shared KV behavior, and scaling exactly. It has been replaced with a `KVCache` adapter so `mlx-lm` still owns attention math.
3. Head dimensions were guessed in multiple places. Gemma exposes per-layer `head_dim`, and this model has mixed dimensions: 50 layers at 256 and 10 layers at 512. The patch path now fails loudly if `.head_dim` is unavailable.
4. QJL residual correction was enabled by default despite being high variance for KV-cache attention. It is now opt-in only, with the scale corrected for explicit experiments.

## Guardrails Added

- Gaussian codebook roundtrip tests verify unit-vector direction preservation.
- Lossless parity verifies cache injection matches baseline greedy decode when no flushing occurs.
- Generation coherence and per-layer head-dim tests catch integration regressions.
- `proof.py` now aborts if lossless parity fails before benchmarking.
