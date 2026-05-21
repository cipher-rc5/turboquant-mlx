# Fused Metal Kernel: Decompress + Unrotate + Norm — Results

Pre- and post-kernel benchmark numbers for the fused KV-cache decompress
kernel landed per `docs/metal_kernel_option_b.md`. Both rows of each pair
were collected on the same Mac with the same model checkpoint
(`mlx-community/gemma-4-31b-it-4bit`) by running:

```sh
uv run python proof.py --model mlx-community/gemma-4-31b-it-4bit --max-tokens 256  --key-bits 4 --value-bits 4 --buffer-size 128
uv run python proof.py --model mlx-community/gemma-4-31b-it-4bit --max-tokens 1024 --key-bits 4 --value-bits 4 --buffer-size 256
```

## Pre-kernel baseline (from `docs/metal_kernel_option_b.md` Section 5.5)

| Config                                  | prompt | gen   | tokens | PPL   |
|-----------------------------------------|-------:|------:|-------:|------:|
| baseline mlx 31B-4bit  (256 tok)        |   77.5 |  32.2 |    256 | 1.154 |
| TQ k4v4 buf128         (256 tok)        |   64.2 |  21.6 |    256 | 1.246 |
| baseline mlx 31B-4bit (1024 tok)        |   73.8 |  30.9 |   1024 | 1.176 |
| TQ k4v4 buf256        (1024 tok)        |   64.1 |  13.4 |   1024 | 1.247 |

## Post-kernel

| Config                                  | prompt | gen   | tokens | PPL   |
|-----------------------------------------|-------:|------:|-------:|------:|
| baseline mlx 31B-4bit  (256 tok)        |  TBD   | TBD   |    256 | TBD   |
| TQ k4v4 buf128         (256 tok)        |  TBD   | TBD   |    256 | TBD   |
| baseline mlx 31B-4bit (1024 tok)        |  TBD   | TBD   |   1024 | TBD   |
| TQ k4v4 buf256        (1024 tok)        |  TBD   | TBD   |   1024 | TBD   |

Acceptance gates (`docs/metal_kernel_option_b.md` §9):

- TQ 1024-token row must reach generation throughput **>= 25 t/s** (up
  from 13.4).
- TQ 256-token row must reach generation throughput **>= 27 t/s** (up
  from 21.6).
- PPL on both TQ rows must stay within 1% of the pre-kernel TQ numbers,
  since the kernel is numerically equivalent within fp16 tolerance.

## Kernel boundary

- Replaces, end-to-end, the work `TurboQuantMSE.decompress` was doing on
  the compressed-key path (unpack -> codebook lookup -> divide by
  sqrt(d) -> matmul by Q -> per-token norm rescale). Four MLX passes
  become one Metal kernel launch.
- Value decompression (`TurboQuantProd`) is unchanged; it stays on the
  pure-MLX path.
- Attention scoring stays inside mlx-lm's
  `scaled_dot_product_attention` — the cache adapter contract is
  untouched.

## Specializations compiled

| bits | head_dim |
|-----:|---------:|
|    3 |      256 |
|    3 |      512 |
|    4 |      256 |
|    4 |      512 |
|    8 |      256 |
|    8 |      512 |

Compilation is lazy (first call per process) and cached on the kernel
object; the six template instantiations are independent.

## Fallback

`TurboQuantMSE.decompress` checks `metal_kernels.has_metal_kernel(...)`
on every call. On non-Metal platforms or unsupported `(bits, head_dim)`
pairs (e.g. head_dim 160 from earlier experiments) the call routes to
`TurboQuantMSE._decompress_mlx`, which is the pre-kernel body moved
verbatim. The existing `tests/test_quantizer.py` and
`tests/test_kv_cache.py` suites exercise that path.
