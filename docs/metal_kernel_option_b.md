# Metal Kernel Option B: Fused Decompress + Unrotate + Norm

**Target codebase**: `cipher-rc5/turboquant-mlx` (already remediated; current decode is correct but slow at long context).
**Status**: Decode generation throughput drops 38 percent going from 256 to 1024 tokens (32.2 -> 13.4 t/s on Gemma 4 31B 4-bit). Baseline mlx-lm only drops 4 percent over the same range.
**Goal**: Close most of that gap by fusing the per-step KV-cache decompression pipeline into one MLX custom Metal kernel.

This document is self-contained. Do not assume access to a prior conversation. All file paths, function names, and line numbers below refer to the working tree of the repository at the time this brief was written. Read those files before changing them; they may have moved.

---

## Section 0: Read this before changing any code

Three pieces of context that drive the design.

1. **mlx-lm still owns attention.** The Section 4 remediation already moved this codebase to a cache-adapter architecture: `src/turboquant_mlx/mlx_cache_adapter.py::TurboQuantKVCache` exposes `.keys` and `.values` properties; mlx-lm calls those and runs its own fused `scaled_dot_product_attention`. Do not break that contract. The kernel produces full un-rotated, norm-restored key tensors that are bit-identical (within fp16 tolerance) to what `TurboQuantMSE.decompress` produces today. Nothing downstream of `_materialise_kv` should need to change.
2. **The bottleneck is `_materialise_kv`, not SDPA.** It is called every decode step and grows linearly in the size of the compressed cache. The MLX baseline does not have this term, which is why baseline t/s is roughly flat across context lengths.
3. **MLX exposes custom Metal kernels via `mx.fast.metal_kernel`.** Read its docstring and at least one example in the mlx repository before writing kernel source. Do not invent argument-passing conventions; follow the documented contract exactly. Reference: `python -c "import mlx.core; help(mlx.core.fast.metal_kernel)"` and the mlx examples directory in the user's mlx checkout.

---

## Section 1: Scope of this kernel

In scope (Option B - fused decompress + unrotate + norm):

- Replace, end to end, the work currently done by `TurboQuantMSE.decompress` in `src/turboquant_mlx/quantizer.py` for **compressed key batches only**.
- The kernel takes packed bit-quantized indices, the codebook centroid table, the per-token rotation inverse operand (the orthogonal matrix Q), the per-token L2 norms, and produces the final un-rotated, norm-restored key tensor `(T, head_dim) float16`.

Out of scope:

- Value decompression. `TurboQuantProd` group quantization is separate; leave it on the existing MLX path.
- Attention scoring. Scoring stays inside mlx-lm's `scaled_dot_product_attention`. Do not write an attention kernel.
- Prefill compression. Prefill goes through standard mlx-lm attention; no kernel needed there.
- Multi-batch. Current cache adapter rejects `batch_size > 1` (see `mlx_cache_adapter.py::update_and_fetch`); preserve that restriction.

---

## Section 2: Mathematics the kernel computes

Notation, fixed for the rest of this document:

- `T` - number of tokens in a single compressed batch (the value of `n * K` in `TurboQuantLayerCache._flush`, where `n` is real tokens and `K` is `n_kv_heads`).
- `D` - head dimension (256 or 512 for Gemma 4 31B 4-bit; read from `key_quantizer.head_dim`).
- `bits` - quantization bits per coordinate (2, 3, or 4 in practice; 8-bit also supported by the existing path and must be supported here).
- `n_bins = 2**bits` - codebook size; centroid table dimension.
- `packed_D = ceil(D * bits / 8)` - byte width of one packed row.
- `inv_sqrt_d = 1.0 / sqrt(D)` - precomputable host-side scalar.

For each token row `t` in `0..T`, the existing `decompress` (see `quantizer.py:159-184`) computes:

```
indices[t, i] = unpack(packed[t], i, bits)             for i in 0..D
k_hat_scaled[t, i] = centroids[indices[t, i]]          # codebook lookup
k_hat_rot[t, i] = k_hat_scaled[t, i] * inv_sqrt_d      # undo sqrt(d) scaling
k_unit[t, j] = sum over i of k_hat_rot[t, i] * Q[i, j] # un-rotate (k_hat_rot @ Q)
out[t, j] = k_unit[t, j] * norms[t]                    # restore L2 norm
```

The kernel fuses all four steps into one launch.

Bit-extract for `unpack(packed_row, i, bits)`, lifted from `src/turboquant_mlx/codebook.py::_unpack_plan` and verified against the test suite:

```
bit_off    = i * bits
lo_byte    = bit_off / 8
lo_shift   = bit_off mod 8
bits_in_lo = min(bits, 8 - lo_shift)
lo_mask    = (1 << bits_in_lo) - 1
lo_val     = (packed_row[lo_byte] >> lo_shift) & lo_mask

# if bits_in_lo < bits, we straddle a byte boundary
hi_byte    = lo_byte + 1
hi_shift   = bits_in_lo
hi_active  = (bits_in_lo < bits) ? 1 : 0
hi_val     = (packed_row[hi_byte] << hi_shift) * hi_active

index      = (lo_val | hi_val) & ((1 << bits) - 1)
```

The Python implementation in `codebook.py::unpack_bits` is the reference. The kernel must produce identical indices for every legal input.

---

## Section 3: Kernel signature and specialization

Specialize per `(bits, head_dim)` pair. For Gemma 4 31B 4-bit, the active combinations are:

```
key kernel:   (bits=3, D=256), (bits=3, D=512),
              (bits=4, D=256), (bits=4, D=512),
              (bits=8, D=256), (bits=8, D=512)
```

Do **not** generate value-decompression kernels in this pass; they are out of scope (see Section 1).

Specialization rationale: known `bits` and `D` allow the Metal compiler to unroll the inner bit-extract and to size tile dimensions at compile time. With only six combinations and one specialization compiled lazily at first use per process, the cost is negligible.

Kernel construction site (Python-side): a new module `src/turboquant_mlx/metal_kernels.py` that exposes:

```python
def get_decompress_kernel(bits: int, head_dim: int):
    """Return a compiled mx.fast.metal_kernel specialized to (bits, head_dim).
    Cached in a module-level dict keyed by (bits, head_dim).
    Raises RuntimeError on non-Metal platforms; callers must check availability
    via has_metal_kernel(bits, head_dim) and fall back to the MLX path on False.
    """
```

Kernel call signature (Python wrapper, returns final `(T, D) float16`):

```python
def decompress_packed_keys(
    packed: mx.array,        # (T, packed_D) uint8
    centroids: mx.array,     # (n_bins,)    float16
    Q: mx.array,             # (D, D)       float16
    norms: mx.array,         # (T,)         float16
    bits: int,
    head_dim: int,
) -> mx.array:               # (T, D)       float16
```

Inside the Python wrapper, compute `inv_sqrt_d = float(1.0 / math.sqrt(head_dim))` and pass it as a kernel constant.

---

## Section 4: Threadgroup layout

Use a standard tiled matmul layout with on-the-fly decode of the left operand.

```
grid           = (ceil(T / TILE_T), ceil(D / TILE_N))
threadgroup    = (TILE_N, TILE_T)        # default: 32 x 8 = 256 threads
TILE_K         = 32                       # inner-K tile width

threadgroup memory:
  Q_tile[TILE_K][TILE_N]      half        # 32 * 32 * 2 = 2 KB
  packed_tile[TILE_T][packed_D] uchar     # packed bits for this row tile

constant memory:
  centroids[n_bins]            half       # 8 to 32 bytes
  norms_chunk[TILE_T]          half       # for this row tile (alternatively
                                          # device-memory load; pick whichever is
                                          # cleanest in the kernel source)

registers / per-thread:
  acc                          half       # one output coordinate per thread
```

Per output element `out[t, j]`, where `t` indexes the row tile and `j` indexes the column tile:

```
acc = 0
for k_tile in range(0, D, TILE_K):
    cooperatively load Q_tile  = Q[k_tile : k_tile + TILE_K, j_block : j_block + TILE_N]
    threadgroup_barrier()

    for kk in range(0, TILE_K):
        i   = k_tile + kk
        idx = extract_bits(packed_tile[t_local], i, bits)
        c   = centroids[idx]
        acc += c * Q_tile[kk, j_local]

    threadgroup_barrier()

acc = acc * inv_sqrt_d * norms[t]
out[t, j] = half(acc)
```

Notes:

- `centroids` is small and read-only; mark it `constant address space` in Metal. The compiler will keep it in fast cache.
- `Q_tile` is the bandwidth bottleneck. Use vectorized loads (`half4` or `half2`) for the cooperative load.
- `packed_tile` is small per row (e.g. 96 bytes at D=256, bits=3). Each row can be staged into threadgroup memory once and re-read across the `TILE_K` loop iterations.
- Hold `inv_sqrt_d * norms[t]` in a per-row scalar after the K loop, so the final multiply is a single fma.
- Choose `TILE_T` such that one threadgroup processes a full row tile of `TILE_T` tokens; this avoids inter-threadgroup synchronization for the per-token norm multiply.

These tile choices are a starting point. Profile and tune; do not ship the first numbers that work.

---

## Section 5: Implementation steps

Do these in order. Each step has a verification gate; do not advance past a failing gate.

### Step 5.1: Add the kernel module skeleton

Create `src/turboquant_mlx/metal_kernels.py` with:

- `has_metal_kernel(bits: int, head_dim: int) -> bool` that returns False on non-Metal platforms and on unsupported `(bits, head_dim)` combinations.
- `get_decompress_kernel(bits, head_dim)` returning a cached `mx.fast.metal_kernel` instance.
- `decompress_packed_keys(...)` Python wrapper described in Section 3.

Gate: `uv run python -c "from turboquant_mlx.metal_kernels import has_metal_kernel; print(has_metal_kernel(3, 256))"` prints True on this Mac.

### Step 5.2: Write and verify the kernel for `(bits=3, head_dim=256)` only

Implement the Metal source as a Python string constant in `metal_kernels.py`. Specialize: hard-code `bits = 3`, `D = 256`, `inv_sqrt_d` baked as a literal `half`.

Add test file `tests/test_metal_kernel.py`:

```python
import math
import mlx.core as mx
import numpy as np
import pytest

from turboquant_mlx.quantizer import TurboQuantMSE
from turboquant_mlx.metal_kernels import has_metal_kernel, decompress_packed_keys

pytestmark = pytest.mark.skipif(
    not has_metal_kernel(3, 256),
    reason="Metal kernel not available on this platform",
)

def test_kernel_matches_reference_bits3_dim256():
    """Fused Metal kernel must match TurboQuantMSE.decompress within fp16 tolerance."""
    T = 64
    D = 256
    rng = np.random.default_rng(0)
    keys = rng.standard_normal((T, D)).astype(np.float32)
    keys /= np.linalg.norm(keys, axis=-1, keepdims=True)
    keys_mx = mx.array(keys.astype(np.float16))

    tq = TurboQuantMSE(head_dim=D, bits=3, use_qjl=False)
    ck = tq.compress(keys_mx)

    ref = tq.decompress(ck)
    out = decompress_packed_keys(
        packed=ck.packed,
        centroids=tq.centroids,
        Q=tq.Q,
        norms=ck.norms,
        bits=3,
        head_dim=D,
    )

    ref_np = np.array(ref.tolist(), dtype=np.float32)
    out_np = np.array(out.tolist(), dtype=np.float32)
    max_abs = float(np.max(np.abs(ref_np - out_np)))
    assert max_abs < 1e-2, f"max abs diff {max_abs:.5f} exceeds 1e-2 tolerance"
```

Gate: `uv run pytest tests/test_metal_kernel.py::test_kernel_matches_reference_bits3_dim256 -v` passes.

If the gate fails:

- Compare a single row from the kernel output and the reference. Look first at unpack: is `indices[0]` identical in both paths? If not, the bit-extract is wrong.
- If indices match but centroid values differ, check that the centroid table is being read with the correct stride; the table is `n_bins` long but Metal may interpret it as a longer buffer if shape is passed wrong.
- If centroid values match but the un-rotation differs, check `Q_tile` loads. Common error: loading transposed Q or off-by-one in tile indexing.
- If the un-rotation is correct but values are still off, the `inv_sqrt_d` scaling or norm multiply has the wrong sign or magnitude.

Do not lower the tolerance to make the test pass. The reference is correct.

### Step 5.3: Extend to the other five `(bits, head_dim)` specializations

Parameterize the kernel source generation. Each specialization is a separate `mx.fast.metal_kernel`; the source string is generated by string-substituting `bits` and `D` into a template.

Extend the test to parameterize over all six pairs:

```python
@pytest.mark.parametrize("bits,head_dim", [
    (3, 256), (3, 512),
    (4, 256), (4, 512),
    (8, 256), (8, 512),
])
def test_kernel_matches_reference(bits, head_dim):
    ...
```

Gate: all six parameterizations pass at `max_abs < 1e-2`.

### Step 5.4: Wire the kernel into `TurboQuantMSE.decompress`

Modify `src/turboquant_mlx/quantizer.py::TurboQuantMSE.decompress` to call the kernel when available and fall back to the existing MLX implementation when not. The fallback must be transparent; the existing tests must still pass on a hypothetical non-Metal machine.

Suggested shape:

```python
def decompress(self, ck: CompressedKey) -> mx.array:
    from .metal_kernels import has_metal_kernel, decompress_packed_keys
    if has_metal_kernel(ck.bits, self.head_dim):
        return decompress_packed_keys(
            packed=ck.packed,
            centroids=self.centroids,
            Q=self.Q,
            norms=ck.norms,
            bits=ck.bits,
            head_dim=self.head_dim,
        )
    return self._decompress_mlx(ck)   # existing implementation moved verbatim

def _decompress_mlx(self, ck: CompressedKey) -> mx.array:
    # body of current decompress, unmodified
    ...
```

Gate: full `uv run pytest tests/` passes with the kernel active, including `test_codebook_roundtrip`, `test_lossless_parity`, and both `test_generation_quality` tests. No test thresholds should be relaxed.

### Step 5.5: Benchmark and document the gain

Re-run the two configurations used to establish the baseline:

```sh
uv run python proof.py --model mlx-community/gemma-4-31b-it-4bit --max-tokens 256  --key-bits 4 --value-bits 4 --buffer-size 128
uv run python proof.py --model mlx-community/gemma-4-31b-it-4bit --max-tokens 1024 --key-bits 4 --value-bits 4 --buffer-size 256
```

Record the new t/s and PPL in `docs/metal_kernel_results.md`. Pre-existing pre-kernel numbers, for comparison:

```
Config                                  prompt   gen    tokens   PPL
baseline mlx 31B-4bit  (256 tok)          77.5   32.2     256   1.154
TQ k4v4 buf128         (256 tok)          64.2   21.6     256   1.246
baseline mlx 31B-4bit (1024 tok)          73.8   30.9    1024   1.176
TQ k4v4 buf256        (1024 tok)          64.1   13.4    1024   1.247
```

Expected post-kernel: the 1024-token TQ row reaches at least 25 t/s. If it does not, do not commit; investigate.

---

## Section 6: Numerical-equivalence tolerances

| Quantity                          | Tolerance vs MLX reference |
|-----------------------------------|----------------------------|
| `max(abs(out - ref))` per element | < 1e-2 (fp16 unit in last) |
| `mean(abs(out - ref))` per row    | < 5e-4                     |
| Per-token cosine similarity       | > 0.999                    |
| `test_lossless_parity` greedy out | exact string match         |

The greedy-output test is the strongest guard. If the kernel produces sequences that diverge from baseline-mlx greedy output at 8 bits with buffer larger than the prompt, the kernel has a numerical bug regardless of what the elementwise comparison says.

---

## Section 7: Validation strategy

Required tests, all must pass with the kernel active:

1. `tests/test_metal_kernel.py::test_kernel_matches_reference[*]` (new, 6 parameterizations).
2. `tests/test_codebook_roundtrip.py::test_unit_vector_roundtrip_preserves_direction[*]` (unchanged thresholds).
3. `tests/test_lossless_parity.py::test_high_bit_no_qjl_matches_baseline` (exact greedy-output match).
4. `tests/test_generation_quality.py::test_4bit_generation_is_coherent_smoke`.
5. `tests/test_generation_quality.py::test_patch_cache_head_dim_matches_attention_layers`.
6. `tests/test_quantizer.py` full suite (10 tests; should be unchanged in behavior).
7. `tests/test_kv_cache.py` full suite (4 tests; should be unchanged).

Run `uv run pytest tests/ -v` end to end before declaring done.

---

## Section 8: Process discipline

1. **One specialization at a time.** Get `(3, 256)` right, then extend. Do not write all six kernel sources before any of them are verified.
2. **Do not relax tolerances.** If a test fails, the kernel is wrong. The reference MLX path was validated against the original 0xSero CUDA implementation; trust it.
3. **Do not modify the public API.** `TurboQuantMSE.decompress` keeps its existing signature. `CompressedKey` keeps its fields. The kernel is an internal optimization.
4. **Do not skip the fallback path.** Some users may run on non-Metal hardware (CI, intel macs, the future). `has_metal_kernel` must return False there and `decompress` must transparently use the MLX path.
5. **Profile before assuming the bottleneck moved.** If after the kernel lands, decode is still well under baseline mlx t/s, the next bottleneck is probably `TurboQuantProd` value decompression. That is out of scope for this pass; report it but do not address it.
6. **No emojis in code or documentation.** Standing project preference.

---

## Section 9: Acceptance criteria

The change is complete when:

1. All tests in Section 7 pass without threshold changes.
2. `proof.py` at `--max-tokens 1024 --key-bits 4 --value-bits 4 --buffer-size 256` reports TurboQuant generation throughput >= 25 t/s, up from the pre-kernel 13.4 t/s.
3. `proof.py` at `--max-tokens 256 --key-bits 4 --value-bits 4 --buffer-size 128` reports TurboQuant generation throughput >= 27 t/s, up from the pre-kernel 21.6 t/s.
4. PPL on both configurations stays at or below the pre-kernel ratio (within 1 percent of the pre-kernel TQ number, since the kernel is numerically equivalent).
5. The change is committed as a single conventional commit: `perf: add fused metal kernel for KV-cache decompression`. The commit body describes the kernel boundary, the specializations, and the measured speedup. No co-author attribution.
6. `docs/metal_kernel_results.md` exists and documents the pre- and post-kernel numbers in a table identical in format to Section 5.5's table.

---

## Section 10: Out of scope (do not attempt)

- A value-decompression kernel for `TurboQuantProd`. Same fusion idea applies, but the math is different (group quantization, not codebook + rotation) and the speedup is smaller per step. Address in a follow-up pass.
- A score-on-packed-bits attention kernel that bypasses materialization entirely (Option C from the design discussion). Breaks the cache-adapter contract and is not necessary to close the measured gap.
- Multi-batch decode. Cache adapter is single-batch by design.
- Re-tuning the existing MLX bit-pack `pack_bits`. Encode is not on the hot path; decode is.
- Generalizing the kernel to non-Gemma head dimensions in this pass. The current six specializations cover every layer of every Gemma 4 31B variant.

---

## Appendix A: Reading list before starting

- `python -c "import mlx.core; help(mlx.core.fast.metal_kernel)"` - the actual API contract.
- Source of `mx.fast.metal_kernel` in the mlx repo (find via `python -c "import mlx.core.fast; print(mlx.core.fast.__file__)"`).
- At least one shipped example kernel in the mlx repo (search for `metal_kernel(` under the mlx source tree).
- `src/turboquant_mlx/quantizer.py::TurboQuantMSE.decompress` - the function this kernel replaces.
- `src/turboquant_mlx/codebook.py::unpack_bits` and `_unpack_plan` - the reference bit-extract logic. The kernel must reproduce these indices exactly.
- `src/turboquant_mlx/mlx_cache_adapter.py` - the consumer of `_materialise_kv`; understand the call pattern so you do not break it.

## Appendix B: Files this work touches

New:

- `src/turboquant_mlx/metal_kernels.py`
- `tests/test_metal_kernel.py`
- `docs/metal_kernel_results.md`

Modified:

- `src/turboquant_mlx/quantizer.py` (only `TurboQuantMSE.decompress`; move the existing body into `_decompress_mlx`, add the kernel dispatch and fallback).

Untouched (do not modify in this pass):

- `src/turboquant_mlx/kv_cache.py`
- `src/turboquant_mlx/patch.py`
- `src/turboquant_mlx/mlx_cache_adapter.py`
- `src/turboquant_mlx/rotation.py`
- `src/turboquant_mlx/codebook.py`
- `src/turboquant_mlx/config.py`
- `proof.py`
- `README.md`

If you find yourself needing to modify a file outside the "Modified" list, stop and report why; the design has drifted.
