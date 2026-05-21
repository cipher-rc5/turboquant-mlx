"""
Fused Metal kernels for TurboQuant KV-cache decompression.

This module implements Option B from `docs/metal_kernel_option_b.md`:
a single custom Metal kernel that fuses unpack -> codebook lookup ->
inv-sqrt(d) scaling -> un-rotation (matmul by Q) -> norm restoration,
replacing four separate MLX passes in `TurboQuantMSE.decompress` with one
kernel launch.

In scope: key decompression for the (bits, head_dim) pairs that occur in
Gemma 4 31B 4-bit. Value decompression and prefill stay on the MLX path.

The kernel is bit-equivalent to the Python reference in
`codebook.py::unpack_bits` and `quantizer.py::TurboQuantMSE.decompress`
modulo fp16 rounding order.

Specialization strategy: per the spec (Section 5.3), the source string is
generated per (bits, head_dim) with the constants substituted as literals.
This sidesteps any concern about constexpr-from-template-parameter being
valid as a threadgroup array dimension on a given MSL toolchain.
"""

from __future__ import annotations

import mlx.core as mx


SUPPORTED_SPECIALIZATIONS: frozenset[tuple[int, int]] = frozenset(
    {
        (3, 256), (3, 512),
        (4, 256), (4, 512),
        (8, 256), (8, 512),
    }
)

TILE_T = 8
TILE_N = 32
TILE_K = 32


def _kernel_source(bits: int, head_dim: int) -> str:
    """Render the kernel body with `bits` and `head_dim` baked in as literals.

    All compile-time-constant expressions (PACKED_D, OUT_MASK, inv_sqrt_d,
    threadgroup-array dimensions) become plain integer/float literals in
    the emitted MSL, so the kernel compiles even on MSL toolchains that
    are strict about what may sit in a `threadgroup` array bound.
    """
    packed_d = (head_dim * bits + 7) // 8
    out_mask = (1 << bits) - 1
    # bake inv_sqrt_d as a literal half; the source uses it directly.
    inv_sqrt_d = 1.0 / (head_dim ** 0.5)

    return f"""
    constexpr uint BITS      = {bits};
    constexpr uint HEAD_DIM  = {head_dim};
    constexpr uint PACKED_D  = {packed_d};
    constexpr uint OUT_MASK  = {out_mask}u;
    constexpr uint TILE_T    = {TILE_T};
    constexpr uint TILE_N    = {TILE_N};
    constexpr uint TILE_K    = {TILE_K};

    uint t        = thread_position_in_grid.y;
    uint j        = thread_position_in_grid.x;
    uint t_local  = thread_position_in_threadgroup.y;
    uint j_local  = thread_position_in_threadgroup.x;
    uint t_block  = threadgroup_position_in_grid.y;
    uint j_block  = threadgroup_position_in_grid.x;

    uint T_rows = packed_shape[0];

    threadgroup half  Q_tile[{TILE_K}][{TILE_N}];
    threadgroup uchar packed_tile[{TILE_T}][{packed_d}];

    uint lid         = t_local * TILE_N + j_local;
    uint num_threads = TILE_T * TILE_N;

    // Cooperatively stage the packed rows for this row tile into threadgroup
    // memory. Each thread loads strided bytes from the global packed buffer.
    for (uint p = lid; p < TILE_T * PACKED_D; p += num_threads) {{
        uint pt = p / PACKED_D;
        uint pb = p % PACKED_D;
        uint global_t = t_block * TILE_T + pt;
        packed_tile[pt][pb] =
            (global_t < T_rows) ? packed[global_t * PACKED_D + pb] : (uchar)0;
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float acc = 0.0f;

    for (uint k_tile = 0; k_tile < HEAD_DIM; k_tile += TILE_K) {{
        uint col_base = j_block * TILE_N;

        // Cooperatively stage Q[k_tile : k_tile+TILE_K, col_base : col_base+TILE_N].
        for (uint p = lid; p < TILE_K * TILE_N; p += num_threads) {{
            uint pk = p / TILE_N;
            uint pn = p % TILE_N;
            uint global_k = k_tile + pk;
            uint global_n = col_base + pn;
            Q_tile[pk][pn] =
                (global_k < HEAD_DIM && global_n < HEAD_DIM)
                    ? Q[global_k * HEAD_DIM + global_n]
                    : (half)0;
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Inner-K accumulation: decode index from packed bits, look up
        // centroid, multiply by Q tile entry, fma into acc.
        for (uint kk = 0; kk < TILE_K; ++kk) {{
            uint i = k_tile + kk;

            uint bit_off    = i * BITS;
            uint lo_byte    = bit_off >> 3;
            uint lo_shift   = bit_off & 7u;
            uint bits_in_lo = metal::min(BITS, 8u - lo_shift);
            uint lo_mask    = (1u << bits_in_lo) - 1u;

            uint lo     = (uint)packed_tile[t_local][lo_byte];
            uint lo_val = (lo >> lo_shift) & lo_mask;

            uint hi_val = 0u;
            if (bits_in_lo < BITS) {{
                uint hi = (uint)packed_tile[t_local][lo_byte + 1];
                hi_val = (hi << bits_in_lo);
            }}
            uint idx = (lo_val | hi_val) & OUT_MASK;

            float c = (float)centroids[idx];
            float q = (float)Q_tile[kk][j_local];
            acc = metal::fma(c, q, acc);
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}

    if (t < T_rows && j < HEAD_DIM) {{
        // inv_sqrt_d baked at source-render time so the final scale is one
        // fma against the per-token norm.
        float scale = {inv_sqrt_d!r}f * (float)norms[t];
        out[t * HEAD_DIM + j] = (half)(acc * scale);
    }}
"""


_KERNEL_CACHE: dict[tuple[int, int], object] = {}


def _get_kernel(bits: int, head_dim: int) -> object:
    key = (bits, head_dim)
    cached = _KERNEL_CACHE.get(key)
    if cached is not None:
        return cached
    kernel = mx.fast.metal_kernel(
        name=f"turboquant_decompress_keys_b{bits}_d{head_dim}",
        input_names=["packed", "centroids", "Q", "norms"],
        output_names=["out"],
        source=_kernel_source(bits, head_dim),
        ensure_row_contiguous=True,
    )
    _KERNEL_CACHE[key] = kernel
    return kernel


def has_metal_kernel(bits: int, head_dim: int) -> bool:
    """Return True iff a fused decompress kernel is available for this
    `(bits, head_dim)` on the current platform.

    The caller must use this to decide between the kernel and the MLX
    fallback; do not call `decompress_packed_keys` without first checking.
    """
    if (bits, head_dim) not in SUPPORTED_SPECIALIZATIONS:
        return False
    try:
        return bool(mx.metal.is_available())
    except Exception:
        return False


def decompress_packed_keys(
    packed: mx.array,
    centroids: mx.array,
    Q: mx.array,
    norms: mx.array,
    bits: int,
    head_dim: int,
) -> mx.array:
    """Fused decompress + unrotate + norm-restore for a batch of packed keys.

    Parameters
    ----------
    packed    : (T, packed_D) uint8     bit-packed codebook indices.
    centroids : (2**bits,)    float16   Lloyd-Max centroid table.
    Q         : (D, D)        float16   orthogonal rotation matrix.
    norms     : (T,)          float16   per-token L2 norm to restore.
    bits      : int                     2..8; must match `(bits, head_dim)` in
                                        `SUPPORTED_SPECIALIZATIONS`.
    head_dim  : int                     `D`.

    Returns
    -------
    out : (T, D) float16 — un-rotated, norm-restored keys, bit-equivalent
        (to fp16 tolerance) with `TurboQuantMSE._decompress_mlx`.
    """
    if (bits, head_dim) not in SUPPORTED_SPECIALIZATIONS:
        raise RuntimeError(
            f"no metal kernel specialization for (bits={bits}, head_dim={head_dim}); "
            f"caller should fall back to the MLX path"
        )

    T = int(packed.shape[0])

    # Round grid up to a multiple of the threadgroup; the kernel bounds-checks
    # the trailing partial tile against packed_shape[0].
    grid_x = ((head_dim + TILE_N - 1) // TILE_N) * TILE_N
    grid_y = ((T + TILE_T - 1) // TILE_T) * TILE_T

    kernel = _get_kernel(bits, head_dim)
    outputs = kernel(
        inputs=[packed, centroids, Q, norms],
        grid=(grid_x, grid_y, 1),
        threadgroup=(TILE_N, TILE_T, 1),
        output_shapes=[(T, head_dim)],
        output_dtypes=[mx.float16],
    )
    return outputs[0]
