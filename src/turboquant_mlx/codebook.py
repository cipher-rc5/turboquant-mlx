# file: src/turboquant_mlx/codebook.py
# description: Lloyd-Max optimal scalar quantizer for Beta(0.5, 0.5) distribution.
#              Replicates 0xSero/turboquant codebook.py with all CUDA/Triton removed.
#              Runs on CPU via numpy; codebooks are pre-baked into mlx arrays at load time.
# reference: arXiv:2504.19874 Algorithm 1; 0xSero/turboquant/turboquant/codebook.py

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import mlx.core as mx
import numpy as np

# ---------------------------------------------------------------------------
# Beta(0.5, 0.5) PDF helper (arcsine distribution)
# ---------------------------------------------------------------------------

def _beta_half_pdf(x: np.ndarray) -> np.ndarray:
    """
    PDF of Beta(0.5, 0.5): f(x) = 1 / (pi * sqrt(x * (1 - x))).
    Numerically stable on (0, 1).
    """
    eps = 1e-12
    x = np.clip(x, eps, 1 - eps)
    return 1.0 / (math.pi * np.sqrt(x * (1.0 - x)))


# ---------------------------------------------------------------------------
# Lloyd-Max iteration
# ---------------------------------------------------------------------------

def lloyd_max(
    n_bins: int,
    n_iter: int = 500,
    n_grid: int = 100_000,
    rng_seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run Lloyd-Max quantizer design for Beta(0.5, 0.5) on [0, 1].

    Returns
    -------
    boundaries : np.ndarray shape (n_bins + 1,)
    centroids  : np.ndarray shape (n_bins,)
    """
    rng = np.random.default_rng(rng_seed)

    # uniform grid for integration approximation
    x_grid = np.linspace(1e-6, 1 - 1e-6, n_grid)
    pdf = _beta_half_pdf(x_grid)
    pdf /= pdf.sum()

    # initialise boundaries uniformly
    boundaries = np.linspace(0.0, 1.0, n_bins + 1)

    for _ in range(n_iter):
        # centroid step: conditional expectation in each bin
        centroids = np.empty(n_bins)
        for k in range(n_bins):
            lo, hi = boundaries[k], boundaries[k + 1]
            mask = (x_grid >= lo) & (x_grid < hi)
            w = pdf[mask]
            if w.sum() < 1e-30:
                centroids[k] = (lo + hi) / 2.0
            else:
                centroids[k] = (x_grid[mask] * w).sum() / w.sum()

        # boundary step: Voronoi midpoints
        new_boundaries = np.empty_like(boundaries)
        new_boundaries[0] = 0.0
        new_boundaries[-1] = 1.0
        for k in range(1, n_bins):
            new_boundaries[k] = (centroids[k - 1] + centroids[k]) / 2.0

        if np.max(np.abs(new_boundaries - boundaries)) < 1e-9:
            break
        boundaries = new_boundaries

    centroids = np.clip(centroids, 0.0, 1.0)
    return boundaries, centroids


# ---------------------------------------------------------------------------
# Codebook cache (in-memory singleton)
# ---------------------------------------------------------------------------

_CACHE: dict[tuple[int, int], tuple[mx.array, mx.array]] = {}

_CODEBOOK_DIR = Path(__file__).parent / "codebooks"


def get_codebook(head_dim: int, bits: int) -> tuple[mx.array, mx.array]:
    """
    Return (boundaries, centroids) as mlx float16 arrays.

    Loads from disk if a JSON file exists at codebooks/d{head_dim}_b{bits}.json,
    otherwise runs Lloyd-Max and caches to disk.

    Parameters
    ----------
    head_dim : int   - dimension of each key/value head (typically 64 or 128)
    bits     : int   - quantization bits (2, 3, or 4)
    """
    key = (head_dim, bits)
    if key in _CACHE:
        return _CACHE[key]

    n_bins = 2 ** bits
    disk_path = _CODEBOOK_DIR / f"d{head_dim}_b{bits}.json"

    if disk_path.exists():
        payload = json.loads(disk_path.read_text())
        boundaries = np.array(payload["boundaries"], dtype=np.float32)
        centroids = np.array(payload["centroids"], dtype=np.float32)
    else:
        boundaries, centroids = lloyd_max(n_bins)
        _CODEBOOK_DIR.mkdir(parents=True, exist_ok=True)
        disk_path.write_text(json.dumps({
            "head_dim": head_dim,
            "bits": bits,
            "boundaries": boundaries.tolist(),
            "centroids": centroids.tolist(),
        }, indent=2))

    # convert to half precision mlx arrays
    b_mx = mx.array(boundaries.astype(np.float16))
    c_mx = mx.array(centroids.astype(np.float16))
    _CACHE[key] = (b_mx, c_mx)
    return b_mx, c_mx


# ---------------------------------------------------------------------------
# Quantize / dequantize (pure mlx, no Metal kernel required)
# ---------------------------------------------------------------------------

def quantize_with_codebook(x: mx.array, boundaries: mx.array) -> mx.array:
    """
    Scalar quantization: map each element of x to a bin index.

    Parameters
    ----------
    x          : (..., D) float16 in [0, 1]
    boundaries : (n_bins + 1,) float16

    Returns
    -------
    indices    : (..., D) uint8
    """
    # bucketize: count how many boundaries each element exceeds
    # boundaries shape (B+1,), x shape (..., D)
    # expand for broadcasting: (..., D, 1) vs (B+1,)
    x_exp = x[..., None]                    # (..., D, 1)
    gt = (x_exp > boundaries[:-1]).sum(axis=-1) - 1  # (..., D)
    gt = mx.clip(gt, 0, boundaries.shape[0] - 2)
    return gt.astype(mx.uint8)


def dequantize_with_codebook(indices: mx.array, centroids: mx.array) -> mx.array:
    """
    Map bin indices back to centroid values.

    Parameters
    ----------
    indices   : (..., D) uint8
    centroids : (n_bins,) float16

    Returns
    -------
    x_hat     : (..., D) float16
    """
    return centroids[indices.astype(mx.int32)]


# ---------------------------------------------------------------------------
# Bit-packing helpers
# ---------------------------------------------------------------------------

# Per-(bits, D) plan caches: precomputed gather/scatter tables that turn the
# pack/unpack loops into a handful of vectorised mlx ops.  Plans are pure
# integer constants so they live on-GPU and never trigger a CPU sync.
_PACK_PLAN: dict[tuple[int, int], dict[str, mx.array]] = {}
_UNPACK_PLAN: dict[tuple[int, int], dict[str, mx.array]] = {}


def _pack_plan(bits: int, d: int) -> dict[str, mx.array]:
    key = (bits, d)
    plan = _PACK_PLAN.get(key)
    if plan is not None:
        return plan

    packed_d = math.ceil(d * bits / 8)
    # For each of the (up to 2) target bytes per source index, record the
    # destination byte and shift amount.  Out-of-range bytes get a sentinel
    # destination so we can mask their contribution to zero.
    lo_byte = np.zeros(d, dtype=np.int32)
    lo_shift = np.zeros(d, dtype=np.int32)
    hi_byte = np.zeros(d, dtype=np.int32)
    hi_shift = np.zeros(d, dtype=np.int32)
    hi_active = np.zeros(d, dtype=np.int32)

    for i in range(d):
        bit_off = i * bits
        lo_byte[i] = bit_off // 8
        lo_shift[i] = bit_off % 8
        bits_in_lo = min(bits, 8 - lo_shift[i])
        if bits_in_lo < bits:
            hi_byte[i] = lo_byte[i] + 1
            hi_shift[i] = bits_in_lo
            hi_active[i] = 1
        else:
            hi_byte[i] = 0  # unused; masked out via hi_active

    plan = {
        "packed_d": mx.array(packed_d, dtype=mx.int32),
        "lo_byte": mx.array(lo_byte),
        "lo_shift": mx.array(lo_shift.astype(np.uint32)),
        "hi_byte": mx.array(hi_byte),
        "hi_shift": mx.array(hi_shift.astype(np.uint32)),
        "hi_active": mx.array(hi_active.astype(np.uint32)),
    }
    _PACK_PLAN[key] = plan
    return plan


def _unpack_plan(bits: int, d: int) -> dict[str, mx.array]:
    key = (bits, d)
    plan = _UNPACK_PLAN.get(key)
    if plan is not None:
        return plan

    lo_byte = np.zeros(d, dtype=np.int32)
    lo_shift = np.zeros(d, dtype=np.int32)
    lo_mask = np.zeros(d, dtype=np.uint32)
    hi_byte = np.zeros(d, dtype=np.int32)
    hi_shift = np.zeros(d, dtype=np.int32)
    hi_active = np.zeros(d, dtype=np.uint32)

    for i in range(d):
        bit_off = i * bits
        lo_byte[i] = bit_off // 8
        lo_shift[i] = bit_off % 8
        bits_in_lo = min(bits, 8 - lo_shift[i])
        lo_mask[i] = (1 << bits_in_lo) - 1
        if bits_in_lo < bits:
            hi_byte[i] = lo_byte[i] + 1
            hi_shift[i] = bits_in_lo
            hi_active[i] = 1

    plan = {
        "lo_byte": mx.array(lo_byte),
        "lo_shift": mx.array(lo_shift.astype(np.uint32)),
        "lo_mask": mx.array(lo_mask),
        "hi_byte": mx.array(hi_byte),
        "hi_shift": mx.array(hi_shift.astype(np.uint32)),
        "hi_active": mx.array(hi_active),
        "out_mask": mx.array(np.uint32((1 << bits) - 1)),
    }
    _UNPACK_PLAN[key] = plan
    return plan


def pack_bits(indices: mx.array, bits: int) -> mx.array:
    """
    Tightly pack uint8 indices (values in [0, 2^bits)) across byte boundaries.

    Pure-MLX implementation: scatter-adds into the target byte array using
    precomputed per-coordinate destination/shift tables.  No CPU sync.

    Returns uint8 array of shape (..., ceil(D * bits / 8)).
    """
    orig_shape = indices.shape
    d = orig_shape[-1]
    plan = _pack_plan(bits, d)
    packed_d = math.ceil(d * bits / 8)

    flat = indices.reshape(-1, d).astype(mx.uint32)        # (N, D)
    n = flat.shape[0]

    # Low and high contributions per (token, src_index), as 32-bit ints.
    lo_contrib = mx.left_shift(flat, plan["lo_shift"])     # (N, D)
    hi_contrib = mx.right_shift(flat, plan["hi_shift"]) * plan["hi_active"]

    # Scatter both contributions into a (N, packed_d) accumulator using
    # mx.add.at-style indexing.  We expand to (N, D) → bin via index_add by
    # building a one-hot scatter through `mx.scatter_add` on a flat (N*packed_d)
    # vector.
    out = mx.zeros((n, packed_d), dtype=mx.uint32)

    row_idx = mx.arange(n, dtype=mx.int32)[:, None]        # (N, 1)
    # scatter low bytes
    out = _scatter_or_row(out, row_idx, plan["lo_byte"], lo_contrib, packed_d)
    # scatter high bytes (only where hi_active==1; contrib already gated)
    out = _scatter_or_row(out, row_idx, plan["hi_byte"], hi_contrib, packed_d)

    out_u8 = (out & 0xFF).astype(mx.uint8)
    new_shape = orig_shape[:-1] + (packed_d,)
    return out_u8.reshape(new_shape)


def _scatter_or_row(
    acc: mx.array,
    row_idx: mx.array,
    col_idx: mx.array,
    contrib: mx.array,
    packed_d: int,
) -> mx.array:
    """
    Add `contrib[n, i]` into `acc[n, col_idx[i]]` for every (n, i).

    Implemented as a dense (D, packed_d) one-hot bridge so it stays a single
    matmul instead of a Python scatter loop:
        acc[n, j] += sum_i contrib[n, i] * onehot[i, j]
    where onehot[i, j] = 1 iff col_idx[i] == j.

    MLX matmul requires floating dtypes; per-byte values stay below 2^16 with
    at most two collisions, so float32 represents every intermediate exactly.
    """
    cols = mx.arange(packed_d, dtype=col_idx.dtype)[None, :]    # (1, packed_d)
    onehot = (col_idx[:, None] == cols).astype(mx.float32)       # (D, packed_d)
    scattered = (contrib.astype(mx.float32) @ onehot).astype(mx.uint32)
    return acc + scattered


def unpack_bits(packed: mx.array, bits: int, original_dim: int) -> mx.array:
    """
    Inverse of pack_bits.  Pure-MLX, no CPU sync.

    Returns uint8 array of shape (..., original_dim).
    """
    orig_shape = packed.shape
    plan = _unpack_plan(bits, original_dim)

    flat = packed.reshape(-1, orig_shape[-1]).astype(mx.uint32)  # (N, packed_d)

    # Gather low and high bytes per output index.
    lo = mx.take(flat, plan["lo_byte"], axis=-1)                  # (N, D)
    hi = mx.take(flat, plan["hi_byte"], axis=-1)                  # (N, D)

    lo_val = mx.right_shift(lo, plan["lo_shift"]) & plan["lo_mask"]
    hi_val = mx.left_shift(hi, plan["hi_shift"]) * plan["hi_active"]

    out = (lo_val | hi_val) & plan["out_mask"]
    new_shape = orig_shape[:-1] + (original_dim,)
    return out.astype(mx.uint8).reshape(new_shape)


# ---------------------------------------------------------------------------
# CLI entry point (pre-generate codebooks)
# ---------------------------------------------------------------------------

def cli_main() -> None:
    parser = argparse.ArgumentParser(description="Pre-generate TurboQuant Lloyd-Max codebooks")
    parser.add_argument("--dims",  nargs="+", type=int, default=[64, 128, 256])
    parser.add_argument("--bits",  nargs="+", type=int, default=[2, 3, 4])
    parser.add_argument("--iter",  type=int, default=500)
    args = parser.parse_args()

    for d in args.dims:
        for b in args.bits:
            print(f"generating codebook: dim={d} bits={b} ...")
            get_codebook(d, b)
    print("done.")


if __name__ == "__main__":
    cli_main()
