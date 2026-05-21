"""
Numerical-equivalence tests for the fused Metal decompress kernel.

Every test compares the kernel output against
`TurboQuantMSE._decompress_mlx`, which is the pure-MLX reference. The
kernel may differ only in fp16 rounding order; the spec sets the tolerance
at 1e-2 absolute.

The whole module skips when no Metal kernel is available (non-Apple-Silicon
CI, future, etc.).
"""

import mlx.core as mx
import numpy as np
import pytest

from turboquant_mlx.metal_kernels import (
    SUPPORTED_SPECIALIZATIONS,
    decompress_packed_keys,
    has_metal_kernel,
)
from turboquant_mlx.quantizer import TurboQuantMSE


pytestmark = pytest.mark.skipif(
    not has_metal_kernel(3, 256),
    reason="Metal kernel not available on this platform",
)


@pytest.mark.parametrize(
    "bits,head_dim",
    sorted(SUPPORTED_SPECIALIZATIONS),
    ids=lambda v: str(v),
)
def test_kernel_matches_reference(bits: int, head_dim: int) -> None:
    """Fused Metal kernel output must match the MLX reference within fp16 tol."""
    T = 64
    rng = np.random.default_rng(0)
    keys = rng.standard_normal((T, head_dim)).astype(np.float32)
    keys /= np.linalg.norm(keys, axis=-1, keepdims=True)
    keys_mx = mx.array(keys.astype(np.float16))

    tq = TurboQuantMSE(head_dim=head_dim, bits=bits, use_qjl=False)
    ck = tq.compress(keys_mx)

    ref = tq._decompress_mlx(ck)
    out = decompress_packed_keys(
        packed=ck.packed,
        centroids=tq.centroids,
        Q=tq.Q,
        norms=ck.norms,
        bits=bits,
        head_dim=head_dim,
    )

    ref_np = np.array(ref.tolist(), dtype=np.float32)
    out_np = np.array(out.tolist(), dtype=np.float32)

    max_abs = float(np.max(np.abs(ref_np - out_np)))
    mean_abs_per_row = float(np.mean(np.abs(ref_np - out_np), axis=-1).max())
    ref_norm = np.linalg.norm(ref_np, axis=-1)
    out_norm = np.linalg.norm(out_np, axis=-1)
    cos = np.sum(ref_np * out_np, axis=-1) / np.maximum(ref_norm * out_norm, 1e-12)
    min_cos = float(cos.min())

    assert max_abs < 1e-2, (
        f"bits={bits} head_dim={head_dim}: max abs diff {max_abs:.5f} > 1e-2"
    )
    assert mean_abs_per_row < 5e-4, (
        f"bits={bits} head_dim={head_dim}: worst-row mean abs diff "
        f"{mean_abs_per_row:.6f} > 5e-4"
    )
    assert min_cos > 0.999, (
        f"bits={bits} head_dim={head_dim}: min per-token cosine "
        f"{min_cos:.6f} <= 0.999"
    )


def test_kernel_dispatch_via_decompress() -> None:
    """`TurboQuantMSE.decompress` must transparently dispatch to the kernel
    when one is available and produce results indistinguishable from
    `_decompress_mlx`."""
    T = 32
    head_dim = 256
    bits = 4
    rng = np.random.default_rng(1)
    keys = rng.standard_normal((T, head_dim)).astype(np.float32)
    keys /= np.linalg.norm(keys, axis=-1, keepdims=True)
    keys_mx = mx.array(keys.astype(np.float16))

    tq = TurboQuantMSE(head_dim=head_dim, bits=bits, use_qjl=False)
    ck = tq.compress(keys_mx)

    via_dispatch = np.array(tq.decompress(ck).tolist(), dtype=np.float32)
    via_mlx = np.array(tq._decompress_mlx(ck).tolist(), dtype=np.float32)

    max_abs = float(np.max(np.abs(via_dispatch - via_mlx)))
    assert max_abs < 1e-2, (
        f"dispatch path disagrees with MLX reference: max abs {max_abs:.5f}"
    )


def test_kernel_handles_short_batches() -> None:
    """T < TILE_T must still produce correct output (partial trailing tile)."""
    head_dim = 256
    bits = 3
    for T in (1, 3, 7, 8, 9, 17):
        rng = np.random.default_rng(100 + T)
        keys = rng.standard_normal((T, head_dim)).astype(np.float32)
        keys /= np.linalg.norm(keys, axis=-1, keepdims=True)
        keys_mx = mx.array(keys.astype(np.float16))

        tq = TurboQuantMSE(head_dim=head_dim, bits=bits, use_qjl=False)
        ck = tq.compress(keys_mx)
        ref = tq._decompress_mlx(ck)
        out = decompress_packed_keys(
            packed=ck.packed,
            centroids=tq.centroids,
            Q=tq.Q,
            norms=ck.norms,
            bits=bits,
            head_dim=head_dim,
        )
        ref_np = np.array(ref.tolist(), dtype=np.float32)
        out_np = np.array(out.tolist(), dtype=np.float32)
        assert out_np.shape == (T, head_dim)
        max_abs = float(np.max(np.abs(ref_np - out_np)))
        assert max_abs < 1e-2, f"T={T}: max abs {max_abs:.5f} > 1e-2"
