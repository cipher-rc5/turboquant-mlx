# file: src/turboquant_mlx/quantizer.py
# description: TurboQuantMSE (Algorithm 1) and TurboQuantProd (Algorithm 2) quantizer
#              pipelines. Direct translation of 0xSero/turboquant quantizer.py; all
#              CUDA/Triton paths replaced with MLX. Operates on float16 mlx arrays.
# reference: arXiv:2504.19874 Algorithms 1 & 2; 0xSero/turboquant/turboquant/quantizer.py

from __future__ import annotations

import math
from dataclasses import dataclass, field

import mlx.core as mx

from .codebook import (
    dequantize_with_codebook,
    get_codebook,
    pack_bits,
    quantize_with_codebook,
    unpack_bits,
)
from .rotation import (
    qjl_decode_correction,
    qjl_encode,
    random_orthogonal,
    random_qjl_matrix,
    rotate,
    rotate_query,
)


# ---------------------------------------------------------------------------
# Compressed key bundle
# ---------------------------------------------------------------------------

@dataclass
class CompressedKey:
    """
    Storage bundle for a batch of compressed keys.

    Attributes
    ----------
    packed          : mlx uint8 array (T, packed_D)  - bit-packed codebook indices
    norms           : mlx float16 array (T,)          - per-token L2 norm before rotation
    qjl_signs       : mlx uint8 array (T, n_sketch)  - QJL residual sign bits (optional)
    residual_norms  : mlx float16 array (T,)          - L2 norm of quantisation residual
    bits            : int                             - bits used for quantisation
    head_dim        : int                             - original head dimension
    """
    packed: mx.array
    norms: mx.array
    qjl_signs: mx.array | None
    residual_norms: mx.array | None
    bits: int
    head_dim: int


@dataclass
class CompressedValue:
    """
    Storage bundle for a batch of compressed values (group quantisation).

    Attributes
    ----------
    packed      : mlx uint8 array (T, packed_D)
    scales      : mlx float16 array (T, n_groups) - per-group scale
    zeros       : mlx float16 array (T, n_groups) - per-group minimum
    bits        : int
    head_dim    : int
    group_size  : int
    """
    packed: mx.array
    scales: mx.array
    zeros: mx.array
    bits: int
    head_dim: int
    group_size: int


# ---------------------------------------------------------------------------
# TurboQuantMSE -- Algorithm 1 (keys, MSE-optimal)
# ---------------------------------------------------------------------------

class TurboQuantMSE:
    """
    Compress keys with:
      1. L2 normalisation + norm storage
      2. Random orthogonal rotation + sqrt(D) scaling -> N(0,1) marginals
      3. Lloyd-Max scalar quantisation per coordinate
      4. Optional QJL residual sketch for unbiased inner-product estimation

    Decode path: rotate query once, lookup centroids, add QJL correction.
    """

    def __init__(
        self,
        head_dim: int,
        bits: int = 3,
        use_qjl: bool = False,
        rotation_seed: int = 42,
        qjl_seed: int = 99,
    ) -> None:
        self.head_dim = head_dim
        self.bits = bits
        self.use_qjl = use_qjl

        self.Q = random_orthogonal(head_dim, seed=rotation_seed)
        self.S = random_qjl_matrix(head_dim, seed=qjl_seed) if use_qjl else None
        self.boundaries, self.centroids = get_codebook(head_dim, bits)

    # ------------------------------------------------------------------
    def compress(self, keys: mx.array) -> CompressedKey:
        """
        Compress a batch of key vectors.

        Parameters
        ----------
        keys : (T, head_dim) float16

        Returns
        -------
        CompressedKey
        """
        # 1. normalise
        norms = mx.linalg.norm(keys, axis=-1, keepdims=True)          # (T, 1)
        norms_scalar = norms.squeeze(-1)                               # (T,)
        safe_norms = mx.where(norms > 1e-8, norms, mx.ones_like(norms))
        k_normed = keys / safe_norms                                   # (T, D)

        # 2. rotate unit vectors, then scale coordinates into the N(0,1) codebook domain
        k_rot = rotate(k_normed, self.Q)                               # (T, D)
        sqrt_d = math.sqrt(self.head_dim)
        k_scaled = mx.clip(k_rot * sqrt_d, -4.0, 4.0)

        # 4. scalar quantise
        indices = quantize_with_codebook(k_scaled, self.boundaries)    # (T, D) uint8
        packed = pack_bits(indices, self.bits)                         # (T, packed_D)

        # 5. QJL residual
        if self.use_qjl:
            k_hat_scaled = dequantize_with_codebook(indices, self.centroids)  # (T, D)
            k_hat_rot = k_hat_scaled / sqrt_d
            residual = k_rot - k_hat_rot                                    # (T, D)
            qjl_signs = qjl_encode(residual, self.S)                       # (T, n_sketch)  # ty:ignore[invalid-argument-type]
            residual_norms = mx.linalg.norm(residual, axis=-1)             # (T,)
        else:
            qjl_signs = None
            residual_norms = None

        return CompressedKey(
            packed=packed,
            norms=norms_scalar,
            qjl_signs=qjl_signs,
            residual_norms=residual_norms,
            bits=self.bits,
            head_dim=self.head_dim,
        )

    # ------------------------------------------------------------------
    def decompress(self, ck: CompressedKey) -> mx.array:
        """
        Reconstruct approximate float16 keys from a CompressedKey.

        Dispatches to the fused Metal kernel when one is available for
        ``(ck.bits, self.head_dim)``; otherwise falls back to the pure-MLX
        implementation. Both paths are numerically equivalent within fp16
        rounding.
        """
        from .metal_kernels import decompress_packed_keys, has_metal_kernel

        if has_metal_kernel(ck.bits, self.head_dim):
            return decompress_packed_keys(
                packed=ck.packed,
                centroids=self.centroids,
                Q=self.Q,
                norms=ck.norms,
                bits=ck.bits,
                head_dim=self.head_dim,
            )
        return self._decompress_mlx(ck)

    # ------------------------------------------------------------------
    def _decompress_mlx(self, ck: CompressedKey) -> mx.array:
        """
        Pure-MLX reference decompress: unpack indices → centroid lookup →
        un-rotate (Q is orthogonal so Q^{-1} = Q^T, and ``rotate(x, Q) =
        x @ Q.T`` so the inverse is ``x_rot @ Q``) → restore original L2 norm.

        Used as the fallback path on non-Metal platforms and as the
        reference oracle by `tests/test_metal_kernel.py`.

        Returns
        -------
        keys : (T, head_dim) float16 — approximate reconstruction of the
            original keys passed to ``compress``.
        """
        from .codebook import dequantize_with_codebook, unpack_bits

        sqrt_d = math.sqrt(self.head_dim)
        indices = unpack_bits(ck.packed, ck.bits, ck.head_dim)         # (T, D)
        k_hat_scaled = dequantize_with_codebook(indices, self.centroids)  # (T, D)
        k_hat_rot = k_hat_scaled / sqrt_d
        k_unit = k_hat_rot @ self.Q
        return (k_unit * ck.norms[:, None]).astype(mx.float16)

    # ------------------------------------------------------------------
    def attention_scores(
        self,
        query: mx.array,
        ck: CompressedKey,
    ) -> mx.array:
        """
        Compute approximate dot-product attention scores without decompressing keys.

        Parameters
        ----------
        query : (n_heads, head_dim) or (1, head_dim) float16
        ck    : CompressedKey with T compressed tokens

        Returns
        -------
        scores : (n_heads, T) or (1, T) float16
        """
        # rotate query into key space
        q_rot = rotate_query(query, self.Q)                   # (..., D)

        # unpack indices -> centroids -> scaled back to rotated unit-vector coordinates
        sqrt_d = math.sqrt(self.head_dim)
        indices = unpack_bits(ck.packed, self.bits, self.head_dim)
        k_hat_scaled = dequantize_with_codebook(indices, self.centroids)  # (T, D)
        k_hat_rot = k_hat_scaled / sqrt_d

        # approximate scores: (q_rot @ k_hat_rot^T) * norms
        scores = q_rot @ k_hat_rot.T                          # (..., T)
        scores = scores * ck.norms[None, :]                   # broadcast norms

        # add QJL correction if present
        if self.use_qjl and ck.qjl_signs is not None and ck.residual_norms is not None:
            n_sketch = self.S.shape[1]  # ty:ignore[unresolved-attribute]
            correction = qjl_decode_correction(q_rot, ck.qjl_signs, self.S)  # ty:ignore[invalid-argument-type]
            # QJL estimator: corr = sum_j sign(r . s_j) * (q . s_j) with unit-norm s_j on S^{d-1}.
            # E[sign(r.s)(q.s)] = sqrt(2/pi) * <q, r/||r||> per column.
            # Summing m columns: corr ~ m * sqrt(2/pi) * <q, r> / ||r||
            # Therefore <q, r> = corr * ||r|| * sqrt(pi/2) / m
            scale = ck.residual_norms * (math.sqrt(math.pi / 2) / n_sketch)
            scores = scores + correction * scale

        return scores


# ---------------------------------------------------------------------------
# TurboQuantProd -- Algorithm 2 (values, group quantisation)
# ---------------------------------------------------------------------------

class TurboQuantProd:
    """
    Compress values with asymmetric group quantisation (min-max per group).
    This is cheaper than key compression and is used for value vectors.

    No rotation is applied -- values are summed with softmax weights, so
    the inner-product-preserving property of the key path is not needed.
    """

    def __init__(
        self,
        head_dim: int,
        bits: int = 2,
        group_size: int = 32,
    ) -> None:
        self.head_dim = head_dim
        self.bits = bits
        self.group_size = group_size
        self.n_bins = 2 ** bits
        import math
        self.n_groups = math.ceil(head_dim / group_size)

    # ------------------------------------------------------------------
    def compress(self, values: mx.array) -> CompressedValue:
        """
        Compress value vectors with per-group min-max quantisation.

        Parameters
        ----------
        values : (T, head_dim) float16

        Returns
        -------
        CompressedValue
        """
        T, D = values.shape
        gs = self.group_size
        n_groups = self.n_groups

        # pad to multiple of group_size
        pad = n_groups * gs - D
        if pad > 0:
            values = mx.pad(values, ((0, 0), (0, pad)))  # ty:ignore[invalid-argument-type]

        v_grouped = values.reshape(T, n_groups, gs)        # (T, G, gs)

        v_min = v_grouped.min(axis=-1, keepdims=True)      # (T, G, 1)
        v_max = v_grouped.max(axis=-1, keepdims=True)

        scale = (v_max - v_min)
        safe_scale = mx.where(scale > 1e-8, scale, mx.ones_like(scale))
        v_normed = (v_grouped - v_min) / safe_scale        # (T, G, gs) in [0,1]

        # uniform quantise to n_bins
        indices = mx.clip(
            (v_normed * (self.n_bins - 1) + 0.5).astype(mx.int32),
            0,
            self.n_bins - 1,
        ).astype(mx.uint8)                                  # (T, G, gs)

        # pack bits
        indices_flat = indices.reshape(T, n_groups * gs)
        packed = pack_bits(indices_flat, self.bits)          # (T, packed_D)

        scales = scale.squeeze(-1).astype(mx.float16)       # (T, G)
        zeros = v_min.squeeze(-1).astype(mx.float16)        # (T, G)

        return CompressedValue(
            packed=packed,
            scales=scales,
            zeros=zeros,
            bits=self.bits,
            head_dim=D,
            group_size=gs,
        )

    # ------------------------------------------------------------------
    def decompress(self, cv: CompressedValue) -> mx.array:
        """
        Decompress value vectors back to float16.

        Parameters
        ----------
        cv : CompressedValue

        Returns
        -------
        values : (T, head_dim) float16
        """
        T = cv.scales.shape[0]
        gs = cv.group_size
        n_groups = cv.scales.shape[1]
        D_padded = n_groups * gs

        indices_flat = unpack_bits(cv.packed, cv.bits, D_padded)  # (T, D_padded)
        indices_grouped = indices_flat.reshape(T, n_groups, gs)

        v_normed = indices_grouped.astype(mx.float16) / (self.n_bins - 1)

        scales = cv.scales[:, :, None]     # (T, G, 1)
        zeros  = cv.zeros[:, :, None]
        v_reconstructed = v_normed * scales + zeros   # (T, G, gs)
        v_flat = v_reconstructed.reshape(T, D_padded)

        return v_flat[:, :cv.head_dim]
