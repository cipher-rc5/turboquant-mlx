# file: src/turboquant_mlx/rotation.py
# description: Random orthogonal rotation matrix (QR of Gaussian) and QJL residual
#              projection for unbiased inner-product estimation.
#              Direct port of 0xSero/turboquant rotation.py; CUDA removed, MLX substituted.
# reference: arXiv:2504.19874 Section 3; 0xSero/turboquant/turboquant/rotation.py

from __future__ import annotations

import math

import mlx.core as mx
import numpy as np


def random_orthogonal(dim: int, seed: int = 42) -> mx.array:
    """
    Generate a random orthogonal matrix Q of shape (dim, dim) via QR decomposition
    of a Gaussian matrix.  Stored as float16.

    Parameters
    ----------
    dim  : int - dimension (head_dim)
    seed : int - for reproducibility across encode/decode

    Returns
    -------
    Q : (dim, dim) float16 mlx array
    """
    rng = np.random.default_rng(seed)
    G = rng.standard_normal((dim, dim)).astype(np.float32)
    Q, _ = np.linalg.qr(G)
    return mx.array(Q.astype(np.float16))


def rotate(x: mx.array, Q: mx.array) -> mx.array:
    """
    Apply orthogonal rotation: x_rot = x @ Q^T.

    Rotated unit-vector coordinates concentrate at scale 1/sqrt(D).  The
    quantizer multiplies them by sqrt(D) before codebook lookup.

    Parameters
    ----------
    x : (..., D)   float16
    Q : (D, D)     float16

    Returns
    -------
    x_rot : (..., D) float16
    """
    return x @ Q.T


def rotate_query(q: mx.array, Q: mx.array) -> mx.array:
    """
    Rotate query in the same basis as keys: q_rot = q @ Q^T.
    During decode, rotate the query once instead of decompressing all keys.
    """
    return q @ Q.T


# ---------------------------------------------------------------------------
# QJL (Quantized Johnson-Lindenstrauss) residual correction
# ---------------------------------------------------------------------------

def random_qjl_matrix(dim: int, n_sketch: int | None = None, seed: int = 99) -> mx.array:
    """
    Generate the QJL sketch matrix S of shape (dim, n_sketch).
    n_sketch defaults to dim (square sketch).

    Each column is a random Gaussian vector; sign bits of S^T x give an
    unbiased estimator of inner products.
    """
    if n_sketch is None:
        n_sketch = dim
    rng = np.random.default_rng(seed)
    S = rng.standard_normal((dim, n_sketch)).astype(np.float32)
    # normalise columns
    S /= np.linalg.norm(S, axis=0, keepdims=True)
    return mx.array(S.astype(np.float16))


def qjl_encode(residual: mx.array, S: mx.array) -> mx.array:
    """
    Compute QJL sign sketch of the residual.

    Parameters
    ----------
    residual : (..., D)       float16 - key minus quantised reconstruction
    S        : (D, n_sketch)  float16 - sketch matrix

    Returns
    -------
    signs : (..., n_sketch) bool (stored as uint8, 0/1)
    """
    projected = residual @ S          # (..., n_sketch)
    return (projected > 0).astype(mx.uint8)


def qjl_decode_correction(
    q_rot: mx.array,
    signs: mx.array,
    S: mx.array,
) -> mx.array:
    """
    Compute approximate inner product correction from QJL sketch.

    Inner product estimate: sum_j sign_j * (q_rot @ s_j)
    This is an unbiased estimator of <q_rot, residual>.

    Parameters
    ----------
    q_rot  : (..., D)           float16 - already-rotated query
    signs  : (T, n_sketch)      uint8   - stored sign bits per token
    S      : (D, n_sketch)      float16 - sketch matrix

    Returns
    -------
    correction : (..., T) float16
    """
    # q_rot @ S : (..., n_sketch)
    q_proj = q_rot @ S
    # signs: (T, n_sketch), cast to float for matmul
    s_f = (signs.astype(mx.float16) * 2 - 1)  # {0,1} -> {-1, +1}
    # dot product over sketch dimension: (..., n_sketch) x (T, n_sketch)^T
    # = (..., T)
    return q_proj @ s_f.T
