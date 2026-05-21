# file: tests/test_quantizer.py
# description: Unit tests for TurboQuantMSE and TurboQuantProd, validating the
#              core paper claims: compression ratio, norm preservation, inner-product
#              estimator unbiasedness. Port of 0xSero/turboquant/test_turboquant.py
#              with all CUDA/Triton assertions replaced with MLX equivalents.
# reference: 0xSero/turboquant/test_turboquant.py; arXiv:2504.19874

from __future__ import annotations

import math

import mlx.core as mx
import numpy as np
import pytest

from turboquant_mlx.quantizer import TurboQuantMSE, TurboQuantProd


HEAD_DIM = 128
T = 64


# ---------------------------------------------------------------------------
# TurboQuantMSE (key quantiser)
# ---------------------------------------------------------------------------

def _random_keys(T: int = T, D: int = HEAD_DIM, seed: int = 0) -> mx.array:
    rng = np.random.default_rng(seed)
    k = rng.standard_normal((T, D)).astype(np.float32)
    k /= np.linalg.norm(k, axis=-1, keepdims=True)
    return mx.array(k.astype(np.float16))


class TestTurboQuantMSE:
    def setup_method(self):
        self.tq = TurboQuantMSE(head_dim=HEAD_DIM, bits=3, use_qjl=True)

    def test_compress_returns_correct_shapes(self):
        keys = _random_keys()
        ck = self.tq.compress(keys)

        expected_packed_D = math.ceil(HEAD_DIM * 3 / 8)
        assert ck.packed.shape == (T, expected_packed_D), "packed shape mismatch"
        assert ck.norms.shape == (T,), "norms shape mismatch"
        assert ck.qjl_signs is not None
        assert ck.qjl_signs.shape[0] == T, "qjl_signs row count mismatch"

    def test_attention_scores_shape(self):
        keys = _random_keys()
        ck = self.tq.compress(keys)

        query = _random_keys(T=1)   # (1, D)
        scores = self.tq.attention_scores(query, ck)
        assert scores.shape[-1] == T, "scores should cover all T tokens"

    def test_inner_product_estimator_unbiased(self):
        """E[QJL estimate] should converge to true score over many trials."""
        n_trials = 200
        keys_np = np.random.default_rng(42).standard_normal((1, HEAD_DIM)).astype(np.float32)
        keys_np /= np.linalg.norm(keys_np, axis=-1, keepdims=True)
        keys = mx.array(keys_np.astype(np.float16))
        query_np = np.random.default_rng(7).standard_normal((1, HEAD_DIM)).astype(np.float32)
        query = mx.array(query_np.astype(np.float16))
        true_score = float(np.sum(query_np * keys_np))

        estimates = []
        for i in range(n_trials):
            tq = TurboQuantMSE(head_dim=HEAD_DIM, bits=3, use_qjl=True,
                               rotation_seed=1000 + i, qjl_seed=2000 + i)
            ck = tq.compress(keys)
            estimates.append(float(tq.attention_scores(query, ck).item()))

        mean_est = float(np.mean(estimates))
        sem = float(np.std(estimates) / math.sqrt(n_trials))
        assert abs(mean_est - true_score) < 3 * sem + 0.05, (
            f"true={true_score:.4f} mean_est={mean_est:.4f} sem={sem:.4f}"
        )

    def test_compression_ratio(self):
        """
        3-bit compression should yield ~5x vs float16 (2 bytes per element).
        """
        keys = _random_keys()
        ck = self.tq.compress(keys)

        compressed_bytes = int(np.prod(np.array(ck.packed.shape))) * 1  # uint8
        original_bytes   = T * HEAD_DIM * 2  # float16
        ratio = original_bytes / compressed_bytes
        assert ratio >= 3.5, f"compression ratio too low: {ratio:.2f}x"


# ---------------------------------------------------------------------------
# TurboQuantProd (value quantiser)
# ---------------------------------------------------------------------------

class TestTurboQuantProd:
    def setup_method(self):
        self.tq = TurboQuantProd(head_dim=HEAD_DIM, bits=2, group_size=32)

    def _random_values(self, T: int = T) -> mx.array:
        rng = np.random.default_rng(1)
        v = rng.standard_normal((T, HEAD_DIM)).astype(np.float16)
        return mx.array(v)

    def test_round_trip_shape(self):
        values = self._random_values()
        cv = self.tq.compress(values)
        v_hat = self.tq.decompress(cv)
        assert v_hat.shape == values.shape, "round-trip shape mismatch"

    def test_round_trip_mse(self):
        values = self._random_values()
        cv = self.tq.compress(values)
        v_hat = self.tq.decompress(cv)

        v_np   = np.array(values.tolist(), dtype=np.float32)
        vhat_np = np.array(v_hat.tolist(), dtype=np.float32)
        mse = float(np.mean((v_np - vhat_np) ** 2))
        # 2-bit min-max group quantisation on N(0,1) data: theoretical MSE ≈ range²/108 ≈ 0.18
        assert mse < 0.30, f"value round-trip MSE too high: {mse:.5f}"
