import math

import mlx.core as mx
import numpy as np
import pytest

from turboquant_mlx.quantizer import TurboQuantMSE


@pytest.mark.parametrize("d,bits,cos_threshold", [
    (160, 4, 0.95),
    (160, 3, 0.90),
    (256, 4, 0.96),
    (256, 3, 0.92),
])
def test_unit_vector_roundtrip_preserves_direction(d, bits, cos_threshold):
    """A unit vector compressed and decoded should retain cosine similarity > threshold."""
    rng = np.random.default_rng(0)
    keys = rng.standard_normal((128, d)).astype(np.float32)
    keys /= np.linalg.norm(keys, axis=-1, keepdims=True)
    keys_mx = mx.array(keys.astype(np.float16))

    tq = TurboQuantMSE(head_dim=d, bits=bits, use_qjl=False)
    ck = tq.compress(keys_mx)

    scores = np.array(tq.attention_scores(keys_mx, ck).tolist(), dtype=np.float32)
    diag = np.diag(scores) / np.array(ck.norms.tolist(), dtype=np.float32)
    mean_cos = diag.mean()
    assert mean_cos > cos_threshold, (
        f"d={d} bits={bits}: mean cosine {mean_cos:.4f} < {cos_threshold}"
    )
