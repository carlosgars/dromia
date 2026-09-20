from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from dromia.models import pmpose


def test_extract_pmpose_diagnostics_preserves_heads_and_affine() -> None:
    count = 17
    instances = SimpleNamespace(
        keypoints=np.arange(count * 2, dtype=np.float32).reshape(1, count, 2),
        keypoint_scores=np.linspace(0.1, 0.9, count, dtype=np.float32),
        keypoints_conf=np.linspace(0.2, 0.8, count, dtype=np.float32),
        keypoints_probs=np.linspace(0.3, 0.7, count, dtype=np.float32),
        keypoints_visible=np.linspace(0.4, 0.6, count, dtype=np.float32),
        keypoints_error=np.linspace(-0.2, 0.2, count, dtype=np.float32),
    )
    result = SimpleNamespace(
        pred_instances=instances,
        pred_fields=SimpleNamespace(heatmaps=np.ones((count, 64, 48), dtype=np.float32)),
        metainfo={
            "input_center": np.asarray([120, 220], dtype=np.float32),
            "input_scale": np.asarray([180, 240], dtype=np.float32),
            "input_size": (192, 256),
        },
    )

    diagnostics = pmpose.extract_pmpose_diagnostics(result)

    np.testing.assert_allclose(diagnostics.presence_probability, instances.keypoints_probs)
    np.testing.assert_allclose(diagnostics.visibility_probability, instances.keypoints_visible)
    np.testing.assert_allclose(diagnostics.normalized_error, instances.keypoints_error)
    np.testing.assert_allclose(diagnostics.input_center_xy, [120, 220])
    np.testing.assert_allclose(diagnostics.input_scale_xy, [180, 240])
    np.testing.assert_allclose(diagnostics.input_size_xy, [192, 256])
