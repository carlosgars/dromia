from __future__ import annotations

import numpy as np

from dromia.models import crops as pose_crops


def test_masked_crop_maps_points_between_spaces() -> None:
    frame = np.zeros((40, 60, 3), dtype=np.uint8)
    mask = np.zeros((40, 60), dtype=np.uint8)
    mask[10:30, 20:50] = 1
    crop = pose_crops.build_masked_crop(
        frame,
        np.asarray([20, 10, 50, 30], dtype=np.float32),
        mask,
        padding=1.0,
        blur_kernel=5,
    )
    global_xy = np.asarray([[25, 15], [40, 20]], dtype=np.float32)
    local = crop.points_global_to_local(global_xy)
    np.testing.assert_allclose(local, [[5, 5], [20, 10]])
    np.testing.assert_allclose(crop.points_local_to_global(local), global_xy)
