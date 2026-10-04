from __future__ import annotations

from pathlib import Path

import numpy as np

from dromia import dto
from dromia.pipeline.native_pose import NativePoseStore


def observation(frame_idx: int, obj_id: int, value: float) -> dto.PoseObservation:
    return dto.PoseObservation(
        frame_idx=frame_idx,
        obj_id=obj_id,
        keypoints_xy=np.full((17, 2), value, dtype=np.float32),
        confidence=np.full(17, value, dtype=np.float32),
        heatmaps=np.full((17, 4, 3), value, dtype=np.float32),
        heatmap_metadata={"heatmap_space": "pmpose_affine_heatmap"},
        crop_xyxy=np.asarray([0, 0, 3, 4], dtype=np.float32),
        bbox_xyxy=np.asarray([0, 0, 3, 4], dtype=np.float32),
        heatmap_confidence=np.full(17, value, dtype=np.float32),
    )


def test_native_pose_store_replays_float32_and_preserves_npz_contract(tmp_path: Path) -> None:
    (tmp_path / "pose").mkdir()
    store = NativePoseStore(tmp_path, [(2, 9), (1, 9)])
    store.add(observation(2, 9, 0.2))
    store.add(observation(1, 9, 0.1))

    replayed = store.get(1, 9)
    assert replayed is not None
    assert replayed.heatmaps is not None
    assert replayed.heatmaps.dtype == np.float32
    np.testing.assert_array_equal(replayed.heatmaps, np.full((17, 4, 3), 0.1, np.float32))

    path = store.finalize(tmp_path)
    assert path == tmp_path / "pose" / "pmpose_native_outputs.npz"
    with np.load(path, allow_pickle=False) as data:
        assert set(data.files) == {
            "frame_indices",
            "object_ids",
            "decoded_keypoints_xy",
            "crop_xyxy",
            "bbox_xyxy",
            "oks_confidence",
            "heatmap_confidence",
            "presence_probability",
            "visibility_probability",
            "normalized_localization_error",
            "native_heatmaps",
            "heatmap_metadata",
            "storage_format",
        }
        np.testing.assert_array_equal(data["frame_indices"], [1, 2])
        assert data["native_heatmaps"].dtype == np.float16
        np.testing.assert_array_equal(data["native_heatmaps"][0], np.float16(0.1))
    assert not list((tmp_path / "pose").glob("*.tmp.npy"))

